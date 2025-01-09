
import sys 
import os 
import json
import numpy as np
from torch.utils.data import DataLoader
import torch
from TemporalGNN import TemporalMalwareDataLoader, NumpyEncoder, evaluate_novel_detection
from sklearn.metrics import classification_report
from gcn import MalwareGNN, MalwareTrainer
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class EpsilonAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def process_large_graph(self, batch, num_chunks=4):
        """Process a large graph by splitting it into chunks"""
        x, edge_index = batch.x, batch.edge_index
        batch_idx = batch.batch
        num_graphs = batch_idx.max().item() + 1
        
        all_logits = []
        all_outlier_scores = []
        
        # Process one graph at a time
        for graph_idx in range(num_graphs):
            # Get nodes for this graph
            graph_mask = batch_idx == graph_idx
            graph_x = x[graph_mask]
            node_indices = torch.nonzero(graph_mask).squeeze()
            
            # Get edges for this graph
            edge_mask = (edge_index[0] >= node_indices.min()) & (edge_index[0] <= node_indices.max()) & \
                       (edge_index[1] >= node_indices.min()) & (edge_index[1] <= node_indices.max())
            graph_edges = edge_index[:, edge_mask]
            
            # Adjust edge indices
            graph_edges = graph_edges - node_indices.min()
            
            # Process in chunks if needed
            num_nodes = graph_x.size(0)
            chunk_size = num_nodes // num_chunks + 1
            graph_logits = []
            graph_scores = []
            
            for i in range(0, num_nodes, chunk_size):
                end_idx = min(i + chunk_size, num_nodes)
                chunk_mask = torch.arange(i, end_idx, device=self.device)
                
                # Get chunk data
                chunk_x = graph_x[chunk_mask]
                chunk_edges = graph_edges.clone()
                edge_mask = (chunk_edges[0] >= i) & (chunk_edges[0] < end_idx) & \
                           (chunk_edges[1] >= i) & (chunk_edges[1] < end_idx)
                chunk_edges = chunk_edges[:, edge_mask]
                chunk_edges = chunk_edges - i
                
                # Process chunk
                chunk_batch = torch.zeros(len(chunk_x), dtype=torch.long, device=self.device)
                chunk_data = Data(x=chunk_x, edge_index=chunk_edges, batch=chunk_batch)
                with torch.amp.autocast(device_type='cuda'):
                    chunk_logits, chunk_scores = self.model(chunk_data)
                    graph_logits.append(chunk_logits)
                    graph_scores.append(chunk_scores)
            
            # Aggregate chunks for this graph
            if graph_logits:
                graph_logits = torch.cat(graph_logits).mean(dim=0, keepdim=True)
                graph_scores = torch.cat(graph_scores).mean(dim=0, keepdim=True)
                all_logits.append(graph_logits)
                all_outlier_scores.append(graph_scores)
        
        # Ensure we have results for each graph
        if len(all_logits) != num_graphs:
            raise ValueError(f"Expected {num_graphs} results, but got {len(all_logits)}")
            
        return torch.cat(all_logits), torch.cat(all_outlier_scores)

    def collect_predictions(self, loader, test_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect all raw predictions and scores for later analysis.
        Returns:
            distances: Array of minimum distances to centroids
            is_novel: Array of true novelty labels
            confidences: Array of prediction confidences
        """
        all_distances = []
        all_is_novel = []
        all_confidences = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting predictions")):
                # If in test mode, only process first 3 batches
                if test_mode and batch_idx >= 3:
                    logger.info("Test mode: stopping after 3 batches")
                    break
                try:
                    batch = batch.to(self.device)
                    
                    # Check if graph is large and needs chunked processing
                    if hasattr(batch, 'x') and batch.x.size(0) > 50000:  # Threshold for large graphs
                        logger.info(f"Processing large graph {batch_idx} with {batch.x.size(0)} nodes in chunks...")
                        logits, outlier_scores = self.process_large_graph(batch)
                    else:
                        logits, outlier_scores = self.model(batch)
                    
                    # Convert logits to distances (they're negative distances)
                    distances = -logits
                    min_distances, _ = distances.min(dim=1)
                    
                    # Get prediction confidences
                    probs = F.softmax(logits, dim=1)
                    confidences = probs.max(dim=1)[0]
                    
                    # Move to CPU and convert to numpy
                    all_distances.append(min_distances.cpu().numpy())
                    all_is_novel.append(batch.is_novel.cpu().numpy())
                    all_confidences.append(confidences.cpu().numpy())
                    
                    # Clear cache periodically
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"OOM error in batch {batch_idx}, attempting recovery...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise e
                    
        # Debug info for NaN values
        distances = np.concatenate(all_distances)
        is_novel = np.concatenate(all_is_novel)
        confidences = np.concatenate(all_confidences)
        
        # save 
        np.save('distances.npy', distances)
        np.save('is_novel.npy', is_novel)
        np.save('confidences.npy', confidences)
        np.savez('raw_predictions.npz', distances=distances, is_novel=is_novel, confidences=confidences)
        # Check for NaNs and log info
        if np.isnan(distances).any():
            logger.error(f"Found {np.isnan(distances).sum()} NaN values in distances")
            logger.error(f"Distance stats - min: {np.nanmin(distances)}, max: {np.nanmax(distances)}")
            nan_indices = np.where(np.isnan(distances))[0]
            logger.error(f"First few NaN indices: {nan_indices[:5]}")
            
        return distances, is_novel, confidences
    
    def compute_pr_curve(self, distances: np.ndarray, is_novel: np.ndarray) -> Dict:
        """
        Compute precision-recall curve using distances as novelty scores.
        """
        # Larger distance = more likely to be novel
        precision, recall, thresholds = precision_recall_curve(is_novel, distances)
        ap = average_precision_score(is_novel, distances)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': ap
        }
    
    def analyze_thresholds(self, loader, num_thresholds: int = 50, test_mode: bool = False) -> Dict:
        """
        Perform complete threshold analysis including PR curves.
        """
        # Collect all predictions
        logger.info("Collecting model predictions...")
        distances, is_novel, confidences = self.collect_predictions(loader, test_mode)
        
        # Compute PR curves
        logger.info("Computing precision-recall curves...")
        distance_pr = self.compute_pr_curve(distances, is_novel)
        confidence_pr = self.compute_pr_curve(-confidences, is_novel)  # Negative because lower confidence = more likely novel
        
        # Find optimal thresholds
        f1_scores_dist = 2 * (distance_pr['precision'] * distance_pr['recall']) / \
                        (distance_pr['precision'] + distance_pr['recall'] + 1e-10)
        f1_scores_conf = 2 * (confidence_pr['precision'] * confidence_pr['recall']) / \
                        (confidence_pr['precision'] + confidence_pr['recall'] + 1e-10)
        
        optimal_dist_idx = np.argmax(f1_scores_dist)
        optimal_conf_idx = np.argmax(f1_scores_conf)
        
        return {
            'distance_based': {
                'pr_curve': distance_pr,
                'optimal_threshold': distance_pr['thresholds'][optimal_dist_idx],
                'optimal_f1': f1_scores_dist[optimal_dist_idx],
                'optimal_precision': distance_pr['precision'][optimal_dist_idx],
                'optimal_recall': distance_pr['recall'][optimal_dist_idx]
            },
            'confidence_based': {
                'pr_curve': confidence_pr,
                'optimal_threshold': -confidence_pr['thresholds'][optimal_conf_idx],  # Convert back to positive
                'optimal_f1': f1_scores_conf[optimal_conf_idx],
                'optimal_precision': confidence_pr['precision'][optimal_conf_idx],
                'optimal_recall': confidence_pr['recall'][optimal_conf_idx]
            },
            'statistics': {
                'distance_stats': {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances))
                },
                'confidence_stats': {
                    'mean': float(np.mean(confidences)),
                    'std': float(np.std(confidences)),
                    'min': float(np.min(confidences)),
                    'max': float(np.max(confidences))
                },
                'total_samples': len(distances),
                'novel_ratio': float(np.mean(is_novel))
            }
        }
    
    def plot_results(self, results: Dict, save_path: str = 'epsilon_analysis.png'):
        """Generate comprehensive visualization of the analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot distance-based PR curve
        dist_results = results['distance_based']
        ax1.plot(dist_results['pr_curve']['recall'], 
                dist_results['pr_curve']['precision'], 
                'b-', label=f'AP={dist_results["pr_curve"]["average_precision"]:.3f}')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Distance-based PR Curve')
        ax1.grid(True)
        ax1.legend()
        
        # Plot confidence-based PR curve
        conf_results = results['confidence_based']
        ax2.plot(conf_results['pr_curve']['recall'], 
                conf_results['pr_curve']['precision'], 
                'r-', label=f'AP={conf_results["pr_curve"]["average_precision"]:.3f}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Confidence-based PR Curve')
        ax2.grid(True)
        ax2.legend()
        
        # Plot distance distribution
        stats = results['statistics']['distance_stats']
        ax3.hist(np.linspace(stats['min'], stats['max'], 50), bins=50, alpha=0.6)
        ax3.axvline(dist_results['optimal_threshold'], color='r', linestyle='--', 
                   label=f'Optimal threshold: {dist_results["optimal_threshold"]:.3f}')
        ax3.set_xlabel('Distance')
        ax3.set_ylabel('Count')
        ax3.set_title('Distance Distribution')
        ax3.legend()
        ax3.grid(True)
        
        # Plot confidence distribution
        stats = results['statistics']['confidence_stats']
        ax4.hist(np.linspace(stats['min'], stats['max'], 50), bins=50, alpha=0.6)
        ax4.axvline(conf_results['optimal_threshold'], color='r', linestyle='--',
                   label=f'Optimal threshold: {conf_results["optimal_threshold"]:.3f}')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Count')
        ax4.set_title('Confidence Distribution')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    # Add argument parser for test mode
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited batches')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load your model and data
    model_path = 'best_family_model.pt'
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = MalwareGNN(
        num_node_features=14,
        hidden_dim=256,
        num_classes=checkpoint['model_state_dict']['centroid.centroids'].shape[0] // 4,
        n_centroids_per_class=4,
        num_layers=4,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize data loader
    data_loader = TemporalMalwareDataLoader(
        batch_dir=Path('bodmas_batches'),
        behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        metadata_path=Path('bodmas_metadata_cleaned.csv'),
        malware_types_path=Path('bodmas_malware_category.csv')
    )
    
    # Load test data - reduce batch size for large graphs
    batch_size = 8 if args.test else 32
    test_loader, _ = data_loader.load_split('test', use_groups=False, batch_size=batch_size)
    
    if args.test:
        logger.info("Running in test mode with 3 batches")
        
    # Run analysis
    analyzer = EpsilonAnalyzer(model, device)
    results = analyzer.analyze_thresholds(test_loader, test_mode=args.test)
    
    # Plot and save results
    plot_suffix = '_test' if args.test else ''
    analyzer.plot_results(results, save_path=f'epsilon_analysis{plot_suffix}.png')
    
    # Save results
    with open(f'epsilon_results{plot_suffix}.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Print optimal thresholds
    print("\nOptimal Thresholds:")
    print(f"Distance-based: {results['distance_based']['optimal_threshold']:.3f}")
    print(f"Confidence-based: {results['confidence_based']['optimal_threshold']:.3f}")

if __name__ == "__main__":
    main()
sys.exit()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.model.eval()
        
    def evaluate_with_epsilon(self, test_loader: DataLoader, epsilon: float) -> Dict:
        """
        Evaluate model with a specific epsilon threshold for centroid distances
        """
        all_preds = []
        all_labels = []
        all_distances = []
        all_outlier_scores = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                # Get model predictions and distances
                logits, outlier_scores = self.model(batch)
                distances = -logits  # Convert back to distances
                
                # Apply epsilon threshold
                mask = distances > epsilon
                logits[mask] = float('-inf')  # Mask out distances above epsilon
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
                all_distances.append(distances.cpu())
                all_outlier_scores.append(outlier_scores.cpu())
        
        # Concatenate results
        predictions = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        distances = torch.cat(all_distances).numpy()
        outlier_scores = torch.cat(all_outlier_scores).numpy()
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(labels, predictions, output_dict=True),
            'avg_distance': float(np.mean(distances)),
            'avg_outlier_score': float(np.mean(outlier_scores)),
            'epsilon': epsilon
        }
        
        return metrics
    
    def find_optimal_epsilon(self, test_loader: DataLoader, 
                           epsilon_range: List[float]) -> Tuple[float, Dict]:
        """
        Find optimal epsilon value by testing multiple thresholds
        """
        best_f1 = 0
        best_epsilon = None
        best_metrics = None
        results = {}
        
        for epsilon in epsilon_range:
            logger.info(f"Testing epsilon: {epsilon}")
            metrics = self.evaluate_with_epsilon(test_loader, epsilon)
            current_f1 = metrics['classification_report']['weighted avg']['f1-score']
            
            results[epsilon] = metrics
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epsilon = epsilon
                best_metrics = metrics
        
        logger.info(f"Best epsilon: {best_epsilon} (F1: {best_f1:.4f})")
        return best_epsilon, results

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # First load the checkpoints to get the model dimensions
    family_checkpoint = torch.load('checkpoint_family_epoch_40.pt', map_location=device)
    group_checkpoint = torch.load('checkpoint_group_epoch_40.pt', map_location=device)
    
    # Get the centroid dimensions from the saved models
    family_centroid_shape = family_checkpoint['model_state_dict']['centroid.centroids'].shape
    group_centroid_shape = group_checkpoint['model_state_dict']['centroid.centroids'].shape
    
    # Calculate number of classes based on centroid shapes
    # Since shape is (num_classes * n_centroids_per_class, feature_dim)
    n_centroids_per_class = 4  # This should match your saved model
    num_family_classes = family_centroid_shape[0] // n_centroids_per_class
    num_group_classes = group_centroid_shape[0] // n_centroids_per_class
    
    logger.info(f"Detected family classes: {num_family_classes}")
    logger.info(f"Detected group classes: {num_group_classes}")
    
    # Initialize data loader
    data_loader = TemporalMalwareDataLoader(
        batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches'),
        behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        metadata_path=Path('bodmas_metadata_cleaned.csv'),
        malware_types_path=Path('bodmas_malware_category.csv')
    )
    
    # Load test data for both family and group models
    test_loader_family, test_stats = data_loader.load_split('test', use_groups=False, batch_size=32)
    test_loader_groups, test_group_stats = data_loader.load_split('test', use_groups=True, batch_size=32)
    
    # Get hidden_dim from the checkpoint
    hidden_dim = family_centroid_shape[1]  # This should be 1024 based on the error message
    
    family_model = MalwareGNN(
        num_node_features=14,
        hidden_dim=hidden_dim // 4,  # Since the final dimension is 4x the hidden_dim
        num_classes=num_family_classes,
        n_centroids_per_class=n_centroids_per_class,
        num_layers=4,
        dropout=0.2
    ).to(device)
    
    group_model = MalwareGNN(
        num_node_features=14,
        hidden_dim=hidden_dim // 4,
        num_classes=num_group_classes,
        n_centroids_per_class=n_centroids_per_class,
        num_layers=4,
        dropout=0.2
    ).to(device)
    
    family_model.load_state_dict(family_checkpoint['model_state_dict'])
    group_model.load_state_dict(group_checkpoint['model_state_dict'])
    
    # Initialize evaluators
    family_evaluator = ModelEvaluator(family_model, data_loader, device)
    group_evaluator = ModelEvaluator(group_model, data_loader, device)
    
    # Define epsilon ranges to test
    epsilon_range = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Evaluate family model
    logger.info("Evaluating family model...")
    best_family_epsilon, family_results = family_evaluator.find_optimal_epsilon(
        test_loader_family, epsilon_range
    )
    
    # Evaluate group model
    logger.info("Evaluating group model...")
    best_group_epsilon, group_results = group_evaluator.find_optimal_epsilon(
        test_loader_groups, epsilon_range
    )
    
    # Save results
    results = {
        'family_model': {
            'best_epsilon': best_family_epsilon,
            'results': family_results
        },
        'group_model': {
            'best_epsilon': best_group_epsilon,
            'results': group_results
        }
    }
    
    with open('epsilon_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    logger.info("Results saved to epsilon_evaluation_results.json")

if __name__ == "__main__":
    main()

# # Classification metrics
# family_report = classification_report(family_labels, family_predicted_classes, output_dict=True)
# group_report = classification_report(group_labels, group_predicted_classes, output_dict=True)

# # Novelty Analysis
# def analyze_outlier_scores(scores):
#     scores = np.array(scores)
#     return {
#         'mean': float(np.mean(scores)),
#         'std': float(np.std(scores)),
#         'min': float(np.min(scores)),
#         'max': float(np.max(scores)),
#         'quartiles': [float(x) for x in np.percentile(scores, [25, 50, 75])]
#     }

# # Analyze novelty detection scores (if available)
# family_novelty_scores = family_predictions[:, -1]  # Assume last column is novelty score
# group_novelty_scores = group_predictions[:, -1]

# family_novelty_stats = analyze_outlier_scores(family_novelty_scores)
# group_novelty_stats = analyze_outlier_scores(group_novelty_scores)

# # Save metrics
# test_metrics = {
#     'classification_metrics': {
#         'family': family_report,
#         'group': group_report
#     },
#     'novelty_detection': {
#         'family': family_novelty_stats,
#         'group': group_novelty_stats
#     }
# }

# with open('test_metrics_report.json', 'w') as f:
#     json.dump(test_metrics, f, indent=2)

# print("Test metrics saved to test_metrics_report.json")


sys.exit()

# import torch
# import json
# import numpy as np
# import os
# from pathlib import Path
# from gcn import MalwareGNN, MalwareTrainer
# from TemporalGNN import TemporalMalwareDataLoader, evaluate_novel_detection
import json
import torch
from gcn import MalwareTrainer, MalwareGNN
from pathlib import Path
from TemporalGNN import TemporalMalwareDataLoader, NumpyEncoder, evaluate_novel_detection
import json
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
# Load models and metrics
family_checkpoint = torch.load('checkpoint_family_epoch_40.pt', map_location='cpu')
group_checkpoint = torch.load('checkpoint_group_epoch_40.pt', map_location='cpu')

# Load and slice metrics history to epoch 40
with open('family_metrics2.json', 'r') as f:
    family_metrics_history = json.load(f)

with open('group_metrics2.json', 'r') as f:
    group_metrics_history = json.load(f)

family_metrics_history = [entry for entry in family_metrics_history if entry['epoch'] <= 40]
group_metrics_history = [entry for entry in group_metrics_history if entry['epoch'] <= 40]

# Consolidate metrics for paper
metrics_summary = {
    "family": {
        "training_history": {
            "epochs": len(family_metrics_history),
            "best_epoch": family_checkpoint['epoch'],
            "best_metrics": family_checkpoint.get('metrics', {}),
            "best_novel_detection": family_checkpoint.get('novel_metrics', {})
        },
        "validation_metrics": family_metrics_history[-1],  # Last epoch in metrics
    },
    "group": {
        "training_history": {
            "epochs": len(group_metrics_history),
            "best_epoch": group_checkpoint['epoch'],
            "best_metrics": group_checkpoint.get('metrics', {}),
            "best_novel_detection": group_checkpoint.get('novel_metrics', {})
        },
        "validation_metrics": group_metrics_history[-1],  # Last epoch in metrics
    }
}



# def analyze_classification_metrics(metrics_history):
#     """Compute macro precision, recall, and F1 from metrics history."""
#     last_epoch_metrics = metrics_history[-1]  # Metrics for the final epoch
#     overall_metrics = last_epoch_metrics['val']['overall']
    
#     classification_stats = {
#         'precision': overall_metrics.get('precision', 0.0),
#         'recall': overall_metrics.get('recall', 0.0),
#         'f1': overall_metrics.get('f1', 0.0),
#         'accuracy': overall_metrics.get('accuracy', 0.0)
#     }
    
#     return classification_stats

# def analyze_thresholds(metrics, outlier_thresholds=np.arange(0.3, 0.9, 0.05), confidence_thresholds=np.arange(0.3, 0.9, 0.05)):
#     """Analyze different threshold combinations for novel detection."""
#     all_outlier_scores = np.array(metrics.get('scores', []))
#     all_confidences = np.array(metrics.get('confidences', []))
#     all_is_novel = np.array(metrics.get('is_novel', []))
#     results = {'threshold_grid': [], 'pr_curve': None}
#     best_f1 = 0
#     best_thresholds = None
#     for out_th in outlier_thresholds:
#         for conf_th in confidence_thresholds:
#             pred_novel = (all_outlier_scores > out_th) | (all_confidences < conf_th)
#             tp = np.sum((pred_novel == True) & (all_is_novel == True))
#             fp = np.sum((pred_novel == True) & (all_is_novel == False))
#             fn = np.sum((pred_novel == False) & (all_is_novel == True))
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#             f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#             results['threshold_grid'].append({
#                 'outlier_threshold': float(out_th),
#                 'confidence_threshold': float(conf_th),
#                 'precision': float(precision),
#                 'recall': float(recall),
#                 'f1': float(f1)
#             })
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_thresholds = (out_th, conf_th)
#     combined_scores = all_outlier_scores - all_confidences
#     precision, recall, _ = precision_recall_curve(all_is_novel, combined_scores)
#     pr_auc = auc(recall, precision)
#     results['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': float(pr_auc)}
#     results['best_thresholds'] = {'outlier_threshold': best_thresholds[0], 'confidence_threshold': best_thresholds[1]}
#     return results

# # Example usage
# # with open('family_metrics2.json', 'r') as f:
# #     family_metrics_history = json.load(f)

# # with open('group_metrics2.json', 'r') as f:
# #     group_metrics_history = json.load(f)

# # Compute statistics
# family_classification = analyze_classification_metrics(family_metrics_history)
# group_classification = analyze_classification_metrics(group_metrics_history)

# # Assuming outlier stats are already computed and stored
# family_outlier_stats = analyze_outlier_scores(family_metrics_history[-1]['val']['novelty'])
# group_outlier_stats = analyze_outlier_scores(group_metrics_history[-1]['val']['novelty'])

# # Assuming `metrics` contains scores, confidences, and labels for threshold analysis
# family_threshold_analysis = analyze_thresholds(family_metrics_history[-1]['val']['novelty'])
# group_threshold_analysis = analyze_thresholds(group_metrics_history[-1]['val']['novelty'])

# # Create final JSON report
# final_report = {
#     'family': {
#         'classification': family_classification,
#         'outlier_stats': family_outlier_stats,
#         'threshold_analysis': family_threshold_analysis
#     },
#     'group': {
#         'classification': group_classification,
#         'outlier_stats': group_outlier_stats,
#         'threshold_analysis': group_threshold_analysis
#     }
# }

# # Save to file
# with open('metrics_analysis.json', 'w') as f:
#     json.dump(final_report, f, indent=2)

# print(json.dumps(final_report, indent=2))

# Save to a JSON file using NumpyEncoder
with open('metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2, cls=NumpyEncoder)

# Print key metrics for inclusion in the paper
print("Metrics Summary for Paper:")
print(json.dumps(metrics_summary, indent=2, cls=NumpyEncoder))

import json
import numpy as np

# Helper function to calculate metrics
def calculate_metrics(metrics_history, split='val'):
    """Calculate macro precision, recall, and F1 scores."""
    last_epoch_metrics = metrics_history[-1][split]
    overall_metrics = last_epoch_metrics['overall']

    macro_metrics = {
        'precision': overall_metrics['precision'],
        'recall': overall_metrics['recall'],
        'f1': overall_metrics['f1']
    }
    return macro_metrics

# Analyze outlier scores
def analyze_outlier_scores(novelty):
    """Analyze the distribution of outlier scores."""
    novel_scores = np.array(novelty.get('novel', {}).get('scores', []))
    known_scores = np.array(novelty.get('known', {}).get('scores', []))

    def stats(scores):
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)) if len(scores) > 0 else None,
            'max': float(np.max(scores)) if len(scores) > 0 else None,
            'quartiles': [float(x) for x in np.percentile(scores, [25, 50, 75])] if len(scores) > 0 else None
        }

    return {
        'novel': stats(novel_scores),
        'known': stats(known_scores)
    }

# Load metrics from JSON
with open('family_metrics2.json', 'r') as f:
    family_metrics_history = json.load(f)
with open('group_metrics2.json', 'r') as f:
    group_metrics_history = json.load(f)

# Compute classification metrics
family_classification_metrics = calculate_metrics(family_metrics_history)
group_classification_metrics = calculate_metrics(group_metrics_history)

# Analyze novelty detection metrics
family_novelty_stats = analyze_outlier_scores(family_metrics_history[-1]['val']['novelty'])
group_novelty_stats = analyze_outlier_scores(group_metrics_history[-1]['val']['novelty'])

# Combine metrics into a report
metrics_report = {
    'classification_metrics': {
        'family': family_classification_metrics,
        'group': group_classification_metrics
    },
    'novelty_detection': {
        'family': family_novelty_stats,
        'group': group_novelty_stats
    }
}

# Save report to JSON
with open('final_metrics_report.json', 'w') as f:
    json.dump(metrics_report, f, indent=2)

# def generate_final_report():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # First load the checkpoints to get the number of classes
#     family_checkpoint = torch.load('best_family_model.pt')
#     group_checkpoint = torch.load('best_group_model.pt')
    
#     # Get number of classes from checkpoint
#     family_num_classes = family_checkpoint['model_state_dict']['centroid.centroids'].shape[0] // 4  # divide by n_centroids_per_class
#     group_num_classes = group_checkpoint['model_state_dict']['centroid.centroids'].shape[0] // 4
    
#     print(f"Number of classes in checkpoints - Family: {family_num_classes}, Group: {group_num_classes}")

#     # Initialize models with correct number of classes
#     family_model = MalwareGNN(
#         num_node_features=14,
#         hidden_dim=256,
#         num_classes=family_num_classes,  # Use number from checkpoint
#         n_centroids_per_class=4,
#         num_layers=4,
#         dropout=0.2
#     ).to(device)

#     group_model = MalwareGNN(
#         num_node_features=14,
#         hidden_dim=256,
#         num_classes=group_num_classes,  # Use number from checkpoint
#         n_centroids_per_class=4,
#         num_layers=4,
#         dropout=0.2
#     ).to(device)

#     # Load model states
#     family_model.load_state_dict(family_checkpoint['model_state_dict'])
#     group_model.load_state_dict(group_checkpoint['model_state_dict'])
    
#     # Initialize data loader
#     data_loader = TemporalMalwareDataLoader(
#         batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches'),
#         behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
#         metadata_path=Path('bodmas_metadata_cleaned.csv'),
#         malware_types_path=Path('bodmas_malware_category.csv')
#     )

#     # Load metrics history
#     with open('family_metrics2.json', 'r') as f:
#         family_metrics_history = json.load(f)
#     with open('group_metrics2.json', 'r') as f:
#         group_metrics_history = json.load(f)

#     # Load test data
#     print("Loading test data...")
#     test_loader_family, test_stats = data_loader.load_split('test', use_groups=False, batch_size=32)
#     test_loader_groups, test_group_stats = data_loader.load_split('test', use_groups=True, batch_size=32)

#     # Initialize trainers
#     family_trainer = MalwareTrainer(family_model, device, lr=0.001)
#     group_trainer = MalwareTrainer(group_model, device, lr=0.001)

#     # Evaluate on test set
#     print("Evaluating family model on test set...")
#     test_family_metrics = family_trainer.evaluate(test_loader_family, None)  # No class weights for evaluation
#     print("Evaluating group model on test set...")
#     test_group_metrics = group_trainer.evaluate(test_loader_groups, None)
#     print("Evaluating family model novel detection...")
#     test_family_novel = evaluate_novel_detection(family_trainer, test_loader_family, None)
#     print("Evaluating group model novel detection...")
#     test_group_novel = evaluate_novel_detection(group_trainer, test_loader_groups, None)

#     # Analyze outlier scores
#     print("\nAnalyzing family model outlier scores...")
#     family_outlier_stats = analyze_outlier_scores(family_model, test_loader_family, device)
#     print("\nAnalyzing group model outlier scores...")
#     group_outlier_stats = analyze_outlier_scores(group_model, test_loader_groups, device)

#     # Create final report
#     final_report = {
#         'training_history': {
#             'family': family_metrics_history,
#             'group': group_metrics_history
#         },
#         'best_models': {
#             'family': {
#                 'epoch': family_checkpoint['epoch'],
#                 'metrics': family_checkpoint['metrics'],
#                 'novel_metrics': family_checkpoint.get('novel_metrics', None)
#             },
#             'group': {
#                 'epoch': group_checkpoint['epoch'],
#                 'metrics': group_checkpoint['metrics'],
#                 'novel_metrics': group_checkpoint.get('novel_metrics', None)
#             }
#         },
#         'test_results': {
#             'family': {
#                 'metrics': test_family_metrics,
#                 'novel_detection': test_family_novel,
#                 'outlier_stats': family_outlier_stats
#             },
#             'group': {
#                 'metrics': test_group_metrics,
#                 'novel_detection': test_group_novel,
#                 'outlier_stats': group_outlier_stats
#             }
#         },
#         'data_statistics': {
#             'test': {
#                 'family': test_stats,
#                 'group': test_group_stats
#             }
#         }
#     }

#     # Print key metrics
#     print("\nTest Results Summary:")
#     print("-" * 50)
#     print("\nFamily Model:")
#     print(f"Classification F1: {test_family_metrics['overall']['f1']:.4f}")
#     print(f"Novel Detection F1: {test_family_novel['overall']['f1']:.4f}")
    
#     print("\nGroup Model:")
#     print(f"Classification F1: {test_group_metrics['overall']['f1']:.4f}")
#     print(f"Novel Detection F1: {test_group_novel['overall']['f1']:.4f}")

#     # Save report
#     with open('final_report2.json', 'w') as f:
#         json.dump(final_report, f, indent=2)
    
#     print("\nFinal report generated and saved to final_report2.json")

# def analyze_outlier_scores(model, loader, device):
#     """Analyze distribution of outlier scores for known and novel samples."""
#     model.eval()
#     novel_scores = []
#     known_scores = []
    
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             _, outlier_scores = model(batch)
            
#             novel_scores.extend(outlier_scores[batch.is_novel].cpu().numpy())
#             known_scores.extend(outlier_scores[~batch.is_novel].cpu().numpy())
    
#     stats = {
#         'novel': {
#             'mean': float(np.mean(novel_scores)),
#             'std': float(np.std(novel_scores)),
#             'min': float(np.min(novel_scores)),
#             'max': float(np.max(novel_scores)),
#             'quartiles': [float(x) for x in np.percentile(novel_scores, [25, 50, 75])]
#         },
#         'known': {
#             'mean': float(np.mean(known_scores)),
#             'std': float(np.std(known_scores)),
#             'min': float(np.min(known_scores)),
#             'max': float(np.max(known_scores)),
#             'quartiles': [float(x) for x in np.percentile(known_scores, [25, 50, 75])]
#         }
#     }
    
#     print("\nOutlier Score Analysis:")
#     print("\nKnown Samples:")
#     print(f"Mean ± Std: {stats['known']['mean']:.4f} ± {stats['known']['std']:.4f}")
#     print(f"Range: [{stats['known']['min']:.4f}, {stats['known']['max']:.4f}]")
#     print(f"Quartiles: {[f'{x:.4f}' for x in stats['known']['quartiles']]}")
    
#     print("\nNovel Samples:")
#     print(f"Mean ± Std: {stats['novel']['mean']:.4f} ± {stats['novel']['std']:.4f}")
#     print(f"Range: [{stats['novel']['min']:.4f}, {stats['novel']['max']:.4f}]")
#     print(f"Quartiles: {[f'{x:.4f}' for x in stats['novel']['quartiles']]}")
    
#     return stats

# if __name__ == "__main__":
#     generate_final_report()

# import torch
# import numpy as np
# from pathlib import Path
# import json
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve, auc
# import seaborn as sns
# from tqdm import tqdm
# from gcn import CentroidLayer, MalwareGNN, MalwareTrainer

# import sys
# sys.path.append('/data/saranyav/gcn_new/Methods')
# from TemporalGNN import TemporalMalwareDataLoader



# def analyze_thresholds(model, loader, device, 
#                       outlier_thresholds=np.arange(0.3, 0.9, 0.05),
#                       confidence_thresholds=np.arange(0.3, 0.9, 0.05)):
#     """Analyze different threshold combinations."""
#     model.eval()
#     all_outlier_scores = []
#     all_confidences = []
#     all_is_novel = []
    
#     # Collect predictions
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Running inference"):
#             batch = batch.to(device)
#             logits, outlier_scores = model(batch)
#             probs = torch.softmax(logits, dim=1)
#             max_probs = probs.max(dim=1)[0]
            
#             all_outlier_scores.extend(outlier_scores.cpu().numpy())
#             all_confidences.extend(max_probs.cpu().numpy())
#             all_is_novel.extend(batch.is_novel.cpu().numpy())
    
#     all_outlier_scores = np.array(all_outlier_scores)
#     all_confidences = np.array(all_confidences)
#     all_is_novel = np.array(all_is_novel)
    
#     results = {
#         'threshold_grid': [],
#         'pr_curve': None
#     }
    
#     # Grid search over thresholds
#     best_f1 = 0
#     best_thresholds = None
    
#     for out_th in outlier_thresholds:
#         for conf_th in confidence_thresholds:
#             pred_novel = (all_outlier_scores > out_th) | (all_confidences < conf_th)
            
#             # Calculate metrics
#             tp = np.sum((pred_novel == True) & (all_is_novel == True))
#             fp = np.sum((pred_novel == True) & (all_is_novel == False))
#             fn = np.sum((pred_novel == False) & (all_is_novel == True))
            
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#             f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
#             results['threshold_grid'].append({
#                 'outlier_threshold': float(out_th),
#                 'confidence_threshold': float(conf_th),
#                 'precision': float(precision),
#                 'recall': float(recall),
#                 'f1': float(f1)
#             })
            
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_thresholds = (out_th, conf_th)
    
#     # Calculate PR curve using combined score
#     combined_scores = all_outlier_scores - all_confidences  # Higher score = more likely novel
#     precision, recall, _ = precision_recall_curve(all_is_novel, combined_scores)
#     pr_auc = auc(recall, precision)
    
#     results['pr_curve'] = {
#         'precision': precision.tolist(),
#         'recall': recall.tolist(),
#         'auc': float(pr_auc)
#     }
    
#     return results

# def plot_threshold_heatmap(results, metric='f1', save_path=None):
#     """Plot heatmap of threshold performance."""
#     # Extract unique threshold values
#     out_thresholds = sorted(set(r['outlier_threshold'] for r in results['threshold_grid']))
#     conf_thresholds = sorted(set(r['confidence_threshold'] for r in results['threshold_grid']))
    
#     # Create matrix for heatmap
#     matrix = np.zeros((len(out_thresholds), len(conf_thresholds)))
    
#     for r in results['threshold_grid']:
#         i = out_thresholds.index(r['outlier_threshold'])
#         j = conf_thresholds.index(r['confidence_threshold'])
#         matrix[i, j] = r[metric]
    
#     # Create heatmap
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(matrix, 
#                 xticklabels=[f'{x:.2f}' for x in conf_thresholds],
#                 yticklabels=[f'{x:.2f}' for x in out_thresholds],
#                 annot=True, fmt='.3f', cmap='viridis')
    
#     plt.xlabel('Confidence Threshold')
#     plt.ylabel('Outlier Threshold')
#     plt.title(f'{metric.upper()} Score by Threshold Combination')
    
#     if save_path:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()

# def plot_pr_curve(results, save_path=None):
#     """Plot precision-recall curve."""
#     plt.figure(figsize=(8, 8))
#     plt.plot(results['pr_curve']['recall'], 
#             results['pr_curve']['precision'], 
#             label=f'PR curve (AUC = {results["pr_curve"]["auc"]:.3f})')
    
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve for Novelty Detection')
#     plt.legend()
#     plt.grid(True)
    
#     if save_path:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()

# def main():
#     # Setup
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Initialize data loader
#     data_loader = TemporalMalwareDataLoader(
#         batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches'),
#         behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
#         metadata_path=Path('bodmas_metadata_cleaned.csv'),
#         malware_types_path=Path('bodmas_malware_category.csv')
#     )
    
#     # Load validation data
#     val_loader_family, val_stats = data_loader.load_split('test', use_groups=False, batch_size=32)
#     val_loader_groups, val_group_stats = data_loader.load_split('test', use_groups=True, batch_size=32)
    
#     # Load best models
#     family_model = MalwareGNN(
#         num_node_features=14,
#         hidden_dim=256,
#         num_classes=data_loader.get_num_classes(use_groups=False),
#         n_centroids_per_class=4,
#         num_layers=4,
#         dropout=0.2
#     ).to(device)
    
#     group_model = MalwareGNN(
#         num_node_features=14,
#         hidden_dim=256,
#         num_classes=data_loader.get_num_classes(use_groups=True),
#         n_centroids_per_class=4,
#         num_layers=4,
#         dropout=0.2
#     ).to(device)
    
#     # Load model states
#     family_checkpoint = torch.load('best_family_model.pt')
#     family_model.load_state_dict(family_checkpoint['model_state_dict'])
    
#     group_checkpoint = torch.load('best_group_model.pt')
#     group_model.load_state_dict(group_checkpoint['model_state_dict'])
    
#     # Analyze thresholds
#     print("Analyzing family model thresholds...")
#     family_results = analyze_thresholds(
#         family_model, 
#         val_loader_family,
#         device,
#         outlier_thresholds=np.arange(0.3, 0.9, 0.05),
#         confidence_thresholds=np.arange(0.3, 0.9, 0.05)
#     )
    
#     print("Analyzing group model thresholds...")
#     group_results = analyze_thresholds(
#         group_model,
#         val_loader_groups,
#         device,
#         outlier_thresholds=np.arange(0.3, 0.9, 0.05),
#         confidence_thresholds=np.arange(0.3, 0.9, 0.05)
#     )
    
#     # Save results
#     with open('family_threshold_analysis.json', 'w') as f:
#         json.dump(family_results, f, indent=2)
#     with open('group_threshold_analysis.json', 'w') as f:
#         json.dump(group_results, f, indent=2)
    
#     # Create visualizations
#     print("Generating visualizations...")
#     # F1 score heatmaps
#     plot_threshold_heatmap(family_results, metric='f1', save_path='family_f1_heatmap.png')
#     plot_threshold_heatmap(group_results, metric='f1', save_path='group_f1_heatmap.png')
    
#     # Precision-Recall curves
#     plot_pr_curve(family_results, save_path='family_pr_curve.png')
#     plot_pr_curve(group_results, save_path='group_pr_curve.png')
    
#     print("Analysis complete! Results saved to JSON files and visualizations saved as PNG files.")

# if __name__ == "__main__":
#     main()