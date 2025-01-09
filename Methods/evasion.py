import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from torch_geometric.data import Data
from gcn import CentroidLayer, MalwareGNN, MalwareTrainer
from TemporalGNN import TemporalMalwareDataLoader, NumpyEncoder
import json
from tqdm import tqdm
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reshape_state_dict(checkpoint_path, target_size=1948):
    """
    Reshape the centroid layer in state dict to match target model size.
    
    Args:
        checkpoint_path: Path to checkpoint file
        target_size: Target number of centroids (default 1948)
    
    Returns:
        Modified state dictionary
    """
    import torch
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    
    # Get current centroid size
    current_centroids = state_dict['centroid.centroids']
    current_size = current_centroids.size(0)
    
    if current_size == target_size:
        return state_dict
        
    # Reshape centroids
    if current_size > target_size:
        # Take first target_size centroids
        state_dict['centroid.centroids'] = current_centroids[:target_size]
    else:
        # Pad with zeros
        new_centroids = torch.zeros(target_size, current_centroids.size(1), 
                                  device=current_centroids.device)
        new_centroids[:current_size] = current_centroids
        state_dict['centroid.centroids'] = new_centroids
        
    return state_dict




class EvasionAnalyzer:
    """Analyze model robustness against evasion techniques."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test_evasion_techniques(self, graph: Data) -> Dict[str, float]:
        """Test various evasion techniques on a single graph."""
        techniques = {
            'control_flow_obfuscation': self._test_control_flow_obfuscation,
            'dead_code_insertion': self._test_dead_code_insertion,
            'api_call_indirection': self._test_api_indirection,
            'feature_manipulation': self._test_feature_manipulation,
            'graph_structure_perturbation': self._test_graph_perturbation
        }

        results = {}
        original_pred, original_conf = self._get_prediction(graph)

        for technique_name, technique_func in techniques.items():
            perturbed_graph = technique_func(graph.clone())
            new_pred, new_conf = self._get_prediction(perturbed_graph)
            
            results[technique_name] = {
                'evasion_success': int(new_pred != original_pred),
                'confidence_drop': float(original_conf - new_conf),
                'detection_score': float(new_conf)
            }

        return results


    def _fix_batch_assignments(self, graph: Data) -> Data:
        """Fix batch assignments to match the number of nodes."""
        num_nodes = graph.x.size(0)
        
        # Create new batch tensor that matches number of nodes
        if hasattr(graph, 'batch') and graph.batch is not None:
            # Get unique batch ids
            unique_batches = torch.unique(graph.batch)
            num_batches = len(unique_batches)
            
            # Create new batch assignments
            new_batch = torch.zeros(num_nodes, device=graph.batch.device, dtype=graph.batch.dtype)
            nodes_per_batch = num_nodes // num_batches
            
            for i in range(num_batches):
                start_idx = i * nodes_per_batch
                end_idx = start_idx + nodes_per_batch if i < num_batches - 1 else num_nodes
                new_batch[start_idx:end_idx] = i
                
            graph.batch = new_batch
            
            # Update ptr if it exists
            if hasattr(graph, 'ptr'):
                new_ptr = torch.zeros(num_batches + 1, device=graph.ptr.device, dtype=graph.ptr.dtype)
                for i in range(num_batches):
                    new_ptr[i + 1] = (new_batch <= i).sum()
                graph.ptr = new_ptr
        
        return graph

    def _get_prediction(self, graph: Data) -> Tuple[int, float]:
        """Get model prediction and confidence with fixed batch assignments."""
        self.model.eval()
        with torch.no_grad():
            # Fix batch assignments
            graph = self._fix_batch_assignments(graph)
            
            # Move to device
            graph = graph.to(self.device)
            
            try:
                logits, _ = self.model(graph)
                probs = F.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs.max(dim=1)[0].item()
                return pred, conf
            except Exception as e:
                print(f"\nError during forward pass: {str(e)}")
                print(f"x shape: {graph.x.shape}")
                print(f"batch shape: {graph.batch.shape}")
                print(f"edge_index shape: {graph.edge_index.shape}")
                if hasattr(graph, 'ptr'):
                    print(f"ptr shape: {graph.ptr.shape}")
                raise e
                
    def _test_control_flow_obfuscation(self, graph: Data) -> Data:
        """Simulate control flow obfuscation attacks."""
        # Clone input data
        edge_index = graph.edge_index.clone()
        x = graph.x.clone()
        batch = graph.batch.clone()
        
        # Calculate number of nodes to add
        num_orig_nodes = x.size(0)
        num_new_nodes = num_orig_nodes // 5  # Add 20% more nodes
        
        # print(f"\nModifying graph:")
        # print(f"Original nodes: {num_orig_nodes}")
        # print(f"Adding nodes: {num_new_nodes}")
        
        # Create new nodes
        new_node_features = torch.zeros((num_new_nodes, x.size(1)), device=x.device)
        
        # Set random features
        for node_idx in range(num_new_nodes):
            num_features = max(1, x.size(1) // 10)
            indices = torch.randint(0, x.size(1), (num_features,), device=x.device)
            new_node_features[node_idx, indices] = 1
        
        # Update node features
        x = torch.cat([x, new_node_features], dim=0)
        
        # Create new batch assignments that match the original pattern
        unique_batches = torch.unique(batch)
        nodes_per_batch = []
        for b in unique_batches:
            nodes_in_b = (batch == b).sum().item()
            nodes_per_batch.append(nodes_in_b)
        
        # Calculate new nodes per batch proportionally
        total_orig_nodes = sum(nodes_per_batch)
        new_batch = []
        remaining_nodes = num_new_nodes
        
        for batch_idx, orig_nodes in enumerate(nodes_per_batch):
            # Distribute new nodes proportionally
            batch_new_nodes = int(num_new_nodes * (orig_nodes / total_orig_nodes))
            if batch_idx == len(nodes_per_batch) - 1:
                batch_new_nodes = remaining_nodes
            remaining_nodes -= batch_new_nodes
            
            # Add batch assignments for new nodes
            new_batch.extend([batch_idx] * batch_new_nodes)
        
        # Convert to tensor and append to original batch
        new_batch_tensor = torch.tensor(new_batch, device=batch.device)
        batch = torch.cat([batch, new_batch_tensor])
        
        # Add edges
        new_edges = []
        for i in range(num_new_nodes):
            new_idx = num_orig_nodes + i
            # Connect to a node in the same batch
            new_batch_idx = new_batch[i]
            valid_targets = (batch[:num_orig_nodes] == new_batch_idx).nonzero().squeeze()
            if valid_targets.numel() > 0:
                target_idx = valid_targets[torch.randint(0, valid_targets.numel(), (1,))]
                new_edges.extend([[new_idx, target_idx.item()], [target_idx.item(), new_idx]])
        
        new_edges = torch.tensor(new_edges, device=edge_index.device).t()
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        
        # Update graph attributes
        graph.x = x
        graph.edge_index = edge_index
        graph.batch = batch
        
        # Update edge_attr if it exists
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            new_edge_attr = torch.zeros((new_edges.size(1), graph.edge_attr.size(1)), 
                                    device=graph.edge_attr.device)
            graph.edge_attr = torch.cat([graph.edge_attr, new_edge_attr])
        
        # Update ptr if it exists
        if hasattr(graph, 'ptr'):
            # Count nodes per batch in the new graph
            unique_batches = torch.unique(batch)
            new_ptr = torch.zeros_like(graph.ptr)
            new_ptr[0] = 0
            for i, b in enumerate(unique_batches):
                if i + 1 < len(new_ptr):
                    new_ptr[i + 1] = (batch <= b).sum()
            graph.ptr = new_ptr
        
        # print(f"\nFinal sizes:")
        # print(f"x: {graph.x.shape}")
        # print(f"edge_index: {graph.edge_index.shape}")
        # print(f"batch: {graph.batch.shape}")
        
        return graph

    def _test_dead_code_insertion(self, graph: Data) -> Data:
        """Simulate dead code insertion attacks."""
        # Insert benign-looking dead code nodes
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()
        
        # Create dead code nodes with common benign patterns
        num_dead_nodes = int(0.2 * x.size(0))  # Add 20% more nodes
        dead_features = torch.zeros((num_dead_nodes, x.size(1)), device=x.device)
        
        # Set features to look like benign operations
        dead_features[:, 2] = 1  # Set some instructions
        dead_features[:, 4] = 1  # Set some register writes
        
        x = torch.cat([x, dead_features], dim=0)
        
        # Connect dead code to maintain graph structure
        new_edges = []
        for i in range(num_dead_nodes):
            src = x.size(0) - num_dead_nodes + i
            dst = torch.randint(0, x.size(0)-num_dead_nodes, (1,)).item()
            new_edges.extend([[src, dst], [dst, src]])
            
        new_edges = torch.tensor(new_edges, device=edge_index.device).t()
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        
        graph.x = x
        graph.edge_index = edge_index
        return graph

    def _test_api_indirection(self, graph: Data) -> Data:
        """Simulate API call indirection attacks."""
        x = graph.x.clone()
        
        # Find API call nodes
        api_nodes = (x[:, 5] > 0)  # external_calls feature
        
        # Replace direct API calls with indirect ones
        x[api_nodes, 5] = 0  # Remove direct external calls
        x[api_nodes, 6] += 1  # Add internal calls instead
        
        graph.x = x
        return graph

    def _test_feature_manipulation(self, graph: Data) -> Data:
        """Test feature manipulation attacks."""
        x = graph.x.clone()
        
        # Slightly modify node features while preserving overall structure
        noise = torch.randn_like(x) * 0.1
        x = x + noise
        x = torch.clamp(x, min=0, max=1)
        
        graph.x = x
        return graph

    def _test_graph_perturbation(self, graph: Data) -> Data:
        """Test structural perturbation attacks."""
        edge_index = graph.edge_index.clone()
        
        # Randomly remove some edges
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > 0.1
        edge_index = edge_index[:, mask]
        
        # Add some random edges
        num_nodes = graph.x.size(0)
        num_new_edges = int(0.1 * edge_index.size(1))
        new_edges = torch.randint(0, num_nodes, (2, num_new_edges), device=edge_index.device)
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        
        graph.edge_index = edge_index
        return graph

def evaluate_security(model, test_loader, device):
    """Evaluate model security against various evasion techniques."""
    analyzer = EvasionAnalyzer(model, device)
    security_metrics = defaultdict(list)
    
    for batch in test_loader:
        for graph in [batch]:  # Process each graph individually
            results = analyzer.test_evasion_techniques(graph)
            
            for technique, metrics in results.items():
                security_metrics[technique].append(metrics)
    
    # Aggregate results
    aggregated_metrics = {}
    for technique, results in security_metrics.items():
        aggregated_metrics[technique] = {
            'evasion_success_rate': np.mean([r['evasion_success'] for r in results]),
            'avg_confidence_drop': np.mean([r['confidence_drop'] for r in results]),
            'avg_detection_score': np.mean([r['detection_score'] for r in results])
        }
    
    return aggregated_metrics

def main():
    """Evaluate model security against evasion techniques."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    data_loader = TemporalMalwareDataLoader(
        batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches'),
        behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        metadata_path=Path('bodmas_metadata_cleaned.csv'),
        malware_types_path=Path('bodmas_malware_category.csv')
    )
    
    # # Load data
    # train_loader_family, train_stats = data_loader.load_split('train', use_groups=False, batch_size=32)
    # train_loader_groups, train_group_stats = data_loader.load_split('train', use_groups=True, batch_size=32)
    # val_loader_family, val_stats = data_loader.load_split('val', use_groups=False, batch_size=32)
    # val_loader_groups, val_group_stats = data_loader.load_split('val', use_groups=True, batch_size=32)
    # test_loader_family, test_stats = data_loader.load_split('test', use_groups=False, batch_size=32)
    # test_loader_groups, test_group_stats = data_loader.load_split('test', use_groups=True, batch_size=32)
    
    # # Get number of classes
    num_family_classes = data_loader.get_num_classes(use_groups=False)
    num_group_classes = data_loader.get_num_classes(use_groups=True)
    
    # Load the best trained models
    logger.info("Loading best models...")
    family_model = MalwareGNN(
        num_node_features=14,
        hidden_dim=256,
        num_classes=38,  # 152/4 centroids per class
        n_centroids_per_class=4
    ).to(device)
    
    # Load state dict
    checkpoint = torch.load('checkpoint_group_epoch_40.pt')
    family_model.load_state_dict(checkpoint['model_state_dict'])
    family_model.eval()

    checkpoint_path = 'checkpoint_group_epoch_40.pt'
    try:
        print("Loading checkpoint...")
        new_state_dict = reshape_state_dict(checkpoint_path)
        
        print("\nLoading reshaped state dict into model...")
        family_model.load_state_dict(new_state_dict)
        print("Successfully reshaped and loaded state dictionary!")
        
        # Verify the model's centroid size
        print("\nVerifying new centroid size...")
        print(f"Current model centroid size: {family_model.centroid.centroids.size()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")



    # Initialize data loader for test set
    data_loader = TemporalMalwareDataLoader(
        batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches'),
        behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        metadata_path=Path('bodmas_metadata_cleaned.csv'),
        malware_types_path=Path('bodmas_malware_category.csv')
    )
    
    test_loader, _ = data_loader.load_split('test', use_groups=False, batch_size=1)  # batch_size=1 for detailed analysis

    # Initialize security analyzer
    analyzer = EvasionAnalyzer(family_model, device)
    
    # Run security evaluation
    logger.info("Starting security evaluation...")
    results = defaultdict(list)
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating security")):
        try:
            # Test each evasion technique
            evasion_results = analyzer.test_evasion_techniques(batch)
            
            # Store results
            for technique, metrics in evasion_results.items():
                results[technique].append(metrics)
            
            # Log progress periodically
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1} samples")
                
                # Show interim results
                for technique, technique_results in results.items():
                    avg_evasion_rate = np.mean([r['evasion_success'] for r in technique_results])
                    avg_conf_drop = np.mean([r['confidence_drop'] for r in technique_results])
                    
                    logger.info(f"\n{technique}:")
                    logger.info(f"  Evasion Success Rate: {avg_evasion_rate:.3f}")
                    logger.info(f"  Average Confidence Drop: {avg_conf_drop:.3f}")
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    # Aggregate final results
    final_results = {
        technique: {
            'evasion_success_rate': np.mean([r['evasion_success'] for r in technique_results]),
            'avg_confidence_drop': np.mean([r['confidence_drop'] for r in technique_results]),
            'avg_detection_score': np.mean([r['detection_score'] for r in technique_results]),
            'std_evasion_success': np.std([r['evasion_success'] for r in technique_results]),
            'std_confidence_drop': np.std([r['confidence_drop'] for r in technique_results]),
            'std_detection_score': np.std([r['detection_score'] for r in technique_results])
        }
        for technique, technique_results in results.items()
    }

    # Save detailed results
    output_dir = Path('security_analysis')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'evasion_analysis.json', 'w') as f:
        json.dump({
            'aggregate_results': final_results,
            'per_sample_results': {
                technique: technique_results 
                for technique, technique_results in results.items()
            }
        }, f, indent=2, cls=NumpyEncoder)

    # Generate summary report
    summary = {
        'overall_robustness': {
            technique: {
                'evasion_resistance': 1 - metrics['evasion_success_rate'],
                'confidence_stability': 1 - metrics['avg_confidence_drop'],
                'detection_reliability': metrics['avg_detection_score']
            }
            for technique, metrics in final_results.items()
        },
        'vulnerability_ranking': sorted(
            final_results.items(),
            key=lambda x: x[1]['evasion_success_rate'],
            reverse=True
        )
    }

    with open(output_dir / 'security_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Log final results
    logger.info("\nFinal Security Analysis Results:")
    for technique, metrics in final_results.items():
        logger.info(f"\n{technique}:")
        logger.info(f"  Evasion Success Rate: {metrics['evasion_success_rate']:.3f} (±{metrics['std_evasion_success']:.3f})")
        logger.info(f"  Average Confidence Drop: {metrics['avg_confidence_drop']:.3f} (±{metrics['std_confidence_drop']:.3f})")
        logger.info(f"  Average Detection Score: {metrics['avg_detection_score']:.3f} (±{metrics['std_detection_score']:.3f})")

if __name__ == "__main__":
    main()