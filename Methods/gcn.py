import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from collections import defaultdict
import os
import glob
import json
import logging
from datetime import datetime
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import sys 
#cross entropy loss
import torch.nn as nn
import torch.nn.functional as F
# from architectures import CentroidLayer, MalwareGNN
# confusion_matrix
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import numpy
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import collections
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool




class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.centroid = CentroidLayer(100, 2, 2, reject_input=False)
        self.conv1 = GCNConv(10, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
import numpy as np
from collections import Counter



class MalwareTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def compute_class_weights(self, loader):
        """Compute class weights with proper handling of unseen classes."""
        num_classes = self.model.num_classes
        if num_classes <= 0:
            logger.error(f"Invalid number of classes: {num_classes}")
            return None
            
        # Count samples per class
        class_counts = torch.zeros(num_classes, device=self.device)
        total_samples = 0
        
        for batch in loader:
            if not hasattr(batch, 'y'):
                continue
            labels = batch.y
            for label in range(num_classes):
                count = (labels == label).sum().item()
                class_counts[label] += count
                total_samples += count
        
        if total_samples == 0:
            logger.error("No valid samples found for computing class weights")
            return None
            
        # Compute weights with smoothing to handle zero counts
        smoothing_factor = 0.1
        smoothed_counts = class_counts + smoothing_factor
        weights = total_samples / (num_classes * smoothed_counts)
        
        # Normalize weights
        weights = weights / weights.sum() * num_classes
        
        logger.info(f"Class counts: {class_counts.tolist()}")
        logger.info(f"Computed weights: {weights.tolist()}")
        
        return weights
        
    def train_epoch(self, loader, class_weights=None):
        """Train for one epoch with robust error handling."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Use default weights if none provided
        if class_weights is None:
            class_weights = torch.ones(self.model.num_classes, device=self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for batch_idx, batch in enumerate(loader):
            try:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                output, outlier_scores = self.model(batch)
                
                # Ensure valid predictions
                if output.size(1) != self.model.num_classes:
                    raise ValueError(f"Model output size mismatch. Expected {self.model.num_classes}, got {output.size(1)}")
                
                # Compute loss
                loss = criterion(output, batch.y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(batch.y).sum().item()
                total += len(batch.y)
                
                # Periodic logging
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Batch {batch_idx + 1}/{len(loader)}: Loss = {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total if total > 0 else 0
        }
    def evaluate(self, loader, class_weights=None):
        """Evaluate model with comprehensive metrics."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_outlier_scores = []
        
        criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Forward pass
                output, outlier_scores = self.model(batch)
                
                # Compute loss
                loss = criterion(output, batch.y)
                
                # Store predictions and labels
                pred = output.argmax(dim=1)
                all_preds.append(pred)
                all_labels.append(batch.y)
                all_outlier_scores.append(outlier_scores)
                
                total_loss += loss.item()
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_outlier_scores = torch.cat(all_outlier_scores)
        
        # Compute comprehensive metrics
        metrics = self.compute_metrics(all_labels, all_preds, self.model.num_classes)
        
        # Add loss and outlier scores to metrics
        metrics['loss'] = total_loss / len(loader)
        metrics['outlier_scores'] = all_outlier_scores.cpu().numpy()
        
        return metrics
        
    def log_metrics(self, metrics, split="val"):
        """Log metrics in a structured format."""
        logger.info(f"\n{split.capitalize()} Metrics:")
        logger.info(f"Overall Performance:")
        logger.info(f"- Loss: {metrics['loss']:.4f}")
        logger.info(f"- Accuracy: {metrics['overall']['accuracy']:.4f}")
        logger.info(f"- Precision: {metrics['overall']['precision']:.4f}")
        logger.info(f"- Recall: {metrics['overall']['recall']:.4f}")
        logger.info(f"- F1-Score: {metrics['overall']['f1']:.4f}")
        
        logger.info("\nPer-Class Performance:")
        for class_idx, class_metrics in metrics['per_class'].items():
            if class_metrics['support'] > 0:  # Only show classes with samples
                logger.info(f"\nClass {class_idx}:")
                logger.info(f"- Support: {class_metrics['support']}")
                logger.info(f"- Precision: {class_metrics['precision']:.4f}")
                logger.info(f"- Recall: {class_metrics['recall']:.4f}")
                logger.info(f"- F1-Score: {class_metrics['f1']:.4f}")

    def compute_metrics(self, y_true, y_pred, num_classes):
        """
        Compute detailed classification metrics.
        Returns both per-class and overall metrics.
        """
        # Initialize metric containers
        metrics = {
            'per_class': {},
            'overall': {}
        }
        
        # Convert to numpy for sklearn metrics
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
        # Per-class metrics
        for class_idx in range(num_classes):
            # True Positives, False Positives, False Negatives
            tp = conf_matrix[class_idx, class_idx]
            fp = conf_matrix[:, class_idx].sum() - tp
            fn = conf_matrix[class_idx, :].sum() - tp
            
            # Handle division by zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['per_class'][class_idx] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(conf_matrix[class_idx, :].sum())
            }
        
        # Overall metrics (weighted by support)
        total_samples = conf_matrix.sum()
        weighted_precision = sum(m['precision'] * m['support'] for m in metrics['per_class'].values()) / total_samples
        weighted_recall = sum(m['recall'] * m['support'] for m in metrics['per_class'].values()) / total_samples
        weighted_f1 = sum(m['f1'] * m['support'] for m in metrics['per_class'].values()) / total_samples
        
        metrics['overall'] = {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1,
            'accuracy': (y_true == y_pred).mean()
        }
        
        return metrics
        
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
    
#     # Get loaders
#     train_loader = data_loader.load_split('train', use_groups=True)
#     val_loader = data_loader.load_split('val', use_groups=True)
    
#     # Initialize model
#     model = MalwareGNN(
#         num_node_features=14,
#         hidden_dim=128,
#         num_classes=18,
#         n_centroids_per_class=2
#     ).to(device)
    
#     # Initialize trainer
#     trainer = MalwareTrainer(model, device)
    
#     # Compute class weights from training data
#     class_weights = trainer.compute_class_weights(train_loader)
    
#     # Training loop
#     num_epochs = 100
#     best_val_acc = 0
    
#     for epoch in range(num_epochs):
#         # Train
#         train_metrics = trainer.train_epoch(train_loader, class_weights)
        
#         # Evaluate
#         val_metrics = trainer.evaluate(val_loader, class_weights)
        
#         # Log progress
#         logger.info(f"Epoch {epoch}:")
#         logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
#         logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
#         logger.info(f"Outliers detected: {val_metrics['outliers']}")
        
#         # Save best model
#         if val_metrics['accuracy'] > best_val_acc:
#             best_val_acc = val_metrics['accuracy']
#             torch.save(model.state_dict(), 'best_model.pt')

# if __name__ == "__main__":
#     main()

# sys.exit(0)
# class CentroidLayer(torch.nn.Module):
#     def __init__(self, input_dim, n_classes, n_centroids_per_class=3):
#         super().__init__()
#         self.input_dim = input_dim
#         self.n_classes = n_classes
#         self.n_centroids_per_class = n_centroids_per_class
        
#         # Initialize centroids
#         n_total_centroids = n_classes * n_centroids_per_class
#         self.centroids = torch.nn.Parameter(
#             torch.randn(n_total_centroids, input_dim) / np.sqrt(input_dim)
#         )
        
#     def forward(self, x):
#         # Compute distances to all centroids
#         expanded_x = x.unsqueeze(1)
#         expanded_centroids = self.centroids.unsqueeze(0)
        
#         # Compute euclidean distances
#         distances = torch.norm(expanded_x - expanded_centroids, dim=2)
        
#         # Reshape distances to group by class
#         distances = distances.view(x.size(0), self.n_classes, self.n_centroids_per_class)
        
#         # Get minimum distance to each class's centroids
#         min_distances, _ = torch.min(distances, dim=2)
        
#         # Convert distances to logits
#         return -min_distances


def prepare_data(base_dir='bodmas_batches_new'):
    """Prepare datasets with temporal ordering."""
    split_files = defaultdict(list)
    family_counts = defaultdict(int)
    
    logger.info("Starting data preparation...")
    logger.info(f"Looking in base directory: {base_dir}")
    
    # First pass: collect family info and timestamps
    all_families = set()
    file_timestamps = {}  # Store timestamps for sorting
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        files = glob.glob(os.path.join(split_dir, 'batch_*.pt'))
        logger.info(f"Found {len(files)} files in {split} split")
        
        for file in files:
            try:
                batch = torch.load(file)
                # Get timestamp from first graph in batch
                if hasattr(batch[0], 'timestamp'):
                    file_timestamps[file] = batch[0].timestamp
                else:
                    logger.warning(f"No timestamp in {file}")
                    continue
                
                for graph in batch:
                    family = getattr(graph, 'family', 'none')
                    if family is None or family == '':
                        family = 'none'
                    all_families.add(family)
                    family_counts[family] += 1
                
                split_files[split].append(file)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                continue

    # Sort files by timestamp within each split
    for split in split_files:
        split_files[split].sort(key=lambda x: file_timestamps.get(x, ''))
        
        # Log temporal range for each split
        if split_files[split]:
            start_time = file_timestamps[split_files[split][0]]
            end_time = file_timestamps[split_files[split][-1]]
            logger.info(f"{split} split temporal range: {start_time} to {end_time}")

    # Create label mapping
    families = sorted(list(all_families))
    family_to_idx = {family: idx for idx, family in enumerate(families)}
    num_classes = len(family_to_idx)

    # Log statistics
    logger.info(f"\nFound {num_classes} unique families:")
    for family, idx in sorted(family_to_idx.items()):
        count = family_counts[family]
        logger.info(f"{family} -> {idx} (count: {count})")
    
    # Save mapping
    mapping_path = os.path.join(base_dir, 'family_mapping.json')
    mapping_data = {
        'family_to_idx': family_to_idx,
        'idx_to_family': {str(idx): family for family, idx in family_to_idx.items()},
        'family_counts': family_counts,
        'num_classes': num_classes,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(mapping_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    return split_files, num_classes, family_to_idx

def load_and_process_batch(batch_file, family_to_idx, batch_size=32):
    """Load batch while preserving temporal order."""
    try:
        if not os.path.exists(batch_file):
            logger.warning(f"Batch file not found: {batch_file}")
            return None
        
        batch = torch.load(batch_file)
        
        # Sort graphs by timestamp if available
        if hasattr(batch[0], 'timestamp'):
            batch = sorted(batch, key=lambda x: x.timestamp)
        
        processed_batch = []
        num_classes = len(family_to_idx)
        
        for graph in batch:
            try:
                # Handle family label
                family = getattr(graph, 'family', 'none')
                if family is None or family == '':
                    family = 'none'
                
                if family not in family_to_idx:
                    logger.warning(f"Unknown family {family} in {batch_file}, setting to 'none'")
                    family = 'none'
                
                # Set label
                graph.y = torch.tensor(family_to_idx[family])
                
                # Validate label
                if graph.y >= num_classes:
                    logger.error(f"Invalid label {graph.y} for family {family} in {batch_file}")
                    continue
                
                # Handle edge attributes if missing
                if graph.edge_index.size(1) == 0:
                    graph.edge_attr = torch.zeros((0, 1))
                elif not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
                    graph.edge_attr = torch.ones((graph.edge_index.size(1), 1))
                
                processed_batch.append(graph)
                
            except Exception as e:
                logger.error(f"Error processing graph in {batch_file}: {e}")
                continue
        
        if not processed_batch:
            logger.warning(f"No valid graphs in {batch_file}")
            return None
            
        # Create DataLoader with shuffle=False to preserve temporal order
        return DataLoader(processed_batch, batch_size=batch_size, shuffle=False)
        
    except Exception as e:
        logger.error(f"Error processing batch file {batch_file}: {e}")
        return None

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch with validation."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data in loader:
        try:
            data = data.to(device)
            
            # Validate labels
            if not torch.all((data.y >= 0) & (data.y < model.num_classes)):
                invalid_indices = torch.where((data.y < 0) | (data.y >= model.num_classes))[0]
                logger.error(f"Invalid labels detected at indices {invalid_indices}: {data.y[invalid_indices]}")
                continue
            
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, data.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        except Exception as e:
            logger.error(f"Error in training batch: {e}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

def evaluate(model, loader, device, family_to_idx, split='val'):
    """Evaluate model performance."""
    if loader is None:
        logger.warning(f"No loader provided for {split} evaluation")
        return 0.0, {}
    
    # Debug model classes
    logger.info(f"Model number of classes: {model.num_classes}")
    logger.info(f"Centroid layer classes: {model.centroid.n_classes}")
    logger.info(f"Number of families in mapping: {len(family_to_idx)}")
    
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []
    
    # Setup per-family tracking
    idx_to_family = {v: k for k, v in family_to_idx.items()}
    family_correct = defaultdict(int)
    family_total = defaultdict(int)
    
    try:
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(device)
                
                # Debug batch labels
                logger.info(f"Batch {batch_idx} labels: {data.y}")
                logger.info(f"Label range: min={data.y.min().item()}, max={data.y.max().item()}")
                
                # Validate labels
                if not torch.all((data.y >= 0) & (data.y < model.num_classes)):
                    invalid_mask = (data.y < 0) | (data.y >= model.num_classes)
                    invalid_labels = data.y[invalid_mask]
                    logger.error(f"Invalid labels in batch {batch_idx}: {invalid_labels}")
                    continue
                
                out = model(data)
                logger.info(f"Model output shape: {out.shape}")
                pred = out.argmax(dim=1)
                
                correct += int((pred == data.y).sum())
                total += data.y.size(0)
                
                predictions.extend(pred.cpu().numpy())
                labels.extend(data.y.cpu().numpy())
                
                # Debug predictions
                logger.info(f"Predictions in batch {batch_idx}: {pred}")
                
                # Update per-family metrics
                for true, pred_val in zip(data.y.cpu().numpy(), pred.cpu().numpy()):
                    true_family = idx_to_family[true]
                    family_total[true_family] += 1
                    if true == pred_val:
                        family_correct[true_family] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Log detailed metrics
        logger.info(f"\n{split.upper()} Results:")
        logger.info(f"Total samples processed: {total}")
        logger.info(f"Total correct predictions: {correct}")
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        
        # Debug prediction distribution
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        logger.info("Prediction distribution:")
        for pred_class, count in zip(unique_preds, pred_counts):
            family = idx_to_family.get(pred_class, "unknown")
            logger.info(f"Class {pred_class} ({family}): {count} predictions")
        
        # Per-family accuracies
        family_accuracies = {
            family: family_correct[family] / family_total[family] 
            if family_total[family] > 0 else 0
            for family in family_to_idx.keys()
        }
        
        logger.info("\nPer-family accuracies:")
        for family, acc in sorted(family_accuracies.items()):
            samples = family_total[family]
            if samples > 0:
                logger.info(f"{family}: {acc:.4f} ({family_correct[family]}/{samples})")
        
        return accuracy, family_accuracies
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        # Print full traceback
        import traceback
        logger.error(traceback.format_exc())
        return 0.0, {}

# def main():
#     # Setup
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")
    
#     # Prepare data with validation
#     split_files, num_classes, family_to_idx = prepare_data()
    
#     # Validate feature dimensions
#     try:
#         first_batch = torch.load(split_files['train'][0])
#         num_features = first_batch[0].x.size(1)
#         logger.info(f"Number of node features: {num_features}")
#     except Exception as e:
#         logger.error(f"Error loading first batch: {e}")
#         return
    
#     # Initialize model
#     model = MalwareGNN(
#         num_node_features=num_features,
#         num_classes=num_classes
#     ).to(device)
    
#     # Training settings
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=5, verbose=True
#     )
#     num_epochs = 100
#     best_val_acc = 0
#     patience = 10
#     patience_counter = 0
    
#     # Training loop
#     logger.info("\nStarting training...")
#     for epoch in range(num_epochs):
#         # Training
#         epoch_losses = []
#         for batch_file in split_files['train']:
#             train_loader = load_and_process_batch(batch_file, family_to_idx)
#             if train_loader:
#                 loss = train_epoch(model, train_loader, optimizer, device)
#                 epoch_losses.append(loss)
        
#         if not epoch_losses:
#             logger.warning(f"No valid training batches in epoch {epoch}")
#             continue
            
#         avg_loss = sum(epoch_losses) / len(epoch_losses)
        
#         # Validation
#         val_accs = []
#         val_metrics = defaultdict(list)
#         for batch_file in split_files['val']:
#             val_loader = load_and_process_batch(batch_file, family_to_idx)
#             if val_loader:
#                 acc, family_accs = evaluate(model, val_loader, device, family_to_idx, 'val')
#                 val_accs.append(acc)
#                 for family, acc in family_accs.items():
#                     val_metrics[family].append(acc)
        
#         if not val_accs:
#             logger.warning(f"No valid validation batches in epoch {epoch}")
#             continue
        
#         val_acc = sum(val_accs) / len(val_accs)
#         logger.info(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        
#         # Update learning rate
#         scheduler.step(val_acc)
        
#         # Save best model and handle early stopping
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#                 'family_to_idx': family_to_idx
#             }, 'best_model.pt')
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 logger.info

# if __name__ == '__main__':
#     main()
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Tuple

class CentroidLayer(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, n_centroids_per_class: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.n_centroids = n_centroids_per_class
        
        # Ensure at least one class
        if num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {num_classes}")
        
        logger.info(f"Initializing CentroidLayer with {num_classes} classes, {n_centroids_per_class} centroids per class")
        
        # Initialize centroids
        self.centroids = nn.Parameter(
            torch.randn(num_classes * n_centroids_per_class, feature_dim)
        )
        # Initialize with smaller values for better gradient flow
        self.centroids.data = self.centroids.data * 0.1
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with robust distance computation.
        Args:
            x: Input tensor of shape (batch_size, feature_dim)
        Returns:
            Tuple of (class_distances, outlier_scores)
        """
        batch_size = x.size(0)
        
        # Compute pairwise distances between inputs and centroids
        # Reshape x to (batch_size, 1, feature_dim)
        x_expanded = x.unsqueeze(1)
        # Reshape centroids to (1, num_total_centroids, feature_dim)
        centroids_expanded = self.centroids.unsqueeze(0)
        
        # Compute distances (batch_size, num_total_centroids)
        distances = torch.cdist(x_expanded, centroids_expanded).squeeze(1)
        
        # Initialize output tensors
        class_distances = torch.zeros(batch_size, self.num_classes, device=x.device)
        outlier_scores = torch.zeros(batch_size, device=x.device)
        
        # For each class, compute minimum distance to its centroids
        for class_idx in range(self.num_classes):
            start_idx = class_idx * self.n_centroids
            end_idx = start_idx + self.n_centroids
            
            # Get minimum distance to any centroid of this class
            class_min_distances = torch.min(distances[:, start_idx:end_idx], dim=1)[0]
            class_distances[:, class_idx] = class_min_distances
        
        # Compute outlier scores as normalized distance to nearest centroid
        min_distances = torch.min(class_distances, dim=1)[0]
        if min_distances.numel() > 0:  # Check if tensor is non-empty
            mean_min_dist = min_distances.mean()
            std_min_dist = min_distances.std() if min_distances.numel() > 1 else torch.tensor(1.0)
            outlier_scores = (min_distances - mean_min_dist) / (std_min_dist + 1e-6)
        
        return -class_distances, outlier_scores  # Negative distances for logits
    
    def _create_class_mask(self, num_classes: int, n_centroids: int) -> torch.Tensor:
        """Create a mask for mapping centroids to classes."""
        mask = torch.zeros(num_classes, num_classes * n_centroids)
        for i in range(num_classes):
            mask[i, i * n_centroids:(i + 1) * n_centroids] = 1.0
        return mask

class MalwareGNN(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int, num_classes: int,
                 n_centroids_per_class: int = 2):
        super().__init__()
        
        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {num_classes}")
            
        logger.info(f"Initializing MalwareGNN with {num_classes} classes")
        
        # Store dimensions
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Centroid layer for behavioral pattern detection
        self.centroid = CentroidLayer(
            feature_dim=hidden_dim,
            num_classes=num_classes,
            n_centroids_per_class=n_centroids_per_class
        )

    def forward(self, data):
        try:
            # Extract features and connectivity
            x, edge_index = data.x, data.edge_index
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            # Validate inputs
            if x.size(0) == 0:
                raise ValueError("Empty feature tensor")
                
            # First GCN layer
            x1 = self.conv1(x, edge_index)
            x1 = self.bn1(x1)
            x1 = F.relu(x1)
            x1 = self.dropout(x1)
            
            # Second GCN layer with residual connection
            x2 = self.conv2(x1, edge_index)
            x2 = self.bn2(x2)
            x2 = F.relu(x2)
            x2 = self.dropout(x2)
            x2 = x2 + x1  # Residual connection
            
            # Third GCN layer
            x3 = self.conv3(x2, edge_index)
            x3 = self.bn3(x3)
            x3 = F.relu(x3)
            x3 = x3 + x2  # Residual connection
            
            # Global pooling
            x = global_mean_pool(x3, batch)
            
            # Centroid-based classification
            logits, outlier_scores = self.centroid(x)
            
            return logits, outlier_scores
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(f"Input shapes - x: {x.shape}, edge_index: {edge_index.shape}")
            raise