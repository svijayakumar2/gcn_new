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
                
                # Handle predictions for novel classes
                probs = F.softmax(output, dim=1)
                max_probs, pred = torch.max(probs, dim=1)
                
                # Mark as novel class if outlier score is high or confidence is low
                novel_mask = (outlier_scores > 0.7) | (max_probs < 0.3)
                # Only compute loss for known classes
                known_mask = batch.y < self.model.num_classes
                
                if known_mask.any():
                    loss = criterion(output[known_mask], batch.y[known_mask])
                    total_loss += loss.item()
                
                # Store predictions and labels
                all_preds.append(pred)
                all_labels.append(batch.y)
                all_outlier_scores.append(outlier_scores)
        
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

    def compute_metrics(self, y_true, y_pred, num_classes):
        """
        Compute detailed classification metrics with novel class handling.
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
        
        # Handle novel classes
        known_mask = y_true < num_classes
        
        # Compute confusion matrix only for known classes
        if known_mask.any():
            known_true = y_true[known_mask]
            known_pred = y_pred[known_mask]
            conf_matrix = confusion_matrix(known_true, known_pred, labels=range(num_classes))
            
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
            if total_samples > 0:
                weighted_precision = sum(m['precision'] * m['support'] for m in metrics['per_class'].values()) / total_samples
                weighted_recall = sum(m['recall'] * m['support'] for m in metrics['per_class'].values()) / total_samples
                weighted_f1 = sum(m['f1'] * m['support'] for m in metrics['per_class'].values()) / total_samples
                accuracy = (known_true == known_pred).mean()
                
                metrics['overall'] = {
                    'precision': weighted_precision,
                    'recall': weighted_recall,
                    'f1': weighted_f1,
                    'accuracy': accuracy
                }
        
        # Add novel class detection metrics
        novel_true = ~known_mask
        novel_pred = y_pred >= num_classes
        
        novel_precision = np.mean(novel_true[novel_pred]) if novel_pred.any() else 0
        novel_recall = np.mean(novel_pred[novel_true]) if novel_true.any() else 0
        novel_f1 = 2 * (novel_precision * novel_recall) / (novel_precision + novel_recall) if (novel_precision + novel_recall) > 0 else 0
        
        metrics['novel_detection'] = {
            'precision': novel_precision,
            'recall': novel_recall,
            'f1': novel_f1,
            'support': int(novel_true.sum())
        }
        
        return metrics
      
    
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
        self.num_known_classes = num_classes
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