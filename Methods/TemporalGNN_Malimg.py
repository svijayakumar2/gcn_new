import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.nn as pyg_nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import os
from pathlib import Path
import random
from PIL import Image
import logging
from collections import defaultdict
import json
from sklearn.metrics import precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalwareGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_layers=4, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes  # Store num_classes as class attribute
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Global pooling
        self.pool = pyg_nn.global_add_pool
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Novelty detection head - outputs distance to nearest centroid
        self.novelty_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = dropout
        
        # Initialize centroids for known classes
        self.register_buffer('class_centroids', torch.zeros(num_classes, hidden_dim))
        self.register_buffer('centroid_counts', torch.zeros(num_classes))
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolution layers
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Get logits and feature representation
        logits = self.classifier(x)
        
        # Compute distance to nearest centroid for novelty detection
        if self.training:
            # During training, update centroids
            with torch.no_grad():
                for i in range(len(self.class_centroids)):
                    mask = (data.y == i) & (~data.is_novel)
                    if mask.any():
                        class_samples = x[mask]
                        self.class_centroids[i] = (
                            (self.class_centroids[i] * self.centroid_counts[i] + class_samples.sum(0)) /
                            (self.centroid_counts[i] + len(class_samples))
                        )
                        self.centroid_counts[i] += len(class_samples)
        
        # Compute distances to centroids
        distances = torch.cdist(x, self.class_centroids)
        min_distances = distances.min(dim=1)[0]
        novelty_scores = self.novelty_detector(x)
        
        return logits, novelty_scores

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalwareTrainer:
    def __init__(self, model, device, lr=0.001, weight_decay=1e-4):
        self.model = model
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=False  # Changed to False to avoid deprecation warning
        )
        
    def compute_class_weights(self, loader):
        """Compute class weights based on class distribution."""
        class_counts = defaultdict(int)
        total_samples = 0
        
        for batch in loader:
            labels = batch.y.cpu().numpy()
            for label in labels:
                if not batch.is_novel.any():  # Only count known samples
                    class_counts[label] += 1
                    total_samples += 1
        
        if total_samples == 0:
            return None
            
        num_classes = len(class_counts)
        weights = torch.zeros(num_classes)
        
        for class_idx in range(num_classes):
            count = class_counts[class_idx]
            if count > 0:
                weights[class_idx] = total_samples / (num_classes * count)
            else:
                weights[class_idx] = 1.0
                
        return weights.to(self.device)
        
    def train_epoch(self, loader, class_weights=None):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        if class_weights is None:
            num_classes = getattr(self.model, 'num_classes', None)
            if num_classes is None:
                # Try to infer from the classifier's output layer
                num_classes = self.model.classifier[-1].out_features
            class_weights = torch.ones(num_classes, device=self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, novelty_scores = self.model(batch)
            
            # Only compute classification loss for known samples
            known_mask = ~batch.is_novel
            if known_mask.any():
                # Classification loss
                cls_loss = criterion(logits[known_mask], batch.y[known_mask])
                
                # Novelty detection loss
                novelty_loss = F.binary_cross_entropy(
                    novelty_scores.squeeze(),
                    batch.is_novel.float()
                )
                
                # Balance losses
                novel_weight = batch.is_novel.float().mean()
                known_weight = 1 - novel_weight
                loss = (known_weight * cls_loss) + (novel_weight * novelty_loss)
                
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                pred = logits.argmax(dim=1)
                correct += pred[known_mask].eq(batch.y[known_mask]).sum().item()
                total += known_mask.sum().item()
                total_loss += loss.item()
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total if total > 0 else 0
        }

    def evaluate(self, loader, class_weights=None):
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            all_preds = []
            all_labels = []
            all_is_novel = []
            all_novelty_scores = []
            
            if class_weights is None:
                class_weights = torch.ones(self.model.num_classes, device=self.device)
                
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    logits, novelty_scores = self.model(batch)
                    
                    # Only evaluate on known samples
                    known_mask = ~batch.is_novel
                    if known_mask.any():
                        # Classification loss
                        loss = criterion(logits[known_mask], batch.y[known_mask])
                        total_loss += loss.item()
                        
                        # Get predictions
                        pred = logits.argmax(dim=1)
                        correct += pred[known_mask].eq(batch.y[known_mask]).sum().item()
                        total += known_mask.sum().item()
                    
                    # Store predictions and scores
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
                    all_is_novel.extend(batch.is_novel.cpu().numpy())
                    all_novelty_scores.extend(novelty_scores.squeeze().cpu().numpy())
            
            # Convert lists to numpy arrays for easier processing
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_is_novel = np.array(all_is_novel)
            all_novelty_scores = np.array(all_novelty_scores)
            
            # Calculate metrics for known samples
            known_mask = ~all_is_novel
            known_metrics = {}
            if known_mask.any():
                known_metrics['f1'] = f1_score(
                    all_labels[known_mask],
                    all_preds[known_mask],
                    average='weighted'
                )
                known_metrics['accuracy'] = (all_preds[known_mask] == all_labels[known_mask]).mean()
                known_metrics['precision'] = precision_score(
                    all_labels[known_mask],
                    all_preds[known_mask],
                    average='weighted'
                )
                known_metrics['recall'] = recall_score(
                    all_labels[known_mask],
                    all_preds[known_mask],
                    average='weighted'
                )
            else:
                known_metrics = {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
                
            # Calculate metrics for novelty detection
            # Use 0.5 as threshold for novelty scores
            novel_preds = (all_novelty_scores > 0.5).astype(int)
            novel_metrics = {
                'precision': precision_score(all_is_novel, novel_preds),
                'recall': recall_score(all_is_novel, novel_preds),
                'f1': f1_score(all_is_novel, novel_preds)
            } if len(all_is_novel) > 0 else {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            metrics = {
                'loss': total_loss / len(loader),
                'accuracy': correct / total if total > 0 else 0,
                'known': known_metrics,
                'novel': novel_metrics,
                'predictions': {
                    'preds': all_preds,
                    'labels': all_labels,
                    'is_novel': all_is_novel,
                    'novelty_scores': all_novelty_scores
                }
            }
            
            return metrics
        
    def log_metrics(self, metrics, split="train"):
        """Log metrics for a given split."""
        logger.info(f"\n{split.capitalize()} Metrics:")
        logger.info(f"Loss: {metrics['loss']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        if 'predictions' in metrics:
            novel_count = sum(metrics['predictions']['is_novel'])
            total_count = len(metrics['predictions']['is_novel'])
            logger.info(f"Novel samples: {novel_count}/{total_count} ({novel_count/total_count*100:.2f}%)")

class MalwareDataLoader:
    def __init__(self, batch_dir: Path, novel_ratio: float = 0.1):
        """Initialize data loader with novel family detection."""
        self.batch_dir = Path(batch_dir)
        self.novel_ratio = novel_ratio
        
        # Load family mappings
        mappings_file = self.batch_dir / 'family_mappings.json'
        if not mappings_file.exists():
            raise FileNotFoundError(f"Family mappings file not found: {mappings_file}")
            
        with open(mappings_file) as f:
            mappings = json.load(f)
            self.family_to_idx = mappings['family_to_idx']
            self.idx_to_family = {int(k): v for k, v in mappings['idx_to_family'].items()}
        
        # Select novel families
        self.novel_families = self._select_novel_families()
        
        # Create mapping for known families only
        all_families = list(self.family_to_idx.keys())
        known_families = [f for f in all_families if f not in self.novel_families]
        self.known_family_to_idx = {f: i for i, f in enumerate(known_families)}
        
        logger.info(f"Selected {len(self.novel_families)} novel families: {sorted(self.novel_families)}")
        logger.info(f"Known families: {sorted(known_families)}")
        logger.info(f"Total known families: {len(self.known_family_to_idx)}")

    def _select_novel_families(self) -> set:
        """Select a portion of families to be treated as novel."""
        all_families = list(self.family_to_idx.keys())
        num_novel = max(1, int(len(all_families) * self.novel_ratio))
        return set(random.sample(all_families, num_novel))

    def load_split(self, split: str, batch_size: int = 32):
        """Load a data split with novelty information."""
        split_dir = self.batch_dir / split
        batch_files = sorted(list(split_dir.glob('batch_*.pt')))
        
        if not batch_files:
            raise ValueError(f"No batch files found in {split_dir}")

        all_graphs = []
        stats = {
            'known_samples': 0,
            'novel_samples': 0,
            'known_families': defaultdict(int),
            'novel_families': defaultdict(int)
        }

        for batch_file in batch_files:
            try:
                logger.info(f"Processing batch file: {batch_file.name}")
                batch_data = torch.load(batch_file)
                
                for graph in batch_data:
                    if not hasattr(graph, 'family'):
                        logger.warning(f"Graph missing family attribute in {batch_file.name}")
                        continue

                    family = graph.family
                    is_novel = family in self.novel_families
                    
                    # Skip novel families during training
                    if split == 'train' and is_novel:
                        continue

                    # Add novelty information to graph
                    graph.is_novel = torch.tensor(is_novel, dtype=torch.bool)
                    
                    # For known families, use new mapping
                    if not is_novel:
                        graph.y = torch.tensor(self.known_family_to_idx[family])
                    else:
                        # For novel families, assign dummy label
                        graph.y = torch.tensor(0)

                    # Update statistics
                    if is_novel:
                        stats['novel_samples'] += 1
                        stats['novel_families'][family] += 1
                    else:
                        stats['known_samples'] += 1
                        stats['known_families'][family] += 1

                    all_graphs.append(graph)

            except Exception as e:
                logger.error(f"Error loading {batch_file}: {str(e)}")
                continue

        if not all_graphs:
            raise ValueError(f"No valid graphs loaded for {split} split")

        logger.info(f"\nSplit statistics for {split}:")
        logger.info(f"Known samples: {stats['known_samples']}")
        logger.info(f"Novel samples: {stats['novel_samples']}")
        logger.info(f"Known families: {len(stats['known_families'])}")
        logger.info(f"Novel families: {len(stats['novel_families'])}")

        loader = DataLoader(
            all_graphs,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=2,
            pin_memory=True,
            #follow_batch=['x', 'edge_index']
        )

        return loader, stats

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize data loader
    data_loader = MalwareDataLoader(
        batch_dir=Path('/data/datasets/malimg/batches'),
        novel_ratio=0.1
    )

    # Load data splits
    train_loader, train_stats = data_loader.load_split('train', batch_size=32)
    val_loader, val_stats = data_loader.load_split('val', batch_size=32)
    test_loader, test_stats = data_loader.load_split('test', batch_size=32)

    # Initialize model
    # Calculate number of known classes (excluding novel families)
    known_families = set(data_loader.family_to_idx.keys()) - data_loader.novel_families
    num_known_classes = len(known_families)
    
    num_features = 14  # From your CFG feature extraction
    model = MalwareGNN(
        num_node_features=num_features,
        hidden_dim=256,
        num_classes=num_known_classes,  # Only use known family count
        num_layers=4,
        dropout=0.2
    ).to(device)

    # Initialize trainer
    trainer = MalwareTrainer(
        model=model,
        device=device,
        #idx_to_family=data_loader.idx_to_family,
        lr=0.001,
        weight_decay=1e-4
    )

    # Training parameters
    num_epochs = 50
    patience = 10
    best_f1 = 0
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Log metrics
        logger.info(f"\nEpoch {epoch + 1}:")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val Known F1: {val_metrics['known']['f1']:.4f}")
        logger.info(f"Val Novel Precision: {val_metrics['novel']['precision']:.4f}")
        logger.info(f"Val Novel Recall: {val_metrics['novel']['recall']:.4f}")

        # Early stopping check
        current_f1 = val_metrics['known']['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_metrics': val_metrics
            }, 'best_model_GNN.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Update learning rate
        trainer.scheduler.step(current_f1)

    # Final evaluation
    logger.info("\nFinal Evaluation on Test Set")
    checkpoint = torch.load('best_model_GNN.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = trainer.evaluate(test_loader)

    # Print final results
    logger.info("\nTest Set Results:")
    logger.info(f"Known Families F1: {test_metrics['known']['f1']:.4f}")
    logger.info(f"Novel Detection Precision: {test_metrics['novel']['precision']:.4f}")
    logger.info(f"Novel Detection Recall: {test_metrics['novel']['recall']:.4f}")

    # save stats to json
    # convert from ndarray first 
    stats = {
        "known_samples": test_stats['known_samples'],
        "novel_samples": test_stats['novel_samples'],
        "known_families": dict(test_stats['known_families']),
        "novel_families": dict(test_stats['novel_families']),
        "known_f1": test_metrics['known']['f1'],
        "known_precision": test_metrics['known']['precision'],
        "known_recall": test_metrics['known']['recall'],
        "novel_precision": test_metrics['novel']['precision'],
        "novel_recall": test_metrics['novel']['recall']
    }

    with open('malimg_gnn_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()  