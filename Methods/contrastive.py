import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch, DataLoader
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from datetime import datetime
from pathlib import Path
import json
import os
import glob
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'malware_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            embeddings: Normalized embeddings (N x D)
            labels: Labels for each embedding (N)
        """
        device = embeddings.device
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create label matrix where (i,j) = 1 if samples i and j have same label
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Remove diagonal elements
        mask = torch.eye(len(embeddings), dtype=torch.bool, device=device)
        label_matrix = label_matrix.masked_fill(mask, False)
        
        # For each sample, compute loss over its positive pairs
        positive_sim = sim_matrix.masked_fill(~label_matrix, float('-inf'))
        negative_sim = sim_matrix.masked_fill(label_matrix, float('-inf'))
        
        # Compute log softmax
        logits = torch.cat([positive_sim, negative_sim], dim=1)
        labels = torch.zeros(len(embeddings), device=device, dtype=torch.long)
        
        return F.cross_entropy(logits, labels)

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning Loss for fine-grained family classification"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Normalized embeddings (N x D)
            labels: Family labels
        """
        device = embeddings.device
        
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create mask for valid positive pairs
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        pos_mask = pos_mask.float()
        
        # Mask out self-contrast
        identity_mask = torch.eye(embeddings.shape[0], dtype=torch.bool, device=device)
        pos_mask.masked_fill_(identity_mask, 0)
        
        # Compute log probability
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute mean of positive pairs
        mean_log_prob = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        
        return -mean_log_prob.mean()

class MalwareGNN(torch.nn.Module):
    def __init__(self, num_features, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Base Graph Neural Network layers
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, embedding_dim)
        
        # Phase-specific heads
        self.binary_head = torch.nn.Linear(embedding_dim, 2)  # Malware vs Benign
        self.family_head = torch.nn.Linear(embedding_dim, embedding_dim)  # For contrastive family learning
        
    def get_embeddings(self, data):
        """Extract normalized graph embeddings"""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global pooling
        embeddings = global_mean_pool(x, batch)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward(self, data, phase='binary'):
        embeddings = self.get_embeddings(data)
        
        if phase == 'binary':
            return self.binary_head(embeddings)
        else:
            # For family classification, use the family head
            return self.family_head(embeddings)

class TemporalFamilyTracker:
    def __init__(self, embedding_dim=256, distance_threshold=0.4):
        self.embedding_dim = embedding_dim
        self.distance_threshold = distance_threshold
        self.family_centroids = {}
        self.temporal_stats = defaultdict(list)
        self.new_families = set()
        
    def update_centroids(self, embeddings, families, timestamps):
        """Update family centroids with temporal weighting"""
        family_updates = defaultdict(list)
        
        for emb, fam, ts in zip(embeddings, families, timestamps):
            family_updates[fam].append({
                'embedding': emb,
                'timestamp': pd.to_datetime(ts)
            })
            
        for family, updates in family_updates.items():
            sorted_updates = sorted(updates, key=lambda x: x['timestamp'])
            
            # Exponential time decay weights
            weights = np.exp(np.linspace(-1, 0, len(sorted_updates)))
            embeddings = torch.stack([u['embedding'] for u in sorted_updates])
            
            # Update centroid with weighted average
            weighted_centroid = torch.sum(embeddings * weights.reshape(-1, 1), dim=0)
            weighted_centroid = weighted_centroid / weights.sum()
            
            self.family_centroids[family] = {
                'centroid': weighted_centroid,
                'last_updated': sorted_updates[-1]['timestamp'],
                'num_samples': len(sorted_updates)
            }
            
    def detect_new_family(self, embedding):
        """Detect if an embedding represents a new family"""
        if not self.family_centroids:
            return True
            
        # Compute distances to all known centroids
        distances = []
        for family, data in self.family_centroids.items():
            dist = torch.norm(embedding - data['centroid'])
            distances.append(dist.item())
            
        min_distance = min(distances)
        return min_distance > self.distance_threshold
    
    def track_evolution(self, embeddings, families, timestamps):
        """Track family evolution over time"""
        for emb, fam, ts in zip(embeddings, families, timestamps):
            self.temporal_stats[fam].append({
                'embedding': emb,
                'timestamp': pd.to_datetime(ts),
                'distance_to_centroid': torch.norm(
                    emb - self.family_centroids[fam]['centroid']
                ).item() if fam in self.family_centroids else None
            })

def train_phase(model, train_loader, criterion, optimizer, device, phase):
    """Train one epoch of a specific phase"""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        if phase == 'binary':
            # Binary classification phase
            logits = model(batch, phase='binary')
            loss = criterion(logits, batch.is_malware)
        else:
            # Family classification phase
            embeddings = model(batch, phase='family')
            loss = criterion(embeddings, batch.family)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, loader, family_tracker, device, phase='binary'):
    """Evaluate model on given loader"""
    model.eval()
    metrics = defaultdict(list)
    all_embeddings = []
    all_families = []
    all_timestamps = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings = model.get_embeddings(batch)
            
            if phase == 'binary':
                logits = model.binary_head(embeddings)
                preds = torch.argmax(logits, dim=1)
                metrics['accuracy'].append(
                    (preds == batch.is_malware).float().mean().item()
                )
            else:
                # Store embeddings and metadata for family analysis
                all_embeddings.extend(embeddings.cpu())
                all_families.extend(batch.family)
                all_timestamps.extend(batch.timestamp)
                
                # Detect new families
                for emb in embeddings:
                    is_new = family_tracker.detect_new_family(emb)
                    metrics['new_family_detection'].append(is_new)
    
    # Update family tracker with new embeddings
    if phase != 'binary' and all_embeddings:
        family_tracker.update_centroids(
            all_embeddings, all_families, all_timestamps
        )
        family_tracker.track_evolution(
            all_embeddings, all_families, all_timestamps
        )
    
    return {k: np.mean(v) for k, v in metrics.items()}

def load_batch(batch_file, device=None):
    """Load and preprocess a single batch file."""
    try:
        if not os.path.exists(batch_file):
            logger.warning(f"Batch file not found: {batch_file}")
            return None
            
        batch_data = torch.load(batch_file)
        if not batch_data:
            logger.warning(f"Empty batch file: {batch_file}")
            return None
            
        processed = []
        for graph in batch_data:
            try:
                if not isinstance(graph, Data):
                    continue
                    
                # Ensure binary label exists (malware vs benign)
                is_malware = getattr(graph, 'is_malware', None)
                if is_malware is None:
                    # Assume if it has a family label, it's malware
                    is_malware = hasattr(graph, 'family')
                graph.is_malware = torch.tensor(int(is_malware), dtype=torch.long)
                
                # Handle family label
                family = getattr(graph, 'family', 'benign')
                graph.family = family if is_malware else 'benign'
                
                # Ensure timestamp exists
                if not hasattr(graph, 'timestamp'):
                    graph.timestamp = pd.Timestamp.now()
                    
                # Verify tensor dimensions
                if graph.x.dim() != 2:
                    continue
                if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
                    continue
                    
                # Verify edge indices
                if graph.edge_index.size(1) > 0:
                    max_idx = graph.edge_index.max().item()
                    if max_idx >= graph.x.size(0):
                        continue
                
                processed.append(graph)
                
            except Exception as e:
                logger.error(f"Error processing graph: {str(e)}")
                continue
                
        if not processed:
            return None
            
        return DataLoader(processed, batch_size=32, shuffle=True)
        
    except Exception as e:
        logger.error(f"Error loading batch {batch_file}: {str(e)}")
        return None

def prepare_temporal_data(base_dir='bodmas_batches'):
    """Prepare datasets with temporal ordering."""
    split_files = defaultdict(list)
    file_timestamps = {}
    
    logger.info("Starting data preparation...")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        # Collect all batch files
        batch_files = glob.glob(os.path.join(split_dir, 'batch_*.pt'))
        
        # Get timestamp from first sample in each batch
        for file_path in batch_files:
            try:
                batch = torch.load(file_path)
                if batch and len(batch) > 0:
                    file_timestamps[file_path] = getattr(batch[0], 'timestamp', None)
                    split_files[split].append(file_path)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
    
    # Sort files by timestamp within each split
    for split in split_files:
        split_files[split].sort(key=lambda x: file_timestamps.get(x, pd.Timestamp.min))
        
    return dict(split_files), file_timestamps

class TemporalDataLoader:
    """Custom data loader for temporal batches"""
    def __init__(self, file_list, batch_size=32, device=None):
        self.file_list = file_list
        self.batch_size = batch_size
        self.device = device
        self.current_loader = None
        self.current_file_idx = 0
        
    def __iter__(self):
        self.current_file_idx = 0
        self.current_loader = None
        return self
        
    def __next__(self):
        if self.current_loader is None or not self.current_loader:
            if self.current_file_idx >= len(self.file_list):
                raise StopIteration
                
            # Load next batch file
            while self.current_file_idx < len(self.file_list):
                next_loader = load_batch(
                    self.file_list[self.current_file_idx],
                    self.device
                )
                self.current_file_idx += 1
                
                if next_loader is not None:
                    self.current_loader = iter(next_loader)
                    break
            
            if self.current_loader is None:
                raise StopIteration
        
        try:
            batch = next(self.current_loader)
            return batch
        except StopIteration:
            self.current_loader = None
            return next(self)
            
    def __len__(self):
        return len(self.file_list)

def main():
    # Configuration
    config = {
        'batch_dir': 'bodmas_batches',
        'embedding_dim': 256,
        'num_epochs_binary': 10,
        'num_epochs_family': 20,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    device = torch.device(config['device'])
    
    # Prepare temporal data
    logger.info("Preparing temporal datasets...")
    split_files, timestamps = prepare_temporal_data(config['batch_dir'])
    
    if not all(split in split_files for split in ['train', 'val', 'test']):
        logger.error("Missing required data splits!")
        return
        
    # Create temporal data loaders
    train_loader = TemporalDataLoader(split_files['train'], config['batch_size'], device)
    val_loader = TemporalDataLoader(split_files['val'], config['batch_size'], device)
    test_loader = TemporalDataLoader(split_files['test'], config['batch_size'], device)
    
    # Get feature dimension from first batch
    try:
        first_batch = next(iter(train_loader))
        num_features = first_batch.x.size(1)
        logger.info(f"Number of features: {num_features}")
    except Exception as e:
        logger.error(f"Error loading first batch: {str(e)}")
        return
    
    # Initialize models and trackers
    model = MalwareGNN(num_features=num_features, embedding_dim=config['embedding_dim']).to(device)
    family_tracker = TemporalFamilyTracker(embedding_dim=config['embedding_dim'])
    
    # Initialize losses
    binary_criterion = torch.nn.CrossEntropyLoss()
    contrastive_criterion = SupConLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Phase 1: Binary Classification
    logger.info("Starting binary classification phase...")
    for epoch in range(config['num_epochs_binary']):
        train_loss = train_phase(
            model, train_loader, binary_criterion, 
            optimizer, device, phase='binary'
        )
        metrics = evaluate(model, val_loader, family_tracker, device, phase='binary')
        logger.info(f"Binary Phase - Epoch {epoch}: Loss={train_loss:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Phase 2: Family Classification
    logger.info("Starting family classification phase...")
    for epoch in range(config['num_epochs_family']):
        train_loss = train_phase(
            model, train_loader, contrastive_criterion, 
            optimizer, device, phase='family'
        )
        metrics = evaluate(model, val_loader, family_tracker, device, phase='family')
        logger.info(
            f"Family Phase - Epoch {epoch}: Loss={train_loss:.4f}, "
            f"New Family Detection Rate={metrics['new_family_detection']:.4f}"
        )
    
    # Final evaluation on test set
    logger.info("Running final evaluation...")
    test_metrics = evaluate(model, test_loader, family_tracker, device, phase='family')
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'family_evolution': family_tracker.temporal_stats,
        'new_families': list(family_tracker.new_families)
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()