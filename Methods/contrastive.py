import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import logging
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
import os
import glob
from typing import Dict, List, Tuple
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import logging
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import pandas as pd



def load_batch(batch_file, family_to_group, batch_size=32):
    """Load and preprocess a single batch file with robust error handling.
    
    Args:
        batch_file (str): Path to the batch file
        family_to_group (dict): Mapping from family names to group IDs
        batch_size (int): Size of batches to create
        
    Returns:
        DataLoader or None: DataLoader containing processed graphs
    """
    try:
        if not os.path.exists(batch_file):
            logger.warning(f"Batch file not found: {batch_file}")
            return None
            
        # Load batch data
        batch_data = torch.load(batch_file)
        if not batch_data:
            logger.warning(f"Empty batch file: {batch_file}")
            return None
            
        processed = []
        
        for graph in batch_data:
            try:
                # Verify it's a PyG Data object
                if not isinstance(graph, Data):
                    logger.error(f"Graph is not a PyG Data object: {type(graph)}")
                    continue
                
                # Create new graph object to ensure clean tensor creation
                new_graph = Data()
                
                # Copy and ensure basic tensors are on CPU initially
                new_graph.x = graph.x.cpu()
                new_graph.edge_index = graph.edge_index.cpu()
                
                # Process family label
                family = getattr(graph, 'family', 'none')
                if not family or family == '':
                    family = 'none'
                
                # Get group ID and create tensors on CPU
                group = family_to_group.get(family, -1)
                new_graph.group = torch.tensor([group], dtype=torch.long)
                new_graph.y = torch.tensor([group], dtype=torch.long)
                new_graph.family = family
                
                # Verify tensor dimensions
                if new_graph.x.dim() != 2:
                    logger.error(f"Unexpected x dimensions: {new_graph.x.shape}")
                    continue
                    
                if new_graph.edge_index.dim() != 2 or new_graph.edge_index.size(0) != 2:
                    logger.error(f"Unexpected edge_index dimensions: {new_graph.edge_index.shape}")
                    continue
                
                # Verify edge indices are within bounds
                if new_graph.edge_index.size(1) > 0:
                    max_idx = new_graph.edge_index.max().item()
                    if max_idx >= new_graph.x.size(0):
                        logger.error(f"Edge indices out of bounds. Max index: {max_idx}, num nodes: {new_graph.x.size(0)}")
                        continue
                
                processed.append(new_graph)
                
            except Exception as e:
                logger.error(f"Error processing graph: {str(e)}")
                continue
                
        if not processed:
            logger.warning(f"No valid graphs found in {batch_file}")
            return None
            
        # Create DataLoader
        try:
            loader = DataLoader(
                processed, 
                batch_size=min(batch_size, len(processed)),
                shuffle=True
            )
            
            # Verify first batch
            sample_batch = next(iter(loader))
            required_attrs = ['x', 'edge_index', 'batch', 'group', 'y', 'family']
            missing_attrs = [attr for attr in required_attrs if not hasattr(sample_batch, attr)]
            
            if missing_attrs:
                logger.error(f"Batch missing required attributes: {missing_attrs}")
                return None
                
            # Reset loader
            loader = DataLoader(
                processed, 
                batch_size=min(batch_size, len(processed)),
                shuffle=True
            )
            
            return loader
            
        except Exception as e:
            logger.error(f"Error creating DataLoader: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error loading batch {batch_file}: {str(e)}")
        return None
    
class ContrastiveTrainer:
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
    def train_epoch(self, train_files: List[str], criterion, family_to_group: Dict[str, int]) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_file in train_files:
            loader = load_batch(batch_file, family_to_group)
            if not loader:
                continue
                
            for batch in loader:
                try:
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    embeddings, group_logits, family_logits = self.model(batch)
                    
                    # Compute loss
                    loss = criterion(
                        embeddings,
                        group_logits,
                        family_logits,
                        batch.family,
                        self.device
                    )
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Log progress
                    if num_batches % 10 == 0:
                        logger.info(f"Batch {num_batches}, Loss: {loss.item():.4f}")
                    
                    # Clear cache periodically
                    if num_batches % 50 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error in training batch: {str(e)}")
                    continue
        
        return total_loss / max(1, num_batches)
    
    @torch.no_grad()
    def evaluate(self, val_files: List[str], criterion, family_to_group: Dict[str, int]) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        metrics = defaultdict(list)
        num_batches = 0
        
        for batch_file in val_files:
            loader = load_batch(batch_file, family_to_group)
            if not loader:
                continue
                
            for batch in loader:
                try:
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    embeddings, group_logits, family_logits = self.model(batch)
                    
                    # Loss
                    loss = criterion(embeddings, group_logits, family_logits, batch.family, self.device)
                    total_loss += loss.item()
                    
                    # Compute accuracies
                    pred_groups = group_logits.argmax(dim=1)
                    true_groups = batch.group
                    
                    # Group accuracy (excluding unknown)
                    known_mask = true_groups != -999
                    if known_mask.any():
                        group_acc = (pred_groups[known_mask] == true_groups[known_mask]).float().mean()
                        metrics['group_acc'].append(group_acc.item())
                    
                    # Family accuracy for correct groups
                    for i, (pred_group, true_group) in enumerate(zip(pred_groups, true_groups)):
                        if pred_group == true_group and true_group != -999:
                            family_logits_group = family_logits[str(pred_group.item())]
                            pred_family = family_logits_group[i].argmax()
                            true_family = batch.y[i]
                            metrics['family_acc'].append((pred_family == true_family).float().item())
                    
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {str(e)}")
                    continue
        
        return {
            'val_loss': total_loss / max(1, num_batches),
            'group_accuracy': np.mean(metrics['group_acc']) if metrics['group_acc'] else 0.0,
            'family_accuracy': np.mean(metrics['family_acc']) if metrics['family_acc'] else 0.0
        }

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import json
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from collections import defaultdict
import os
import glob
from typing import List, Dict, Tuple
import sys
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, Batch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'contrastive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, family_to_group, alpha=0.3):
        super().__init__()
        self.family_to_group = family_to_group
        self.alpha = alpha
        
        # Create mappings
        self.group_to_families = defaultdict(list)
        for family, group in family_to_group.items():
            self.group_to_families[group].append(family)
            
        self.family_to_idx = {}
        for group_id, families in self.group_to_families.items():
            self.family_to_idx[group_id] = {
                fam: idx for idx, fam in enumerate(sorted(families))
            }
    
    def forward(self, embeddings, group_logits, family_logits, true_families, device):
        # Convert true_families to list if it's not already
        if not isinstance(true_families, list):
            true_families = [true_families]
            
        # Convert family labels to group labels with proper error handling
        true_groups = []
        for fam in true_families:
            group = self.family_to_group.get(fam, -1)
            true_groups.append(group)
        
        true_groups = torch.tensor(true_groups, dtype=torch.long).to(device)
        
        # Contrastive loss computation
        embeddings_norm = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t())
        
        # Create masks for positive pairs (same group)
        pos_mask = torch.eq(true_groups.unsqueeze(0), true_groups.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # Remove self-pairs
        
        # Get negative pairs (different groups)
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)
        
        # Temperature scaling
        sim_matrix = sim_matrix / 0.07  # temperature parameter
        
        # Compute contrastive loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log((exp_sim * neg_mask).sum(dim=1, keepdim=True))
        contrastive_loss = -(log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        
        # Group classification loss
        group_loss = F.cross_entropy(
            group_logits, 
            true_groups,
            label_smoothing=0.1,
            ignore_index=-1
        )
        
        # Family classification loss
        family_loss = 0
        valid_samples = 0
        
        for group_id in self.group_to_families:
            group_mask = (true_groups == group_id)
            if not group_mask.any():
                continue
                
            group_logits_subset = family_logits[str(group_id)][group_mask]
            
            true_indices = []
            for fam in true_families:
                if self.family_to_group.get(fam) == group_id:
                    idx = self.family_to_idx[group_id].get(fam, 0)
                    true_indices.append(idx)
            
            if not true_indices:
                continue
                
            true_indices = torch.tensor(true_indices, dtype=torch.long).to(device)
            
            group_weight = len(self.group_to_families[group_id]) / len(self.family_to_group)
            family_loss += group_weight * F.cross_entropy(
                group_logits_subset,
                true_indices,
                label_smoothing=0.1
            )
            valid_samples += 1
        
        if valid_samples > 0:
            family_loss /= valid_samples
            
        # Combine losses with regularization
        total_loss = contrastive_loss.mean() + self.alpha * group_loss + (1 - self.alpha) * family_loss
        l2_reg = 0.001 * torch.norm(embeddings, p=2, dim=1).mean()
        total_loss += l2_reg
        
        return total_loss

class ContrastiveGNN(torch.nn.Module):
    def __init__(self, num_features, num_groups=16, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Base GNN layers
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, embedding_dim)
        
        # Classification heads
        self.group_classifier = torch.nn.Linear(embedding_dim, num_groups)
        self.family_classifiers = torch.nn.ModuleDict()
    
    def add_family_classifier(self, group_id: str, num_families: int):
        self.family_classifiers[str(group_id)] = torch.nn.Linear(
            self.embedding_dim, num_families
        )
    
    def get_embeddings(self, data):
        device = next(self.parameters()).device
        
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        embeddings = global_mean_pool(x, batch)
        return embeddings
    
    def forward(self, data):
        embeddings = self.get_embeddings(data)
        
        # Group classification
        group_logits = self.group_classifier(embeddings)
        
        # Family classification
        family_logits = {}
        for group_id in self.family_classifiers:
            family_logits[group_id] = self.family_classifiers[group_id](embeddings)
            
        return embeddings, group_logits, family_logits

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_file in train_loader:
        batch_loader = load_batch(batch_file, criterion.family_to_group)
        if not batch_loader:
            continue
        
        for batch in batch_loader:
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                embeddings, group_logits, family_logits = model(batch)
                loss = criterion(embeddings, group_logits, family_logits, batch.family, device)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    logger.info(f"Batch {num_batches}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error in batch: {str(e)}")
                continue
    
    return total_loss / max(1, num_batches)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch_file in val_loader:
            batch_loader = load_batch(batch_file, criterion.family_to_group)
            if not batch_loader:
                continue
            
            for batch in batch_loader:
                try:
                    batch = batch.to(device)
                    embeddings, group_logits, family_logits = model(batch)
                    
                    # Compute loss
                    loss = criterion(embeddings, group_logits, family_logits, batch.family, device)
                    total_loss += loss.item()
                    
                    # Compute metrics
                    pred_groups = group_logits.argmax(dim=1)
                    true_groups = batch.group
                    
                    # Group accuracy
                    valid_mask = true_groups != -1
                    if valid_mask.any():
                        group_acc = (pred_groups[valid_mask] == true_groups[valid_mask]).float().mean()
                        metrics['group_acc'].append(group_acc.item())
                        
                except Exception as e:
                    logger.error(f"Error in validation: {str(e)}")
                    continue
    
    return {
        'val_loss': total_loss / max(1, len(metrics['group_acc'])),
        'group_acc': np.mean(metrics['group_acc']) if metrics['group_acc'] else 0.0
    }

def prepare_data(base_dir='bodmas_batches'):
    split_files = defaultdict(list)
    file_timestamps = {}
    
    logger.info("Starting data preparation...")
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        batch_files = glob.glob(os.path.join(split_dir, 'batch_*.pt'))
        
        for file_path in batch_files:
            try:
                batch = torch.load(file_path)
                if batch and len(batch) > 0:
                    file_timestamps[file_path] = getattr(batch[0], 'timestamp', None)
                    split_files[split].append(file_path)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
    
    for split in split_files:
        split_files[split].sort(key=lambda x: file_timestamps.get(x, pd.Timestamp.min))
    
    return dict(split_files), file_timestamps

def main():
    config = {
        'batch_dir': '/data/saranyav/gcn_new/bodmas_batches',
        'behavioral_groups': '/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json',
        'embedding_dim': 256,
        'num_epochs': 100,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    try:
        device = torch.device(config['device'])
        logger.info(f"Using device: {device}")
        
        # Load group mappings
        with open(config['behavioral_groups']) as f:
            group_data = json.load(f)
            
        family_to_group = {}
        for group_id, families in group_data.items():
            for family in families:
                family_to_group[family] = int(group_id)
        
        # Prepare data
        split_files, file_timestamps = prepare_data(config['batch_dir'])
        
        if not any(split_files.values()):
            logger.error("No data found!")
            return
            
        # Get feature dimension
        first_batch = torch.load(split_files['train'][0])
        num_features = first_batch[0].x.size(1)
        num_groups = len(set(family_to_group.values()))
        logger.info(f"Features: {num_features}, Groups: {num_groups}")
        
        # Initialize model and training components
        model = ContrastiveGNN(
            num_features=num_features,
            num_groups=num_groups,
            embedding_dim=config['embedding_dim']
        ).to(device)
        
        for group_id in set(family_to_group.values()):
            group_families = [f for f, g in family_to_group.items() if g == group_id]
            model.add_family_classifier(str(group_id), len(group_families))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = ContrastiveLoss(family_to_group)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(config['num_epochs']):
            train_loss = train_epoch(model, split_files['train'], optimizer, criterion, device)
            val_metrics = validate(model, split_files['val'], criterion, device)
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_metrics['val_loss']:.4f}, "
                f"Val Group Acc={val_metrics['group_acc']:.4f}"
            )
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics
                }, 'best_contrastive_model.pt')
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)