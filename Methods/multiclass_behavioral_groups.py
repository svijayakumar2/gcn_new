import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import os
import glob
import json
from collections import defaultdict
from typing import Optional, Dict, List
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class BehavioralGroupEncoder:
    def __init__(self, behavioral_groups_path: str):
        """Initialize the encoder with behavioral groups mapping."""
        try:
            with open(behavioral_groups_path, 'r') as f:
                self.behavioral_groups = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading behavioral groups from {behavioral_groups_path}: {str(e)}")
            raise
            
        # Create family to group mapping
        self.family_to_group = {}
        for group_id, families in self.behavioral_groups.items():
            for family in families:
                self.family_to_group[family.lower()] = int(group_id)
        
        self.num_groups = len(self.behavioral_groups)
        print(f"Initialized encoder with {self.num_groups} behavioral groups")
        
    def get_num_classes(self) -> int:
        """Return number of behavioral groups (classes)."""
        return self.num_groups
        
    def transform(self, family: str) -> int:
        """Convert family name to behavioral group ID."""
        if family is None or family == 'benign':
            return -1  # Special case for benign samples
        
        family = family.lower()
        if family not in self.family_to_group:
            # Handle specific special cases
            if family == 'unknown':
                return -2
            if family.startswith('malware'):
                return -2
            
            if not hasattr(self, '_unknown_families_warned'):
                self._unknown_families_warned = set()
            
            if family not in self._unknown_families_warned:
                print(f"Warning: Unknown family '{family}', treating as unknown")
                self._unknown_families_warned.add(family)
            return -2  # Special case for unknown families
            
        return self.family_to_group[family]
        
    def get_group_weights(self, families: List[str]) -> torch.Tensor:
        """Compute class weights based on family distribution."""
        groups = [self.transform(f) for f in families if f not in (None, 'benign')]
        weights = compute_class_weight('balanced', classes=np.unique(groups), y=groups)
        return torch.FloatTensor(weights)


class MalwareGNN(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling and classification
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.classifier(x)


def load_batch(batch_file: str, encoder: BehavioralGroupEncoder, 
               batch_size: int = 32, device: torch.device = torch.device('cpu')) -> Optional[DataLoader]:
    """Load a batch file and convert families to behavioral groups."""
    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}")
        return None

    batch = torch.load(batch_file)
    processed = []
    group_counts = defaultdict(int)
    unknown_families = set()

    for graph in batch:
        try:
            # Convert family to behavioral group
            family = getattr(graph, 'family', None)
            group_id = encoder.transform(family)
            
            if group_id == -1:  # benign
                continue
            if group_id == -2:  # unknown family
                unknown_families.add(family)
                continue
                
            group_counts[group_id] += 1
            graph.y = torch.tensor(group_id, dtype=torch.long).to(device)
            
            # Move graph data to device
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            graph.edge_attr = graph.edge_attr.to(device)
            processed.append(graph)
            
        except Exception as e:
            print(f"Error processing graph: {str(e)}")
            continue

    if not processed:
        return None
        
    print(f"\nProcessed batch statistics:")
    print(f"Total samples: {len(processed)}")
    print(f"Group distribution: {dict(group_counts)}")
    if unknown_families:
        print(f"Unknown families: {unknown_families}")
        
    return DataLoader(processed, batch_size=batch_size, shuffle=True)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, encoder: BehavioralGroupEncoder, 
                device: torch.device, epochs: int = 20):
    """Train model with behavioral group classification and handle class imbalance."""
    print("\nStarting training with the following data distribution:")
    
    # Analyze class distribution in training data
    train_dist = defaultdict(int)
    for batch in train_loader:
        for label in batch.y.cpu().numpy():
            train_dist[int(label)] += 1
    
    total_samples = sum(train_dist.values())
    print("\nTraining data distribution:")
    for group_id, count in sorted(train_dist.items()):
        percentage = (count / total_samples) * 100
        print(f"Group {group_id}: {count} samples ({percentage:.1f}%)")
        
    # Compute inverse frequency weights for loss function
    class_weights = torch.zeros(encoder.get_num_classes(), device=device)
    for group_id, count in train_dist.items():
        class_weights[group_id] = total_samples / (len(train_dist) * count)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Compute class weights for weighted loss
    all_train_families = []
    for batch in train_loader:
        for graph in batch.to_data_list():
            if hasattr(graph, 'family'):
                all_train_families.append(graph.family)
                
    class_weights = encoder.get_group_weights(all_train_families).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training phase
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            total_loss += loss.item()
            
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        group_correct = defaultdict(int)
        group_total = defaultdict(int)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
                
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
                
                # Per-group accuracy
                for p, t in zip(pred, batch.y):
                    group_id = t.item()
                    group_total[group_id] += 1
                    if p == t:
                        group_correct[group_id] += 1
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print("\nPer-group validation accuracy:")
        for group_id in sorted(group_total.keys()):
            group_acc = 100 * group_correct[group_id] / group_total[group_id]
            print(f"Group {group_id}: {group_acc:.2f}% ({group_correct[group_id]}/{group_total[group_id]})")
        print("-" * 80)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get the absolute path to the base directory
    base_dir = os.path.abspath('/data/saranyav/gcn_new/bodmas_batches')
    print(f"Base directory: {base_dir}")
    behav_path = os.path.abspath('/data/saranyav/gcn_new/behavioral_analysis')
    # Check if directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    behavioral_groups_path = os.path.join(behav_path, 'behavioral_groups.json')
    encoder = BehavioralGroupEncoder(behavioral_groups_path)
    
    # Load data
    batch_files = {split: sorted(glob.glob(os.path.join(base_dir, split, 'batch_*.pt')))
                   for split in ['train', 'val']}
    
    train_loader = load_batch(batch_files['train'][0], encoder, device=device)
    val_loader = load_batch(batch_files['val'][0], encoder, device=device)
    
    if train_loader is None or val_loader is None:
        print("Error: Could not load data")
        return
        
    # Get feature dimensionality from first batch
    sample_batch = next(iter(train_loader))
    num_features = sample_batch.x.size(1)
    num_classes = encoder.get_num_classes()
    
    print(f"\nModel Configuration:")
    print(f"Number of input features: {num_features}")
    print(f"Number of classes (behavioral groups): {num_classes}")
    
    # Initialize model
    model = MalwareGNN(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=128
    ).to(device)
    
    # Use AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Train model
    train_model(model, train_loader, val_loader, optimizer, encoder, device, epochs=20)
    behav_path = os.path.abspath('/data/saranyav/gcn_new/behavioral_analysis')
    # Check if directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    behavioral_groups_path = os.path.join(behav_path, 'behavioral_groups.json')
    
    # Initialize behavioral group encoder
    if not os.path.exists(behavioral_groups_path):
        print(f"Behavioral groups file not found at: {behavioral_groups_path}")
        print("Current working directory:", os.getcwd())
        raise FileNotFoundError(f"Behavioral groups file not found: {behavioral_groups_path}")
        
    encoder = BehavioralGroupEncoder(behavioral_groups_path)
    
    # Load data
    batch_files = {split: sorted(glob.glob(os.path.join(base_dir, split, 'batch_*.pt')))
                   for split in ['train', 'val']}
    
    train_loader = load_batch(batch_files['train'][0], encoder, device=device)
    val_loader = load_batch(batch_files['val'][0], encoder, device=device)
    
    if train_loader is None or val_loader is None:
        print("Error: Could not load data")
        return
    
    # Initialize model
    num_features = train_loader.dataset.num_features
    num_classes = encoder.get_num_classes()
    model = MalwareGNN(num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(model, train_loader, val_loader, optimizer, encoder, device)


if __name__ == '__main__':
    main()
    
