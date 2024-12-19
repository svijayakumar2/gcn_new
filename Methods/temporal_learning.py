import torch
import logging
from torch_geometric.data import DataLoader
from collections import defaultdict
from datetime import datetime
import os
import glob
import json
import numpy as np
from architectures import PhasedGNN, PhasedTraining
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=256, embedding_dim=128):
        super().__init__()
        # Initial feature processing
        self.feat_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        # GNN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Normalization layers
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.norm3 = torch.nn.LayerNorm(hidden_dim)
        
        # Attention weights
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Tanh()
        )
        
        # Final projection layers
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Initial feature processing
        x = self.feat_encoder(x)
        
        # GNN layers with residual connections
        x1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        x2 = F.relu(self.norm2(self.conv2(x1, edge_index))) + x1
        x3 = F.relu(self.norm3(self.conv3(x2, edge_index))) + x2
        
        # Attention-based pooling (replacing scatter operations)
        attention_weights = self.attention(x3)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Manual pooling by batch
        max_batch = int(batch.max().item() + 1)
        pooled = []
        
        for b in range(max_batch):
            mask = (batch == b)
            if mask.any():
                # Weight and sum the nodes for this graph
                batch_nodes = x3[mask]
                batch_weights = attention_weights[mask]
                pooled.append((batch_nodes * batch_weights).sum(dim=0))
        
        if pooled:
            x = torch.stack(pooled)
        else:
            # Fallback if batch is empty
            x = x3.mean(dim=0, keepdim=True)
        
        # Project to embedding space
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)

def load_batch(batch_file, family_to_idx, batch_size=32, device='cpu'):
    """Load and preprocess a single batch file."""
    try:
        if not os.path.exists(batch_file):
            logger.warning(f"Batch file not found: {batch_file}")
            return None
            
        batch = torch.load(batch_file)
        processed = []
        
        for graph in batch:
            try:
                # Skip empty graphs
                if graph.edge_index.size(1) == 0:
                    logger.debug(f"Skipping empty graph")
                    continue
                    
                # Process family label
                family = getattr(graph, 'family', 'none')
                if not family or family == '':
                    family = 'none'
                if family not in family_to_idx:
                    family = 'none'
                    
                # Set label and move tensors to device
                graph.y = torch.tensor(family_to_idx[family]).to(device)
                graph.x = graph.x.to(device)
                graph.edge_index = graph.edge_index.to(device)
                graph.edge_attr = graph.edge_attr.to(device)
                
                processed.append(graph)
                
            except Exception as e:
                logger.error(f"Error processing graph: {str(e)}")
                continue
                
        if not processed:
            return None
            
        return DataLoader(processed, batch_size=batch_size, shuffle=False)
        
    except Exception as e:
        logger.error(f"Error loading batch {batch_file}: {str(e)}")
        return None


def calculate_temporal_weight(timestamp, current_time, decay_factor=0.1):
    """Calculate weight based on temporal distance."""
    time_diff = (current_time - timestamp).total_seconds()
    return np.exp(-decay_factor * time_diff)

class TemporalCentroidTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.prototypes = {}  # Family -> centroid embedding
        self.prototype_updates = defaultdict(int)
        self.margin = 0.3
        self.confidence_threshold = 0.7
        # Track moving statistics for distance thresholding
        self.distance_stats = {
            'mean': None,
            'std': None,
            'n_updates': 0
        }
        
    def update_distance_stats(self, distances):
        """Update running statistics for distance-based thresholding"""
        batch_mean = distances.mean().item()
        batch_std = distances.std().item()
        
        if self.distance_stats['mean'] is None:
            self.distance_stats['mean'] = batch_mean
            self.distance_stats['std'] = batch_std
        else:
            # Exponential moving average
            alpha = 0.1
            self.distance_stats['mean'] = (1 - alpha) * self.distance_stats['mean'] + alpha * batch_mean
            self.distance_stats['std'] = (1 - alpha) * self.distance_stats['std'] + alpha * batch_std
        
        self.distance_stats['n_updates'] += 1

    def update_prototypes(self, embeddings, families):
        """Update prototype centroids for each family in batch"""
        unique_families = torch.unique(families)
        
        for family in unique_families:
            family_str = str(family.item())
            family_mask = (families == family)
            family_embeddings = embeddings[family_mask]
            
            # Compute new centroid for this family
            new_centroid = family_embeddings.mean(dim=0)
            
            if family_str not in self.prototypes:
                self.prototypes[family_str] = new_centroid.detach()
            else:
                # Exponential moving average update
                momentum = 0.9
                old_centroid = self.prototypes[family_str]
                updated_centroid = momentum * old_centroid + (1 - momentum) * new_centroid
                self.prototypes[family_str] = updated_centroid.detach()
            
            self.prototype_updates[family_str] += 1

    def compute_distances(self, embeddings):
        """Compute distances between embeddings and all prototypes"""
        if not self.prototypes:
            return None, None
            
        prototype_tensor = torch.stack(list(self.prototypes.values())).to(self.device)
        
        # Compute cosine distances
        similarities = torch.matmul(embeddings, prototype_tensor.T)
        distances = 1 - similarities  # Convert to distances
        
        return distances, list(self.prototypes.keys())

    def train_batch(self, batch, phase):
        self.model.train()
        batch = batch.to(self.device)
        embeddings = self.model(batch)  # [batch_size, embedding_dim]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Update prototypes with new embeddings
        self.update_prototypes(embeddings, batch.y)
        
        # Compute distances to all prototypes
        distances, prototype_families = self.compute_distances(embeddings)
        if distances is None:
            return torch.tensor(0.0, device=self.device)
        
        # Update distance statistics
        self.update_distance_stats(distances)
        
        # Compute loss
        loss = torch.tensor(0.0, device=self.device)
        
        for i, (emb, family) in enumerate(zip(embeddings, batch.y)):
            family_str = str(family.item())
            
            if family_str in prototype_families:
                prototype_idx = prototype_families.index(family_str)
                
                # Positive pair loss
                pos_dist = distances[i, prototype_idx]
                
                # Negative pair losses (to all other prototypes)
                neg_mask = torch.ones(len(prototype_families), dtype=torch.bool, device=self.device)
                neg_mask[prototype_idx] = False
                neg_dists = distances[i, neg_mask]
                
                # Contrastive loss with margin
                neg_loss = torch.max(torch.zeros_like(neg_dists), self.margin - neg_dists)
                loss += pos_dist + neg_loss.mean()
        
        return loss / len(batch)

    def evaluate(self, batch, phase):
        self.model.eval()
        batch = batch.to(self.device)
        
        metrics = {
            'existing': {'correct': 0, 'total': 0},
            'new': {'correct': 0, 'total': 0},
            'per_family': defaultdict(lambda: {'correct': 0, 'total': 0})
        }
        
        with torch.no_grad():
            embeddings = self.model(batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Get distances to prototypes
            distances, prototype_families = self.compute_distances(embeddings)
            if distances is None:
                return metrics
            
            # Compute dynamic threshold for new family detection
            threshold = (self.distance_stats['mean'] + 
                       2 * self.distance_stats['std'])  # 2 sigma threshold
            
            for i, (_, true_family) in enumerate(zip(embeddings, batch.y)):
                true_family_str = str(true_family.item())
                min_dist, pred_idx = distances[i].min(dim=0)
                
                # Check if this is a new family
                is_new_family = true_family_str not in prototype_families
                
                if is_new_family:
                    metrics['new']['total'] += 1
                    # Consider it correct if distance is above threshold
                    if min_dist > threshold:
                        metrics['new']['correct'] += 1
                else:
                    metrics['existing']['total'] += 1
                    predicted_family = prototype_families[pred_idx]
                    if predicted_family == true_family_str and min_dist <= threshold:
                        metrics['existing']['correct'] += 1
                
                # Update per-family metrics
                metrics['per_family'][true_family_str]['total'] += 1
                if (is_new_family and min_dist > threshold) or \
                   (not is_new_family and predicted_family == true_family_str and min_dist <= threshold):
                    metrics['per_family'][true_family_str]['correct'] += 1
        
        return metrics

def print_eval_metrics(metrics):
    """Print evaluation metrics with family details"""
    print("\n=== Evaluation Results ===")
    
    # Print overall metrics
    if metrics['existing']['total'] > 0:
        existing_acc = metrics['existing']['correct'] / metrics['existing']['total']
        print(f"\nExisting Families: {existing_acc:.2%} ({metrics['existing']['correct']}/{metrics['existing']['total']})")
    
    if metrics['new']['total'] > 0:
        new_acc = metrics['new']['correct'] / metrics['new']['total']
        print(f"New Families: {new_acc:.2%} ({metrics['new']['correct']}/{metrics['new']['total']})")
    
    # Print per-family details
    print("\nPer-family Performance (min 5 samples):")
    family_stats = []
    for family, stats in metrics['per_family'].items():
        if stats['total'] >= 5:
            acc = stats['correct'] / stats['total']
            family_stats.append((family, acc, stats['total']))
    
    # Sort by accuracy
    family_stats.sort(key=lambda x: x[1], reverse=True)
    
    # Print top and bottom 5
    print("\nTop 5 Performing Families:")
    for family, acc, total in family_stats[:5]:
        print(f"Family {family}: {acc:.2%} ({total} samples)")
        
    print("\nBottom 5 Performing Families:")
    for family, acc, total in family_stats[-5:]:
        print(f"Family {family}: {acc:.2%} ({total} samples)")
        
def prepare_data(base_dir='bodmas_batches'):
    """Prepare datasets with temporal ordering."""
    split_files = defaultdict(list)
    family_counts = defaultdict(int)
    all_families = set()
    file_timestamps = {}
    
    logger.info("Starting data preparation...")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        # Collect all batch files
        for file in glob.glob(os.path.join(split_dir, 'batch_*.pt')):
            try:
                batch = torch.load(file)
                file_timestamps[file] = getattr(batch[0], 'timestamp', None)
                
                # Count families
                for graph in batch:
                    family = getattr(graph, 'family', 'none')
                    if not family or family == '':
                        family = 'none'
                    all_families.add(family)
                    family_counts[family] += 1
                    
                split_files[split].append(file)
                
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                continue

    # Debug prints
    print("All families found:", all_families)
    print("Family counts:", family_counts)
    
    # Create family mapping
    families = sorted(list(all_families))
    family_to_idx = {family: idx for idx, family in enumerate(families)}
    
    print("Family to idx mapping:", family_to_idx)
    
    # Save mapping
    mapping = {
        'family_to_idx': family_to_idx,
        'idx_to_family': {str(idx): family for family, idx in family_to_idx.items()},
        'family_counts': family_counts,
        'timestamp': datetime.now().isoformat()
    }

    print("Final mapping to be saved:", mapping)
    
    with open(os.path.join(base_dir, 'family_mapping.json'), 'w') as f:
        json.dump(mapping, f, indent=2)
    return split_files, len(families), family_to_idx



def train_epoch(trainer, split_files, family_to_idx, phase, device):
    """Train for one epoch."""
    total_loss = 0
    batches = 0
    
    for batch_file in split_files:
        loader = load_batch(batch_file, family_to_idx, device=device)
        if not loader:
            continue
            
        for batch in loader:
            loss = trainer.train_batch(batch, phase)
            total_loss += loss
            batches += 1
            
    return total_loss / max(1, batches)


def evaluate(trainer, split_files, family_to_idx, phase, device):
    """Evaluate model performance"""
    total_metrics = {
        'existing': {'correct': 0, 'total': 0},
        'new': {'correct': 0, 'total': 0},
        'per_family': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    for batch_file in split_files:
        loader = load_batch(batch_file, family_to_idx, device=device)
        if not loader:
            continue
            
        for batch in loader:
            # Get batch metrics from trainer
            batch_metrics = trainer.evaluate(batch, phase)
            
            # Accumulate metrics
            total_metrics['existing']['correct'] += batch_metrics['existing']['correct']
            total_metrics['existing']['total'] += batch_metrics['existing']['total']
            total_metrics['new']['correct'] += batch_metrics['new']['correct']
            total_metrics['new']['total'] += batch_metrics['new']['total']
            
            # Accumulate per-family metrics
            for family, stats in batch_metrics['per_family'].items():
                total_metrics['per_family'][family]['correct'] += stats['correct']
                total_metrics['per_family'][family]['total'] += stats['total']
    
    return total_metrics



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    split_files, num_families, family_to_idx = prepare_data()
    first_batch = torch.load(split_files['train'][0])
    num_features = first_batch[0].x.size(1)
    
    model = ImprovedGNN(
        num_node_features=num_features,
        hidden_dim=256,
        embedding_dim=128
    ).to(device)
    
    trainer = TemporalCentroidTrainer(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_metrics = {
        'existing_acc': 0.0,
        'new_acc': 0.0,
        'combined_acc': 0.0
    }
    epochs = 20
    
    logger.info("Starting training...")
    logger.info(f"Total families in dataset: {num_families}")
        # In main(), before the training loop:
    print("Checking data splits:")
    for split in ['train', 'val', 'test']:
        if split in split_files:
            print(f"{split} files: {len(split_files[split])}")
            # Load first batch as a test
            first_file = split_files[split][0]
            test_batch = load_batch(first_file, family_to_idx, device=device)
            if test_batch is not None:
                for batch in test_batch:
                    print(f"First {split} batch size: {len(batch.y)}")
                    print(f"Unique families in batch: {len(torch.unique(batch.y))}")
                    break
        
    for epoch in range(epochs):
        print(f"\n=== Starting Epoch {epoch+1} ===")
        # Train
        model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"Training files: {len(split_files['train'])}")  # Debug print
        
        for batch_idx, batch_file in enumerate(split_files['train']):
            print(f"\nProcessing batch file {batch_idx+1}/{len(split_files['train'])}")  # Debug print
            loader = load_batch(batch_file, family_to_idx, device=device)
            if not loader:
                print(f"Skipped batch file {batch_idx+1} - no valid data")  # Debug print
                continue   
    

            
            for batch in loader:
                optimizer.zero_grad()
                loss = trainer.train_batch(batch, 'train')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()  # Convert loss to float here
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        # Validate
        model.eval()
        val_metrics = evaluate(trainer, split_files['val'], family_to_idx, 'val', device)
        
        # Calculate accuracies
        existing_acc = (val_metrics['existing']['correct'] / val_metrics['existing']['total'] 
                       if val_metrics['existing']['total'] > 0 else 0)
        new_acc = (val_metrics['new']['correct'] / val_metrics['new']['total'] 
                  if val_metrics['new']['total'] > 0 else 0)
        total_samples = val_metrics['existing']['total'] + val_metrics['new']['total']
        total_correct = val_metrics['existing']['correct'] + val_metrics['new']['correct']
        combined_acc = total_correct / total_samples if total_samples > 0 else 0
        
        # Print epoch results
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        print(f"Training Loss: {avg_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"\nValidation Results:")
        print(f"Existing Families: {existing_acc:.2%} ({val_metrics['existing']['correct']}/{val_metrics['existing']['total']})")
        print(f"New Families: {new_acc:.2%} ({val_metrics['new']['correct']}/{val_metrics['new']['total']})")
        print(f"Combined Accuracy: {combined_acc:.2%}")
        
        scheduler.step()
        
        # Save best model based on combined accuracy
        if combined_acc > best_metrics['combined_acc']:
            best_metrics = {
                'existing_acc': existing_acc,
                'new_acc': new_acc,
                'combined_acc': combined_acc,
                'epoch': epoch + 1
            }
            
            save_dir = "trained_model"
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'prototypes': trainer.prototypes,
                'prototype_updates': trainer.prototype_updates,
                'best_metrics': best_metrics,
                'epoch': epoch + 1
            }, os.path.join(save_dir, "prototype_gnn_best.pt"))
    
    print(f"\n=== Training Complete ===")
    print(f"Best model achieved at epoch {best_metrics['epoch']}:")
    print(f"- Combined Accuracy: {best_metrics['combined_acc']:.2%}")
    print(f"- Existing Families: {best_metrics['existing_acc']:.2%}")
    print(f"- New Families: {best_metrics['new_acc']:.2%}")

if __name__ == '__main__':
    main()

