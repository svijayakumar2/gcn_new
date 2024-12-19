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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, embedding_dim=64):
        super().__init__()
        # Multiple GNN layers with residual connections
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Projection layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, embedding_dim)
        
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GNN layer
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.dropout(x1)
        
        # Second GNN layer with residual
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.dropout(x2)
        x2 = x2 + x1  # Residual connection
        
        # Third GNN layer with residual
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3)
        x3 = x3 + x2  # Residual connection
        
        # Global pooling with attention
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x3, batch)
        
        # Final projection
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
    

def calculate_temporal_weight(timestamp, current_time, decay_factor=0.1):
    """Calculate weight based on temporal distance."""
    time_diff = (current_time - timestamp).total_seconds()
    return np.exp(-decay_factor * time_diff)

class TemporalCentroidTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.centroids = {}
        self.sample_counts = defaultdict(int)
        self.momentum = 0.9
        self.min_samples = 2
        self.margin = 0.3
        self.confidence_threshold = 0.6

    def train_batch(self, batch, phase):
        self.model.train()
        batch = batch.to(self.device)
        embeddings = self.model(batch)
        
        loss = 0
        batch_size = embeddings.size(0)
        
        for i, (emb, family) in enumerate(zip(embeddings, batch.y)):
            family_str = str(family.item())
            self.sample_counts[family_str] += 1
            
            # Dynamic momentum based on sample count
            momentum = min(0.99, self.momentum + 0.1 * (1.0 - 1.0/self.sample_counts[family_str]))
            
            # Update centroid with dynamic momentum
            if family_str not in self.centroids:
                self.centroids[family_str] = emb.detach()
            else:
                self.centroids[family_str] = (
                    momentum * self.centroids[family_str] + 
                    (1 - momentum) * emb.detach()
                )
            
            # Enhanced contrastive loss
            anchor = emb
            positive = self.centroids[family_str]
            
            # Weighted positive distance based on sample count
            pos_weight = 1.0 / np.sqrt(self.sample_counts[family_str])
            pos_dist = pos_weight * (1 - F.cosine_similarity(anchor, positive.unsqueeze(0)))
            
            # Hard negative mining
            neg_dists = []
            for other_family, other_centroid in self.centroids.items():
                if other_family != family_str:
                    neg_dist = 1 - F.cosine_similarity(anchor, other_centroid.unsqueeze(0))
                    neg_dists.append(neg_dist)
            
            if neg_dists:
                # Use top-k hardest negatives
                k = min(3, len(neg_dists))
                hardest_negs = sorted(neg_dists)[:k]
                neg_loss = sum(max(0, self.margin - neg_dist) for neg_dist in hardest_negs) / k
                loss += pos_dist + neg_loss

        return loss.item() / batch_size if batch_size > 0 else 0.0

    def evaluate(self, batch, phase):
        self.model.eval()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(batch)
            correct = 0
            total = len(batch)
            
            for emb, true_family in zip(embeddings, batch.y):
                true_family = str(true_family.item())
                
                # Calculate similarities with confidence scores
                similarities = []
                for family, centroid in self.centroids.items():
                    sim = F.cosine_similarity(emb, centroid.unsqueeze(0))
                    # Adjust similarity based on sample count
                    confidence = sim * (1 + np.log(self.sample_counts[family]))
                    similarities.append((family, confidence))
                
                if not similarities:
                    continue
                
                # Find best match with confidence threshold
                similarities.sort(key=lambda x: x[1], reverse=True)
                best_family, confidence = similarities[0]
                
                if confidence > self.confidence_threshold:
                    if best_family == true_family:
                        correct += 1
                else:
                    # Consider new family prediction
                    if true_family not in self.centroids or self.sample_counts[true_family] < self.min_samples:
                        correct += 1
            
            return correct, total, correct/total if total > 0 else 0.0
    # def __init__(self, model, device):
    #     self.model = model
    #     self.device = device
    #     self.centroids = {}  # Family -> centroid mapping
    #     self.temporal_weights = {}  # Family -> timestamp mapping

    # def train_batch(self, batch, phase):
    #     """Similar interface to your existing PhasedTraining.train_batch()"""
    #     self.model.train()
    #     batch = batch.to(self.device)
    #     embeddings = self.model(batch)
        
    #     loss = 0
    #     for i, (emb, family) in enumerate(zip(embeddings, batch.y)):
    #         family_str = str(family.item())
            
    #         # Update or initialize centroid
    #         if family_str not in self.centroids:
    #             self.centroids[family_str] = emb.detach()
    #         else:
    #             # Temporal update (keeping your data structure)
    #             old_weight = 0.7
    #             self.centroids[family_str] = (old_weight * self.centroids[family_str] + 
    #                                         (1-old_weight) * emb.detach())
            
    #         # Contrastive loss calculation
    #         pos_dist = F.cosine_similarity(emb, self.centroids[family_str].unsqueeze(0))
    #         neg_dists = []
    #         for f, c in self.centroids.items():
    #             if f != family_str:
    #                 neg_dists.append(F.cosine_similarity(emb, c.unsqueeze(0)))
            
    #         if neg_dists:
    #             loss += -pos_dist + torch.stack(neg_dists).mean()

    #     return loss.item() / len(batch)

    # def evaluate(self, batch, phase):
    #     """Keep similar interface to your PhasedTraining.evaluate()"""
    #     self.model.eval()
    #     batch = batch.to(self.device)
        
    #     with torch.no_grad():
    #         embeddings = self.model(batch)
    #         correct = 0
    #         total = len(batch)
            
    #         for emb, true_family in zip(embeddings, batch.y):
    #             true_family = str(true_family.item())
                
    #             # Find closest centroid
    #             best_dist = float('-inf')
    #             best_family = None
                
    #             for family, centroid in self.centroids.items():
    #                 dist = F.cosine_similarity(emb, centroid.unsqueeze(0))
    #                 if dist > best_dist:
    #                     best_dist = dist
    #                     best_family = family
                
    #             if best_family == true_family:
    #                 correct += 1
            
    #         return correct, total, correct/total
        
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
                
                # Handle edge attributes
                if graph.edge_index.size(1) == 0:
                    graph.edge_attr = torch.zeros((0, 1)).to(device)
                else:
                    graph.edge_attr = torch.ones((graph.edge_index.size(1), 1)).to(device)
                    
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
    """Evaluate on a split with weighted metrics."""
    family_correct = defaultdict(int)
    family_total = defaultdict(int)
    total_correct = 0
    total_preds = 0
    recall_sum = 0
    num_batches = 0
    
    for batch_file in split_files:
        loader = load_batch(batch_file, family_to_idx, device=device)
        if not loader:
            continue
            
        for batch in loader:
            correct, total, recall = trainer.evaluate(batch, phase)
            total_correct += correct
            total_preds += total
            recall_sum += recall
            num_batches += 1
            
            # Track results per family
            for fam in batch.y:
                family_total[fam.item()] += 1

    # Calculate regular precision
    precision = total_correct / max(1, total_preds)
    
    # Calculate weighted precision
    weighted_precision = 0
    total_samples = sum(family_total.values())
    
    if total_samples > 0:
        for family, count in family_total.items():
            weight = count / total_samples
            weighted_precision += precision * weight  # Using overall precision as an approximation

    # Keep original return format
    return precision, recall_sum / max(1, num_batches)






def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data using your existing prepare_data
    split_files, num_families, family_to_idx = prepare_data()
    
    # Get feature dimensions
    first_batch = torch.load(split_files['train'][0])
    num_features = first_batch[0].x.size(1)
    
    # Initialize improved model and trainer
    model = ImprovedGNN(
        num_node_features=num_features,
        hidden_dim=256,  # Increased capacity
        embedding_dim=128  # Increased embedding dimension
    ).to(device)
    
    trainer = TemporalCentroidTrainer(model, device)
    
    # Use AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         factor=0.5, patience=2)
    
    best_precision = 0
    epochs = 20  # Increase epochs
    
    for epoch in range(epochs):
        # Train
        model.train()
        loss = train_epoch(trainer, split_files['train'], family_to_idx, 'centroid', device)
        
        # Validate
        model.eval()
        precision, recall = evaluate(trainer, split_files['val'], 
                                   family_to_idx, 'centroid', device)
        
        # Learning rate scheduling
        scheduler.step(precision)
        
        logger.info(f"Epoch {epoch}: loss={loss:.4f}, "
                   f"precision={precision:.4f}, recall={recall:.4f}")
        
        # Save best model
        if precision > best_precision:
            best_precision = precision
            # Save model as before
            save_dir = "trained_model"
            os.makedirs(save_dir, exist_ok=True)
            
            model_path = os.path.join(save_dir, "centroid_gnn_final.pt")
            mapping_path = os.path.join(save_dir, "family_mapping.json")
            
            # Save model, centroids, and mapping
            torch.save({
                'model_state_dict': model.state_dict(),
                'centroids': trainer.centroids,
                'num_features': num_features,
                'num_families': num_families,
            }, model_path)
            
            # Save mapping
            with open(mapping_path, 'w') as f:
                json.dump({
                    'family_to_idx': family_to_idx,
                    'idx_to_family': {str(idx): family 
                                    for family, idx in family_to_idx.items()},
                }, f, indent=2)
    
    logger.info(f"Model and mapping saved to {save_dir}/")

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, embedding_dim):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global mean pooling
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)
        
        # Project to embedding space
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings
    
if __name__ == '__main__':
    main()


# def evaluate(trainer, split_files, family_to_idx, phase, device):
#     """Evaluate on a split."""
#     total_correct = 0
#     total_preds = 0
#     recall_sum = 0
#     num_batches = 0
    
#     for batch_file in split_files:
#         loader = load_batch(batch_file, family_to_idx, device=device)
#         if not loader:
#             continue
#         for batch in loader:
#             correct, total, recall = trainer.evaluate(batch, phase)
#             total_correct += correct
#             total_preds += total
#             recall_sum += recall
#             num_batches += 1
    
#     precision = total_correct / max(1, total_preds)
#     recall = recall_sum / max(1, num_batches)
#     return precision, recall