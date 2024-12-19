import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import logging
import os
import glob
from tqdm import tqdm
from collections import defaultdict
import random
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MalwareGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=256):
        super().__init__()
        # GNN layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Global pooling attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        # Final projection
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Attention pooling
        weights = self.attention(x)
        weights = F.softmax(weights, dim=0)

        # Pool by batch
        max_batch = int(batch.max().item() + 1)
        pooled = []
        for b in range(max_batch):
            mask = (batch == b)
            if mask.any():
                batch_x = x[mask]
                batch_weights = weights[mask]
                pooled.append((batch_x * batch_weights).sum(dim=0))

        if not pooled:
            return None

        # Final embedding
        x = torch.stack(pooled)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)

class ExemplarDetector:
    def __init__(self, model, device, similarity_threshold=0.75, max_exemplars=5):
        self.model = model
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.max_exemplars = max_exemplars
        self.family_exemplars = defaultdict(list)

    def update_exemplars(self, embeddings, families):
        for emb, family in zip(embeddings, families):
            family = str(family.item())
            exemplars = self.family_exemplars[family]
            
            if len(exemplars) < self.max_exemplars:
                exemplars.append(emb.detach())
            else:
                # Replace least similar exemplar if new embedding is more representative
                sims = torch.stack([F.cosine_similarity(emb, ex.to(self.device), dim=0) 
                                  for ex in exemplars])
                worst_idx = sims.argmin()
                mean_sim = F.cosine_similarity(emb, 
                                             torch.stack(exemplars).mean(dim=0).to(self.device), 
                                             dim=0)
                if sims[worst_idx] < mean_sim:
                    exemplars[worst_idx] = emb.detach()

    def get_similarity(self, embedding, family):
        if family not in self.family_exemplars:
            return 0.0
        
        exemplars = self.family_exemplars[family]
        if not exemplars:
            return 0.0
            
        sims = torch.stack([F.cosine_similarity(embedding, ex.to(self.device), dim=0) 
                           for ex in exemplars])
        return 0.7 * sims.max() + 0.3 * sims.mean()

    def predict(self, embedding):
        max_sim = -1
        pred_family = "new"
        
        for family in self.family_exemplars:
            sim = self.get_similarity(embedding, family)
            if sim > max_sim:
                max_sim = sim
                pred_family = family
                
        if max_sim < self.similarity_threshold:
            return "new", max_sim
        return pred_family, max_sim
    
    def evaluate_batch(self, batch):
        """Evaluate a batch of graphs"""
        self.model.eval()
        metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            embeddings = self.model(batch)
            if embeddings is None:
                return metrics
            
            for emb, true_family in zip(embeddings, batch.y):
                true_family_str = str(true_family.item())
                pred_family, confidence = self.predict(emb)
                
                # Track metrics
                if true_family_str not in self.family_exemplars:
                    # Should be detected as new
                    metrics['new']['total'] += 1
                    if pred_family == "new":
                        metrics['new']['correct'] += 1
                else:
                    # Should be classified correctly
                    metrics['known']['total'] += 1
                    if pred_family == true_family_str:
                        metrics['known']['correct'] += 1
                
                # Track per-family metrics
                metrics[true_family_str]['total'] += 1
                if (true_family_str not in self.family_exemplars and pred_family == "new") or \
                   (true_family_str in self.family_exemplars and pred_family == true_family_str):
                    metrics[true_family_str]['correct'] += 1
        
        return metrics

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

def train_step(detector, batch, optimizer):
    detector.model.train()
    optimizer.zero_grad()
    
    # Get embeddings
    embeddings = detector.model(batch)
    if embeddings is None:
        return 0.0
        
    # Compute triplet loss
    loss = torch.tensor(0.0, device=detector.device)
    
    for i, (anchor, family) in enumerate(zip(embeddings, batch.y)):
        family = str(family.item())
        
        # Skip if no exemplars for this family yet
        if family not in detector.family_exemplars or not detector.family_exemplars[family]:
            continue
            
        # Get positive exemplar (centroid of family)
        positives = torch.stack(detector.family_exemplars[family])
        positive = positives.mean(dim=0).to(detector.device)
        
        # Get negative exemplar
        neg_families = [f for f in detector.family_exemplars if f != family]
        if neg_families:
            neg_family = random.choice(neg_families)
            negatives = torch.stack(detector.family_exemplars[neg_family])
            negative = negatives.mean(dim=0).to(detector.device)
            
            # Triplet margin loss
            d_pos = 1 - F.cosine_similarity(anchor, positive, dim=0)
            d_neg = 1 - F.cosine_similarity(anchor, negative, dim=0)
            loss += F.relu(d_pos - d_neg + 0.3)  # margin = 0.3

    if loss > 0:
        loss.backward()
        optimizer.step()
        
    # Update exemplars
    detector.update_exemplars(embeddings.detach(), batch.y)
    
    return loss.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data preparation
    base_dir = 'bodmas_batches'
    splits = ['train', 'val', 'test']
    batch_files = {split: [] for split in splits}
    family_to_idx = {}
    
    # Process each split
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            batch_files[split] = sorted(glob.glob(os.path.join(split_dir, 'batch_*.pt')))
            logger.info(f"Found {len(batch_files[split])} {split} batches")

            # Collect family information
            for file in batch_files[split]:
                try:
                    batch = torch.load(file)
                    for graph in batch:
                        family = getattr(graph, 'family', 'none')
                        if family and family != '':
                            if family not in family_to_idx:
                                family_to_idx[family] = len(family_to_idx)
                except Exception as e:
                    logger.error(f"Error processing file {file}: {str(e)}")
                    continue

    # Initialize model
    first_batch = torch.load(batch_files['train'][0])
    num_features = first_batch[0].x.size(1)
    
    model = MalwareGNN(num_features).to(device)
    detector = ExemplarDetector(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    num_epochs = 5
    best_acc = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        num_batches = 0
        
        train_iterator = tqdm(batch_files['train'], desc=f"Epoch {epoch+1}")
        for batch_file in train_iterator:
            # Fixed: Properly pass batch_size and device
            loader = load_batch(batch_file, family_to_idx, batch_size=32, device=device)
            if not loader:
                continue
                
            for batch in loader:
                loss = train_step(detector, batch, optimizer)
                total_loss += loss
                num_batches += 1
                train_iterator.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Validate
        model.eval()
        val_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for batch_file in batch_files['val']:
                # Fixed: Properly pass batch_size and device
                loader = load_batch(batch_file, family_to_idx, batch_size=32, device=device)
                if not loader:
                    continue
                    
                for batch in loader:
                    batch_metrics = detector.evaluate_batch(batch)
                    
                    # Aggregate metrics
                    for key in batch_metrics:
                        val_metrics[key]['correct'] += batch_metrics[key]['correct']
                        val_metrics[key]['total'] += batch_metrics[key]['total']
        
        # Calculate accuracies
        for key in val_metrics:
            if val_metrics[key]['total'] > 0:
                val_metrics[key]['accuracy'] = val_metrics[key]['correct'] / val_metrics[key]['total']
        
        # Print metrics
        if val_metrics['known']['total'] > 0:
            acc = val_metrics['known']['accuracy']
            logger.info(f"Known Family Accuracy: {acc:.2%}")
            
        if val_metrics['new']['total'] > 0:
            new_acc = val_metrics['new']['accuracy']
            logger.info(f"New Family Detection: {new_acc:.2%}")
            
        scheduler.step()
        
        # Save best model
        combined_acc = ((val_metrics['known']['correct'] + val_metrics['new']['correct']) / 
                       (val_metrics['known']['total'] + val_metrics['new']['total']) 
                       if (val_metrics['known']['total'] + val_metrics['new']['total']) > 0 else 0)
                       
        if combined_acc > best_acc:
            best_acc = combined_acc
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'family_exemplars': detector.family_exemplars,
                'accuracy': combined_acc,
                'epoch': epoch
            }, os.path.join(save_dir, 'best_model.pt'))
            logger.info(f"Saved new best model with accuracy {combined_acc:.2%}")

    logger.info("Training complete!")

if __name__ == '__main__':
    main()