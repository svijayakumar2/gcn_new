import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
import os
import glob
from collections import defaultdict
from typing import List, Dict, Set
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'malware_dml_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MalwareDML(torch.nn.Module):
    def __init__(self, num_features, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # GNN layers
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, embedding_dim)
        
        # Residual projection
        self.residual_proj = torch.nn.Linear(128, embedding_dim)
        
        # Attention mechanism
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, data):
        # Skip edge_attr, just use x, edge_index, and batch
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        # First GNN layer
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        
        # Second GNN layer with residual connection
        h2 = self.conv2(h1, edge_index)
        h2 = h2 + self.residual_proj(h1)
        h2 = F.relu(h2)
        
        # Attention-weighted pooling
        weights = self.attention(h2)
        weights = torch.sigmoid(weights)
        
        # Global pooling
        embeddings = global_mean_pool(h2 * weights, batch)
        return F.normalize(embeddings, p=2, dim=1)  # L2 normalize embeddings

class ProtoLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, families):
        loss = 0
        unique_families = list(set(families))
        num_samples = len(embeddings)
        
        if num_samples == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Get prototypes for each family
        prototypes = {}
        for family in unique_families:
            mask = torch.tensor([f == family for f in families], device=embeddings.device)
            if not mask.any():
                continue
            family_embeddings = embeddings[mask]
            prototypes[family] = family_embeddings.mean(0)
        
        # Compute loss for each sample
        for i, (emb, family) in enumerate(zip(embeddings, families)):
            if family not in prototypes:
                continue
            
            # Positive pair
            pos_dist = F.cosine_similarity(
                emb.unsqueeze(0),
                prototypes[family].unsqueeze(0)
            )
            
            # Negative pairs
            neg_dists = []
            for neg_family, neg_proto in prototypes.items():
                if neg_family != family:
                    neg_dist = F.cosine_similarity(
                        emb.unsqueeze(0),
                        neg_proto.unsqueeze(0)
                    )
                    neg_dists.append(neg_dist)
            
            if not neg_dists:
                continue
                
            neg_dists = torch.stack(neg_dists)
            
            # InfoNCE loss computation
            logits = torch.cat([pos_dist.unsqueeze(0), neg_dists]) / self.temperature
            loss += -pos_dist / self.temperature + torch.logsumexp(logits / self.temperature, dim=0)
        
        return loss / num_samples

class MalwareClassifier:
    def __init__(self, distance_threshold=0.8):
        self.prototypes = {}
        self.distance_threshold = distance_threshold
        self.family_stats = defaultdict(lambda: {'count': 0, 'embeddings': []})

    def update_prototypes(self, embeddings: torch.Tensor, families: List[str]):
        """Update prototypes with new embeddings."""
        with torch.no_grad():
            for emb, family in zip(embeddings, families):
                self.family_stats[family]['count'] += 1
                self.family_stats[family]['embeddings'].append(emb)
                
                # Update prototype
                if family not in self.prototypes:
                    self.prototypes[family] = emb
                else:
                    # Exponential moving average update
                    alpha = 0.1
                    self.prototypes[family] = (1 - alpha) * self.prototypes[family] + alpha * emb

    def predict(self, embeddings: torch.Tensor) -> List[str]:
        """Predict family labels including unknown detection."""
        predictions = []
        for emb in embeddings:
            best_sim = -1
            best_family = "unknown"
            
            for family, prototype in self.prototypes.items():
                sim = F.cosine_similarity(emb.unsqueeze(0), prototype.unsqueeze(0))
                if sim > best_sim:
                    best_sim = sim
                    best_family = family
            
            # Unknown family detection
            if best_sim < self.distance_threshold:
                predictions.append("unknown")
            else:
                predictions.append(best_family)
                
        return predictions

def load_batch(batch_file: str, batch_size: int = 32) -> DataLoader:
    """Load and preprocess a batch file."""
    try:
        if not os.path.exists(batch_file):
            logger.warning(f"Batch file not found: {batch_file}")
            return None
            
        batch_data = torch.load(batch_file)
        if not batch_data:
            return None
            
        processed = []
        for graph in batch_data:
            if not isinstance(graph, Data):
                continue
                
            # Ensure required attributes
            if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                continue
                
            # Add family attribute if missing
            if not hasattr(graph, 'family'):
                graph.family = 'unknown'
                
            # No need to add edge_attr as GCNConv doesn't require it
            # Ensure tensors are float
            graph.x = graph.x.float()
            
            processed.append(graph)
            
        if not processed:
            return None
            
        return DataLoader(processed, batch_size=min(batch_size, len(processed)), shuffle=True)
        
    except Exception as e:
        logger.error(f"Error loading batch {batch_file}: {str(e)}")
        return None

def evaluate_model(model: MalwareDML, 
                  classifier: MalwareClassifier, 
                  test_files: List[str], 
                  device: torch.device) -> Dict:
    """Evaluate model performance."""
    model.eval()
    metrics = defaultdict(list)
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch_file in test_files:
            batch_loader = load_batch(batch_file)
            if not batch_loader:
                continue
                
            for batch in batch_loader:
                batch = batch.to(device)
                embeddings = model(batch)
                predictions = classifier.predict(embeddings)
                
                # Store predictions
                all_preds.extend(predictions)
                all_true.extend(batch.family)
                
                # Compute batch metrics
                for pred, true in zip(predictions, batch.family):
                    metrics['accuracy'].append(pred == true)
                    metrics['unknown_rate'].append(pred == 'unknown')
    
    # Compute final metrics
    results = {
        'accuracy': np.mean(metrics['accuracy']),
        'unknown_rate': np.mean(metrics['unknown_rate'])
    }
    
    # Compute precision, recall, f1 for known families
    known_preds = [p for p, t in zip(all_preds, all_true) if p != 'unknown' and t != 'unknown']
    known_true = [t for p, t in zip(all_preds, all_true) if p != 'unknown' and t != 'unknown']
    
    if known_preds:
        precision, recall, f1, _ = precision_recall_fscore_support(
            known_true, known_preds, average='weighted'
        )
        results.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results

def train_model(model: MalwareDML,
                classifier: MalwareClassifier,
                train_files: List[str],
                val_files: List[str],
                device: torch.device,
                num_epochs: int = 100,
                batch_size: int = 32) -> Dict:
    """Train the model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = ProtoLoss(temperature=0.5)
    best_val_acc = 0
    training_history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Training loop
        for batch_file in train_files:
            batch_loader = load_batch(batch_file, batch_size)
            if not batch_loader:
                continue
                
            for batch in batch_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                embeddings = model(batch)
                loss = criterion(embeddings, batch.family)
                
                loss.backward()
                optimizer.step()
                
                # Update prototypes
                classifier.update_prototypes(embeddings.detach(), batch.family)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Validation
        val_metrics = evaluate_model(model, classifier, val_files, device)
        
        # Logging
        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch}:")
        logger.info(f"Train Loss: {avg_loss:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
            }, 'best_dml_model.pt')
        
        # Store history
        training_history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_metrics': val_metrics
        })
    
    return training_history

def main():
    # Configuration
    config = {
        'batch_dir': '/data/saranyav/gcn_new/bodmas_batches',  # Update with your path
        'embedding_dim': 256,
        'num_epochs': 100,
        'batch_size': 32,
        'output_dir': 'dml_analysis',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Setup
    device = torch.device(config['device'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data splits
    splits = ['train', 'val', 'test']
    split_files = {}
    for split in splits:
        split_dir = os.path.join(config['batch_dir'], split)
        if os.path.exists(split_dir):
            split_files[split] = glob.glob(os.path.join(split_dir, 'batch_*.pt'))
    
    # Initialize model and classifier
    try:
        first_batch = torch.load(split_files['train'][0])
        num_features = first_batch[0].x.size(1)
        model = MalwareDML(num_features=num_features, embedding_dim=config['embedding_dim']).to(device)
        classifier = MalwareClassifier(distance_threshold=0.8)
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return
    
    # Train model
    logger.info("Starting training...")
    try:
        history = train_model(
            model=model,
            classifier=classifier,
            train_files=split_files['train'],
            val_files=split_files['val'],
            device=device,
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size']
        )
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return
    
    # Final evaluation
    logger.info("Running final evaluation...")
    try:
        # Load best model
        checkpoint = torch.load('best_dml_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = evaluate_model(
            model=model,
            classifier=classifier,
            test_files=split_files['test'],
            device=device
        )
        
        # Save results
        results = {
            'test_metrics': test_metrics,
            'training_history': history,
            'config': config
        }
        
        with open(output_dir / 'dml_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info("Final Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")