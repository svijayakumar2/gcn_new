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
from architectures import CentroidLayer, MalwareGNN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


def prepare_data(base_dir='bodmas_batches_test'):
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

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare data with validation
    split_files, num_classes, family_to_idx = prepare_data()
    
    # Validate feature dimensions
    try:
        first_batch = torch.load(split_files['train'][0])
        num_features = first_batch[0].x.size(1)
        logger.info(f"Number of node features: {num_features}")
    except Exception as e:
        logger.error(f"Error loading first batch: {e}")
        return
    
    # Initialize model
    model = MalwareGNN(
        num_node_features=num_features,
        num_classes=num_classes
    ).to(device)
    
    # Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    num_epochs = 100
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    # Training loop
    logger.info("\nStarting training...")
    for epoch in range(num_epochs):
        # Training
        epoch_losses = []
        for batch_file in split_files['train']:
            train_loader = load_and_process_batch(batch_file, family_to_idx)
            if train_loader:
                loss = train_epoch(model, train_loader, optimizer, device)
                epoch_losses.append(loss)
        
        if not epoch_losses:
            logger.warning(f"No valid training batches in epoch {epoch}")
            continue
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Validation
        val_accs = []
        val_metrics = defaultdict(list)
        for batch_file in split_files['val']:
            val_loader = load_and_process_batch(batch_file, family_to_idx)
            if val_loader:
                acc, family_accs = evaluate(model, val_loader, device, family_to_idx, 'val')
                val_accs.append(acc)
                for family, acc in family_accs.items():
                    val_metrics[family].append(acc)
        
        if not val_accs:
            logger.warning(f"No valid validation batches in epoch {epoch}")
            continue
        
        val_acc = sum(val_accs) / len(val_accs)
        logger.info(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model and handle early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'family_to_idx': family_to_idx
            }, 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info

if __name__ == '__main__':
    main()