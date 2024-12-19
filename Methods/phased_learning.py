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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    split_files, num_families, family_to_idx = prepare_data()
    
    if not any(split_files.values()):
        logger.error("No data found!")
        return
        
    # Get feature dimensions
    first_batch = torch.load(split_files['train'][0])
    num_features = first_batch[0].x.size(1)
    logger.info(f"Features: {num_features}, Families: {num_families}")
    
    # Initialize model and trainer
    model = PhasedGNN(num_node_features=num_features, 
                      num_families=num_families).to(device)
    trainer = PhasedTraining(model, device)
    
    # Training loop
    phases = ['family',  'novelty']
    epochs = 3
        
    for phase in phases:
        logger.info(f"\nStarting {phase} phase")
        
        for epoch in range(epochs):
            # Train
            loss = train_epoch(trainer, split_files['train'], 
                            family_to_idx, phase, device)
                        
            # Validate
            precision, recall = evaluate(trainer, split_files['val'], 
                                    family_to_idx, phase, device)

            logger.info(f"Epoch {epoch}: loss={loss:.4f}, "
                        f"precision={precision:.4f}, recall={recall:.4f}")
                        
            logger.info(f"Epoch {epoch}: loss={loss:.4f}, "
                        f"precision={precision:.4f}, recall={recall:.4f}")

    save_dir = "trained_model"
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "phased_gnn_final.pt")
    mapping_path = os.path.join(save_dir, "family_mapping.json")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_features': num_features,
        'num_families': num_families,
    }, model_path)
    
    # Save mapping
    with open(mapping_path, 'w') as f:
        json.dump({
            'family_to_idx': family_to_idx,
            'idx_to_family': {str(idx): family for family, idx in family_to_idx.items()},
        }, f, indent=2)
    
    logger.info(f"Model and mapping saved to {save_dir}/")
    
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