import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

import torch
import logging
from pathlib import Path
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
from torch_geometric.data import DataLoader
import traceback

def load_checkpoint_safely(model, checkpoint_path):
    """Safely load a checkpoint if it exists."""
    device = next(model.parameters()).device  # Get device from model
    
    if not checkpoint_path.exists():
        logger.warning(f"No checkpoint found at {checkpoint_path}")
        return False
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return False



class ContrastivePhasedGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_families, num_groups, 
                 embedding_dim=128, hidden_dim=256, temperature=0.07):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Base GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        
        # Hierarchical classification heads
        self.group_classifier = torch.nn.Linear(embedding_dim, num_groups)
        self.family_classifiers = torch.nn.ModuleDict()
        
        # Create separate family classifier for each behavioral group
        for group_id in range(num_groups):
            self.family_classifiers[str(group_id)] = torch.nn.Linear(embedding_dim, num_families)
        
        # Novelty detection head
        self.novelty_net = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )
        
        # Prototype memories - will be moved to correct device in forward pass
        self.register_buffer('group_prototypes', torch.zeros(num_groups, embedding_dim))
        self.register_buffer('family_prototypes', torch.zeros(num_families, embedding_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_families))

    def to(self, device):
        # Override to() to ensure all components move to the same device
        super().to(device)
        self.conv1 = self.conv1.to(device)
        self.conv2 = self.conv2.to(device)
        self.group_classifier = self.group_classifier.to(device)
        for key in self.family_classifiers:
            self.family_classifiers[key] = self.family_classifiers[key].to(device)
        self.novelty_net = self.novelty_net.to(device)
        return self

        
    def get_embeddings(self, data):
        device = data.x.device  # Get device from input data
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GNN layers with residual connection
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = self.conv2(h1, edge_index)
        
        # Global pooling
        x = global_mean_pool(h2, batch)
        return x
        
    def contrastive_loss(self, embeddings, labels, groups):
        """
        Compute contrastive loss considering both family and behavioral group structure
        """
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        # Ensure labels and groups are on the correct device
        labels = labels.to(device)
        groups = groups.to(device)
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()) / self.temperature
        
        # Create masks for positive pairs (same family AND same group)
        labels = labels.contiguous().view(-1, 1)
        groups = groups.contiguous().view(-1, 1)
        
        family_mask = torch.eq(labels, labels.t()).float()
        group_mask = torch.eq(groups, groups.t()).float()
        pos_mask = family_mask * group_mask
        
        # Remove self-contrast
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask.masked_fill_(mask, 0)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        loss = -(log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        return loss.mean()
    
    def forward(self, data, phase='family'):
        device = data.x.device
        # Move model buffers to the same device as input
        self.group_prototypes = self.group_prototypes.to(device)
        self.family_prototypes = self.family_prototypes.to(device)
        self.prototype_counts = self.prototype_counts.to(device)
        
        embeddings = self.get_embeddings(data)
        
        if phase == 'family':
            # Hierarchical classification
            group_logits = self.group_classifier(embeddings)
            
            # Get predicted groups for family classification
            if self.training:
                groups = data.group.to(device)  # Use true groups during training
            else:
                groups = group_logits.argmax(dim=1)
            
            # Family classification within groups
            family_logits = {}
            for group_id in range(len(self.family_classifiers)):
                group_mask = (groups == group_id)
                if group_mask.any():
                    group_embeddings = embeddings[group_mask]
                    family_logits[str(group_id)] = self.family_classifiers[str(group_id)](group_embeddings)
            
            return embeddings, group_logits, family_logits
            
        else:  # Novelty detection phase
            # Get group predictions
            group_logits = self.group_classifier(embeddings)
            groups = group_logits.argmax(dim=1)
            
            # Get group context
            group_contexts = self.group_prototypes[groups]
            
            # Concatenate embeddings with group context
            novelty_features = torch.cat([embeddings, group_contexts], dim=1)
            novelty_scores = self.novelty_net(novelty_features)
            
            # Calculate distances to prototypes within predicted group
            distances = []
            for i, group in enumerate(groups):
                group_families_mask = (data.group.to(device) == group)
                group_prototypes = self.family_prototypes[group_families_mask]
                distance = torch.cdist(embeddings[i:i+1], group_prototypes)
                distances.append(distance.min())
            distances = torch.stack(distances)
            
            return novelty_scores, distances

class PhasedTraining:
    def __init__(self, model, device, lr=0.001):
        self.device = device  # Store device in the trainer
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.novelty_threshold = 0.5
    
    def get_device(self):
        return self.device
        
    def train_batch(self, batch, phase):
        # Move batch to device
        batch = batch.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        if phase == 'family':
            # Get predictions
            embeddings, group_logits, family_logits = self.model(batch, phase='family')
            
            # Group classification loss
            group_loss = F.cross_entropy(group_logits, batch.group)
            
            # Family classification loss
            family_loss = 0
            num_groups = 0
            for group_id, logits in family_logits.items():
                group_mask = (batch.group == int(group_id))
                if group_mask.any():
                    family_loss += F.cross_entropy(logits, batch.y[group_mask])
                    num_groups += 1
            if num_groups > 0:
                family_loss /= num_groups
            
            # Contrastive loss
            contrast_loss = self.model.contrastive_loss(embeddings, batch.y, batch.group)
            
            # Combined loss
            loss = group_loss + family_loss + 0.5 * contrast_loss
            
            # Update prototypes
            with torch.no_grad():
                self.model.update_prototypes(embeddings, batch.y, batch.group)
                
        else:
            # Novelty detection training
            novelty_scores, distances = self.model(batch, phase='novelty')
            
            # Consider samples far from prototypes as novel
            novel_targets = (distances > self.novelty_threshold).float()
            loss = F.binary_cross_entropy(novelty_scores, novel_targets)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def load_batch(batch_file, family_to_idx, family_to_group, device, batch_size=32):
    """Load and preprocess a batch with group labels with proper error handling."""
    try:
        if not os.path.exists(batch_file):
            logger.error(f"Batch file not found: {batch_file}")
            return None
            
        batch = torch.load(batch_file)
        processed = []
        
        for graph in batch:
            try:
                # Verify all required attributes exist
                if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                    logger.warning(f"Graph missing required attributes")
                    continue
                
                # Verify x and edge_index are valid tensors
                if not isinstance(graph.x, torch.Tensor) or not isinstance(graph.edge_index, torch.Tensor):
                    logger.warning(f"Invalid tensor types in graph")
                    continue
                
                # Get family and group labels
                family = getattr(graph, 'family', 'unknown')
                if family == 'unknown' or family not in family_to_idx:
                    logger.warning(f"Unknown family: {family}")
                    continue
                    
                family_idx = family_to_idx[family]
                group_idx = family_to_group.get(family, -1)
                if group_idx == -1:
                    logger.warning(f"No group mapping for family: {family}")
                    continue
                
                # Create a new graph object with the required attributes
                processed_graph = type(graph)()
                
                # Copy and move tensors to device
                processed_graph.x = graph.x.to(device)
                processed_graph.edge_index = graph.edge_index.to(device)
                processed_graph.y = torch.tensor(family_idx).to(device)
                processed_graph.group = torch.tensor(group_idx).to(device)
                
                # Handle batch attribute if it exists
                if hasattr(graph, 'batch'):
                    if isinstance(graph.batch, torch.Tensor):
                        processed_graph.batch = graph.batch.to(device)
                    else:
                        processed_graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(device)
                
                # Copy any other relevant attributes
                for attr in ['edge_attr', 'pos']:
                    if hasattr(graph, attr) and isinstance(getattr(graph, attr), torch.Tensor):
                        setattr(processed_graph, attr, getattr(graph, attr).to(device))
                
                processed.append(processed_graph)
                
            except Exception as e:
                logger.error(f"Error processing graph: {str(e)}")
                logger.debug(traceback.format_exc())
                continue
                
        if not processed:
            logger.warning(f"No valid graphs found in batch file: {batch_file}")
            return None
            
        logger.info(f"Successfully processed {len(processed)} graphs from {batch_file}")
        return DataLoader(processed, batch_size=batch_size, shuffle=True)
        
    except Exception as e:
        logger.error(f"Error loading batch {batch_file}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def prepare_data(base_dir='bodmas_batches', groups_file='/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'):
    """Prepare datasets with group and family mappings with improved error handling."""
    try:
        # Load behavioral groups
        if not os.path.exists(groups_file):
            raise FileNotFoundError(f"Groups file not found: {groups_file}")
            
        with open(groups_file, 'r') as f:
            group_data = json.load(f)
        
        # Create mappings
        family_to_group = {}
        for group_id, families in group_data.items():
            for family in families:
                family_to_group[family] = int(group_id)
                
        split_files = defaultdict(list)
        family_counts = defaultdict(int)
        group_counts = defaultdict(int)
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_dir = Path(base_dir) / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
                
            # Collect and process batch files
            batch_files = list(split_dir.glob('batch_*.pt'))
            if not batch_files:
                logger.warning(f"No batch files found in {split_dir}")
                continue
                
            # Verify each batch file
            for file in batch_files:
                try:
                    batch = torch.load(file)
                    if not isinstance(batch, (list, tuple)) or not batch:
                        logger.warning(f"Invalid batch format in {file}")
                        continue
                        
                    # Verify first graph in batch has required attributes
                    first_graph = batch[0]
                    if not hasattr(first_graph, 'x') or not hasattr(first_graph, 'edge_index'):
                        logger.warning(f"Missing required attributes in {file}")
                        continue
                        
                    # Count families and groups
                    valid_graphs = 0
                    for graph in batch:
                        family = getattr(graph, 'family', 'unknown')
                        if family != 'unknown':
                            family_counts[family] += 1
                            group = family_to_group.get(family, -1)
                            if group != -1:
                                group_counts[group] += 1
                                valid_graphs += 1
                                
                    if valid_graphs > 0:
                        split_files[split].append(str(file))
                        logger.info(f"Added {file} to {split} split with {valid_graphs} valid graphs")
                        
                except Exception as e:
                    logger.error(f"Error processing batch file {file}: {str(e)}")
                    continue
        
        if not any(split_files.values()):
            raise ValueError("No valid batch files found in any split!")
        
        # Create index mappings
        families = sorted(family_counts.keys())
        family_to_idx = {fam: idx for idx, fam in enumerate(families)}
        num_groups = max(group_counts.keys()) + 1
        
        # Log dataset statistics
        logger.info(f"Found {len(families)} families across {num_groups} groups")
        logger.info(f"Family distribution: {dict(family_counts)}")
        logger.info(f"Group distribution: {dict(group_counts)}")
        logger.info(f"Split sizes: {dict((k, len(v)) for k, v in split_files.items())}")
        
        return (split_files, family_to_idx, family_to_group, 
                len(families), num_groups)
                
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def train_epoch(trainer, split_files, family_to_idx, family_to_group, phase):
    """Train for one epoch."""
    device = trainer.get_device()  # Get device from trainer
    total_loss = 0
    num_batches = 0
    
    for batch_file in split_files:
        loader = load_batch(batch_file, family_to_idx, family_to_group, device)
        if not loader:
            continue
            
        for batch in loader:
            try:
                loss = trainer.train_batch(batch, phase)
                total_loss += loss
                num_batches += 1
                
                if num_batches % 10 == 0:
                    logger.info(f"Phase: {phase}, Batch: {num_batches}, "
                              f"Average Loss: {total_loss/num_batches:.4f}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    logger.warning("GPU OOM, skipping batch")
                    continue
                else:
                    logger.error(f"Error in batch: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
    
    return total_loss / max(1, num_batches)

def evaluate(trainer, split_files, family_to_idx, family_to_group, phase):
    """Evaluate the model."""
    device = trainer.get_device()  # Get device from trainer
    total_correct = 0
    total_samples = 0
    recall_sum = 0
    num_batches = 0
    
    group_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    family_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    trainer.model.eval()
    
    with torch.no_grad():
        for batch_file in split_files:
            loader = load_batch(batch_file, family_to_idx, family_to_group, device)
            if not loader:
                continue
                
            for batch in loader:
                try:
                    batch = batch.to(device)
                    correct, total, recall = trainer.evaluate(batch, phase)
                    
                    total_correct += correct
                    total_samples += total
                    recall_sum += recall
                    num_batches += 1
                    
                    if phase == 'family':
                        embeddings, group_logits, family_logits = trainer.model(batch, phase='family')
                        group_preds = group_logits.argmax(dim=1)
                        
                        for i, (pred_group, true_group) in enumerate(zip(
                            group_preds, batch.group)):
                            group_metrics[true_group.item()]['total'] += 1
                            if pred_group == true_group:
                                group_metrics[true_group.item()]['correct'] += 1
                                
                            if pred_group == true_group:
                                family_logits_group = family_logits[str(pred_group.item())]
                                family_pred = family_logits_group.argmax(dim=1)
                                family = batch.y[i].item()
                                family_metrics[family]['total'] += 1
                                if family_pred == batch.y[i]:
                                    family_metrics[family]['correct'] += 1
                                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        logger.warning("GPU OOM during evaluation, skipping batch")
                        continue
                    else:
                        logger.error(f"Error in evaluation batch: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
    
    accuracy = total_correct / max(1, total_samples)
    recall = recall_sum / max(1, num_batches)
    
    group_accuracies = {
        group: metrics['correct'] / max(1, metrics['total'])
        for group, metrics in group_metrics.items()
    }
    
    family_accuracies = {
        family: metrics['correct'] / max(1, metrics['total'])
        for family, metrics in family_metrics.items()
    }
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'group_accuracies': group_accuracies,
        'family_accuracies': family_accuracies
    }


def main():
    """Main training and evaluation function."""
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'training_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Setup device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU
        device = torch.device('cuda:0')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    try:
        # Load data and mappings
        data_info = prepare_data()
        split_files, family_to_idx, family_to_group, num_families, num_groups = data_info
        
        if not any(split_files.values()):
            logger.error("No data found!")
            return
            
        # Get feature dimensions from first batch
        first_batch = torch.load(split_files['train'][0])
        num_features = first_batch[0].x.size(1)
        logger.info(f"Features: {num_features}, Families: {num_families}, Groups: {num_groups}")
        
        # Initialize model
        model = ContrastivePhasedGNN(
            num_node_features=num_features,
            num_families=num_families,
            num_groups=num_groups
        ).to(device)
        
        trainer = PhasedTraining(model, device)
        
        # Training configuration
        config = {
            'phases': ['family', 'novelty'],
            'epochs_per_phase': 10,
            'save_dir': Path('trained_models'),
            'batch_size': 32,
            'early_stopping_patience': 5
        }
        
        # Create save directory
        config['save_dir'].mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(config['save_dir'] / 'config.json', 'w') as f:
            json.dump({
                'num_features': num_features,
                'num_families': num_families,
                'num_groups': num_groups,
                'epochs_per_phase': config['epochs_per_phase'],
                'batch_size': config['batch_size']
            }, f, indent=2)
        
        # Training loop
        best_metrics = {'family': 0.0, 'novelty': 0.0}
        patience_counter = {'family': 0, 'novelty': 0}
        
        for phase in config['phases']:
            logger.info(f"\nStarting {phase} phase")
            
            for epoch in range(config['epochs_per_phase']):
                try:
                    # Train
                    loss = train_epoch(trainer, split_files['train'], 
                                     family_to_idx, family_to_group, phase)
                    
                    # Evaluate
                    metrics = evaluate(trainer, split_files['val'], 
                                    family_to_idx, family_to_group, phase)
                    
                    logger.info(
                        f"Epoch {epoch}: loss={loss:.4f}, accuracy={metrics['accuracy']:.4f}, "
                        f"recall={metrics['recall']:.4f}"
                    )
                    
                    # Early stopping check
                    if metrics['accuracy'] > best_metrics[phase]:
                        best_metrics[phase] = metrics['accuracy']
                        patience_counter[phase] = 0
                        
                        # Save best model
                        save_path = config['save_dir'] / f'best_model_{phase}.pt'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': trainer.optimizer.state_dict(),
                            'metrics': metrics,
                            'config': {
                                'num_features': num_features,
                                'num_families': num_families,
                                'num_groups': num_groups
                            }
                        }, save_path)
                        
                        logger.info(f"Saved best {phase} model to {save_path}")
                        
                        # Save mappings
                        mapping_path = config['save_dir'] / f'mappings_{phase}.json'
                        with open(mapping_path, 'w') as f:
                            json.dump({
                                'family_to_idx': family_to_idx,
                                'family_to_group': family_to_group,
                                'idx_to_family': {str(v): k for k, v in family_to_idx.items()}
                            }, f, indent=2)
                    else:
                        patience_counter[phase] += 1
                        if patience_counter[phase] >= config['early_stopping_patience']:
                            logger.info(f"Early stopping triggered for {phase} phase")
                            break
                            
                except Exception as e:
                    logger.error(f"Error in epoch {epoch}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Clear GPU memory after each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final evaluation on test set
        logger.info("\nFinal Evaluation on Test Set:")
        for phase in config['phases']:
            try:
                checkpoint_path = config['save_dir'] / f'best_model_{phase}.pt'
                
                if not checkpoint_path.exists():
                    logger.warning(f"No checkpoint found at {checkpoint_path}")
                    continue
                    
                # Load checkpoint
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {str(e)}")
                    continue
                
                # Evaluate
                metrics = evaluate(trainer, split_files['test'], 
                                family_to_idx, family_to_group, phase)
                
                logger.info(f"\n{phase.capitalize()} Phase Test Results:")
                logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"Recall: {metrics['recall']:.4f}")
                
                # Log group performance
                logger.info("\nGroup Performance:")
                for group, acc in metrics['group_accuracies'].items():
                    logger.info(f"Group {group}: {acc:.4f}")
                    
                # Log family performance
                logger.info("\nTop 10 Family Performance:")
                top_families = sorted(
                    metrics['family_accuracies'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for family, acc in top_families:
                    logger.info(f"Family {family}: {acc:.4f}")
                    
            except Exception as e:
                logger.error(f"Error in final evaluation of {phase} phase: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Final GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

if __name__ == "__main__":
    main()

# # Define model and trainer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ContrastivePhasedGNN(num_node_features=4, num_families=5, num_groups=3)
# trainer = PhasedTraining(model, device)

# # Example usage
# data = torch.load('data.pt')
# train_loader = DataLoader(data['train'], batch_size=64, shuffle=True)
# test_loader = DataLoader(data['test'], batch_size=64)

# for epoch in range(3):
#     for batch in train_loader:
#         loss = trainer.train_batch(batch, phase='family')
#     print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
#     total_correct, total_preds, recall = 0, 0, 0
#     for batch in test_loader:
#         correct, total, recall = trainer.evaluate(batch, phase='family')
#         total_correct += correct
#         total_preds += total
#     print(f"Test Accuracy: {total_correct / total_preds:.4f}, Recall: {recall:.4f}")
