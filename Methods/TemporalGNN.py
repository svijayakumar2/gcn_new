import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

import pandas as pd
from torch_geometric.data import Data, DataLoader
import numpy as np
# nn.CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

CONFIDENCE_THRESHOLD = .8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from gcn import CentroidLayer, MalwareGNN, MalwareTrainer

# Helper class for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
    
def train_and_evaluate(trainer, train_loader, val_loader, class_weights, 
                      model_name, early_stopping_patience=5, num_epochs=100):
    """Train and evaluate with detailed metrics tracking."""
    best_f1 = 0
    patience_counter = 0
    metrics_history = []
    
    for epoch in range(num_epochs):
        # Training
        train_metrics = trainer.evaluate(train_loader, class_weights)
        trainer.log_metrics(train_metrics, split="train")
        
        # Validation
        val_metrics = trainer.evaluate(val_loader, class_weights)
        trainer.log_metrics(val_metrics, split="val")
        
        # Save metrics history
        metrics_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # Save metrics to file
        with open(f'{model_name}_metrics2.json', 'w') as f:
            json.dump(metrics_history, f, indent=2, cls=NumpyEncoder)
        
        # Early stopping based on F1 score
        current_f1 = val_metrics['overall']['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'metrics': val_metrics,
                'epoch': epoch
            }, f'best_{model_name}_model.pt')
            logger.info(f"New best model saved with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return metrics_history

class TemporalMalwareDataLoader:
    def __init__(self, 
                 batch_dir: Path,
                 behavioral_groups_path: Path,
                 metadata_path: Path,
                 malware_types_path: Path):
        self.batch_dir = Path(batch_dir)
        
        # Load behavioral groups
        with open(behavioral_groups_path) as f:
            behavioral_groups = json.load(f)
            
        # Create mappings with a special "unknown" group
        self.family_to_group = {}
        self.group_to_families = defaultdict(set)
        
        # Add unknown group
        self.UNKNOWN_GROUP_ID = -1
        self.NOVEL_GROUP_ID = len(behavioral_groups)
        
        # Initialize family index mapping
        self.family_to_idx = {}
        next_family_idx = 0
        
        # Process known families
        for group_id, families in behavioral_groups.items():
            group_id = int(group_id)
            for family in families:
                family = family.lower()
                self.family_to_group[family] = group_id
                self.group_to_families[group_id].add(family)
                
                # Assign family index
                if family not in self.family_to_idx:
                    self.family_to_idx[family] = next_family_idx
                    next_family_idx += 1
        
        self.num_known_families = next_family_idx
        self.num_known_groups = len(behavioral_groups)
        
        # Initialize trackers
        self.novel_families = set()
        self.family_similarity_matrix = defaultdict(lambda: defaultdict(float))
        self.metadata_df = self._load_metadata(metadata_path, malware_types_path)
        
        # Initialize statistics with timezone-aware timestamps
        default_min_ts = pd.Timestamp.min.tz_localize('UTC')
        default_max_ts = pd.Timestamp.max.tz_localize('UTC')
        
        self.family_first_seen = defaultdict(lambda: default_max_ts)
        self.family_last_seen = defaultdict(lambda: default_min_ts)
        self.family_counts = defaultdict(int)
        self.group_counts = defaultdict(int)
        self.family_behavioral_drift = defaultdict(list)
        self.family_feature_history = defaultdict(list)
        
        logger.info(f"Initialized with {self.num_known_families} known families and {self.num_known_groups} behavioral groups")

    def _process_graph(self, graph: Data, use_groups: bool = False) -> Optional[Data]:
        """Process a single graph with proper handling of unknown families."""
        try:
            family = graph.family.lower() if hasattr(graph, 'family') else 'unknown'
            
            # Standardize timestamp
            timestamp = self._standardize_timestamp(graph.timestamp)
            if timestamp is None:
                return None

            # Update statistics
            self.family_first_seen[family] = min(self.family_first_seen[family], timestamp)
            self.family_last_seen[family] = max(self.family_last_seen[family], timestamp)
            self.family_counts[family] += 1
            
            # Handle group assignment
            if family in self.family_to_group:
                group = self.family_to_group[family]
                self.group_counts[group] += 1
            else:
                # New family
                if family not in self.family_to_idx:
                    self.family_to_idx[family] = len(self.family_to_idx)
                    self.novel_families.add(family)
                    logger.info(f"New family detected: {family} (idx: {self.family_to_idx[family]})")
                group = self.NOVEL_GROUP_ID
            
            # Set graph attributes
            graph.group = torch.tensor(group, dtype=torch.long)
            graph.y = torch.tensor(
                self.family_to_idx[family] if not use_groups else group,
                dtype=torch.long
            )
            graph.is_novel = torch.tensor(family in self.novel_families, dtype=torch.bool)
            graph.timestamp = timestamp
            
            return graph
            
        except Exception as e:
            logger.error(f"Error processing graph: {str(e)}")
            return None

    def get_num_classes(self, use_groups: bool = False) -> int:
        """Get the current number of classes."""
        if use_groups:
            return self.num_known_groups #+ 1  # +1 for novel group
        else:
            return len(self.family_to_idx)  # Includes both known and novel families
        
    
    def analyze_behavioral_evolution(self) -> Dict:
        """Analyze evolutionary patterns in behavioral features."""
        evolution_metrics = {
            'novel_families': list(self.novel_families),
            'behavioral_drift': {},
            'potential_variants': [],
            'convergent_evolution': []
        }
        
        # Analyze feature evolution for each family
        for family, history in self.family_feature_history.items():
            if len(history) < 2:
                continue
                
            # Sort by timestamp
            sorted_history = sorted(history, key=lambda x: x['timestamp'])
            
            # Calculate behavioral drift
            feature_drifts = []
            for i in range(1, len(sorted_history)):
                prev_features = torch.tensor(sorted_history[i-1]['features'])
                curr_features = torch.tensor(sorted_history[i]['features'])
                drift = torch.norm(curr_features - prev_features).item()
                feature_drifts.append(drift)
            
            evolution_metrics['behavioral_drift'][family] = {
                'mean_drift': np.mean(feature_drifts),
                'max_drift': max(feature_drifts),
                'trend': np.polyfit(range(len(feature_drifts)), feature_drifts, 1)[0]
            }
            
            # Identify potential variants
            if evolution_metrics['behavioral_drift'][family]['max_drift'] > 0.5:
                evolution_metrics['potential_variants'].append(family)
        
        return evolution_metrics
    
    
    def _standardize_timestamp(self, ts: str) -> pd.Timestamp:
        """Convert timestamp string to pandas Timestamp with UTC timezone."""
        try:
            # Parse timestamp with pandas
            dt = pd.to_datetime(ts)
            # Ensure UTC timezone
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            else:
                dt = dt.tz_convert('UTC')
            return dt
        except Exception as e:
            logger.error(f"Error parsing timestamp {ts}: {str(e)}")
            return None

      
    def _load_metadata(self, metadata_path: Path, malware_types_path: Path) -> pd.DataFrame:
        """Load and merge metadata."""
        metadata_df = pd.read_csv(metadata_path)
        malware_types_df = pd.read_csv(malware_types_path)
        
        # Prepare for merge
        metadata_df['filename'] = metadata_df['sha']
        malware_types_df['filename'] = malware_types_df['sha256'].apply(lambda x: Path(x).stem)
        
        # Merge
        merged_df = pd.merge(
            metadata_df,
            malware_types_df[['filename', 'category']].rename(columns={'category': 'malware_type'}),
            on='filename',
            how='left'
        )
        
        return merged_df
        
    def load_split(self, split: str = 'train', batch_size: int = 32, 
                use_groups: bool = False) -> Tuple[DataLoader, Dict]:
        """Load a data split maintaining temporal order and return statistics."""
        split_dir = self.batch_dir / split
        batch_files = sorted(list(split_dir.glob('batch_*.pt')))
        
        logger.info(f"Loading {len(batch_files)} batch files from {split_dir}")
        
        all_graphs = []
        skipped = 0
        
        # Track statistics
        split_stats = {
            'known_families': defaultdict(int),
            'novel_families': defaultdict(int),
            'known_groups': defaultdict(int),
            'novel_groups': defaultdict(int),
            'total_samples': 0,
            'novel_samples': 0
        }
        
        for batch_file in batch_files:
            try:
                batch_data = torch.load(batch_file)
                
                # Process each graph
                for graph in batch_data:
                    processed = self._process_graph(graph, use_groups)
                    if processed is not None:
                        all_graphs.append(processed)
                        
                        # Update statistics
                        split_stats['total_samples'] += 1
                        family = graph.family.lower() if hasattr(graph, 'family') else 'unknown'
                        
                        if family in self.family_to_group:
                            split_stats['known_families'][family] += 1
                            split_stats['known_groups'][self.family_to_group[family]] += 1
                        else:
                            split_stats['novel_families'][family] += 1
                            split_stats['novel_samples'] += 1
                            split_stats['novel_groups'][self.NOVEL_GROUP_ID] += 1
                    else:
                        skipped += 1
                        
            except Exception as e:
                logger.error(f"Error loading {batch_file}: {str(e)}")
                continue
        
        # Sort by timestamp
        all_graphs.sort(key=lambda x: x.timestamp)
        
        # Log split statistics
        logger.info(f"\nSplit statistics for {split}:")
        logger.info(f"Total samples: {split_stats['total_samples']}")
        logger.info(f"Novel samples: {split_stats['novel_samples']} "
                f"({split_stats['novel_samples']/split_stats['total_samples']*100:.2f}%)")
        logger.info(f"Known families: {len(split_stats['known_families'])}")
        logger.info(f"Novel families: {len(split_stats['novel_families'])}")
        
        # Create loader
        # In load_split method:
        loader = DataLoader(
            all_graphs, 
            batch_size=batch_size, 
            shuffle=(split == 'train'),
            num_workers=4,  # Add parallel data loading
            pin_memory=True  # Better GPU transfer
        )

        return loader, split_stats

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'families': {
                'total': len(self.family_counts),
                'counts': dict(self.family_counts),
                'temporal_range': {
                    family: {
                        'first_seen': self.family_first_seen[family],
                        'last_seen': self.family_last_seen[family]
                    }
                    for family in self.family_counts
                }
            },
            'groups': {
                'total': len(self.group_counts),
                'counts': dict(self.group_counts)
            }
        }
        return stats

def analyze_behavioral_evolution(family_preds, group_preds, family_outliers, group_outliers, stats):
    """Analyze evolutionary patterns in behavioral predictions."""
    # Track movement between behavioral groups
    group_transitions = defaultdict(int)
    
    # Identify potentially novel behaviors (high outlier scores)
    novel_behaviors = []
    for idx, (f_score, g_score) in enumerate(zip(family_outliers, group_outliers)):
        if f_score > 0.8 and g_score > 0.8:  # High outlier scores indicate novelty
            novel_behaviors.append({
                'family_pred': family_preds[idx],
                'group_pred': group_preds[idx],
                'family_outlier_score': f_score,
                'group_outlier_score': g_score
            })
    
    if novel_behaviors:
        logger.info("\nPotential Novel Behaviors Detected:")
        for behavior in novel_behaviors:
            logger.info(f"Family: {behavior['family_pred']}, "
                       f"Group: {behavior['group_pred']}, "
                       f"Novelty Scores: {behavior['family_outlier_score']:.3f}, "
                       f"{behavior['group_outlier_score']:.3f}")

def evaluate_novel_detection(trainer, loader, class_weights) -> Dict:
    """Evaluate model's performance on novel family detection."""
    trainer.model.eval()
    
    novel_detection_metrics = {
        'novel': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'per_family': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    }
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(trainer.device)
            
            # Get model predictions and outlier scores
            logits, outlier_scores = trainer.model(batch)
            confidences = F.softmax(logits, dim=1).max(dim=1)[0]
            preds = logits.argmax(dim=1)
            
            # Novel detection using both outlier scores and confidence
            confidence_threshold = 0.8  # Can be tuned
            outlier_threshold = 0.7     # Can be tuned
            
            # A sample is considered novel if either:
            # 1. It has high outlier score OR
            # 2. It has low confidence in its prediction
            pred_novel = (outlier_scores > outlier_threshold) | (confidences < confidence_threshold)
            
            # Update counters for novel detection
            for pred, is_novel in zip(pred_novel, batch.is_novel):
                pred = pred.item()
                true_novel = is_novel.item()
                
                # Update global novel detection metrics
                if true_novel and pred:
                    novel_detection_metrics['novel']['tp'] += 1
                elif true_novel and not pred:
                    novel_detection_metrics['novel']['fn'] += 1
                elif not true_novel and pred:
                    novel_detection_metrics['novel']['fp'] += 1
                else:
                    novel_detection_metrics['novel']['tn'] += 1
                
                # Update per-family metrics
                family = batch.family[0] if hasattr(batch, 'family') else 'unknown'
                if isinstance(family, str):
                    family = family.lower()
                
                if true_novel and pred:
                    novel_detection_metrics['per_family'][family]['tp'] += 1
                elif true_novel and not pred:
                    novel_detection_metrics['per_family'][family]['fn'] += 1
                elif not true_novel and pred:
                    novel_detection_metrics['per_family'][family]['fp'] += 1
                else:
                    novel_detection_metrics['per_family'][family]['tn'] += 1
    
    # Calculate metrics
    def calculate_metrics(stats):
        tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'raw_counts': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        }
    
    # Calculate overall metrics
    metrics = {
        'overall': calculate_metrics(novel_detection_metrics['novel']),
        'per_family': {
            family: calculate_metrics(stats) 
            for family, stats in novel_detection_metrics['per_family'].items()
        }
    }
    
    return metrics
import os 
import glob

def cleanup_old_checkpoints(name, keep_last_n=5):
    """Keep only the N most recent checkpoints."""
    checkpoints = sorted(glob.glob(f'checkpoint_{name}_epoch_*.pt'))
    if len(checkpoints) > keep_last_n:
        for checkpoint in checkpoints[:-keep_last_n]:
            os.remove(checkpoint)
            logger.info(f"Removed old checkpoint: {checkpoint}")

def resume_from_checkpoint(checkpoint_path, model, trainer, start_epoch):
    """Resume training from a checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load metrics history if available
    # metrics_history = []
    # metrics_file = checkpoint_path.replace('.pt', '_metrics.json')
    # if os.path.exists(metrics_file):
    #     with open(metrics_file, 'r') as f:
    #         metrics_history = json.load(f)
    
    return model, trainer#, metrics_history


def save_checkpoint(epoch, model, optimizer, metrics, novel_metrics, name, is_best=False):
    """Save training checkpoint and optionally mark as best model."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'novel_metrics': novel_metrics
    }
    
    # Save periodic checkpoint
    torch.save(checkpoint, f'checkpoint_{name}_epoch_{epoch}.pt')
    logger.info(f"Saved checkpoint at epoch {epoch}")
    
    # Optionally save as best model
    if is_best:
        torch.save(checkpoint, f'best_{name}_model.pt')
        logger.info(f"Saved as best model")



def main():
    # Setup
    # Memory optimization config
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory allocator config
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data loader
    data_loader = TemporalMalwareDataLoader(
        batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches'),
        behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        metadata_path=Path('bodmas_metadata_cleaned.csv'),
        malware_types_path=Path('bodmas_malware_category.csv')
    )
    
    # Load data
    train_loader_family, train_stats = data_loader.load_split('train', use_groups=False, batch_size=32)
    train_loader_groups, train_group_stats = data_loader.load_split('train', use_groups=True, batch_size=32)
    val_loader_family, val_stats = data_loader.load_split('val', use_groups=False, batch_size=32)
    val_loader_groups, val_group_stats = data_loader.load_split('val', use_groups=True, batch_size=32)
    
    # Get number of classes
    num_family_classes = data_loader.get_num_classes(use_groups=False)
    num_group_classes = data_loader.get_num_classes(use_groups=True)
    
    if num_family_classes == 0 or num_group_classes == 0:
        raise ValueError("No classes found in the dataset")
    
    # Initialize models
    # Initialize models with better parameters
    family_model = MalwareGNN(
        num_node_features=14,
        hidden_dim=256,  # Increased from 128
        num_classes=num_family_classes,
        n_centroids_per_class=4,  # Increased from 2
        num_layers=4,  # New parameter
        dropout=0.2  # New parameter
    ).to(device)

    group_model = MalwareGNN(
        num_node_features=14,
        hidden_dim=256,
        num_classes=num_group_classes,
        n_centroids_per_class=4,
        num_layers=4,
        dropout=0.2
    ).to(device)
    
    # Initialize trainers with better parameters
    family_trainer = MalwareTrainer(
        model=family_model, 
        device=device,
        lr=0.001,
        weight_decay=1e-4
    )
    group_trainer = MalwareTrainer(
        model=group_model, 
        device=device,
        lr=0.001,
        weight_decay=1e-4
    )
    # Compute class weights
    family_weights = family_trainer.compute_class_weights(train_loader_family)
    group_weights = group_trainer.compute_class_weights(train_loader_groups)
    
    if family_weights is None or group_weights is None:
        raise ValueError("Failed to compute class weights")
    
    # Training parameters
    num_epochs = 50
    # early stopping: less stopping is 
    early_stopping_patience = 15
    best_family_f1 = 0
    best_family_acc = 0
    best_group_f1 = 0
    best_group_acc = 0
    family_patience = 0
    group_patience = 0
    
    # Metrics history
    family_metrics_history = []
    group_metrics_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Train and evaluate family model
        family_train_metrics = family_trainer.train_epoch(train_loader_family, family_weights)
        family_val_metrics = family_trainer.evaluate(val_loader_family, family_weights)
        family_novel_metrics = evaluate_novel_detection(family_trainer, val_loader_family, family_weights)
        
        # Train and evaluate group model
        group_train_metrics = group_trainer.train_epoch(train_loader_groups, group_weights)
        group_val_metrics = group_trainer.evaluate(val_loader_groups, group_weights)
        group_novel_metrics = evaluate_novel_detection(group_trainer, val_loader_groups, group_weights)
        
        # Save metrics history
        family_metrics_history.append({
            'epoch': epoch,
            'train': family_train_metrics,
            'val': family_val_metrics,
            'novel_detection': family_novel_metrics
        })
        
        group_metrics_history.append({
            'epoch': epoch,
            'train': group_train_metrics,
            'val': group_val_metrics,
            'novel_detection': group_novel_metrics
        })
        
        # Save metrics to file
        with open('family_metrics2.json', 'w') as f:
            json.dump(family_metrics_history, f, indent=2, cls=NumpyEncoder)
        with open('group_metrics2.json', 'w') as f:
            json.dump(group_metrics_history, f, indent=2, cls=NumpyEncoder)
        
        # Early stopping based only on F1 score
        current_family_f1 = family_val_metrics['overall']['f1']
        current_group_f1 = group_val_metrics['overall']['f1']
        
        # Save best family model
        if current_family_f1 > best_family_f1:
            best_family_f1 = current_family_f1
            family_patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': family_model.state_dict(),
                'optimizer_state_dict': family_trainer.optimizer.state_dict(),
                'metrics': family_val_metrics,
                'novel_metrics': family_novel_metrics
            }, 'best_family_model.pt')
            logger.info(f"New best family model saved with F1: {best_family_f1:.4f}")
        else:
            family_patience += 1
        
        # Save best group model
        if current_group_f1 > best_group_f1:
            best_group_f1 = current_group_f1
            group_patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': group_model.state_dict(),
                'optimizer_state_dict': group_trainer.optimizer.state_dict(),
                'metrics': group_val_metrics,
                'novel_metrics': group_novel_metrics
            }, 'best_group_model.pt')
            logger.info(f"New best group model saved with F1: {best_group_f1:.4f}")
        else:
            group_patience += 1
                
        # Early stopping check
        if family_patience >= early_stopping_patience and group_patience >= early_stopping_patience:
            if epoch < 30:  # Minimum number of epochs
                logger.info("Continuing training despite early stopping trigger")
                family_patience = 0
                group_patience = 0
            else:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Compute optimal thresholds periodically
        if epoch % 5 == 0:
            family_thresholds = family_trainer.compute_optimal_thresholds(val_loader_family)
            group_thresholds = group_trainer.compute_optimal_thresholds(val_loader_groups)
            family_trainer.best_thresholds = family_thresholds
            group_trainer.best_thresholds = group_thresholds

        if epoch % 10 == 0:
            cleanup_old_checkpoints('family', keep_last_n=5)
            cleanup_old_checkpoints('group', keep_last_n=5)
            save_checkpoint(
                epoch=epoch,
                model=family_model,
                optimizer=family_trainer.optimizer,
                metrics=family_val_metrics,
                novel_metrics=family_novel_metrics,
                name='family'
            )
            save_checkpoint(
                epoch=epoch,
                model=group_model,
                optimizer=group_trainer.optimizer,
                metrics=group_val_metrics,
                novel_metrics=group_novel_metrics,
                name='group'
            )
        
        # Save best models as before
        if current_family_f1 > best_family_f1:
            best_family_f1 = current_family_f1
            family_patience = 0
            save_checkpoint(
                epoch=epoch,
                model=family_model,
                optimizer=family_trainer.optimizer,
                metrics=family_val_metrics,
                novel_metrics=family_novel_metrics,
                name='family',
                is_best=True
            )
        
        # Similar for group model
        if current_group_f1 > best_group_f1:
            best_group_f1 = current_group_f1
            group_patience = 0
            save_checkpoint(
                epoch=epoch,
                model=group_model,
                optimizer=group_trainer.optimizer,
                metrics=group_val_metrics,
                novel_metrics=group_novel_metrics,
                name='group',
                is_best=True
            )
        # Step the schedulers
        family_trainer.scheduler.step(family_val_metrics['overall']['f1'])
        group_trainer.scheduler.step(group_val_metrics['overall']['f1'])

    # Load test data and evaluate
    test_loader_family, test_stats = data_loader.load_split('test', use_groups=False, batch_size=32)
    test_loader_groups, test_group_stats = data_loader.load_split('test', use_groups=True, batch_size=32)
    
    # Load best models
    family_checkpoint = torch.load('best_family_model.pt')
    family_model.load_state_dict(family_checkpoint['model_state_dict'])
    
    group_checkpoint = torch.load('best_group_model.pt')
    group_model.load_state_dict(group_checkpoint['model_state_dict'])
    
    # Final test evaluation
    test_family_metrics = family_trainer.evaluate(test_loader_family, family_weights)
    test_group_metrics = group_trainer.evaluate(test_loader_groups, group_weights)
    test_family_novel = evaluate_novel_detection(family_trainer, test_loader_family, family_weights)
    test_group_novel = evaluate_novel_detection(group_trainer, test_loader_groups, group_weights)
    
    # Save final report
    final_report = {
        'training_history': {
            'family': family_metrics_history,
            'group': group_metrics_history
        },
        'best_models': {
            'family': {
                'epoch': family_checkpoint['epoch'],
                'metrics': family_checkpoint['metrics'],
                'novel_metrics': family_checkpoint['novel_metrics']
            },
            'group': {
                'epoch': group_checkpoint['epoch'],
                'metrics': group_checkpoint['metrics'],
                'novel_metrics': group_checkpoint['novel_metrics']
            }
        },
        'test_results': {
            'family': {
                'metrics': test_family_metrics,
                'novel_detection': test_family_novel
            },
            'group': {
                'metrics': test_group_metrics,
                'novel_detection': test_group_novel
            }
        },
        'data_statistics': {
            'train': {
                'family': train_stats,
                'group': train_group_stats
            },
            'val': {
                'family': val_stats,
                'group': val_group_stats
            },
            'test': {
                'family': test_stats,
                'group': test_group_stats
            }
        }
    }
    
    with open('final_report2.json', 'w') as f:
        json.dump(final_report, f, indent=2, cls=NumpyEncoder)



if __name__ == "__main__":
    main()

