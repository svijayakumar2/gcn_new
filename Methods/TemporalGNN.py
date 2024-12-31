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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from gcn import CentroidLayer, MalwareGNN, MalwareTrainer


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
            return self.num_known_groups + 1  # +1 for novel group
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
                  use_groups: bool = False) -> DataLoader:
        """Load a data split maintaining temporal order."""
        split_dir = self.batch_dir / split
        batch_files = sorted(list(split_dir.glob('batch_*.pt')))
        
        logger.info(f"Loading {len(batch_files)} batch files from {split_dir}")
        
        all_graphs = []
        skipped = 0
        
        for batch_file in batch_files:
            try:
                batch_data = torch.load(batch_file)
                
                # Process each graph
                for graph in batch_data:
                    processed = self._process_graph(graph, use_groups)
                    if processed is not None:
                        all_graphs.append(processed)
                    else:
                        skipped += 1
                        
            except Exception as e:
                logger.error(f"Error loading {batch_file}: {str(e)}")
                continue
        
        # Sort by timestamp
        all_graphs.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Loaded {len(all_graphs)} graphs from {split} split "
                   f"({skipped} skipped)")
        
        # Create loader
        loader = DataLoader(all_graphs, batch_size=batch_size, shuffle=False)
        return loader
        
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

def main():
    print("="*50)
    print("Starting main function")
    print("="*50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Initializing data loader...")
    data_loader = TemporalMalwareDataLoader(
        batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches_new'),
        behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        metadata_path=Path('bodmas_metadata_cleaned.csv'),
        malware_types_path=Path('bodmas_malware_category.csv')
    )
    
    # Load data first to determine number of classes
    print("Loading training data...")
    train_loader_family = data_loader.load_split('train', use_groups=False, batch_size=32)
    train_loader_groups = data_loader.load_split('train', use_groups=True, batch_size=32)
    
    print("Loading validation data...")
    val_loader_family = data_loader.load_split('val', use_groups=False, batch_size=32)
    val_loader_groups = data_loader.load_split('val', use_groups=True, batch_size=32)
    
    # Get number of classes after loading data
    num_family_classes = data_loader.get_num_classes(use_groups=False)
    num_group_classes = data_loader.get_num_classes(use_groups=True)
    
    logger.info(f"Total number of family classes (including novel): {num_family_classes}")
    logger.info(f"Total number of behavioral groups (including novel): {num_group_classes}")
    
    if num_family_classes == 0 or num_group_classes == 0:
        raise ValueError("No classes found in the dataset")
    
    # Now initialize models with correct number of classes
    print("Initializing models...")
    family_model = MalwareGNN(
        num_node_features=14,
        hidden_dim=128,
        num_classes=num_family_classes,
        n_centroids_per_class=2
    ).to(device)
    
    group_model = MalwareGNN(
        num_node_features=14,
        hidden_dim=128,
        num_classes=num_group_classes,
        n_centroids_per_class=2
    ).to(device)
    
    # Initialize trainers
    family_trainer = MalwareTrainer(family_model, device)
    group_trainer = MalwareTrainer(group_model, device)
    
    # Compute class weights after data is loaded
    family_weights = family_trainer.compute_class_weights(train_loader_family)
    group_weights = group_trainer.compute_class_weights(train_loader_groups)
    
    if family_weights is None or group_weights is None:
        raise ValueError("Failed to compute class weights")
    
    # Training parameters
    num_epochs = 100
    best_family_acc = 0
    best_group_acc = 0
    
    # Print key information before training
    logger.info(f"Starting training with:")
    logger.info(f"- Number of family classes: {num_family_classes}")
    logger.info(f"- Number of group classes: {num_group_classes}")
    logger.info(f"- Device: {device}")
    logger.info(f"- Batch size: 32")
    logger.info(f"- Number of epochs: {num_epochs}")
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train family classification
        logger.info("Training family classification...")
        family_train_metrics = family_trainer.train_epoch(train_loader_family, family_weights)
        family_val_metrics = family_trainer.evaluate(val_loader_family, family_weights)
        
        # Train behavioral group classification
        logger.info("Training behavioral group classification...")
        group_train_metrics = group_trainer.train_epoch(train_loader_groups, group_weights)
        group_val_metrics = group_trainer.evaluate(val_loader_groups, group_weights)
        
        # Log metrics
        logger.info(f"\nFamily Classification:")

        logger.info(f"Train - Loss: {family_train_metrics['loss']:.4f}, "
                   f"Accuracy: {family_train_metrics['accuracy']:.4f}")
        logger.info(f"Val - Loss: {family_val_metrics['loss']:.4f}, "
                   f"Accuracy: {family_val_metrics['accuracy']:.4f}")
        
        logger.info(f"\nBehavioral Group Classification:")
        logger.info(f"Train - Loss: {group_train_metrics['loss']:.4f}, "
                   f"Accuracy: {group_train_metrics['accuracy']:.4f}")
        logger.info(f"Val - Loss: {group_val_metrics['loss']:.4f}, "
                   f"Accuracy: {group_val_metrics['accuracy']:.4f}")
        
        # Save best models
        if family_val_metrics['accuracy'] > best_family_acc:
            best_family_acc = family_val_metrics['accuracy']
            torch.save(family_model.state_dict(), 'best_family_model.pt')
            
        if group_val_metrics['accuracy'] > best_group_acc:
            best_group_acc = group_val_metrics['accuracy']
            torch.save(group_model.state_dict(), 'best_group_model.pt')
        
        # Analyze behavioral evolution every 5 epochs
        if epoch % 5 == 0:
            evolution_metrics = data_loader.analyze_behavioral_evolution()
            logger.info("\nEvolutionary Analysis:")
            logger.info(f"Novel families: {len(evolution_metrics['novel_families'])}")
            
            for family, drift in evolution_metrics['behavioral_drift'].items():
                if drift['trend'] > 0.1:
                    logger.info(f"Family {family} showing significant evolution:")
                    logger.info(f"- Mean drift: {drift['mean_drift']:.3f}")
                    logger.info(f"- Trend: {drift['trend']:.3f}")

if __name__ == "__main__":
    print("Script is starting...")
    main()
    print("Script has finished.")