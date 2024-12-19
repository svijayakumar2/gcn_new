import json
import logging
import torch
import pandas as pd
from collections import defaultdict
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import os
import glob




def log_temporal_metrics(metrics: dict, evolution: list, epoch: int):
    """Log temporal performance metrics and family evolution events."""
    logger.info(f"\nEpoch {epoch} Temporal Metrics:")
    
    # Log rolling metrics
    logger.info("7-day Rolling Averages:")
    logger.info(f"Known Family Accuracy: {metrics['known_accuracy'].mean():.4f}")
    logger.info(f"New Family Detection Rate: {metrics['new_detection_rate'].mean():.4f}")
    logger.info(f"False Positive Rate: {metrics['false_positive_rate'].mean():.4f}")
    
    # Log recent evolution events
    recent_events = [e for e in evolution if (pd.Timestamp.now() - e['timestamp']).days <= 7]
    if recent_events:
        logger.info("\nRecent Family Evolution Events:")
        for event in recent_events:
            logger.info(f"{event['timestamp']}: {event['family']} - {event['event']}")

def get_predictions(group_logits: torch.Tensor, family_logits: dict) -> List[str]:
    """Get family predictions from model outputs."""
    predicted_groups = group_logits.argmax(dim=1)
    predicted_families = []
    
    for i, group_id in enumerate(predicted_groups):
        group_id = group_id.item()
        family_logits_group = family_logits[str(group_id)][i]
        family_idx = family_logits_group.argmax().item()
        predicted_family = group_mappings['group_to_families'][group_id][family_idx]
        predicted_families.append(predicted_family)
    
    return predicted_families

def get_metrics(batch, new_family_flags: List[bool]) -> Tuple[List[bool], List[bool], List[bool]]:
    """Compute metrics for both known and new family detection."""
    correct_known = []  # Correct classifications for known families
    correct_new = []    # Correct new family detections
    false_new = []      # False new family detections
    
    for true_family, is_new in zip(batch.family, new_family_flags):
        if is_new:
            # For samples flagged as new families
            correct_new.append(true_family not in known_families)
            false_new.append(true_family in known_families)
            correct_known.append(False)
        else:
            # For samples predicted as known families
            correct_new.append(False)
            false_new.append(false_new)
            correct_known.append(
                predicted_family == true_family if true_family in known_families else False
            )
    
    return correct_known, correct_new, false_new

def evaluate_temporal(model: torch.nn.Module, 
                     classifier: TemporalMalwareClassifier,
                     loader: DataLoader,
                     device: torch.device) -> dict:
    """Evaluate model with temporal metrics."""
    model.eval()
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            embeddings, group_logits, family_logits = model(batch)
            
            # Detect new families
            new_family_flags = classifier.detect_new_families(embeddings)
            
            # Get predictions and compute metrics
            predictions = get_predictions(group_logits, family_logits)
            correct_known, correct_new, false_new = get_metrics(batch, new_family_flags)
            
            # Store results
            for i, (pred, true_fam, is_new) in enumerate(zip(predictions, batch.family, new_family_flags)):
                metrics['timestamp'].append(batch.timestamp[i])
                metrics['true_family'].append(true_fam)
                metrics['predicted_family'].append(pred)
                metrics['is_new'].append(is_new)
                metrics['correct_known'].append(correct_known[i])
                metrics['correct_new'].append(correct_new[i])
                metrics['false_new'].append(false_new[i])
    
    # Compute aggregate metrics
    results = {
        'known_accuracy': np.mean([m for m, n in zip(metrics['correct_known'], 
                                                    metrics['is_new']) if not n]),
        'new_detection_rate': np.mean(metrics['correct_new']),
        'false_positive_rate': np.mean(metrics['false_new']),
        'temporal_metrics': {
            'timestamps': metrics['timestamp'],
            'accuracies': metrics['correct_known']
        }
    }
    
    return results

class HierarchicalMalwareClassifier:
    def __init__(self, behavioral_groups_path: str):
        self.behavioral_groups = self._load_groups(behavioral_groups_path)
        self.group_mappings = self._create_mappings()
        
    def _load_groups(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
            
    def _create_mappings(self):
        """Create bidirectional mappings between families and groups."""
        family_to_group = {}
        group_to_families = defaultdict(list)
        
        for group_id, families in self.behavioral_groups.items():
            for family in families:
                family_to_group[family] = int(group_id)
                group_to_families[int(group_id)].append(family)
                
        return {
            'family_to_group': family_to_group,
            'group_to_families': dict(group_to_families)
        }
        
    def get_family_group(self, family: str) -> int:
        return self.group_mappings['family_to_group'].get(family, -1)
    
class HierarchicalGNN(torch.nn.Module):
    def __init__(self, num_features, num_groups, num_families):
        super().__init__()
        
        # Base GNN layers remain the same
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 256)
        
        # Hierarchical classification heads
        self.group_classifier = torch.nn.Linear(256, num_groups)
        self.family_classifiers = torch.nn.ModuleDict()
        
        # Create separate classifier for each behavioral group
        for group_id in range(num_groups):
            num_families_in_group = len(group_mappings['group_to_families'][group_id])
            self.family_classifiers[str(group_id)] = torch.nn.Linear(256, num_families_in_group)
            
    def forward(self, data):
        # Base GNN encoding
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        
        # Global pooling
        x = global_mean_pool(x, data.batch)
        
        # Group prediction
        group_logits = self.group_classifier(x)
        
        # Family predictions for each group
        family_logits = {}
        for group_id in self.family_classifiers:
            family_logits[group_id] = self.family_classifiers[group_id](x)
            
        return group_logits, family_logits

class HierarchicalLoss(torch.nn.Module):
    def __init__(self, group_mappings, alpha=0.3):
        super().__init__()
        self.group_mappings = group_mappings
        self.alpha = alpha
        
    def forward(self, group_logits, family_logits, true_families, device):
        # Convert family labels to group labels
        true_groups = torch.tensor([
            self.group_mappings['family_to_group'].get(fam, -1) 
            for fam in true_families
        ]).to(device)
        
        # Group classification loss
        group_loss = F.cross_entropy(group_logits, true_groups)
        
        # Family classification loss per group
        family_loss = 0
        valid_samples = 0
        
        for group_id, group_families in self.group_mappings['group_to_families'].items():
            # Get samples belonging to this group
            group_mask = (true_groups == group_id)
            if not group_mask.any():
                continue
                
            # Get family predictions for this group
            group_logits = family_logits[str(group_id)][group_mask]
            
            # Convert family labels to group-specific indices
            family_to_idx = {fam: idx for idx, fam in enumerate(group_families)}
            true_indices = torch.tensor([
                family_to_idx[fam] for fam in true_families[group_mask]
            ]).to(device)
            
            # Compute loss for this group
            family_loss += F.cross_entropy(group_logits, true_indices) * group_mask.sum()
            valid_samples += group_mask.sum()
            
        if valid_samples > 0:
            family_loss /= valid_samples
            
        return self.alpha * group_loss + (1 - self.alpha) * family_loss
    
def train_hierarchical(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        group_logits, family_logits = model(batch)
        loss = criterion(group_logits, family_logits, batch.family, device)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_hierarchical(model, loader, group_mappings, device):
    model.eval()
    group_correct = 0
    family_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            group_logits, family_logits = model(batch)
            
            # Group accuracy
            pred_groups = group_logits.argmax(dim=1)
            true_groups = torch.tensor([
                group_mappings['family_to_group'].get(fam, -1) 
                for fam in batch.family
            ]).to(device)
            group_correct += (pred_groups == true_groups).sum().item()
            
            # Family accuracy
            for i, (pred_group, true_group) in enumerate(zip(pred_groups, true_groups)):
                if pred_group == true_group:
                    group_families = group_mappings['group_to_families'][pred_group.item()]
                    family_logits_group = family_logits[str(pred_group.item())][i]
                    pred_family_idx = family_logits_group.argmax().item()
                    pred_family = group_families[pred_family_idx]
                    if pred_family == batch.family[i]:
                        family_correct += 1
                        
            total += len(batch.family)
            
    return {
        'group_accuracy': group_correct / total,
        'family_accuracy': family_correct / total
    }

class TemporalMalwareClassifier(HierarchicalMalwareClassifier):
    def __init__(self, behavioral_groups_path: str, embedding_dim: int = 256):
        super().__init__(behavioral_groups_path)
        self.embedding_dim = embedding_dim
        self.family_centroids = {}
        self.temporal_statistics = defaultdict(list)
        self.distance_threshold = None  # Will be set during training
        
    def update_centroids(self, model, loader, device):
        """Update family centroids using current model embeddings."""
        model.eval()
        family_embeddings = defaultdict(list)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                embeddings = model.get_embeddings(batch)  # New method to get embeddings
                
                for emb, family, timestamp in zip(embeddings, batch.family, batch.timestamp):
                    family_embeddings[family].append({
                        'embedding': emb.cpu().numpy(),
                        'timestamp': pd.to_datetime(timestamp)
                    })
        
        # Update centroids with temporal weighting
        for family, embeds in family_embeddings.items():
            # Sort by timestamp
            sorted_embeds = sorted(embeds, key=lambda x: x['timestamp'])
            
            # Apply temporal weighting (more recent samples weighted higher)
            weights = np.exp(np.linspace(-1, 0, len(sorted_embeds)))
            weighted_embeddings = np.vstack([e['embedding'] for e in sorted_embeds])
            weighted_centroid = np.average(weighted_embeddings, weights=weights, axis=0)
            
            self.family_centroids[family] = {
                'centroid': weighted_centroid,
                'last_updated': sorted_embeds[-1]['timestamp'],
                'num_samples': len(sorted_embeds)
            }
    
    def compute_distance_threshold(self, model, loader, device, percentile=95):
        """Compute distance threshold for new family detection."""
        distances = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                embeddings = model.get_embeddings(batch)
                
                for emb, family in zip(embeddings, batch.family):
                    if family in self.family_centroids:
                        centroid = self.family_centroids[family]['centroid']
                        distance = np.linalg.norm(emb.cpu().numpy() - centroid)
                        distances.append(distance)
        
        self.distance_threshold = np.percentile(distances, percentile)
        logger.info(f"Set distance threshold to {self.distance_threshold:.4f}")
        
    def detect_new_families(self, embeddings: torch.Tensor, confidence_threshold: float = 0.9) -> List[bool]:
        """Detect potential new families based on distance to existing centroids."""
        embeddings_np = embeddings.cpu().numpy()
        new_family_flags = []
        
        for embedding in embeddings_np:
            # Compute distances to all centroids
            distances = {
                family: np.linalg.norm(embedding - data['centroid'])
                for family, data in self.family_centroids.items()
            }
            
            min_distance = min(distances.values())
            new_family_flags.append(min_distance > self.distance_threshold)
            
        return new_family_flags
    
    def analyze_temporal_performance(self, results_dict: Dict):
        """Analyze classification performance over time."""
        df = pd.DataFrame(results_dict)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        # Compute rolling metrics
        window_size = '7D'  # 7-day window
        metrics = {
            'known_accuracy': df['correct_known'].rolling(window_size).mean(),
            'new_detection_rate': df['correct_new'].rolling(window_size).mean(),
            'false_positive_rate': df['false_new'].rolling(window_size).mean()
        }
        
        # Analyze family evolution
        family_first_seen = {}
        family_evolution = []
        
        for _, row in df.iterrows():
            if row['predicted_family'] not in family_first_seen:
                family_first_seen[row['predicted_family']] = row['timestamp']
                family_evolution.append({
                    'timestamp': row['timestamp'],
                    'family': row['predicted_family'],
                    'event': 'new_family'
                })
        
        return metrics, family_evolution

class TemporalGNN(torch.nn.Module):
    def __init__(self, num_features, num_groups, num_families, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Base GNN layers
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, embedding_dim)
        
        # Classification heads
        self.group_classifier = torch.nn.Linear(embedding_dim, num_groups)
        self.family_classifiers = torch.nn.ModuleDict()
        
        for group_id in range(num_groups):
            num_families_in_group = len(group_mappings['group_to_families'][group_id])
            self.family_classifiers[str(group_id)] = torch.nn.Linear(embedding_dim, num_families_in_group)
    
    def get_embeddings(self, data):
        """Get graph embeddings before classification."""
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        return global_mean_pool(x, data.batch)
    
    def forward(self, data):
        embeddings = self.get_embeddings(data)
        group_logits = self.group_classifier(embeddings)
        
        family_logits = {}
        for group_id in self.family_classifiers:
            family_logits[group_id] = self.family_classifiers[group_id](embeddings)
        
        return embeddings, group_logits, family_logits

def train_temporal(model, classifier, train_loader, val_loader, optimizer, criterion, 
                  device, num_epochs=100):
    """Training loop with temporal analysis."""
    results = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            embeddings, group_logits, family_logits = model(batch)
            
            # Update centroids periodically
            if epoch % 5 == 0:
                classifier.update_centroids(model, train_loader, device)
                classifier.compute_distance_threshold(model, val_loader, device)
            
            # Detect new families
            new_family_flags = classifier.detect_new_families(embeddings)
            
            # Compute loss only for known families
            known_mask = ~torch.tensor(new_family_flags).to(device)
            if known_mask.any():
                loss = criterion(
                    group_logits[known_mask], 
                    family_logits[known_mask], 
                    batch.family[known_mask], 
                    device
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Record results
            results.extend([{
                'timestamp': ts,
                'true_family': true_fam,
                'predicted_family': pred_fam,
                'is_new': is_new,
                'correct_known': correct_known,
                'correct_new': correct_new,
                'false_new': false_new
            } for ts, true_fam, pred_fam, is_new, correct_known, correct_new, false_new in 
                zip(batch.timestamp, batch.family, get_predictions(group_logits, family_logits),
                    new_family_flags, *get_metrics(batch, new_family_flags))])
        
        # Analyze temporal performance
        if epoch % 10 == 0:
            metrics, evolution = classifier.analyze_temporal_performance(results)
            log_temporal_metrics(metrics, evolution, epoch)
    
    return results


class FamilyDriftAnalyzer:
    """Analyze and track malware family evolution over time."""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.family_trajectories = defaultdict(list)
        self.drift_metrics = defaultdict(dict)
        
    def track_family_drift(self, family: str, embedding: np.ndarray, timestamp: pd.Timestamp):
        """Track a family's position in embedding space over time."""
        self.family_trajectories[family].append({
            'embedding': embedding,
            'timestamp': timestamp
        })
        
        # Keep trajectories sorted by timestamp
        self.family_trajectories[family].sort(key=lambda x: x['timestamp'])
        
        # Update drift metrics if we have enough data
        if len(self.family_trajectories[family]) > 1:
            self._update_drift_metrics(family)
    
    def _update_drift_metrics(self, family: str):
        """Compute drift metrics for a family."""
        trajectory = self.family_trajectories[family]
        
        # Get time-ordered embeddings
        embeddings = np.vstack([t['embedding'] for t in trajectory])
        timestamps = np.array([t['timestamp'] for t in trajectory])
        
        # Compute drift metrics
        self.drift_metrics[family].update({
            'total_drift': self._compute_total_drift(embeddings),
            'drift_velocity': self._compute_drift_velocity(embeddings, timestamps),
            'drift_acceleration': self._compute_drift_acceleration(embeddings, timestamps),
            'stability_periods': self._identify_stability_periods(embeddings, timestamps),
            'major_shifts': self._detect_major_shifts(embeddings, timestamps)
        })
    
    def _compute_total_drift(self, embeddings: np.ndarray) -> float:
        """Compute total drift as path length in embedding space."""
        return np.sum(np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1))
    
    def _compute_drift_velocity(self, embeddings: np.ndarray, 
                              timestamps: np.ndarray) -> np.ndarray:
        """Compute drift velocity over time."""
        time_deltas = np.diff(timestamps).astype('timedelta64[s]').astype(float)
        displacement = embeddings[1:] - embeddings[:-1]
        
        return displacement / time_deltas[:, np.newaxis]
    
    def _compute_drift_acceleration(self, embeddings: np.ndarray, 
                                  timestamps: np.ndarray) -> np.ndarray:
        """Compute drift acceleration over time."""
        velocities = self._compute_drift_velocity(embeddings, timestamps)
        time_deltas = np.diff(timestamps[1:]).astype('timedelta64[s]').astype(float)
        
        return np.diff(velocities, axis=0) / time_deltas[:, np.newaxis]
    
    def _identify_stability_periods(self, embeddings: np.ndarray, 
                                  timestamps: np.ndarray, 
                                  threshold: float = 0.1) -> List[Dict]:
        """Identify periods where family behavior remains stable."""
        velocities = self._compute_drift_velocity(embeddings, timestamps)
        speeds = np.linalg.norm(velocities, axis=1)
        
        stable_periods = []
        current_period = None
        
        for i, (speed, ts) in enumerate(zip(speeds, timestamps[1:])):
            if speed < threshold:
                if current_period is None:
                    current_period = {'start': timestamps[i], 'count': 1}
                current_period['count'] += 1
                current_period['end'] = ts
            elif current_period is not None:
                stable_periods.append(current_period)
                current_period = None
        
        if current_period is not None:
            stable_periods.append(current_period)
            
        return stable_periods
    
    def _detect_major_shifts(self, embeddings: np.ndarray, 
                           timestamps: np.ndarray, 
                           threshold: float = 0.5) -> List[Dict]:
        """Detect major behavioral shifts in family evolution."""
        velocities = self._compute_drift_velocity(embeddings, timestamps)
        accelerations = self._compute_drift_acceleration(embeddings, timestamps)
        
        major_shifts = []
        
        for i, (vel, acc, ts) in enumerate(zip(velocities[1:], accelerations, timestamps[2:])):
            # Detect sudden changes in behavior
            velocity_magnitude = np.linalg.norm(vel)
            acceleration_magnitude = np.linalg.norm(acc)
            
            if velocity_magnitude > threshold and acceleration_magnitude > threshold:
                # Analyze the nature of the shift
                shift_analysis = self._analyze_behavioral_shift(
                    embeddings[i:i+3],  # Look at before, during, and after shift
                    velocities[i:i+2]
                )
                
                major_shifts.append({
                    'timestamp': ts,
                    'magnitude': velocity_magnitude,
                    'acceleration': acceleration_magnitude,
                    'analysis': shift_analysis
                })
        
        return major_shifts
    
    def _analyze_behavioral_shift(self, embeddings: np.ndarray, 
                                velocities: np.ndarray) -> Dict:
        """Analyze the nature of a behavioral shift."""
        # Compute direction of change
        direction = velocities[0] / np.linalg.norm(velocities[0])
        
        # Analyze if the change is temporary or permanent
        temporary = np.dot(velocities[0], velocities[1]) < 0
        
        # Compute behavioral distance
        distance = np.linalg.norm(embeddings[2] - embeddings[0])
        
        return {
            'temporary': temporary,
            'distance': distance,
            'direction': direction
        }
    
    def get_family_evolution_summary(self, family: str) -> Dict:
        """Get comprehensive evolution summary for a family."""
        if family not in self.drift_metrics:
            return None
            
        metrics = self.drift_metrics[family]
        trajectory = self.family_trajectories[family]
        
        total_time = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
        average_velocity = metrics['total_drift'] / total_time.total_seconds()
        
        return {
            'first_seen': trajectory[0]['timestamp'],
            'last_seen': trajectory[-1]['timestamp'],
            'total_drift': metrics['total_drift'],
            'average_velocity': average_velocity,
            'stability_periods': len(metrics['stability_periods']),
            'major_shifts': len(metrics['major_shifts']),
            'evolution_phases': self._identify_evolution_phases(family)
        }
    
    def _identify_evolution_phases(self, family: str) -> List[Dict]:
        """Identify distinct phases in family evolution."""
        metrics = self.drift_metrics[family]
        shifts = metrics['major_shifts']
        
        phases = []
        last_phase_end = self.family_trajectories[family][0]['timestamp']
        
        for shift in shifts:
            phases.append({
                'start': last_phase_end,
                'end': shift['timestamp'],
                'duration': shift['timestamp'] - last_phase_end,
                'stability': self._compute_phase_stability(family, last_phase_end, shift['timestamp'])
            })
            last_phase_end = shift['timestamp']
        
        # Add final phase
        final_timestamp = self.family_trajectories[family][-1]['timestamp']
        phases.append({
            'start': last_phase_end,
            'end': final_timestamp,
            'duration': final_timestamp - last_phase_end,
            'stability': self._compute_phase_stability(family, last_phase_end, final_timestamp)
        })
        
        return phases
    
    def _compute_phase_stability(self, family: str, 
                               start: pd.Timestamp, 
                               end: pd.Timestamp) -> float:
        """Compute stability metric for a specific phase."""
        trajectory = self.family_trajectories[family]
        phase_embeddings = [
            t['embedding'] for t in trajectory 
            if start <= t['timestamp'] <= end
        ]
        
        if len(phase_embeddings) < 2:
            return 1.0
            
        embeddings = np.vstack(phase_embeddings)
        centroid = np.mean(embeddings, axis=0)
        
        # Compute average distance from centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return 1.0 / (1.0 + np.mean(distances))
    
def train_with_drift_analysis(model, classifier, train_loader, optimizer, drift_analyzer):
    for batch in train_loader:
        # Your existing training code here
        embeddings, group_logits, family_logits = model(batch)
        
        # Track drift for each sample
        for emb, family, timestamp in zip(embeddings, batch.family, batch.timestamp):
            drift_analyzer.track_family_drift(
                family, 
                emb.detach().cpu().numpy(),
                pd.to_datetime(timestamp)
            )
        
        # Periodically analyze drift patterns
        if batch_idx % 100 == 0:
            for family in drift_analyzer.family_trajectories:
                summary = drift_analyzer.get_family_evolution_summary(family)
                if summary['major_shifts']:
                    logger.info(f"Family {family} evolution summary:")
                    logger.info(f"Total drift: {summary['total_drift']:.4f}")
                    logger.info(f"Major behavioral shifts: {len(summary['major_shifts'])}")
                    for phase in summary['evolution_phases']:
                        logger.info(f"Phase: {phase['start']} to {phase['end']}")
                        logger.info(f"Stability: {phase['stability']:.4f}")

import argparse
import logging
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'malware_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Malware Family Evolution Analysis')
    parser.add_argument('--batch_dir', type=str, default='bodmas_batches',
                       help='Directory containing processed batches')
    parser.add_argument('--output_dir', type=str, default='evolution_analysis',
                       help='Directory for saving results')
    parser.add_argument('--behavioral_groups', type=str, default='behavioral_analysis/behavioral_groups.json',
                       help='Path to behavioral groups JSON')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Dimension of graph embeddings')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing components...")
    
    # Initialize classifier and drift analyzer
    classifier = TemporalMalwareClassifier(
        behavioral_groups_path=args.behavioral_groups,
        embedding_dim=args.embedding_dim
    )
    
    drift_analyzer = FamilyDriftAnalyzer(embedding_dim=args.embedding_dim)

    # Load data
    logger.info("Loading data...")
    train_loader = load_temporal_data(f"{args.batch_dir}/train")
    val_loader = load_temporal_data(f"{args.batch_dir}/val")
    test_loader = load_temporal_data(f"{args.batch_dir}/test")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    num_features = next(iter(train_loader)).x.size(1)  # Get feature dimension from data
    model = TemporalGNN(
        num_features=num_features,
        num_groups=len(classifier.group_mappings['group_to_families']),
        num_families=len(classifier.family_centroids),
        embedding_dim=args.embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = HierarchicalLoss(classifier.group_mappings)

    # Training loop with drift analysis
    logger.info("Starting training with drift analysis...")
    best_val_acc = 0
    best_model_path = output_dir / 'best_model.pt'

    for epoch in range(100):  # 100 epochs
        # Train
        train_results = train_temporal(
            model=model,
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            drift_analyzer=drift_analyzer
        )

        # Validate
        val_metrics = evaluate_temporal(model, classifier, val_loader, device)
        
        # Log metrics
        logger.info(f"Epoch {epoch}:")
        logger.info(f"Known Family Accuracy: {val_metrics['known_accuracy']:.4f}")
        logger.info(f"New Family Detection Rate: {val_metrics['new_detection_rate']:.4f}")
        logger.info(f"False Positive Rate: {val_metrics['false_positive_rate']:.4f}")

        # Save best model
        if val_metrics['known_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['known_accuracy']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': best_val_acc
            }, best_model_path)

        # Analyze and save drift patterns
        if epoch % 10 == 0:  # Every 10 epochs
            save_drift_analysis(drift_analyzer, output_dir / f'drift_analysis_epoch_{epoch}')

    # Final evaluation on test set
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_temporal(model, classifier, test_loader, device)
    logger.info("\nFinal Test Results:")
    logger.info(f"Known Family Accuracy: {test_metrics['known_accuracy']:.4f}")
    logger.info(f"New Family Detection Rate: {test_metrics['new_detection_rate']:.4f}")
    logger.info(f"False Positive Rate: {test_metrics['false_positive_rate']:.4f}")

    # Save final evolution analysis
    save_final_analysis(
        classifier=classifier,
        drift_analyzer=drift_analyzer,
        test_metrics=test_metrics,
        output_dir=output_dir
    )

def save_drift_analysis(drift_analyzer, output_path):
    """Save drift analysis results and visualizations."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summaries for each family
    summaries = {}
    for family in drift_analyzer.family_trajectories:
        summaries[family] = drift_analyzer.get_family_evolution_summary(family)

    with open(output_path / 'evolution_summaries.json', 'w') as f:
        json.dump(summaries, f, indent=2, default=str)

    # Create visualizations
    plot_drift_patterns(drift_analyzer, output_path)

def plot_drift_patterns(drift_analyzer, output_path):
    """Create visualizations of drift patterns."""
    # Plot total drift over time
    plt.figure(figsize=(12, 8))
    for family in drift_analyzer.family_trajectories:
        timestamps = [t['timestamp'] for t in drift_analyzer.family_trajectories[family]]
        drifts = np.cumsum([np.linalg.norm(t['embedding']) 
                           for t in drift_analyzer.family_trajectories[family]])
        plt.plot(timestamps, drifts, label=family)
    
    plt.xlabel('Time')
    plt.ylabel('Cumulative Drift')
    plt.title('Family Evolution Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path / 'drift_patterns.png')
    plt.close()

def save_final_analysis(classifier, drift_analyzer, test_metrics, output_dir):
    """Save final analysis results."""
    results = {
        'test_metrics': test_metrics,
        'family_evolution': {
            family: drift_analyzer.get_family_evolution_summary(family)
            for family in drift_analyzer.family_trajectories
        },
        'behavioral_groups': classifier.group_mappings
    }

    with open(output_dir / 'final_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()