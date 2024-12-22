import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import json
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import List, Dict, Tuple
import os
import glob
# hierarchical loss
import sys 
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, Batch



class HierarchicalLoss(torch.nn.Module):
    def __init__(self, family_to_group, alpha=0.3):
        """
        Initialize hierarchical loss.
        
        Args:
            family_to_group (dict): Mapping from family names to group IDs
            alpha (float): Weight between group loss and family loss (default: 0.3)
        """
        super().__init__()
        self.family_to_group = family_to_group
        self.alpha = alpha
        
        # Create reverse mapping (group to families)
        self.group_to_families = defaultdict(list)
        for family, group in family_to_group.items():
            self.group_to_families[group].append(family)
            
        # Create family to index mappings for each group
        self.family_to_idx = {}
        for group_id, families in self.group_to_families.items():
            self.family_to_idx[group_id] = {
                fam: idx for idx, fam in enumerate(sorted(families))
            }
    
    def forward(self, embeddings, group_logits, family_logits, true_families, device):
        """
        Compute hierarchical loss.
        
        Args:
            embeddings: Graph embeddings from the model
            group_logits: Predicted group probabilities
            family_logits: Dictionary of family probabilities per group
            true_families: List of true family names
            device: Device to put tensors on
        """
        # Convert true_families to list if it's not already
        if not isinstance(true_families, list):
            true_families = [true_families]
            
        # Convert family labels to group labels with proper error handling
        true_groups = []
        for fam in true_families:
            group = self.family_to_group.get(fam, -1)
            true_groups.append(group)
        
        true_groups = torch.tensor(true_groups, dtype=torch.long).to(device)
        
        # Group classification loss
        group_loss = F.cross_entropy(
            group_logits, 
            true_groups,
            label_smoothing=0.1,
            ignore_index=-1  # Ignore any invalid groups
        )
        
        # Family classification loss
        family_loss = 0
        valid_samples = 0
        
        for group_id in self.group_to_families:
            # Get samples belonging to this group
            group_mask = (true_groups == group_id)
            if not group_mask.any():
                continue
                
            # Get family predictions for this group
            group_logits_subset = family_logits[str(group_id)][group_mask]
            
            # Get true family indices for this group
            true_indices = []
            for fam in true_families:
                if self.family_to_group.get(fam) == group_id:
                    idx = self.family_to_idx[group_id].get(fam, 0)
                    true_indices.append(idx)
            
            if not true_indices:
                continue
                
            true_indices = torch.tensor(true_indices, dtype=torch.long).to(device)
            
            # Weight loss by group size
            group_weight = len(self.group_to_families[group_id]) / len(self.family_to_group)
            family_loss += group_weight * F.cross_entropy(
                group_logits_subset,
                true_indices,
                label_smoothing=0.1
            )
            valid_samples += 1
        
        if valid_samples > 0:
            family_loss /= valid_samples
        
        # Combine losses with alpha weighting
        total_loss = self.alpha * group_loss + (1 - self.alpha) * family_loss
        
        # Add L2 regularization
        l2_reg = 0.001 * torch.norm(embeddings, p=2, dim=1).mean()
        total_loss += l2_reg
        
        return total_loss
    
def save_analysis_results(output_dir: Path, classifier,
                            drift_analyzer, final_metrics: dict):
    """Save analysis results to output directory."""
    # Save family centroids
    with open(output_dir / 'family_centroids.json', 'w') as f:
        json.dump(classifier.family_centroids, f)
    
    # Save drift metrics
    with open(output_dir / 'drift_metrics.json', 'w') as f:
        json.dump(drift_analyzer.drift_metrics, f)
    
    # Save final metrics
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f)
    


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'malware_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FamilyDriftAnalyzer:
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.family_trajectories = defaultdict(list)
        self.drift_metrics = defaultdict(dict)
    
    @torch.no_grad()  # Ensure no gradients are tracked for the entire method
    def track_family_drift(self, family: str, embedding: torch.Tensor, timestamp: pd.Timestamp):
        """Track a family's position in embedding space over time."""
        # Safely convert embedding to numpy
        if torch.is_tensor(embedding):
            embedding_np = embedding.cpu().numpy()
        else:
            embedding_np = embedding
            
        self.family_trajectories[family].append({
            'embedding': embedding_np,
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
            velocity_magnitude = np.linalg.norm(vel)
            acceleration_magnitude = np.linalg.norm(acc)
            
            if (velocity_magnitude is not None and acceleration_magnitude is not None and velocity_magnitude > threshold and acceleration_magnitude > threshold):
                shift_analysis = self._analyze_behavioral_shift(
                    embeddings[i:i+3],
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
        direction = velocities[0] / np.linalg.norm(velocities[0])
        temporary = np.dot(velocities[0], velocities[1]) < 0
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
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return 1.0 / (1.0 + np.mean(distances))



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
            velocity_magnitude = np.linalg.norm(vel)
            acceleration_magnitude = np.linalg.norm(acc)
            
            if (velocity_magnitude is not None and acceleration_magnitude is not None and velocity_magnitude > threshold and acceleration_magnitude > threshold):
                shift_analysis = self._analyze_behavioral_shift(
                    embeddings[i:i+3],
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
        direction = velocities[0] / np.linalg.norm(velocities[0])
        temporary = np.dot(velocities[0], velocities[1]) < 0
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
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return 1.0 / (1.0 + np.mean(distances))


# class TemporalMalwareClassifier:
#     def __init__(self, behavioral_groups_path: str, embedding_dim: int = 256):
#         self.group_mappings = self._load_groups(behavioral_groups_path)
#         self.embedding_dim = embedding_dim
#         self.family_centroids = {}
#         self.temporal_statistics = defaultdict(list)
#         self.distance_threshold = None


class TemporalMalwareClassifier:
    def __init__(self, behavioral_groups_path: str, embedding_dim: int = 256):
        try:
            self.group_mappings = self._load_groups(behavioral_groups_path)
        except Exception as e:
            logger.error(f"Error loading behavioral groups from {behavioral_groups_path}: {str(e)}")
            raise
        self.embedding_dim = embedding_dim
        self.family_centroids = {}
        self.temporal_statistics = defaultdict(list)
        self.distance_threshold = None
        # Add monitoring for unknown families
        self.unknown_families_count = 0
        self.unknown_families_set = set()
    
    def _load_groups(self, path: str) -> dict:
        """Load behavioral groups from JSON file with support for unknown families.
        
        Args:
            path (str): Path to JSON file containing group mappings in format:
                       {"group_id": ["family1", "family2", ...], ...}
            
        Returns:
            dict: Dictionary containing:
                - family_to_group: Maps family names to group IDs
                - group_to_families: Maps group IDs to lists of family names
        """
        try:
            with open(path, 'r') as f:
                group_data = json.load(f)
                
            family_to_group = {}
            group_to_families = {}
            
            # Initialize special group for unknown families
            unknown_group_id = -999  # Use a special ID for unknown group
            group_to_families[unknown_group_id] = []
            
            # Load known groups
            for group_id, families in group_data.items():
                group_id = int(group_id)  # Convert string group ID to int
                group_to_families[group_id] = families
                for family in families:
                    family_to_group[family] = group_id
            
            logger.info(f"Initially loaded {len(family_to_group)} families in {len(group_to_families)-1} groups")
            
            return {
                'family_to_group': family_to_group,
                'group_to_families': group_to_families
            }
            
        except Exception as e:
            logger.error(f"Error loading groups from {path}: {str(e)}")
            raise
            
    def get_or_add_family_group(self, family_name: str) -> int:
        """Get group ID for a family, adding to unknown group if not found."""
        if family_name not in self.group_mappings['family_to_group']:
            self.group_mappings['family_to_group'][family_name] = -999
            self.group_mappings['group_to_families'][-999].append(family_name)
            self.unknown_families_count += 1
            self.unknown_families_set.add(family_name)
            logger.warning(
                f"Added unknown family '{family_name}' to unknown group "
                f"(Total unknown families: {len(self.unknown_families_set)})"
            )
        return self.group_mappings['family_to_group'][family_name]

    def detect_new_families(self, embeddings: torch.Tensor) -> List[bool]:
        """Detect potential new families based on distance to existing centroids."""
        with torch.no_grad():  # Ensure no gradients are tracked
            # Move to CPU and convert to numpy
            embeddings_np = embeddings.detach().cpu().numpy()
            new_family_flags = []
            
            for embedding in embeddings_np:
                distances = {
                    family: np.linalg.norm(embedding - data['centroid'])
                    for family, data in self.family_centroids.items()
                }
                
                min_distance = min(distances.values()) if distances else float('inf')
                new_family_flags.append(min_distance is not None and self.distance_threshold is not None and min_distance > self.distance_threshold)
                
            return new_family_flags
    
    def update_centroids(self, model, loader, device):
        """Update family centroids using current model embeddings."""
        model.eval()  # Set model to evaluation mode
        family_embeddings = defaultdict(list)
        
        with torch.no_grad():  # Ensure no gradients are tracked
            for batch in loader:
                batch = batch.to(device)
                embeddings = model.get_embeddings(batch)
                
                # Safely convert embeddings to numpy
                embeddings_np = embeddings.cpu().numpy()
                
                for emb, family, timestamp in zip(embeddings_np, batch.family, batch.timestamp):
                    family_embeddings[family].append({
                        'embedding': emb,
                        'timestamp': pd.to_datetime(timestamp)
                    })
        
        # Process the collected embeddings
        for family, embeds in family_embeddings.items():
            sorted_embeds = sorted(embeds, key=lambda x: x['timestamp'])
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
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():  # Ensure no gradients are tracked
            for batch in loader:
                batch = batch.to(device)
                embeddings = model.get_embeddings(batch)
                
                # Safely convert embeddings to numpy
                embeddings_np = embeddings.cpu().numpy()
                
                for emb, family in zip(embeddings_np, batch.family):
                    if family in self.family_centroids:
                        centroid = self.family_centroids[family]['centroid']
                        distance = np.linalg.norm(emb - centroid)
                        distances.append(distance)
        
        if distances:
            self.distance_threshold = np.percentile(distances, percentile)
            logger.info(f"Set distance threshold to {self.distance_threshold:.4f}")
        else:
            logger.warning("No distances computed for threshold calculation")

            
    # def update_centroids(self, model, loader, device):
    #     """Update family centroids using current model embeddings."""
    #     model.eval()  # Set model to evaluation mode
    #     family_embeddings = defaultdict(list)
        
    #     with torch.no_grad():  # Ensure no gradients are tracked
    #         for batch in loader:
    #             batch = batch.to(device)
    #             embeddings = model.get_embeddings(batch)
                
    #             # Safely convert embeddings to numpy
    #             embeddings_np = embeddings.cpu().numpy()
                
    #             for emb, family, timestamp in zip(embeddings_np, batch.family, batch.timestamp):
    #                 family_embeddings[family].append({
    #                     'embedding': emb,
    #                     'timestamp': pd.to_datetime(timestamp)
    #                 })
        
    #     # Process the collected embeddings
    #     for family, embeds in family_embeddings.items():
    #         sorted_embeds = sorted(embeds, key=lambda x: x['timestamp'])
    #         weights = np.exp(np.linspace(-1, 0, len(sorted_embeds)))
    #         weighted_embeddings = np.vstack([e['embedding'] for e in sorted_embeds])
    #         weighted_centroid = np.average(weighted_embeddings, weights=weights, axis=0)
            
    #         self.family_centroids[family] = {
    #             'centroid': weighted_centroid,
    #             'last_updated': sorted_embeds[-1]['timestamp'],
    #             'num_samples': len(sorted_embeds)
    #         }
    
    def compute_distance_threshold(self, model, loader, device, percentile=95):
        """Compute distance threshold for new family detection."""
        distances = []
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():  # Ensure no gradients are tracked
            for batch in loader:
                batch = batch.to(device)
                embeddings = model.get_embeddings(batch)
                
                # Safely convert embeddings to numpy
                embeddings_np = embeddings.cpu().numpy()
                
                for emb, family in zip(embeddings_np, batch.family):
                    if family in self.family_centroids:
                        centroid = self.family_centroids[family]['centroid']
                        distance = np.linalg.norm(emb - centroid)
                        distances.append(distance)
        
        if distances:
            self.distance_threshold = np.percentile(distances, percentile)
            logger.info(f"Set distance threshold to {self.distance_threshold:.4f}")
        else:
            logger.warning("No distances computed for threshold calculation")



def evaluate_predictions(group_logits, family_logits, true_families, 
                       new_family_flags, group_mappings, classifier=None):
    """Evaluate predictions for a batch with unknown family handling."""
    pred_groups = group_logits.argmax(dim=1)
    metrics = defaultdict(list)
    
    for i, (pred_group, true_family, is_new) in enumerate(
            zip(pred_groups, true_families, new_family_flags)):
        
        # Use the classifier's unknown family handling if available
        if classifier and hasattr(classifier, 'get_or_add_family_group'):
            true_group = classifier.get_or_add_family_group(true_family)
        else:
            # Fallback to default behavior with unknown group
            true_group = group_mappings['family_to_group'].get(true_family, -999)
        
        # Group accuracy
        group_correct = (pred_group.item() == true_group)
        metrics['group_accuracy'].append(group_correct)
        
        # Family prediction accuracy (only for known families)
        if not is_new and true_group != -999:
            family_logits_group = family_logits[str(pred_group.item())][i]
            pred_family_idx = family_logits_group.argmax().item()
            pred_family = group_mappings['group_to_families'][pred_group.item()][pred_family_idx]
            metrics['accuracy'].append(pred_family == true_family)
        
        # Unknown family detection
        is_unknown = true_group == -999
        metrics['unknown_family_detection'].append(is_new == is_unknown)
    
    return {k: np.mean(v) for k, v in metrics.items()}

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class HierarchicalMalwareGNN(torch.nn.Module):
    def __init__(self, num_features, num_groups=16, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Base Graph Neural Network layers
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, embedding_dim)
        
        # Group classification head
        self.group_classifier = torch.nn.Linear(embedding_dim, num_groups)
        
        # Separate classifiers for each behavioral group
        self.family_classifiers = torch.nn.ModuleDict()
    
    def add_family_classifier(self, group_id: str, num_families: int):
        """Dynamically add a family classifier for a behavioral group"""
        self.family_classifiers[str(group_id)] = torch.nn.Linear(
            self.embedding_dim, num_families
        )
    
    def get_embeddings(self, data):
        """Extract graph embeddings"""
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Move graph data to correct device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        
        # Forward pass through GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global pooling
        embeddings = global_mean_pool(x, batch)
        return embeddings
    
    def forward(self, data):
        device = next(self.parameters()).device
        
        # Get embeddings
        embeddings = self.get_embeddings(data)
        
        # Group classification
        group_logits = self.group_classifier(embeddings)
        
        # Family classification for each group
        family_logits = {}
        for group_id in self.family_classifiers:
            family_logits[group_id] = self.family_classifiers[group_id](embeddings)
            
        return embeddings, group_logits, family_logits


def evaluate(model, split_files, family_to_group, device, criterion, batch_size=32):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_file in split_files:
            # Load batch data
            batch_loader = load_batch(batch_file, family_to_group, batch_size=batch_size)
            if not batch_loader:
                continue
            
            for batch in batch_loader:
                try:
                    # Move batch to device
                    batch = batch.to(device)
                    
                    # Forward pass
                    embeddings, group_logits, family_logits = model(batch)
                    
                    # Get predictions
                    pred_groups = group_logits.argmax(dim=1)
                    true_groups = torch.tensor([
                        family_to_group.get(fam, -1) for fam in batch.family
                    ]).to(device)
                    
                    # Compute metrics
                    correct += (pred_groups == true_groups).sum().item()
                    total += len(true_groups)
                    
                    # Compute loss
                    loss = criterion(embeddings, group_logits, family_logits, batch.family, device)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue
    
    accuracy = correct / max(1, total)
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, accuracy



def evaluate(model, split_files, family_to_group, device, criterion, batch_size=32):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_file in split_files:
            # Load batch data
            batch_loader = load_batch(batch_file, family_to_group, batch_size=batch_size)
            if not batch_loader:
                continue
            
            for batch in batch_loader:
                try:
                    # Move batch to device
                    batch = batch.to(device)
                    
                    # Forward pass
                    embeddings, group_logits, family_logits = model(batch)
                    
                    # Get predictions
                    pred_groups = group_logits.argmax(dim=1)
                    true_groups = torch.tensor([
                        family_to_group.get(fam, -1) for fam in batch.family
                    ]).to(device)
                    
                    # Compute metrics
                    correct += (pred_groups == true_groups).sum().item()
                    total += len(true_groups)
                    
                    # Compute loss
                    loss = criterion(embeddings, group_logits, family_logits, batch.family, device)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue
    
    accuracy = correct / max(1, total)
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, accuracy

                   
def load_batch(batch_file, family_to_group, batch_size=32):
    """Load and preprocess a single batch file with robust error handling.
    
    Args:
        batch_file (str): Path to the batch file
        family_to_group (dict): Mapping from family names to group IDs
        batch_size (int): Size of batches to create
        
    Returns:
        DataLoader or None: DataLoader containing processed graphs, or None if processing fails
    """
    try:
        if not os.path.exists(batch_file):
            logger.warning(f"Batch file not found: {batch_file}")
            return None
            
        # Load batch from file
        batch_data = torch.load(batch_file)
        #print("Batch families:", [getattr(g, 'family', 'none') for g in batch_data])
        if not batch_data:
            logger.warning(f"Empty batch file: {batch_file}")
            return None
            
        processed = []
        
        for graph in batch_data:
            try:
                # Verify it's a PyG Data object
                if not isinstance(graph, Data):
                    logger.error(f"Graph is not a PyG Data object: {type(graph)}")
                    continue
                
                # Process family label
                family = getattr(graph, 'family', 'none')
                if not family or family == '':
                    family = 'none'
                
                # Get group ID safely with debug logging
                group = family_to_group.get(family, -1)
                logger.debug(f"Family: {family}, Group: {group}")
                
                # Convert group to tensor explicitly and ensure it's properly shaped
                graph.group = torch.tensor(group, dtype=torch.long)
                graph.y = torch.tensor(group, dtype=torch.long)  # Add y for compatibility
                graph.family = family  # Keep original family name
                
                # Debug log tensor shapes
                logger.debug(f"Graph tensor shapes - x: {graph.x.shape}, edge_index: {graph.edge_index.shape}")
                
                # Handle edge attributes safely
                if graph.edge_index.size(1) == 0:
                    graph.edge_attr = torch.zeros((0, 1))
                else:
                    graph.edge_attr = torch.ones((graph.edge_index.size(1), 1))
                
                # Verify tensor dimensions
                if graph.x.dim() != 2:
                    logger.error(f"Unexpected x dimensions: {graph.x.shape}")
                    continue
                    
                if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
                    logger.error(f"Unexpected edge_index dimensions: {graph.edge_index.shape}")
                    continue
                
                # Verify edge indices are within bounds
                if graph.edge_index.size(1) > 0:
                    max_idx = graph.edge_index.max().item()
                    if max_idx >= graph.x.size(0):
                        logger.error(f"Edge indices out of bounds. Max index: {max_idx}, num nodes: {graph.x.size(0)}")
                        continue
                
                processed.append(graph)
                
            except Exception as e:
                logger.error(f"Error processing graph: {str(e)}")
                continue
                
        if not processed:
            logger.warning(f"No valid graphs found in {batch_file}")
            return None
            
        # Create DataLoader with additional checks
        try:
            loader = DataLoader(
                processed, 
                batch_size=min(batch_size, len(processed)),
                shuffle=False
            )
            
            # Verify first batch to ensure proper formatting
            sample_batch = next(iter(loader))
            required_batch_attrs = ['x', 'edge_index', 'batch']
            missing_batch_attrs = [
                attr for attr in required_batch_attrs 
                if not hasattr(sample_batch, attr)
            ]
            
            if missing_batch_attrs:
                logger.error(f"Batch missing required attributes: {missing_batch_attrs}")
                return None
                
            # Reset loader iterator
            loader = DataLoader(
                processed, 
                batch_size=min(batch_size, len(processed)),
                shuffle=False
            )
            
            return loader
            
        except Exception as e:
            logger.error(f"Error creating DataLoader: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error loading batch {batch_file}: {str(e)}")
        return None

def validate_model(model, val_loader, criterion, classifier, device):
    """Validate model and return accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Get valid groups (excluding unknown group -999)
    valid_groups = set(classifier.group_mappings['family_to_group'].values())
    valid_groups.discard(-999)  # Remove unknown group if present
    
    with torch.no_grad():
        for batch_file in val_loader:
            batch_loader = load_batch(batch_file, classifier.group_mappings['family_to_group'])
            if not batch_loader:
                continue
                
            for batch in batch_loader:
                batch = batch.to(device)
                
                # Forward pass
                embeddings, group_logits, family_logits = model(batch)
                
                # Zero out logits for invalid groups to prevent predicting them
                invalid_mask = torch.ones_like(group_logits, dtype=torch.bool)
                for i in valid_groups:
                    invalid_mask[:, i] = False
                group_logits[invalid_mask] = float('-inf')
                
                # Get predictions
                pred_groups = group_logits.argmax(dim=1)
                
                # Get true labels
                true_groups = torch.tensor([
                    classifier.group_mappings['family_to_group'].get(fam, -999) 
                    for fam in batch.family
                ]).to(device)
                
                # Only count accuracy for known groups
                mask = true_groups != -999
                if mask.any():
                    correct += (pred_groups[mask] == true_groups[mask]).sum().item()
                    total += mask.sum().item()
                
                # Calculate loss
                loss = criterion(embeddings, group_logits, family_logits,
                               batch.family, device)
                total_loss += loss.item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    
    logger.info(f"\nValidation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return {'loss': avg_loss, 'accuracy': accuracy}

def train_temporal(model, classifier, train_loader, val_loader, optimizer, criterion, 
                  device, drift_analyzer, num_epochs=10):
    """Training loop with temporal analysis."""
    best_val_loss = float('inf')
    results = []
    
    # Ensure model is on correct device
    model = model.to(device)
    
    # Convert loaders to lists if they aren't already
    train_files = train_loader if isinstance(train_loader, (list, tuple)) else [train_loader]
    val_files = val_loader if isinstance(val_loader, (list, tuple)) else [val_loader]
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Training
        for batch_file in train_files:
            # Load batch data with new robust loader
            batch_loader = load_batch(
                batch_file, 
                classifier.group_mappings['family_to_group']
            )
            
            if not batch_loader:
                logger.warning(f"Skipping invalid batch file: {batch_file}")
                continue
            
            for batch_data in batch_loader:
                try:
                    # Move batch to device
                    batch_data = batch_data.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    embeddings, group_logits, family_logits = model(batch_data)
                    
                    # Ensure family data is properly formatted for loss computation
                    batch_families = batch_data.family if isinstance(batch_data.family, list) else [batch_data.family]
                    
                    # Debug information
                    logger.debug(f"Batch shapes - embeddings: {embeddings.shape}, group_logits: {group_logits.shape}")
                    logger.debug(f"Number of families in batch: {len(batch_families)}")
                    
                    # Compute loss with proper tensor handling
                    loss = criterion(
                        embeddings, 
                        group_logits, 
                        family_logits, 
                        batch_families,
                        device
                    )
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue
        
        if num_batches == 0:
            logger.error("No valid batches processed in epoch")
            continue
            
        avg_train_loss = total_loss / num_batches
        
        # Validation
        val_metrics = validate_model(
            model, val_files, criterion, classifier, device
        )
        
        # Log progress
        logger.info(f"Epoch {epoch}:")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Save best model
        if val_metrics.get('val_loss') is not None and (best_val_loss is None or val_metrics['val_loss'] < best_val_loss):
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_model.pt')
        
        results.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_metrics': val_metrics
        })
    
    return results

def validate_model(model, val_loader, criterion, classifier, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch_file in val_loader:
            batch_loader = load_batch(
                batch_file,
                classifier.group_mappings['family_to_group'],
                #device=device
            )
            if not batch_loader:
                continue
                
            for batch in batch_loader:
                batch = batch.to(device)
                
                # Forward pass
                embeddings, group_logits, family_logits = model(batch)
                
                # Detect new families
                new_family_flags = classifier.detect_new_families(embeddings)
                
                # Compute loss
                loss = criterion(embeddings, group_logits, family_logits,
                               batch.family, device)
                total_loss += loss.item()
                num_batches += 1
                
                # Compute batch metrics
                batch_metrics = evaluate_predictions(
                    group_logits, family_logits,
                    batch.family, new_family_flags,
                    classifier.group_mappings
                )
                
                for k, v in batch_metrics.items():
                    metrics[k].append(v)
    
    # Compute average metrics
    avg_metrics = {
        'val_loss': total_loss / max(1, num_batches)
    }
    avg_metrics.update({
        k: np.mean(v) for k, v in metrics.items()
    })
    
    return avg_metrics

def prepare_data(base_dir='bodmas_batches'):
    """Prepare datasets with temporal ordering."""
    split_files = defaultdict(list)
    file_timestamps = {}
    
    logger.info("Starting data preparation...")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        # Collect all batch files
        batch_files = glob.glob(os.path.join(split_dir, 'batch_*.pt'))
        
        # Add each file to the appropriate split
        for file_path in batch_files:
            try:
                # Just get the first sample's timestamp without loading entire batch
                batch = torch.load(file_path)
                if batch and len(batch) > 0:
                    file_timestamps[file_path] = getattr(batch[0], 'timestamp', None)
                    split_files[split].append(file_path)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
    
    # Sort files by timestamp within each split
    for split in split_files:
        split_files[split].sort(key=lambda x: file_timestamps.get(x, pd.Timestamp.min))
    
    return dict(split_files), file_timestamps

def evaluate(model, split_files, family_to_group, device, criterion, batch_size=32):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_file in split_files:
            batch_loader = load_batch(batch_file, family_to_group, batch_size=batch_size)
            if not batch_loader:
                continue
                
            for batch in batch_loader:
                # Forward pass
                embeddings, group_logits, family_logits = model(batch)
                
                # Get predictions
                pred_groups = group_logits.argmax(dim=1)
                true_groups = torch.tensor([
                    family_to_group.get(fam, -1) for fam in batch.family
                ]).to(device)
                
                # Compute metrics
                correct += (pred_groups == true_groups).sum().item()
                total += len(true_groups)
                
                # Compute loss
                loss = criterion(embeddings, group_logits, family_logits, batch.family, device)
                total_loss += loss.item()
                num_batches += 1
    
    accuracy = correct / max(1, total)
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, accuracy

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Malware Family Evolution Analysis')
    parser.add_argument('--batch_dir', type=str, default='bodmas_batches',
                       help='Directory containing processed batches')
    parser.add_argument('--behavioral_groups', type=str, required=True,
                       help='Path to behavioral groups JSON')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Dimension of graph embeddings')
    parser.add_argument('--output_dir', type=str, default='evolution_analysis',
                       help='Directory for saving results')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    args = parser.parse_args()

    try:
        # Set up output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up device
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")

        # Initialize temporal components
        logger.info("Initializing temporal components...")
        classifier = TemporalMalwareClassifier(
            behavioral_groups_path=args.behavioral_groups,
            embedding_dim=args.embedding_dim
        )
        drift_analyzer = FamilyDriftAnalyzer(embedding_dim=args.embedding_dim)

        # Prepare data
        logger.info("Preparing data...")
        split_files, file_timestamps = prepare_data(args.batch_dir)
        
        if not any(split_files.values()):
            logger.error("No data found!")
            return

        # Get feature dimension from first batch
        try:
            first_batch = torch.load(split_files['train'][0])
            num_features = first_batch[0].x.size(1)
            logger.info(f"Number of features: {num_features}")
        except Exception as e:
            logger.error(f"Error loading first batch: {str(e)}")
            return

        # Initialize model
        logger.info("Initializing model...")
        model = HierarchicalMalwareGNN(
            num_features=num_features,
            num_groups=len(set(classifier.group_mappings['family_to_group'].values())),
            embedding_dim=args.embedding_dim
        )

        # Add family classifiers for each group
        for group_id, families in classifier.group_mappings['group_to_families'].items():
            model.add_family_classifier(str(group_id), len(families))

        # Move model to device
        model = model.to(device)
        logger.info(f"Model moved to {device}")

        # Initialize optimizer and criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = HierarchicalLoss(classifier.group_mappings['family_to_group'])

        # Training with temporal analysis
        logger.info("Starting training with temporal analysis...")
        # Get file lists for each split
        train_files = split_files['train']
        val_files = split_files['val']
        test_files = split_files['test']

        logger.info(f"Number of training files: {len(train_files)}")
        logger.info(f"Number of validation files: {len(val_files)}")
        logger.info(f"Number of test files: {len(test_files)}")

        results = train_temporal(
            model=model,
            classifier=classifier,
            train_loader=train_files,
            val_loader=val_files,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            drift_analyzer=drift_analyzer,
            num_epochs=args.num_epochs
        )

        # Load best model for final evaluation
        logger.info("Loading best model for final evaluation...")
        try:
            checkpoint = torch.load('best_model.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            return
        
        # Final test evaluation
        logger.info("Running final evaluation...")
        test_loss, test_acc = evaluate(
            model=model,
            split_files=split_files['test'],
            family_to_group=classifier.group_mappings['family_to_group'],
            device=device,
            criterion=criterion,
            batch_size=args.batch_size
        )
        
        # Save final results
        logger.info("Saving analysis results...")
        final_metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'training_history': results
        }
        
        try:
            save_analysis_results(output_dir, classifier, drift_analyzer, final_metrics)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
        
        # Log final results
        logger.info("\nFinal Test Results:")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)