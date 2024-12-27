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
    


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'malware_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



def save_analysis_results(output_dir: Path, classifier, final_metrics: dict):
    """Save analysis results to output directory with proper handling of numpy arrays."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.float32):
            return float(obj)
        return obj
    
    # Save family centroids
    centroids_dict = {}
    for family, data in classifier.family_centroids.items():
        centroids_dict[family] = {
            'centroid': convert_to_serializable(data['centroid']),
            'last_updated': data['last_updated'].isoformat(),
            'num_samples': data['num_samples']
        }
    
    with open(output_dir / 'family_centroids.json', 'w') as f:
        json.dump(centroids_dict, f, indent=2)
    
    # Save drift metrics
    # drift_metrics_serializable = convert_to_serializable(drift_analyzer.drift_metrics)
    # with open(output_dir / 'drift_metrics.json', 'w') as f:
    #     json.dump(drift_metrics_serializable, f, indent=2)
    
    # Save final metrics
    final_metrics_serializable = convert_to_serializable(final_metrics)
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics_serializable, f, indent=2)

# class FamilyDriftAnalyzer:
#     def __init__(self, embedding_dim: int = 256):
#         self.embedding_dim = embedding_dim
#         self.family_trajectories = defaultdict(list)
#         self.drift_metrics = defaultdict(dict)
    
#     @torch.no_grad()  # Ensure no gradients are tracked for the entire method
#     def track_family_drift(self, family: str, embedding: torch.Tensor, timestamp: pd.Timestamp):
#         """Track a family's position in embedding space over time."""
#         # Safely convert embedding to numpy
#         if torch.is_tensor(embedding):
#             embedding_np = embedding.cpu().numpy()
#         else:
#             embedding_np = embedding
            
#         self.family_trajectories[family].append({
#             'embedding': embedding_np,
#             'timestamp': timestamp
#         })
        
#         # Keep trajectories sorted by timestamp
#         self.family_trajectories[family].sort(key=lambda x: x['timestamp'])
        
#         # Update drift metrics if we have enough data
#         if len(self.family_trajectories[family]) > 1:
#             self._update_drift_metrics(family)
            
#     def _update_drift_metrics(self, family: str):
#         """Compute drift metrics for a family."""
#         trajectory = self.family_trajectories[family]
        
#         # Get time-ordered embeddings
#         embeddings = np.vstack([t['embedding'] for t in trajectory])
#         timestamps = np.array([t['timestamp'] for t in trajectory])
        
#         # Compute drift metrics
#         self.drift_metrics[family].update({
#             'total_drift': self._compute_total_drift(embeddings),
#             'drift_velocity': self._compute_drift_velocity(embeddings, timestamps),
#             'drift_acceleration': self._compute_drift_acceleration(embeddings, timestamps),
#             'stability_periods': self._identify_stability_periods(embeddings, timestamps),
#             'major_shifts': self._detect_major_shifts(embeddings, timestamps)
#         })
    
#     def _compute_total_drift(self, embeddings: np.ndarray) -> float:
#         """Compute total drift as path length in embedding space."""
#         return np.sum(np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1))
        
#     def _compute_drift_velocity(self, embeddings: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
#         """Compute drift velocity over time."""
#         time_deltas = np.diff(timestamps).astype('timedelta64[s]').astype(float)
#         displacement = embeddings[1:] - embeddings[:-1]
        
#         # Handle zero time deltas
#         time_deltas = np.where(time_deltas == 0, np.inf, time_deltas)
#         return np.divide(displacement, time_deltas[:, np.newaxis], out=np.zeros_like(displacement), where=time_deltas[:, np.newaxis]!=0)
    
#     def _compute_drift_acceleration(self, embeddings: np.ndarray, 
#                                   timestamps: np.ndarray) -> np.ndarray:
#         """Compute drift acceleration over time."""
#         velocities = self._compute_drift_velocity(embeddings, timestamps)
#         time_deltas = np.diff(timestamps[1:]).astype('timedelta64[s]').astype(float)
#         return np.diff(velocities, axis=0) / time_deltas[:, np.newaxis]
    
#     def _identify_stability_periods(self, embeddings: np.ndarray, 
#                                   timestamps: np.ndarray, 
#                                   threshold: float = 0.1) -> List[Dict]:
#         """Identify periods where family behavior remains stable."""
#         velocities = self._compute_drift_velocity(embeddings, timestamps)
#         speeds = np.linalg.norm(velocities, axis=1)
        
#         stable_periods = []
#         current_period = None
        
#         for i, (speed, ts) in enumerate(zip(speeds, timestamps[1:])):
#             if speed < threshold:
#                 if current_period is None:
#                     current_period = {'start': timestamps[i], 'count': 1}
#                 current_period['count'] += 1
#                 current_period['end'] = ts
#             elif current_period is not None:
#                 stable_periods.append(current_period)
#                 current_period = None
        
#         if current_period is not None:
#             stable_periods.append(current_period)
            
#         return stable_periods
    
#     def _detect_major_shifts(self, embeddings: np.ndarray, 
#                            timestamps: np.ndarray, 
#                            threshold: float = 0.5) -> List[Dict]:
#         """Detect major behavioral shifts in family evolution."""
#         velocities = self._compute_drift_velocity(embeddings, timestamps)
#         accelerations = self._compute_drift_acceleration(embeddings, timestamps)
        
#         major_shifts = []
        
#         for i, (vel, acc, ts) in enumerate(zip(velocities[1:], accelerations, timestamps[2:])):
#             velocity_magnitude = np.linalg.norm(vel)
#             acceleration_magnitude = np.linalg.norm(acc)
            
#             if (velocity_magnitude is not None and acceleration_magnitude is not None and velocity_magnitude > threshold and acceleration_magnitude > threshold):
#                 shift_analysis = self._analyze_behavioral_shift(
#                     embeddings[i:i+3],
#                     velocities[i:i+2]
#                 )
                
#                 major_shifts.append({
#                     'timestamp': ts,
#                     'magnitude': velocity_magnitude,
#                     'acceleration': acceleration_magnitude,
#                     'analysis': shift_analysis
#                 })
        
#         return major_shifts
        
#     def _analyze_behavioral_shift(self, embeddings: np.ndarray, velocities: np.ndarray) -> Dict:
#         """Enhanced behavioral shift analysis with pattern detection"""
#         # Compute basic metrics
#         direction = self._normalize_vector(velocities[0])
#         magnitude = np.linalg.norm(velocities[0])
        
#         # Analyze trajectory patterns
#         trajectory_length = np.sum([np.linalg.norm(v) for v in velocities])
#         straightness = np.linalg.norm(embeddings[-1] - embeddings[0]) / (trajectory_length + 1e-10)
        
#         # Detect oscillation patterns
#         velocity_angles = [self._angle_between(velocities[i], velocities[i+1]) 
#                         for i in range(len(velocities)-1)]
#         is_oscillating = np.mean(velocity_angles) > np.pi/2
        
#         return {
#             'magnitude': float(magnitude),
#             'straightness': float(straightness),
#             'is_oscillating': bool(is_oscillating),
#             'direction': direction.tolist()
#         }
    
#     def get_family_evolution_summary(self, family: str) -> Dict:
#         """Get comprehensive evolution summary for a family."""
#         if family not in self.drift_metrics:
#             return None
            
#         metrics = self.drift_metrics[family]
#         trajectory = self.family_trajectories[family]
        
#         total_time = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
#         average_velocity = metrics['total_drift'] / total_time.total_seconds()
        
#         return {
#             'first_seen': trajectory[0]['timestamp'],
#             'last_seen': trajectory[-1]['timestamp'],
#             'total_drift': metrics['total_drift'],
#             'average_velocity': average_velocity,
#             'stability_periods': len(metrics['stability_periods']),
#             'major_shifts': len(metrics['major_shifts']),
#             'evolution_phases': self._identify_evolution_phases(family)
#         }

#     def _identify_evolution_phases(self, family: str) -> List[Dict]:
#         """Identify distinct phases in family evolution."""
#         metrics = self.drift_metrics[family]
#         shifts = metrics['major_shifts']
        
#         phases = []
#         last_phase_end = self.family_trajectories[family][0]['timestamp']
        
#         for shift in shifts:
#             phases.append({
#                 'start': last_phase_end,
#                 'end': shift['timestamp'],
#                 'duration': shift['timestamp'] - last_phase_end,
#                 'stability': self._compute_phase_stability(family, last_phase_end, shift['timestamp'])
#             })
#             last_phase_end = shift['timestamp']
        
#         final_timestamp = self.family_trajectories[family][-1]['timestamp']
#         phases.append({
#             'start': last_phase_end,
#             'end': final_timestamp,
#             'duration': final_timestamp - last_phase_end,
#             'stability': self._compute_phase_stability(family, last_phase_end, final_timestamp)
#         })
        
#         return phases

#     def _compute_phase_stability(self, family: str, 
#                                start: pd.Timestamp, 
#                                end: pd.Timestamp) -> float:
#         """Compute stability metric for a specific phase."""
#         trajectory = self.family_trajectories[family]
#         phase_embeddings = [
#             t['embedding'] for t in trajectory 
#             if start <= t['timestamp'] <= end
#         ]
        
#         if len(phase_embeddings) < 2:
#             return 1.0
            
#         embeddings = np.vstack(phase_embeddings)
#         centroid = np.mean(embeddings, axis=0)
#         distances = np.linalg.norm(embeddings - centroid, axis=1)
#         return 1.0 / (1.0 + np.mean(distances))



class FamilyDriftAnalyzer:
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
        
        # Get time-ordered embeddings and timestamps
        embeddings = np.vstack([t['embedding'] for t in trajectory])
        timestamps = np.array([t['timestamp'] for t in trajectory])
        
        # If all timestamps are identical but we have multiple samples,
        # artificially create small time differences
        if len(timestamps) > 1 and np.all(timestamps == timestamps[0]):
            # Create evenly spaced timestamps within the same second
            base_timestamp = timestamps[0]
            microsecond_offsets = np.linspace(0, 999999, len(timestamps))
            timestamps = np.array([
                base_timestamp + pd.Timedelta(microseconds=offset) 
                for offset in microsecond_offsets
            ])
        
        # Now compute drift metrics with the adjusted timestamps
        self.drift_metrics[family].update({
            'total_drift': self._compute_total_drift(embeddings),
            'drift_velocity': self._compute_drift_velocity(embeddings, timestamps),
            'drift_acceleration': self._compute_drift_acceleration(embeddings, timestamps),
            'stability_periods': self._identify_stability_periods(embeddings, timestamps),
            'major_shifts': self._detect_major_shifts(embeddings, timestamps)
        })
    
    def _compute_total_drift(self, embeddings: np.ndarray) -> float:
        """Compute total drift as path length in embedding space."""
        return float(np.sum(np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)))
    
    def _compute_drift_velocity(self, embeddings: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Compute drift velocity over time with safe division."""
        time_deltas = np.diff(timestamps).astype('timedelta64[s]').astype(float)
        displacement = embeddings[1:] - embeddings[:-1]
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        safe_time_deltas = time_deltas + epsilon
        
        # Compute velocities
        velocities = displacement / safe_time_deltas[:, np.newaxis]
        
        # Zero out velocities where time delta was too small
        velocities[time_deltas < epsilon] = 0
        
        return velocities
    
    def _compute_drift_acceleration(self, embeddings: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Compute drift acceleration over time."""
        velocities = self._compute_drift_velocity(embeddings, timestamps)
        time_deltas = np.diff(timestamps[1:]).astype('timedelta64[s]').astype(float)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        safe_time_deltas = time_deltas + epsilon
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0) / safe_time_deltas[:, np.newaxis]
        
        # Zero out accelerations where time delta was too small
        accelerations[time_deltas < epsilon] = 0
        
        return accelerations
    
    def _identify_stability_periods(self, embeddings: np.ndarray, timestamps: np.ndarray, threshold: float = 0.1) -> List[Dict]:
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
    
    def _detect_major_shifts(self, embeddings: np.ndarray, timestamps: np.ndarray, threshold: float = 0.5) -> List[Dict]:
        """Detect major behavioral shifts in family evolution."""
        velocities = self._compute_drift_velocity(embeddings, timestamps)
        accelerations = self._compute_drift_acceleration(embeddings, timestamps)
        
        major_shifts = []
        
        if len(accelerations) > 0:  # Only process if we have acceleration data
            for i, (vel, acc, ts) in enumerate(zip(velocities[1:], accelerations, timestamps[2:])):
                velocity_magnitude = np.linalg.norm(vel)
                acceleration_magnitude = np.linalg.norm(acc)
                
                if velocity_magnitude > threshold and acceleration_magnitude > threshold:
                    shift_analysis = self._analyze_behavioral_shift(
                        embeddings[i:i+3],
                        velocities[i:i+2]
                    )
                    
                    major_shifts.append({
                        'timestamp': ts,
                        'magnitude': float(velocity_magnitude),
                        'acceleration': float(acceleration_magnitude),
                        'analysis': shift_analysis
                    })
        
        return major_shifts
    
    def _analyze_behavioral_shift(self, embeddings: np.ndarray, velocities: np.ndarray) -> Dict:
        """Analyze the nature of a behavioral shift."""
        v0_norm = np.linalg.norm(velocities[0])
        if v0_norm > 0:
            direction = velocities[0] / v0_norm
        else:
            direction = np.zeros_like(velocities[0])
            
        temporary = bool(np.dot(velocities[0], velocities[1]) < 0)
        distance = float(np.linalg.norm(embeddings[2] - embeddings[0]))
        
        return {
            'temporary': temporary,
            'distance': distance,
            'direction': direction.tolist()  # Convert to list for JSON serialization
        }

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
    @torch.no_grad()
    def detect_new_families(self, embeddings: torch.Tensor) -> List[bool]:
        """Detect potential new families based on distance to existing centroids."""
        # Safely convert embeddings to numpy
        embeddings_np = embeddings.detach().cpu().numpy()
        new_family_flags = []
        
        # If we have no centroids or no distance threshold yet, treat all as known families
        if not self.family_centroids or self.distance_threshold is None:
            return [False] * len(embeddings_np)
        
        for embedding in embeddings_np:
            # Compute distances to all existing centroids
            distances = {
                family: np.linalg.norm(embedding - data['centroid'])
                for family, data in self.family_centroids.items()
            }
            
            # Get minimum distance if we have any centroids
            if distances:
                min_distance = min(distances.values())
                # Compare with threshold only if we have both values
                is_new = min_distance > self.distance_threshold
            else:
                is_new = True  # If no centroids, treat as new family
                
            new_family_flags.append(is_new)
        
        return new_family_flags

    def _compute_local_density(self, embedding, k=5):
        """Compute local density score using k-nearest neighbors"""
        if not self.family_centroids:
            return 0
        
        # Get distances to all centroids
        distances = [
            np.linalg.norm(embedding - data['centroid'])
            for data in self.family_centroids.values()
        ]
        
        # Use k nearest neighbors for density estimation
        knn_distances = sorted(distances)[:k]
        return 1.0 / (np.mean(knn_distances) + 1e-10)
    
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
        
        # Add residual projection layer
        self.residual_proj = torch.nn.Linear(128, embedding_dim)
        
        # Add attention layer
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        # Group classification head
        self.group_classifier = torch.nn.Linear(embedding_dim, num_groups)
        
        # Separate classifiers for each behavioral group
        self.family_classifiers = torch.nn.ModuleDict()

    def add_family_classifier(self, group_id: str, num_families: int):
        """Dynamically add a family classifier for a behavioral group"""
        self.family_classifiers[str(group_id)] = torch.nn.Linear(
            self.embedding_dim, num_families
        )
    
    def _weighted_pool(self, x, weights, batch):
        """Custom weighted pooling"""
        weights = torch.sigmoid(weights)  # Ensure weights are in [0,1]
        weighted_x = x * weights
        return global_mean_pool(weighted_x, batch)
    
    def get_embeddings(self, data):
        """Enhanced embedding extraction with attention and residual connections"""
        device = next(self.parameters()).device
        x = data.x.to(device) 
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        
        # First GCN layer
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        
        # Second GCN layer with residual connection
        h2 = self.conv2(h1, edge_index)
        h2 = h2 + self.residual_proj(h1)  # Add residual
        h2 = F.relu(h2)
        
        # Global pooling with attention
        node_weights = self.attention(h2)
        embeddings = self._weighted_pool(h2, node_weights, batch)
        
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

def evaluate_detailed(model, split_files, classifier, device, criterion, batch_size=32):
    """Evaluate the model with detailed metrics per family and group."""
    model.eval()
    metrics = {
        'total_loss': 0,
        'num_batches': 0,
        'predictions': []  # Store all predictions for confusion matrix
    }
    
    with torch.no_grad():
        for batch_file in split_files:
            batch_loader = load_batch(batch_file, classifier.group_mappings['family_to_group'], batch_size=batch_size)
            if not batch_loader:
                continue
                
            for batch in batch_loader:
                batch = batch.to(device)
                
                # Forward pass
                embeddings, group_logits, family_logits = model(batch)
                
                # Get group predictions
                pred_groups = group_logits.argmax(dim=1)
                true_groups = torch.tensor([
                    classifier.group_mappings['family_to_group'].get(fam, -1) 
                    for fam in batch.family
                ]).to(device)
                
                # Get family predictions for each group
                pred_families = []
                for i, (pred_group, true_family) in enumerate(zip(pred_groups, batch.family)):
                    group_id = str(pred_group.item())
                    if group_id in family_logits:
                        family_logits_group = family_logits[group_id][i]
                        pred_family_idx = family_logits_group.argmax().item()
                        if pred_group.item() in classifier.group_mappings['group_to_families']:
                            families = classifier.group_mappings['group_to_families'][pred_group.item()]
                            if pred_family_idx < len(families):
                                pred_family = families[pred_family_idx]
                            else:
                                pred_family = 'unknown'
                        else:
                            pred_family = 'unknown'
                    else:
                        pred_family = 'unknown'
                    pred_families.append(pred_family)
                
                # Store predictions
                for pred_group, true_group, pred_family, true_family in zip(
                    pred_groups, true_groups, pred_families, batch.family
                ):
                    metrics['predictions'].append({
                        'pred_group': pred_group.item(),
                        'true_group': true_group.item(),
                        'pred_family': pred_family,
                        'true_family': true_family
                    })
                
                # Compute loss
                loss = criterion(embeddings, group_logits, family_logits, batch.family, device)
                metrics['total_loss'] += loss.item()
                metrics['num_batches'] += 1
    
    return compute_final_metrics(metrics, classifier)

def compute_final_metrics(metrics, classifier):
    """Compute final metrics including precision, recall, and F1 score."""
    final_metrics = {
        'avg_loss': float(metrics['total_loss']) / max(1, metrics['num_batches']),
        'group_metrics': {},
        'family_metrics': {},
        'overall': {}
    }
    
    # Compute group-level metrics
    all_group_preds = []
    all_group_true = []
    
    for pred in metrics['predictions']:
        if pred['true_group'] != -1:  # Skip unknown groups
            all_group_preds.append(pred['pred_group'])
            all_group_true.append(pred['true_group'])
    
    if all_group_preds:
        group_precision, group_recall, group_f1, _ = precision_recall_fscore_support(
            all_group_true, all_group_preds, average='weighted', zero_division=0
        )
        group_accuracy = accuracy_score(all_group_true, all_group_preds)
        
        final_metrics['overall']['group'] = {
            'precision': float(group_precision),
            'recall': float(group_recall),
            'f1': float(group_f1),
            'accuracy': float(group_accuracy)
        }
    
    # Compute per-group metrics
    unique_groups = set(pred['true_group'] for pred in metrics['predictions'])
    for group_id in unique_groups:
        if group_id == -1:  # Skip unknown group
            continue
            
        # Binary classification for this group
        true_labels = []
        pred_labels = []
        for pred in metrics['predictions']:
            if pred['true_group'] == group_id or pred['pred_group'] == group_id:
                true_labels.append(1 if pred['true_group'] == group_id else 0)
                pred_labels.append(1 if pred['pred_group'] == group_id else 0)
        
        if true_labels:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='binary', zero_division=0
            )
            final_metrics['group_metrics'][str(group_id)] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
    
    # Compute family-level overall metrics
    all_family_preds = []
    all_family_true = []
    
    for pred in metrics['predictions']:
        if pred['true_family'] != 'unknown':
            all_family_preds.append(pred['pred_family'])
            all_family_true.append(pred['true_family'])
    
    if all_family_preds:
        family_precision, family_recall, family_f1, _ = precision_recall_fscore_support(
            all_family_true, all_family_preds, average='weighted', zero_division=0
        )
        family_accuracy = accuracy_score(all_family_true, all_family_preds)
        
        final_metrics['overall']['family'] = {
            'precision': float(family_precision),
            'recall': float(family_recall),
            'f1': float(family_f1),
            'accuracy': float(family_accuracy)
        }
    
    # Compute per-family metrics
    unique_families = set(pred['true_family'] for pred in metrics['predictions'])
    for family in unique_families:
        if family == 'unknown':
            continue
            
        # Binary classification for this family
        true_labels = []
        pred_labels = []
        for pred in metrics['predictions']:
            if pred['true_family'] == family or pred['pred_family'] == family:
                true_labels.append(1 if pred['true_family'] == family else 0)
                pred_labels.append(1 if pred['pred_family'] == family else 0)
        
        if true_labels:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='binary', zero_division=0
            )
            final_metrics['family_metrics'][family] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'total_samples': sum(1 for pred in metrics['predictions'] 
                                   if pred['true_family'] == family)
            }
    
    return final_metrics

def log_evaluation_results(metrics: dict):
    """Log detailed evaluation results."""
    logger.info("\nEvaluation Results:")
    
    # Log overall metrics
    if 'overall' in metrics:
        for level in ['group', 'family']:
            if level in metrics['overall']:
                logger.info(f"\n{level.capitalize()} Level Overall Metrics:")
                for metric, value in metrics['overall'][level].items():
                    logger.info(f"{metric}: {value:.4f}")
    
    # Log group metrics
    logger.info("\nBehavioral Group Metrics:")
    for group_id, group_metrics in metrics['group_metrics'].items():
        logger.info(f"\nGroup {group_id}:")
        for metric, value in group_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    # Log family metrics
    logger.info("\nFamily Metrics:")
    for family, family_metrics in metrics['family_metrics'].items():
        logger.info(f"\nFamily {family}:")
        for metric, value in family_metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric}: {value:.4f}")
            else:
                logger.info(f"{metric}: {value}")

def train_temporal(model, classifier, train_loader, val_loader, optimizer, criterion, device, drift_analyzer, num_epochs=10):
    best_val_loss = float('inf')
    results = []
    model = model.to(device)
    
    train_files = train_loader if isinstance(train_loader, (list, tuple)) else [train_loader]
    val_files = val_loader if isinstance(val_loader, (list, tuple)) else [val_loader]
    
    # Store all embeddings during training
    all_embeddings = defaultdict(list)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_file in train_files:
            batch_loader = load_batch(batch_file, classifier.group_mappings['family_to_group'])
            if not batch_loader:
                continue
            
            for batch_data in batch_loader:
                try:
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    
                    embeddings, group_logits, family_logits = model(batch_data)
                    batch_families = batch_data.family if isinstance(batch_data.family, list) else [batch_data.family]
                    
                    # Store embeddings and timestamps
                    embeddings_cpu = embeddings.detach().cpu().numpy()
                    for emb, fam, ts in zip(embeddings_cpu, batch_families, batch_data.timestamp):
                        all_embeddings[fam].append({
                            'embedding': emb,
                            'timestamp': pd.to_datetime(ts)
                        })
                    
                    loss = criterion(embeddings, group_logits, family_logits, batch_families, device)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue
        
        if num_batches == 0:
            continue
            
        avg_train_loss = total_loss / num_batches
        val_metrics = validate_model(model, val_files, criterion, classifier, device)
        
        logger.info(f"Epoch {epoch}:")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        if val_metrics['val_loss'] < best_val_loss:
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

    # Compute drift metrics and update centroids after all epochs
    logger.info("Computing final drift metrics and centroids...")
    
    # First update centroids as it's simpler
    logger.info("Computing centroids...")
    for family, samples in all_embeddings.items():
        embeddings = np.vstack([s['embedding'] for s in samples])
        centroid = np.mean(embeddings, axis=0)
        last_timestamp = max(s['timestamp'] for s in samples)
        classifier.family_centroids[family] = {
            'centroid': centroid,
            'last_updated': last_timestamp,
            'num_samples': len(samples)
        }

    # Then do drift analysis for any families with >1 sample, with a progress log
    logger.info("Computing drift metrics...")
    total_families = len(all_embeddings)
    for idx, (family, samples) in enumerate(all_embeddings.items(), 1):
        if len(samples) > 1:  # Only compute drift for families with multiple samples
            sorted_samples = sorted(samples, key=lambda x: x['timestamp'])
            for sample in sorted_samples:
                drift_analyzer.track_family_drift(family, sample['embedding'], sample['timestamp'])
        if idx % 10 == 0:  # Log progress every 10 families
            logger.info(f"Processed drift metrics for {idx}/{total_families} families")

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

def main():
    # Define configuration directly instead of using command-line arguments
    config = {
        'batch_dir': '/data/saranyav/gcn_new/bodmas_batches',
        'behavioral_groups': '/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json',
        'embedding_dim': 256,
        'num_epochs': 10,
        'batch_size': 32,
        'output_dir': 'evolution_analysis',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    try:
        # Set up output directory
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up device
        device = torch.device(config['device'])
        logger.info(f"Using device: {device}")

        # Initialize temporal components
        logger.info("Initializing temporal components...")
        classifier = TemporalMalwareClassifier(
            behavioral_groups_path=config['behavioral_groups'],
            embedding_dim=config['embedding_dim']
        )
        #drift_analyzer = FamilyDriftAnalyzer(embedding_dim=config['embedding_dim'])

        # Prepare data
        logger.info("Preparing data...")
        split_files, file_timestamps = prepare_data(config['batch_dir'])
        
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
            embedding_dim=config['embedding_dim']
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
            #drift_analyzer=drift_analyzer,
            num_epochs=config['num_epochs']
        )

        # Load best model for final evaluation
        logger.info("Loading best model for final evaluation...")
        try:
            checkpoint = torch.load('best_model.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            return
        
        # Run detailed evaluation
        logger.info("Running final evaluation...")
        test_metrics = evaluate_detailed(
            model=model,
            split_files=test_files,
            classifier=classifier,
            device=device,
            criterion=criterion,
            batch_size=config['batch_size']
        )
        
        # Save final results
        logger.info("Saving analysis results...")
        final_metrics = {
            'test_metrics': test_metrics,
            'training_history': results,
            'config': config  # Save configuration for reference
        }
        
        try:
            save_analysis_results(output_dir, classifier, final_metrics)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
        
        # Log detailed evaluation results
        log_evaluation_results(test_metrics)

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)