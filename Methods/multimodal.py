import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import numpy as np
from collections import defaultdict
import pefile
import json
import logging
import os
from pathlib import Path
from tqdm import tqdm
import gzip
from typing import Dict, List, Optional
import concurrent.futures

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from collections import defaultdict

class BehavioralAttention(nn.Module):
    """Attention mechanism that learns to focus on behavioral patterns"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        weights = self.attention(x)
        weighted = x * weights
        return weighted.sum(dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from collections import defaultdict
import datetime
from typing import Dict, List, Tuple

class TemporalEncoder(nn.Module):
    """Encode temporal information from timestamps"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),  # 5 temporal features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def forward(self, timestamps: List[str]) -> torch.Tensor:
        # Convert timestamps to temporal features
        temporal_features = []
        reference_time = datetime.datetime(2019, 1, 1)  # Adjust based on your data

        for ts in timestamps:
            try:
                # Parse timestamp like "2019-09-05 00:01:50 UTC"
                dt = datetime.datetime.strptime(ts.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
                
                # Extract temporal features
                time_diff = (dt - reference_time).total_seconds() / (24 * 3600)  # Days since reference
                month = dt.month / 12  # Normalize month
                day = dt.day / 31  # Normalize day
                hour = dt.hour / 24  # Normalize hour
                weekday = dt.weekday() / 7  # Normalize weekday
                
                temporal_features.append([time_diff, month, day, hour, weekday])
            except Exception as e:
                # Use default features if parsing fails
                temporal_features.append([0, 0, 0, 0, 0])

        # Convert to tensor and encode
        temp_tensor = torch.tensor(temporal_features, dtype=torch.float32)
        return self.time_mlp(temp_tensor)

class MalwareTypeEncoder(nn.Module):
    """Encode malware type information"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.type_embedding = nn.Embedding(10, hidden_dim)  # Assume max 10 malware types
        self.type_to_idx = {}  # Will be populated during forward pass
        
    def forward(self, malware_types: List[str]) -> torch.Tensor:
        # Convert malware types to indices
        indices = []
        for mtype in malware_types:
            if mtype not in self.type_to_idx:
                self.type_to_idx[mtype] = len(self.type_to_idx)
            indices.append(self.type_to_idx[mtype])
        
        type_tensor = torch.tensor(indices, dtype=torch.long)
        # Move to same device as model
        type_tensor = type_tensor.to(next(self.type_embedding.parameters()).device)
        return self.type_embedding(type_tensor)

class TemporalAttention(nn.Module):
    """Attention mechanism that considers temporal proximity"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, query, keys, time_diffs):
        # Combine feature similarity with temporal proximity
        attention_input = torch.cat([
            query.unsqueeze(1).expand(-1, keys.size(0), -1),
            keys.unsqueeze(0).expand(query.size(0), -1, -1)
        ], dim=2)
        
        # Calculate attention scores
        scores = self.attention(attention_input).squeeze(-1)
        
        # Apply temporal weighting
        temporal_weight = 1.0 / (1.0 + time_diffs)
        scores = scores * temporal_weight
        
        # Normalize scores
        attention_weights = F.softmax(scores, dim=1)
        return attention_weights

class HierarchicalMalwareGNN(nn.Module):
    def __init__(self, gnn_input_dim, static_input_dim, hidden_dim=256, n_behavioral_groups=20):
        super().__init__()
        
        # Feature Encoders
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN layers with dimension matching for residual connections
        self.conv_layers = nn.ModuleList([
            GCNConv(gnn_input_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        
        # Dimension matching layers for residual connections
        self.dim_match = nn.ModuleList([
            nn.Linear(gnn_input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal and Type Encoders
        self.temporal_encoder = TemporalEncoder(hidden_dim)
        self.type_encoder = MalwareTypeEncoder(hidden_dim)
        
        # Multi-head attention for different aspects
        self.behavioral_attention = BehavioralAttention(hidden_dim)
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Pattern Recognition Components
        self.pattern_detector = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim)
        ])
        
        self.structure_detector = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim)
        ])
        
        # Fusion Components
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Output Components
        self.group_classifier = nn.Linear(hidden_dim, n_behavioral_groups)
        self.temporal_predictor = nn.Linear(hidden_dim, 1)
        
    def forward(self, graph_data, static_features):
        # Process static features
        static_embed = self.static_encoder(static_features)
        
        # Process temporal and type information
        temporal_embed = self.temporal_encoder(graph_data.timestamp)
        type_embed = self.type_encoder(graph_data.malware_type)
        
        # Process graph features
        x, edge_index = graph_data.x, graph_data.edge_index
        batch = graph_data.batch if hasattr(graph_data, 'batch') else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # GNN layers with proper dimension matching for residual connections
        for i, (conv, dim_match) in enumerate(zip(self.conv_layers, self.dim_match)):
            x_new = conv(x, edge_index)
            x_res = dim_match(x)
            x = F.relu(x_new) + x_res
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Extract patterns
        pattern_features = x
        for layer in self.pattern_detector:
            pattern_features = F.relu(layer(pattern_features))
        
        # Extract structural patterns
        structure_features = x
        for layer in self.structure_detector:
            structure_features = F.relu(layer(structure_features))
        
        # Pool graph-level representations
        graph_embed = global_mean_pool(x, batch)
        pattern_embed = global_mean_pool(pattern_features, batch)
        
        # Debug print dimensions
        print(f"Dimensions before concatenation:")
        print(f"graph_embed: {graph_embed.shape}")
        print(f"static_embed: {static_embed.shape}")
        print(f"temporal_embed: {temporal_embed.shape}")
        print(f"type_embed: {type_embed.shape}")
        print(f"pattern_embed: {pattern_embed.shape}")
        
        # Combine all features
        combined_features = torch.cat([
            graph_embed,
            static_embed,
            temporal_embed,
            type_embed,
            pattern_embed
        ], dim=1)
        
        # Final fusion
        fused = self.feature_fusion(combined_features)
        
        # Outputs
        group_logits = self.group_classifier(fused)
        time_pred = self.temporal_predictor(fused)
        
        return fused, group_logits, time_pred

class EnhancedExemplarDetector:
    """Exemplar detector with behavioral group awareness"""
    def __init__(self, model, device, behavioral_groups, similarity_threshold=0.75):
        self.model = model
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.behavioral_groups = behavioral_groups
        
        # Maintain exemplars at both family and group levels
        self.family_exemplars = defaultdict(list)
        self.group_exemplars = defaultdict(list)
        
        # Track family-to-group mapping
        self.family_to_group = {}
        for group_id, families in behavioral_groups.items():
            for family in families:
                self.family_to_group[family] = group_id

    def update_exemplars(self, embeddings, families, group_preds):
        """Update both family and behavioral group exemplars"""
        for emb, family, group_pred in zip(embeddings, families, group_preds):
            family_str = str(family.item())
            group_id = self.family_to_group.get(family_str, None)
            
            # Update family exemplars
            self._update_exemplar_set(self.family_exemplars[family_str], emb)
            
            # Update group exemplars if family has a known group
            if group_id is not None:
                self._update_exemplar_set(self.group_exemplars[group_id], emb)
    
    def _update_exemplar_set(self, exemplars, new_emb, max_exemplars=5):
        """Helper method to update a set of exemplars"""
        if len(exemplars) < max_exemplars:
            exemplars.append(new_emb.detach())
        else:
            # Replace least representative exemplar
            sims = torch.stack([F.cosine_similarity(new_emb, ex.to(self.device), dim=0) 
                              for ex in exemplars])
            worst_idx = sims.argmin()
            mean_sim = F.cosine_similarity(new_emb, 
                                         torch.stack(exemplars).mean(dim=0).to(self.device),
                                         dim=0)
            if sims[worst_idx] < mean_sim:
                exemplars[worst_idx] = new_emb.detach()

    def predict(self, embedding, group_logits):
        """Hierarchical prediction using both family and group exemplars"""
        # First try to match with family exemplars
        max_family_sim = -1
        pred_family = "new"
        
        for family in self.family_exemplars:
            sim = self._compute_similarity(embedding, self.family_exemplars[family])
            if sim > max_family_sim:
                max_family_sim = sim
                pred_family = family
        
        # If no good family match, try matching behavioral group
        if max_family_sim < self.similarity_threshold:
            pred_group = group_logits.argmax().item()
            group_sim = self._compute_similarity(embedding, self.group_exemplars[pred_group])
            
            if group_sim > self.similarity_threshold:
                return f"new_group_{pred_group}", group_sim
            return "new", group_sim
            
        return pred_family, max_family_sim
    
    def _compute_similarity(self, embedding, exemplars):
        """Compute similarity to a set of exemplars"""
        if not exemplars:
            return 0.0
        
        sims = torch.stack([F.cosine_similarity(embedding, ex.to(self.device), dim=0) 
                           for ex in exemplars])
        return 0.7 * sims.max() + 0.3 * sims.mean()

class TemporalExemplarDetector(EnhancedExemplarDetector):
    """Enhanced detector with temporal awareness"""
    def __init__(self, model, device, behavioral_groups, similarity_threshold=0.75,
                 temporal_window=30):  # temporal_window in days
        super().__init__(model, device, behavioral_groups, similarity_threshold)
        self.temporal_window = temporal_window
        self.exemplar_timestamps = defaultdict(list)
        
    def update_exemplars(self, embeddings, families, timestamps, group_preds):
        """Update exemplars with temporal information"""
        for emb, family, ts, group_pred in zip(embeddings, families, timestamps, group_preds):
            family_str = str(family.item())
            group_id = self.family_to_group.get(family_str, None)
            
            # Parse timestamp
            timestamp = datetime.datetime.strptime(ts.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
            
            # Update family exemplars with timestamp
            self._update_exemplar_set(
                self.family_exemplars[family_str],
                self.exemplar_timestamps[family_str],
                emb, timestamp
            )
            
            # Update group exemplars
            if group_id is not None:
                self._update_exemplar_set(
                    self.group_exemplars[group_id],
                    self.exemplar_timestamps[f"group_{group_id}"],
                    emb, timestamp
                )
    
    def _update_exemplar_set(self, exemplars, timestamps, new_emb, new_timestamp, 
                            max_exemplars=5):
        """Update exemplar set considering temporal information"""
        if len(exemplars) < max_exemplars:
            exemplars.append(new_emb.detach())
            timestamps.append(new_timestamp)
        else:
            # Consider both similarity and temporal relevance
            current_time = datetime.datetime.now()
            time_diffs = [(current_time - ts).days for ts in timestamps]
            
            # Compute combined score (similarity + temporal)
            scores = []
            for i, ex in enumerate(exemplars):
                sim = F.cosine_similarity(new_emb, ex.to(self.device), dim=0)
                temporal_weight = np.exp(-time_diffs[i] / self.temporal_window)
                scores.append(sim.item() * temporal_weight)
            
            # Replace worst exemplar if new one is better
            worst_idx = np.argmin(scores)
            if scores[worst_idx] < 0.5:  # Threshold for replacement
                exemplars[worst_idx] = new_emb.detach()
                timestamps[worst_idx] = new_timestamp

    def predict(self, embedding, group_logits, current_time=None):
        """Predict with temporal awareness"""
        if current_time is None:
            current_time = datetime.datetime.now()
            
        # Try family matching with temporal weighting
        max_score = -1
        pred_family = "new"
        
        for family in self.family_exemplars:
            # Compute similarity with temporal weighting
            sims = []
            for ex, ts in zip(self.family_exemplars[family], 
                            self.exemplar_timestamps[family]):
                sim = F.cosine_similarity(embedding, ex.to(self.device), dim=0)
                days_diff = (current_time - ts).days
                temporal_weight = np.exp(-days_diff / self.temporal_window)
                weighted_sim = sim.item() * temporal_weight
                sims.append(weighted_sim)
            
            if sims:
                score = max(sims)
                if score > max_score:
                    max_score = score
                    pred_family = family
        
        # If no good match, try group matching
        if max_score < self.similarity_threshold:
            pred_group = group_logits.argmax().item()
            group_sims = []
            
            for ex, ts in zip(self.group_exemplars[pred_group],
                            self.exemplar_timestamps[f"group_{pred_group}"]):
                sim = F.cosine_similarity(embedding, ex.to(self.device), dim=0)
                days_diff = (current_time - ts).days
                temporal_weight = np.exp(-days_diff / self.temporal_window)
                weighted_sim = sim.item() * temporal_weight
                group_sims.append(weighted_sim)
            
            if group_sims and max(group_sims) > self.similarity_threshold:
                return f"new_group_{pred_group}", max(group_sims)
            return "new", max_score
            
        return pred_family, max_score
    
class TemporalEncoder(nn.Module):
    """Encode temporal information from timestamps"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),  # 5 temporal features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def forward(self, timestamps: List[str]) -> torch.Tensor:
        # Convert timestamps to temporal features
        temporal_features = []
        reference_time = datetime.datetime(2019, 1, 1)  # Adjust based on your data

        for ts in timestamps:
            try:
                # Parse timestamp like "2019-09-05 00:01:50 UTC"
                dt = datetime.datetime.strptime(ts.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
                
                # Extract temporal features
                time_diff = (dt - reference_time).total_seconds() / (24 * 3600)  # Days since reference
                month = dt.month / 12  # Normalize month
                day = dt.day / 31  # Normalize day
                hour = dt.hour / 24  # Normalize hour
                weekday = dt.weekday() / 7  # Normalize weekday
                
                temporal_features.append([time_diff, month, day, hour, weekday])
            except Exception as e:
                # Use default features if parsing fails
                temporal_features.append([0, 0, 0, 0, 0])

        # Convert to tensor and encode
        temp_tensor = torch.tensor(temporal_features, dtype=torch.float32)
        # Move to same device as model
        temp_tensor = temp_tensor.to(next(self.time_mlp.parameters()).device)
        return self.time_mlp(temp_tensor)

        
    def update_exemplars(self, embeddings, families, timestamps, group_preds):
        """Update exemplars with temporal information"""
        for emb, family, ts, group_pred in zip(embeddings, families, timestamps, group_preds):
            family_str = str(family.item())
            group_id = self.family_to_group.get(family_str, None)
            
            # Parse timestamp
            timestamp = datetime.datetime.strptime(ts.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
            
            # Update family exemplars with timestamp
            self._update_exemplar_set(
                self.family_exemplars[family_str],
                self.exemplar_timestamps[family_str],
                emb, timestamp
            )
            
            # Update group exemplars
            if group_id is not None:
                self._update_exemplar_set(
                    self.group_exemplars[group_id],
                    self.exemplar_timestamps[f"group_{group_id}"],
                    emb, timestamp
                )
    
    def _update_exemplar_set(self, exemplars, timestamps, new_emb, new_timestamp, 
                            max_exemplars=5):
        """Update exemplar set considering temporal information"""
        if len(exemplars) < max_exemplars:
            exemplars.append(new_emb.detach())
            timestamps.append(new_timestamp)
        else:
            # Consider both similarity and temporal relevance
            current_time = datetime.datetime.now()
            time_diffs = [(current_time - ts).days for ts in timestamps]
            
            # Compute combined score (similarity + temporal)
            scores = []
            for i, ex in enumerate(exemplars):
                sim = F.cosine_similarity(new_emb, ex.to(self.device), dim=0)
                temporal_weight = np.exp(-time_diffs[i] / self.temporal_window)
                scores.append(sim.item() * temporal_weight)
            
            # Replace worst exemplar if new one is better
            worst_idx = np.argmin(scores)
            if scores[worst_idx] < 0.5:  # Threshold for replacement
                exemplars[worst_idx] = new_emb.detach()
                timestamps[worst_idx] = new_timestamp

    def predict(self, embedding, group_logits, current_time=None):
        """Predict with temporal awareness"""
        if current_time is None:
            current_time = datetime.datetime.now()
            
        # Try family matching with temporal weighting
        max_score = -1
        pred_family = "new"
        
        for family in self.family_exemplars:
            # Compute similarity with temporal weighting
            sims = []
            for ex, ts in zip(self.family_exemplars[family], 
                            self.exemplar_timestamps[family]):
                sim = F.cosine_similarity(embedding, ex.to(self.device), dim=0)
                days_diff = (current_time - ts).days
                temporal_weight = np.exp(-days_diff / self.temporal_window)
                weighted_sim = sim.item() * temporal_weight
                sims.append(weighted_sim)
            
            if sims:
                score = max(sims)
                if score > max_score:
                    max_score = score
                    pred_family = family
        
        # If no good match, try group matching
        if max_score < self.similarity_threshold:
            pred_group = group_logits.argmax().item()
            group_sims = []
            
            for ex, ts in zip(self.group_exemplars[pred_group],
                            self.exemplar_timestamps[f"group_{pred_group}"]):
                sim = F.cosine_similarity(embedding, ex.to(self.device), dim=0)
                days_diff = (current_time - ts).days
                temporal_weight = np.exp(-days_diff / self.temporal_window)
                weighted_sim = sim.item() * temporal_weight
                group_sims.append(weighted_sim)
            
            if group_sims and max(group_sims) > self.similarity_threshold:
                return f"new_group_{pred_group}", max(group_sims)
            return "new", max_score
            
        return pred_family, max_score




def load_behavioral_groups(groups_path: str) -> Dict:
    """Load behavioral groups from JSON file"""
    with open(groups_path, 'r') as f:
        return json.load(f)

class StaticFeatureExtractor:
    """Extract static features from PE files with caching"""
    def __init__(self, feature_dim=43, cache_dir="static_features_cache"):
        self.feature_dim = feature_dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, exe_hash):
        """Get path for cached features"""
        return self.cache_dir / f"{exe_hash}_features.pt"
        
    def extract_features(self, exe_path):
        try:
            # Get hash from filename (assumes format: {hash}_refang.exe)
            exe_hash = os.path.basename(exe_path).split('_')[0]
            cache_path = self._get_cache_path(exe_hash)
            
            # Check cache first
            if cache_path.exists():
                return torch.load(cache_path)
            
            # Extract features if not cached
            pe = pefile.PE(exe_path)
            features = []
            
            # Header features
            features.extend([
                pe.FILE_HEADER.Machine,
                pe.FILE_HEADER.NumberOfSections,
                pe.FILE_HEADER.TimeDateStamp,
                pe.FILE_HEADER.Characteristics,
            ])
            
            # Section features
            section_features = []
            for section in pe.sections:
                section_features.extend([
                    section.Misc_VirtualSize,
                    section.SizeOfRawData,
                    section.Characteristics,
                    section.get_entropy()
                ])
            
            # Pad or truncate to fixed length
            section_features = (section_features + [0] * 32)[:32]
            features.extend(section_features)
            
            # Import features
            import_features = defaultdict(int)
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    for imp in entry.imports:
                        if imp.name:
                            import_features[imp.name.decode()] += 1
            
            # Get top imported functions
            common_imports = ['CreateFileA', 'WriteFile', 'RegOpenKeyExA', 
                            'LoadLibraryA', 'GetProcAddress']
            for imp in common_imports:
                features.append(import_features[imp])
            
            # Add entropy and size features
            features.append(sum(s.get_entropy() for s in pe.sections) / len(pe.sections))
            features.append(os.path.getsize(exe_path))
            
            # Convert to tensor and cache
            features_tensor = torch.tensor(features, dtype=torch.float32)
            torch.save(features_tensor, cache_path)
            
            return features_tensor
            
        except Exception as e:
            logging.error(f"Error extracting static features from {exe_path}: {str(e)}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
    
    def extract_features_batch(self, exe_paths, num_workers=4):
        """Extract features for multiple files in parallel"""
        from concurrent.futures import ThreadPoolExecutor
        
        results = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_path = {executor.submit(self.extract_features, path): path 
                            for path in exe_paths}
            
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    features = future.result()
                    results[path] = features
                except Exception as e:
                    logging.error(f"Error processing {path}: {str(e)}")
                    results[path] = torch.zeros(self.feature_dim, dtype=torch.float32)
        
        return results

def clean_graph_indices(graph):
    """Clean and validate graph edge indices with proper shape handling"""
    try:
        num_nodes = graph.x.size(0)
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
        
        # Print initial shapes for debugging
        print(f"Initial shapes:")
        print(f"Edge index shape: {edge_index.shape}")
        if edge_attr is not None:
            print(f"Edge attr shape: {edge_attr.shape}")
        print(f"Num nodes: {num_nodes}")
        
        # Find valid edges (within bounds and no self-loops)
        valid_mask = (edge_index[0] >= 0) & (edge_index[0] < num_nodes) & \
                    (edge_index[1] >= 0) & (edge_index[1] < num_nodes) & \
                    (edge_index[0] != edge_index[1])
        
        # Print debug info
        invalid_count = (~valid_mask).sum().item()
        if invalid_count > 0:
            print(f"Found {invalid_count} invalid edges in graph")
            
        # Keep only valid edges
        edge_index = edge_index[:, valid_mask]
        
        # Handle edge attributes carefully
        if edge_attr is not None:
            # Reshape edge_attr if needed
            if len(edge_attr.shape) == 2 and edge_attr.shape[0] != edge_index.shape[1]:
                print(f"Reshaping edge_attr from {edge_attr.shape} to match edge_index")
                # If edge_attr is [N, 1], reshape to [N]
                if edge_attr.shape[1] == 1:
                    edge_attr = edge_attr.squeeze(-1)
                # Take only the first N values where N is the number of valid edges
                edge_attr = edge_attr[:edge_index.shape[1]]
            
            # Apply mask to edge attributes
            try:
                if len(valid_mask) == len(edge_attr):
                    edge_attr = edge_attr[valid_mask]
                else:
                    print(f"Warning: edge_attr length ({len(edge_attr)}) doesn't match mask length ({len(valid_mask)})")
                    # Create new edge attributes of the correct length
                    edge_attr = torch.ones(edge_index.shape[1], dtype=edge_attr.dtype, 
                                         device=edge_attr.device)
            except Exception as e:
                print(f"Error handling edge attributes: {str(e)}")
                # Fall back to creating new edge attributes
                edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32,
                                     device=edge_index.device)
        
        # Update graph
        graph.edge_index = edge_index
        if edge_attr is not None:
            graph.edge_attr = edge_attr
            
        # Print final shapes
        print(f"Final shapes:")
        print(f"Edge index shape: {graph.edge_index.shape}")
        print(f"Edge attr shape: {graph.edge_attr.shape}")
        
        # Verify after cleaning
        assert edge_index.max() < num_nodes, f"Max index {edge_index.max()} >= num_nodes {num_nodes}"
        assert edge_index.min() >= 0, f"Min index {edge_index.min()} < 0"
        assert (edge_index[0] != edge_index[1]).all(), "Self-loops found after cleaning"
        assert graph.edge_index.shape[1] == graph.edge_attr.shape[0], "Edge index and attr shapes don't match"
        
        return graph, invalid_count
        
    except Exception as e:
        print(f"Error in clean_graph_indices: {str(e)}")
        print(f"Shapes at error:")
        print(f"Edge index: {edge_index.shape}")
        if edge_attr is not None:
            print(f"Edge attr: {edge_attr.shape}")
        return None, -1

def validate_graph(graph):
    """Validate graph structure allowing for the specific edge attribute pattern"""
    try:
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        
        # Print debug info
        print(f"\nValidating graph structure:")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Edge index shape: {graph.edge_index.shape}")
        print(f"Edge attr shape: {graph.edge_attr.shape}")
        print(f"Node feature shape: {graph.x.shape}")
        
        # Check basic structure
        if graph.edge_index.size(0) != 2:
            print(f"Invalid edge_index shape: {graph.edge_index.shape}")
            return False
            
        # Check for invalid indices
        if graph.edge_index.min() < 0:
            print(f"Found negative edge indices: min={graph.edge_index.min()}")
            return False
            
        if graph.edge_index.max() >= num_nodes:
            print(f"Found out-of-bounds edge indices: max={graph.edge_index.max()}, num_nodes={num_nodes}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating graph: {str(e)}")
        return False

def prepare_batch(batch_file: str, exe_dir: str, family_to_idx: Dict, 
                 static_extractor: StaticFeatureExtractor, device: str) -> Optional[DataLoader]:
    try:
        graphs = torch.load(batch_file)
        processed_graphs = []
        skipped_count = 0
        
        for graph_idx, graph in enumerate(graphs):
            try:
                print(f"\nProcessing graph {graph_idx} from {batch_file}")
                
                # Basic attribute check
                required_attrs = ['edge_attr', 'edge_index', 'x', 'sha']
                if not all(hasattr(graph, attr) for attr in required_attrs):
                    missing_attrs = [attr for attr in required_attrs if not hasattr(graph, attr)]
                    print(f"Skipping graph {graph_idx} - missing attributes: {missing_attrs}")
                    skipped_count += 1
                    continue
                
                # Strict validation
                if not validate_graph(graph):
                    print(f"Skipping graph {graph_idx} - failed validation")
                    skipped_count += 1
                    continue
                
                # Process family label
                family = getattr(graph, 'family', 'none')
                if not family or family == '':
                    family = 'none'
                if family not in family_to_idx:
                    family = 'none'
                
                # Move tensors to device
                graph.y = torch.tensor(family_to_idx[family]).to(device)
                graph.x = graph.x.to(device)
                graph.edge_index = graph.edge_index.to(device)
                
                # Handle edge attributes specifically
                graph.edge_attr = graph.edge_attr.to(device)
                
                # Extract static features
                exe_hash = graph.sha
                exe_path = os.path.join(exe_dir, f"{exe_hash}_refang.exe")
                static_features = static_extractor.extract_features(exe_path)
                if len(static_features.shape) == 1:
                    static_features = static_features.unsqueeze(0)
                graph.static_features = static_features.to(device)
                
                # Set default values for optional attributes
                if not hasattr(graph, 'malware_type'):
                    graph.malware_type = ['unknown']
                if not hasattr(graph, 'timestamp'):
                    graph.timestamp = ['2019-01-01 00:00:00 UTC']
                
                processed_graphs.append(graph)
                
            except Exception as e:
                print(f"Error processing graph {graph_idx}: {str(e)}")
                skipped_count += 1
                continue
        
        if not processed_graphs:
            print(f"No valid graphs found in batch file: {batch_file}")
            return None
            
        print(f"Batch summary:")
        print(f"Total graphs: {len(graphs)}")
        print(f"Processed: {len(processed_graphs)}")
        print(f"Skipped: {skipped_count}")
        
        return DataLoader(processed_graphs, batch_size=32, shuffle=True)
        
    except Exception as e:
        print(f"Error loading batch {batch_file}: {str(e)}")
        return None
    
def train_epoch(model: HierarchicalMalwareGNN, detector: TemporalExemplarDetector, 
                loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        embeddings, group_logits, time_pred = model(batch, batch.static_features)
        
        loss = torch.tensor(0.0, device=detector.device)
        
        for i, (anchor, family, timestamp) in enumerate(zip(embeddings, batch.y, batch.timestamp)):
            family = str(family.item())
            
            if family not in detector.family_exemplars or not detector.family_exemplars[family]:
                continue
            
            # Get positive exemplar with temporal weighting
            positives = torch.stack(detector.family_exemplars[family])
            pos_timestamps = detector.exemplar_timestamps[family]
            
            # Calculate temporal weights for positives
            current_time = datetime.datetime.strptime(timestamp.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
            pos_weights = [np.exp(-(current_time - ts).days / detector.temporal_window) 
                          for ts in pos_timestamps]
            pos_weights = torch.tensor(pos_weights, device=detector.device)
            
            # Weighted positive centroid
            positive = (positives * pos_weights.unsqueeze(1)).sum(dim=0) / pos_weights.sum()
            
            # Get negative exemplar from different group
            family_group = detector.family_to_group.get(family, None)
            
            if family_group is not None:
                neg_groups = [g for g in detector.group_exemplars.keys() 
                            if g != family_group and detector.group_exemplars[g]]
                
                if neg_groups:
                    neg_group = np.random.choice(neg_groups)
                    negatives = torch.stack(detector.group_exemplars[neg_group])
                    neg_timestamps = detector.exemplar_timestamps[f"group_{neg_group}"]
                    
                    # Calculate temporal weights for negatives
                    neg_weights = [np.exp(-(current_time - ts).days / detector.temporal_window) 
                                 for ts in neg_timestamps]
                    neg_weights = torch.tensor(neg_weights, device=detector.device)
                    
                    # Weighted negative centroid
                    negative = (negatives * neg_weights.unsqueeze(1)).sum(dim=0) / neg_weights.sum()
                    
                    # Triplet margin loss with temporal weighting
                    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=0)
                    d_neg = 1 - F.cosine_similarity(anchor, negative, dim=0)
                    loss += F.relu(d_pos - d_neg + 0.3)
        
        # Add classification and temporal prediction losses
        if hasattr(batch, 'group'):
            loss += F.cross_entropy(group_logits, batch.group)
        
        if hasattr(batch, 'time_delta'):
            loss += F.mse_loss(time_pred, batch.time_delta)
        
        if loss > 0:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Update exemplars with temporal information
        detector.update_exemplars(
            embeddings.detach(), 
            batch.y,
            batch.timestamp,
            group_logits.detach()
        )
    
    return total_loss

def evaluate(model: HierarchicalMalwareGNN, detector: TemporalExemplarDetector, 
            loader: DataLoader) -> Dict:
    model.eval()
    metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with torch.no_grad():
        for batch in loader:
            embeddings, group_logits, time_pred = model(batch, batch.static_features)
            
            for emb, true_family, timestamp in zip(embeddings, batch.y, batch.timestamp):
                true_family_str = str(true_family.item())
                current_time = datetime.datetime.strptime(
                    timestamp.replace(" UTC", ""), 
                    "%Y-%m-%d %H:%M:%S"
                )
                pred_family, confidence = detector.predict(emb, group_logits, current_time)
                
                # Update metrics as before
                if true_family_str not in detector.family_exemplars:
                    metrics['new']['total'] += 1
                    if 'new' in pred_family:
                        metrics['new']['correct'] += 1
                else:
                    metrics['known']['total'] += 1
                    if pred_family == true_family_str:
                        metrics['known']['correct'] += 1
                
                metrics[true_family_str]['total'] += 1
                if pred_family == true_family_str:
                    metrics[true_family_str]['correct'] += 1
    
    return metrics

def main():
    # Setup paths
    base_dir = Path('bodmas_batches')
    exe_dir = Path('/data/datasets/bodmas_exes/refanged_exes')
    groups_file = 'behavioral_analysis/behavioral_groups.json'
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load behavioral groups
    behavioral_groups = load_behavioral_groups(groups_file)
    
    # Initialize components with correct dimensions
    static_extractor = StaticFeatureExtractor(feature_dim=43)  # Set to actual dimension
    
    # Collect family information and create mapping
    family_to_idx = {}
    splits = ['train', 'val', 'test']
    batch_files = {split: [] for split in splits}
    
    for split in splits:
        split_dir = base_dir / split
        if split_dir.exists():
            batch_files[split] = sorted(split_dir.glob('batch_*.pt'))
            
            # Collect families from each batch
            for batch_file in batch_files[split]:
                try:
                    batch = torch.load(batch_file)
                    for graph in batch:
                        family = getattr(graph, 'family', 'none')
                        if family and family != '':
                            if family not in family_to_idx:
                                family_to_idx[family] = len(family_to_idx)
                except Exception as e:
                    logger.error(f"Error processing {batch_file}: {str(e)}")
                    continue
    
    # Initialize model with correct dimensions
    first_batch = torch.load(batch_files['train'][0])
    gnn_input_dim = first_batch[0].x.size(1)
    
    model = HierarchicalMalwareGNN(
        gnn_input_dim=gnn_input_dim,
        static_input_dim=43,  # Match actual static feature dimension
        n_behavioral_groups=len(behavioral_groups)
    ).to(device)
    
    
    detector = TemporalExemplarDetector(
        model=model,
        device=device,
        behavioral_groups=behavioral_groups,
        temporal_window=30  # 30-day window
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Training loop
    num_epochs = 5
    best_acc = 0
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        num_batches = 0
        
        train_iterator = tqdm(batch_files['train'], desc=f"Epoch {epoch+1}")
        for batch_file in train_iterator:
            loader = prepare_batch(
                batch_file, 
                exe_dir, 
                family_to_idx,
                static_extractor,
                device
            )
            
            if not loader:
                continue
            
            loss = train_epoch(model, detector, loader, optimizer)
            total_loss += loss
            num_batches += 1
            train_iterator.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Validate
        val_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for batch_file in batch_files['val']:
            loader = prepare_batch(
                batch_file,
                exe_dir,
                family_to_idx,
                static_extractor,
                device
            )
            
            if not loader:
                continue
            
            batch_metrics = evaluate(model, detector, loader)
            
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'family_exemplars': detector.family_exemplars,
                'group_exemplars': detector.group_exemplars,
                'accuracy': combined_acc,
                'epoch': epoch
            }, save_dir / 'best_model.pt')
            logger.info(f"Saved new best model with accuracy {combined_acc:.2%}")
    
    logger.info("Training complete!")

if __name__ == '__main__':
    main()