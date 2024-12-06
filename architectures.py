import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import numpy as np
from torch_geometric.nn import BatchNorm
import logging
# logger
import os 
logger = logging.getLogger(__name__)
import sys 
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)



# class NoveltyEvaluator:
#     """Evaluate model performance including goodware detection."""
#     def __init__(self, known_families):
#         self.known_families = set(known_families)
#         self.results = {
#             'true_positives': 0,  # Correctly identified new malware families
#             'false_positives': 0,  # Known malware/goodware marked as new
#             'true_negatives': 0,  # Known samples correctly classified
#             'false_negatives': 0,  # New families missed
#             'goodware_correct': 0,  # Correctly identified goodware
#             'goodware_incorrect': 0,  # Goodware misclassified as malware
#             'malware_as_goodware': 0,  # Malware misclassified as goodware
#             'novel_families_found': set(),
#             'misclassified_known': defaultdict(int)
#         }
    
#     def update(self, predictions, true_families):
#         """Update metrics including goodware classification."""
#         for pred, true_family in zip(predictions, true_families):
#             is_actually_goodware = true_family in ['none', 'goodware']
#             is_actually_new = not is_actually_goodware and true_family not in self.known_families
            
#             was_flagged_goodware = pred['is_goodware'].item()
#             was_flagged_new = pred['is_novel'].item()
            
#             if is_actually_goodware:
#                 if was_flagged_goodware:
#                     self.results['goodware_correct'] += 1
#                 else:
#                     self.results['goodware_incorrect'] += 1
#             else:  # Malware sample
#                 if was_flagged_goodware:
#                     self.results['malware_as_goodware'] += 1
#                 elif is_actually_new:
#                     if was_flagged_new:
#                         self.results['true_positives'] += 1
#                         self.results['novel_families_found'].add(true_family)
#                     else:
#                         self.results['false_negatives'] += 1
#                 else:
#                     if was_flagged_new:
#                         self.results['false_positives'] += 1
#                         self.results['misclassified_known'][true_family] += 1
#                     else:
#                         self.results['true_negatives'] += 1
    
#     def get_metrics(self):
#         """Calculate comprehensive metrics."""
#         # Novelty detection metrics
#         tp = self.results['true_positives']
#         fp = self.results['false_positives']
#         tn = self.results['true_negatives']
#         fn = self.results['false_negatives']
        
#         # Goodware classification metrics
#         gc = self.results['goodware_correct']
#         gi = self.results['goodware_incorrect']
#         mg = self.results['malware_as_goodware']
        
#         novelty_metrics = {
#             'novelty_precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
#             'novelty_recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
#             'novel_families_found': list(self.results['novel_families_found'])
#         }
        
#         goodware_metrics = {
#             'goodware_accuracy': gc / (gc + gi) if (gc + gi) > 0 else 0,
#             'goodware_false_positive_rate': mg / (mg + tn) if (mg + tn) > 0 else 0,
#         }
        
#         return {
#             **novelty_metrics,
#             **goodware_metrics,
#             'most_confused_known': dict(sorted(
#                 self.results['misclassified_known'].items(),
#                 key=lambda x: x[1],
#                 reverse=True
#             )[:5])
#         }

class PhasedGNN(nn.Module):
    def __init__(self, num_node_features: int, num_families: int, hidden_dim: int = 64):
        super().__init__()
        self.num_families = num_families
        
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Centroid-based classification
        self.centroid_layer = CentroidClassifier(
            hidden_dim,
            num_families,
            goodware_centroids=5,
            centroids_per_family=3
        )
        
        self.current_phase = 'family'  # ['family', 'goodware', 'novelty']

    def encode(self, x, edge_index, batch):
        """Get graph embeddings"""
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.bn3(self.conv3(x, edge_index))
        
        # Global pooling
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding

    def forward(self, data):
        # Get graph embeddings
        embeddings = self.encode(data.x, data.edge_index, data.batch)
        
        # Get predictions based on current phase
        if self.current_phase == 'family':
            # Regular family classification
            logits = self.centroid_layer(embeddings)
            return logits
            
        elif self.current_phase == 'goodware':
            # Binary goodware vs malware classification
            logits = self.centroid_layer(embeddings)
            goodware_logits = logits[:, 0]  # First centroid is goodware
            malware_logits = torch.logsumexp(logits[:, 1:], dim=1)  # Combine all malware families
            return torch.stack([goodware_logits, malware_logits], dim=1)
            
        else:  # novelty phase
            # Full classification plus novelty detection
            logits, novelty_scores = self.centroid_layer(embeddings, return_novelty=True)
            return logits, novelty_scores

    def set_phase(self, phase):
        assert phase in ['family', 'goodware', 'novelty']
        self.current_phase = phase

class CentroidClassifier(nn.Module):
    def __init__(self, embed_dim, num_families, goodware_centroids=5, centroids_per_family=3):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_families = num_families
        self.centroids_per_family = centroids_per_family
        
        # Initialize centroids for malware families
        self.malware_centroids = nn.Parameter(
            torch.randn(num_families * centroids_per_family, embed_dim) / np.sqrt(embed_dim)
        )
        
        # Separate centroids for goodware
        self.goodware_centroids = nn.Parameter(
            torch.randn(goodware_centroids, embed_dim) / np.sqrt(embed_dim)
        )
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor([1.0]))
        
        # Learnable novelty threshold
        self.novelty_threshold = nn.Parameter(torch.tensor([10.0]))

    def forward(self, x, return_novelty=False):
        batch_size = x.size(0)
        
        # Calculate distances
        goodware_dists = self._compute_distances(x, self.goodware_centroids)
        malware_dists = self._compute_distances(x, self.malware_centroids)
        
        # Reshape malware distances by family
        malware_dists = malware_dists.view(
            batch_size, self.num_families, self.centroids_per_family
        )
        
        # Get minimum distance to each family
        min_goodware_dist = goodware_dists.min(dim=1)[0]
        min_malware_dists = malware_dists.min(dim=2)[0]
        
        # Convert distances to logits
        goodware_logits = -min_goodware_dist / self.temperature
        malware_logits = -min_malware_dists / self.temperature
        
        # Combine logits
        logits = torch.cat([
            goodware_logits.unsqueeze(1),
            malware_logits
        ], dim=1)
        
        if return_novelty:
            # Calculate novelty scores
            all_dists = torch.cat([min_goodware_dist.unsqueeze(1), min_malware_dists], dim=1)
            min_dists = all_dists.min(dim=1)[0]
            novelty_scores = min_dists / self.novelty_threshold
            return logits, novelty_scores
            
        return logits

    def _compute_distances(self, x, centroids):
        """Compute squared Euclidean distances between samples and centroids"""
        x_expanded = x.unsqueeze(1)
        centroids_expanded = centroids.unsqueeze(0)
        return torch.sum((x_expanded - centroids_expanded) ** 2, dim=2)

class PhasedTraining:
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_batch(self, batch, phase):
        self.model.train()
        self.model.set_phase(phase)
        self.optimizer.zero_grad()
        
        if phase == 'family':
            # Standard family classification
            logits = self.model(batch)
            loss = F.cross_entropy(logits, batch.y)
            
        elif phase == 'goodware':
            # Binary classification
            logits = self.model(batch)
            is_goodware = (batch.y == 0).float()
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], is_goodware)
            
        else:  # novelty phase
            logits, novelty_scores = self.model(batch)
            # Classification loss
            ce_loss = F.cross_entropy(logits, batch.y)
            # Novelty loss
            is_novel = (batch.y >= self.model.num_families).float()
            novelty_loss = F.binary_cross_entropy_with_logits(novelty_scores, is_novel)
            loss = ce_loss + novelty_loss
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    @torch.no_grad()
    def evaluate(self, batch, phase):
        self.model.eval()
        self.model.set_phase(phase)
        
        if phase == 'family':
            # Same as before
            logits = self.model(batch)
            preds = logits.argmax(dim=1)
            precision_correct = (preds == batch.y).sum().item()
            total_predictions = len(preds)
            classes = torch.unique(batch.y)
            recalls = []
            for c in classes:
                class_mask = batch.y == c
                class_total = class_mask.sum().item()
                class_correct = (preds[class_mask] == c).sum().item()
                recalls.append(class_correct / class_total if class_total > 0 else 0)
            return precision_correct, total_predictions, sum(recalls) / len(recalls)
                        
        elif phase == 'goodware':
            # Load mapping file to get none_idx
            with open('/data/saranyav/gcn_new/bodmas_batches_test/family_mapping.json', 'r') as f:
                mapping = json.load(f)
                none_idx = mapping['family_to_idx']['none']
                
            logits = self.model(batch)
            is_goodware = (batch.y == none_idx)
            preds = logits[:, 0] > 0
            
            precision_correct = (preds & is_goodware).sum().item()
            total_predictions = preds.sum().item()
            goodware_correct = (preds & is_goodware).sum().item()
            total_goodware = is_goodware.sum().item()
            recall = goodware_correct / total_goodware if total_goodware > 0 else 0
            return precision_correct, total_predictions, recall
            
        else:  # novelty phase
            embeddings = self.model.get_embedding(batch)
            distances = self.model.get_distances(embeddings)  # Distance to nearest centroid
            is_novel = distances > self.model.epsilon  # Beyond threshold = novel
            
            # Something really is novel if it's outside our known families
            truly_novel = batch.y >= self.model.num_families
            
            precision_correct = (is_novel & truly_novel).sum().item()
            total_predictions = is_novel.sum().item()
            recall = precision_correct / truly_novel.sum().item() if truly_novel.sum().item() > 0 else 0
            
            return precision_correct, total_predictions, recall

# sys.exit(0) 
# class PhasedTraining:
#     """Handles phased training for malware GNN"""
#     def __init__(self, model, device, phases=['basic', 'goodware', 'novelty'], epochs_per_phase=30):
#         self.model = model
#         self.device = device
#         self.phases = phases
#         self.epochs_per_phase = epochs_per_phase
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
#         # Map phase names to functions
#         self.phase_functions = {
#             'basic': self._train_basic,
#             'goodware': self._train_goodware,
#             'novelty': self._train_novelty
#         }
        
#     def train_batch(self, phase, batch):
#         """Train a single batch using the appropriate phase function"""
#         self.model.train()
#         loss = self.phase_functions[phase](batch)
        
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         return loss.item()
        
#     def validate_batch(self, phase, batch):
#         """Validate a single batch"""
#         self.model.eval()
#         with torch.no_grad():
#             if phase == 'basic':
#                 # Basic family classification
#                 logits = self.model(batch)
#                 pred = logits.argmax(dim=1)
#                 correct = (pred == batch.y).sum().item()
#                 return correct, batch.y.size(0)
                
#             elif phase == 'goodware':
#                 # Binary goodware vs malware
#                 logits = self.model(batch)
#                 pred = (F.softmax(logits, dim=1)[:, 0] > 0.5).float()  # goodware probability
#                 is_goodware = (batch.y == 0).float()
#                 correct = (pred == is_goodware).sum().item()
#                 return correct, batch.y.size(0)
                
#             else:  # novelty phase
#                 logits, distances = self.model(batch)
#                 predictions = self.model.centroid.predict_with_novelty(
#                     self.model.get_embedding(batch)
#                 )
#                 # Consider prediction correct if:
#                 # - Correctly identified as novel
#                 # - OR correctly classified if not novel
#                 novel_mask = batch.y >= self.model.num_classes - 1
#                 correct = ((predictions['is_novel'] & novel_mask) | 
#                           ((~predictions['is_novel']) & (~novel_mask) & 
#                            (predictions['predictions'] == batch.y))).sum().item()
#                 return correct, batch.y.size(0)

#     def _train_basic(self, batch):
#         """Basic malware family classification"""
#         logits = self.model(batch)
#         return F.cross_entropy(logits, batch.y)
        
#     def _train_goodware(self, batch):
#         """Train to distinguish goodware vs malware"""
#         logits = self.model(batch)
#         is_goodware = (batch.y == 0).float()
#         return F.binary_cross_entropy_with_logits(
#             logits[:, 0],  # goodware logit
#             is_goodware
#         )
        
#     def _train_novelty(self, batch):
#         """Train with novelty detection"""
#         logits, distances = self.model(batch)
#         # Regular classification loss
#         classification_loss = F.cross_entropy(logits, batch.y)
        
#         # Novelty detection loss
#         predictions = self.model.centroid.predict_with_novelty(
#             self.model.get_embedding(batch)
#         )
#         novel_mask = (batch.y >= self.model.num_classes - 1).float()
#         novelty_loss = F.binary_cross_entropy(
#             predictions['novelty_scores'],
#             novel_mask
#         )
        
#         return classification_loss + novelty_loss
    
# class CentroidLayer(torch.nn.Module):
#     def __init__(self, input_dim, n_classes, n_centroids_per_class=3, 
#                  goodware_centroids=5, temperature=1.0, novelty_threshold=None):
#         super().__init__()
#         self.input_dim = input_dim
#         self.n_classes = n_classes  # Total number of classes including goodware
#         self.n_malware_classes = n_classes - 1  # Number of malware classes
#         self.n_centroids_per_class = n_centroids_per_class
#         self.goodware_centroids = goodware_centroids
#         self.temperature = temperature
        
#         # Initialize centroids for malware families (n_classes - 1 because one class is goodware)
#         self.malware_centroids = torch.nn.Parameter(
#             torch.randn(self.n_malware_classes * n_centroids_per_class, input_dim) / np.sqrt(input_dim)
#         )
        
#         # Separate centroids for goodware
#         self.goodware_centroids_param = torch.nn.Parameter(
#             torch.randn(goodware_centroids, input_dim) / np.sqrt(input_dim)
#         )
        
#         if novelty_threshold is None:
#             self.register_buffer('novelty_threshold', torch.tensor(float('inf')))
#         else:
#             self.register_buffer('novelty_threshold', torch.tensor(novelty_threshold))
    
#     def forward(self, x, return_distances=False):
#         batch_size = x.size(0)
        
#         # Expand dimensions for broadcasting
#         expanded_x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
#         expanded_malware_centroids = self.malware_centroids.unsqueeze(0)  # [1, n_mal_centroids, input_dim]
#         expanded_goodware_centroids = self.goodware_centroids_param.unsqueeze(0)  # [1, goodware_centroids, input_dim]
        
#         # Compute distances
#         malware_distances = torch.sum((expanded_x - expanded_malware_centroids) ** 2, dim=2)
#         goodware_distances = torch.sum((expanded_x - expanded_goodware_centroids) ** 2, dim=2)
        
#         # Reshape malware distances - now using n_malware_classes
#         malware_distances = malware_distances.view(batch_size, self.n_malware_classes, self.n_centroids_per_class)
        
#         # Get minimum distances
#         min_malware_distances = torch.min(malware_distances, dim=2)[0]  # [batch_size, n_malware_classes]
#         min_goodware_distances = torch.min(goodware_distances, dim=1)[0]  # [batch_size]
        
#         # Convert to logits
#         malware_logits = -min_malware_distances / self.temperature
#         goodware_logits = -min_goodware_distances / self.temperature
        
#         # Calculate novelty scores
#         all_min_distances = torch.cat([
#             min_goodware_distances.unsqueeze(1),
#             min_malware_distances
#         ], dim=1)  # [batch_size, n_classes]
        
#         min_overall_distances = torch.min(all_min_distances, dim=1)[0]
#         novelty_logits = -(min_overall_distances / self.novelty_threshold).unsqueeze(1)
        
#         # Combine logits: [goodware, malware_families, novel]
#         logits = torch.cat([
#             goodware_logits.unsqueeze(1),  # [batch_size, 1]
#             malware_logits,                # [batch_size, n_malware_classes]
#             novelty_logits                 # [batch_size, 1]
#         ], dim=1)
        
#         if return_distances:
#             return logits, {
#                 'malware_distances': min_malware_distances,
#                 'goodware_distances': min_goodware_distances,
#                 'min_overall_distances': min_overall_distances,
#                 'novelty_scores': -novelty_logits.squeeze(1)
#             }
        
#         return logits
    
#     def predict_with_novelty(self, x):
#         """Make predictions including goodware and novelty detection."""
#         logits, distances = self.forward(x, return_distances=True)
#         predictions = logits.argmax(dim=1)
#         novelty_scores = distances['novelty_scores']
        
#         is_goodware = predictions == 0
#         is_novel = (novelty_scores > 1.0) | (predictions == self.n_classes)
        
#         return {
#             'predictions': predictions,
#             'is_goodware': is_goodware,
#             'is_novel': is_novel,
#             'novelty_scores': novelty_scores,
#             'goodware_confidence': torch.softmax(logits, dim=1)[:, 0],
#             'distances': distances
#         }


class SimpleGCN(torch.nn.Module):

    def __init__(self, *, num_features, hidden_channels, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels * 2, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)],
                      dim=-1)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)

        return x


# +

class SelectiveCentroidLayer(nn.Module):
    def __init__(self, input_dim, n_classes, n=None, norm='l2', lc=1, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.n = 1 if n is None else int(n)
        self._norm = norm
        self._lc = lc
        self._epsilon = epsilon
        # Check norm compatibility.
        if norm != 'l2':
            # For now we only support the l2 norm.
            raise ValueError(f'{norm} norm not supported')
        #remove the build in tensorflow
        self.centroids = nn.Parameter(torch.Tensor(self.n_classes, self.n, input_dim).uniform_(-1, 1))
        self.rejective_centroids = nn.Parameter(torch.Tensor(self.n_classes * self.n, input_dim).uniform_(-1, 1))
        self.rejective_centroids.requires_grad = False
        class_centroid_norm = torch.max(torch.norm(self.centroids, p=2, dim=-1))
        rejective_centroid_norms = torch.norm(self.rejective_centroids, p=2, dim=-1, keepdim=True) + 1e-16
        # initialize rejective centroids outside the class centroids.
        self.rejective_centroids.data = 2.0 * class_centroid_norm * self.rejective_centroids.data / rejective_centroid_norms
        self.centroid_norm = nn.Parameter(torch.ones(1), requires_grad=False)
        self.std_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.ac_temp = nn.Parameter(torch.ones(1), requires_grad=False)
        self.running_mean = nn.Parameter(torch.ones(1), requires_grad=False)
        self.running_var = nn.Parameter(torch.zeros(0), requires_grad=False)
        self.closest_centroid = None
    def get_closest_centroid(self):
        return self.closest_centroid
    @property
    def norm(self):
        return self._norm
    @property
    def frozen(self):
        return self._lc_frozen
    def forward(self, x, return_dists=False, training=True):
        # Calculate distance from x to each centroid.
        self.centroid_norm.data = torch.max(torch.norm(self.centroids, dim=2, p=2))
        centroids = self.centroids / (self.centroid_norm + 1e-16)
        # This has shape (batch_size, n_classes, centroids_per_class)
        dist_to_centroids = torch.sqrt(
            torch.sum((centroids[None] - x[:, None, None])**2,
                      dim=-1))
        # This has shape (batch_size, n_classes).
        min_dist_to_class_centroid = torch.min(dist_to_centroids, dim=2).values
        # Now we want to find the closest centroid for each point.
        total_centroids = self.n_classes * self.n
        # This has shape (batch_size, total_centroids).
        closest_centroid_mask = torch.nn.functional.one_hot(
            torch.argmin(dist_to_centroids.view(-1, total_centroids),
                         dim=1),total_centroids)
        # This has shape (batch_size, centroid_dim).
        closest_centroid = torch.sum(
                closest_centroid_mask[:, :, None] *
                torch.reshape(centroids, (total_centroids, -1))[None],
                axis=1)
        self.closest_centroid = closest_centroid_mask
        # Now we compute the distance to rejective_centroids
        rejective_centroids = self.rejective_centroids
        dist_to_rej_centroids = torch.min(torch.sqrt(
            torch.sum((rejective_centroids[None] - x[:, None])**2,
                      dim=-1)), dim=-1).values
        # Now we want the distance to the boundary between the closest centroid
        # and each other rejective centroid.
        # Hyperplanes are of the form w @ x + b = 0. The normal vector w is in
        # the direction of the line connecting our closest centroid with another
        # centroid. We select b such that the hyperplane bisects this line.
        # I.e., if x_0 is a point in the hyperplane, then b = -(w @ x_0).
        # This has shape (batch_size, n_classes * n_centroids, centroid_dim).
        w = closest_centroid[:, None] - rejective_centroids[None]
        bisecting_point = (closest_centroid[:, None] +
                           rejective_centroids[None]) / 2
        b = -torch.sum(w * bisecting_point, dim=-1)
        # calculate hyperplane distances 
        hyperplane_distances = (
            torch.abs(torch.sum(w * x[:, None], dim=-1) +
                      b)) / (torch.sqrt(torch.sum(w**2, dim=-1)) + 1e-9)
        # calculate the smallest distance from the current point to the hyperplane 
        # between the current centroid and all rejective centroids
        rej_hyperplane_distances = torch.min(hyperplane_distances, dim=-1).values
        # assign logits according to centroid distance for each class
        y = -min_dist_to_class_centroid
        min_dist = torch.min(min_dist_to_class_centroid, dim=1).values
        removed_min_dist_to_class_centroid = torch.where(
            min_dist_to_class_centroid == min_dist[:, None], 
            torch.zeros_like(min_dist_to_class_centroid) + np.inf, 
            min_dist_to_class_centroid)
        second_min_dist = torch.min(removed_min_dist_to_class_centroid, dim=1).values
        # accept if min_dist < (1-self._epsilon)/(1+self._epsilon) * second_min_dist
        y_bot = -(1-self._epsilon)/(1+self._epsilon) * second_min_dist
        if return_dists:
            dists = rej_hyperplane_distances
            return torch.cat([y, dists[:, None]], dim=1)
        # we accept if smallest distance to class centroids is smaller than 
        # the smallest distance to rejective centroids
        # i.e dist_to_rej_centroids > closest_centroid_dist
        accept_score = min_dist - self._epsilon * (min_dist + second_min_dist)
        soft_accept_score = torch.sigmoid(accept_score / self.ac_temp)
        return torch.cat([y, y_bot[:, None], soft_accept_score[:, None]], dim=1)
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_classes': self.n_classes,
            'n': self.n,
            'norm': self._norm
        })
        return config
        


class GCNCentroid(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, epsilon = 0.1):
        super(GCNCentroid, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #self.centroid = CentroidLayer(2 * hidden_channels, num_classes, n_centroids_per_class=2, 
        #                              epsilon = epsilon, ac_std_lim=ac_std_lim, reject_input=reject_input)
        self.centroid = SelectiveCentroidLayer(2 * hidden_channels, num_classes, epsilon = epsilon)
    def get_closest_centroid(self):
        return self.centroid.closest_centroid
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.batchnorm1(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.batchnorm2(x)
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)],
                      dim=-1)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.centroid(x)
        return x, self.centroid.get_closest_centroid()



class MalwareGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64):
        super().__init__()
        self.num_classes = num_classes
        
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Centroid layer
        self.centroid = CentroidLayer(
            input_dim=hidden_dim,
            n_classes=num_classes,
            n_centroids_per_class=3
        )
    
    def get_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        
        x = global_mean_pool(x, batch)
        return x
    
    def forward(self, data):
        embedding = self.get_embedding(data)
        return self.centroid(embedding)
    


