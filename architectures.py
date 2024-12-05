import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import numpy as np
from torch_geometric.nn import BatchNorm


# # -

# class CentroidLayer(torch.nn.Module):
#   def __init__(self, input_dim, n_classes, n_centroids_per_class=None, ac_std_lim=5.0, reject_input=False, **kwargs):

#     super().__init__(**kwargs)

#     self.n_classes = n_classes
#     self.n = n_centroids_per_class or 1
#     self.input_dim = int(input_dim)
    
#     self.centroids = torch.nn.Parameter(torch.randn(self.n_classes, self.n, self.input_dim))
#     self.std_scale = torch.nn.Parameter(torch.tensor(1.0))
#     self.ac_temp = torch.nn.Parameter(torch.tensor(1.0))

#     running_mean = torch.tensor(torch.tensor(1.0))
#     running_var = torch.tensor(torch.tensor(0.0))
#     ac_std_lim = torch.tensor(torch.tensor(ac_std_lim))
    
#     self.register_buffer('running_mean', running_mean)
#     self.register_buffer('running_var', running_var)
#     self.register_buffer('ac_std_lim', ac_std_lim)

#     self.reject_input = reject_input
#     self.relu = torch.nn.ReLU()
  
#   def forward(self, x):

#       # This has shape (batch_size, n_classes, centroids_per_class)
#       # Note: the [None] notation adds an extra dimension that we can
#       #    broadcast over.

#       dist_to_centroids = torch.sqrt(
#           torch.sum((self.centroids[None] - x[:, None, None])**2,
#                         dim=-1))

#       # This is the min distance to class centroids and has shape (batch_size, n_classes).
#       dist, _ = torch.min(dist_to_centroids, dim=2)
#       y = -dist

#       if self.reject_input:

#         if self.training:
#           mean_dist = torch.mean(torch.min(dist, dim=1)[0]).detach()
#           var_dist = torch.var(torch.min(dist, dim=1)[0]).detach()
#           self.running_mean = self.running_mean * 0.9 + mean_dist * 0.1
#           self.running_var = self.running_var * 0.9 + var_dist * 0.1

#         max_ac_dist = self.running_mean + torch.clip(self.relu(self.std_scale), min=0., max=self.ac_std_lim) * torch.sqrt(self.running_var)

#         # we accept if the distance is smaller than the max_ac_dist
#         # that is, if max_ac_dist - dist > 0, accept(x) = 1
#         accept_score = max_ac_dist - torch.min(dist, dim=1, keepdims=True)[0].detach()
#         soft_accept_score = accept_score / self.ac_temp
#         soft_accept_score = soft_accept_score.sigmoid()


#         return torch.cat([y, soft_accept_score], dim=1)

#       return y

class CentroidLayer(torch.nn.Module):
    """
    Neurosymbolic centroid-based classification layer.
    Instead of learning a linear transformation, it learns prototype vectors (centroids)
    for each class and classifies based on distance to these prototypes.
    """
    def __init__(self, input_dim, n_classes, n_centroids_per_class=3, distance_temp=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_centroids_per_class = n_centroids_per_class
        self.distance_temp = distance_temp  # Temperature for softening distance calculations
        
        # Initialize centroids matrix: (n_classes * n_centroids_per_class, input_dim)
        # Each class gets multiple centroids to capture different variations
        self.centroids = torch.nn.Parameter(
            torch.randn(n_classes * n_centroids_per_class, input_dim) / np.sqrt(input_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings of shape (batch_size, input_dim)
        Returns:
            logits: Classification logits of shape (batch_size, n_classes)
        """
        batch_size = x.size(0)
        
        # Compute pairwise distances between inputs and all centroids
        # Expand dimensions to enable broadcasting
        expanded_x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        expanded_centroids = self.centroids.unsqueeze(0)  # (1, n_classes * n_centroids, input_dim)
        
        # Compute squared Euclidean distance
        squared_diff = (expanded_x - expanded_centroids) ** 2  # (batch_size, n_classes * n_centroids, input_dim)
        distances = torch.sum(squared_diff, dim=2)  # (batch_size, n_classes * n_centroids)
        
        # Reshape distances to group centroids by class
        distances = distances.view(batch_size, self.n_classes, self.n_centroids_per_class)
        
        # For each class, use the minimum distance to any of its centroids
        # This implements a "closest prototype" rule per class
        min_class_distances, _ = torch.min(distances, dim=2)  # (batch_size, n_classes)
        
        # Convert distances to logits using negative distance and temperature
        # Lower distance = higher logit
        logits = -min_class_distances / self.distance_temp
        
        return logits
    
    def get_centroids_by_class(self):
        """
        Returns the centroids grouped by class for interpretation.
        """
        return self.centroids.view(self.n_classes, self.n_centroids_per_class, self.input_dim)
    
    def get_nearest_training_samples(self, embeddings, labels, k=5):
        """
        Find training samples closest to each centroid for interpretation.
        
        Args:
            embeddings: Training sample embeddings (n_samples, input_dim)
            labels: True labels for training samples (n_samples,)
            k: Number of nearest neighbors to return per centroid
        Returns:
            Dictionary mapping (class_idx, centroid_idx) to list of k nearest sample indices
        """
        nearest_samples = {}
        
        # For each class and its centroids
        for class_idx in range(self.n_classes):
            class_mask = (labels == class_idx)
            if not torch.any(class_mask):
                continue
                
            class_embeddings = embeddings[class_mask]
            class_indices = torch.where(class_mask)[0]
            
            # Get this class's centroids
            start_idx = class_idx * self.n_centroids_per_class
            end_idx = start_idx + self.n_centroids_per_class
            class_centroids = self.centroids[start_idx:end_idx]
            
            # For each centroid of this class
            for centroid_idx, centroid in enumerate(class_centroids):
                # Compute distances to all samples of this class
                distances = torch.norm(class_embeddings - centroid.unsqueeze(0), dim=1)
                
                # Get k nearest samples
                k_distances, k_indices = torch.topk(distances, min(k, len(distances)), largest=False)
                nearest_samples[(class_idx, centroid_idx)] = {
                    'indices': class_indices[k_indices].cpu().numpy(),
                    'distances': k_distances.cpu().numpy()
                }
        
        return nearest_samples


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
