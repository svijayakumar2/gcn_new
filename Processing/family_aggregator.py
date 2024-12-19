import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

# Set up logging similar to your temporal.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BehavioralSimilarityComputer:
    """Compute similarity between malware family behavioral profiles."""
    
    def __init__(self, family_distributions: Dict):
        self.family_distributions = family_distributions
        
    def compute_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """
        Compute similarity score between two family behavioral profiles using
        Jensen-Shannon divergence for distributions and cosine similarity for patterns.
        Returns a score between 0 (completely different) and 1 (identical).
        """
        # Compute similarities for each component
        similarities = [
            self._compare_feature_distributions(
                profile1['feature_stats'], 
                profile2['feature_stats']
            ),
            self._compare_behavior_patterns(
                profile1['behavior_patterns'], 
                profile2['behavior_patterns']
            ),
            self._compare_graph_characteristics(
                profile1['graph_characteristics'], 
                profile2['graph_characteristics']
            )
        ]
        
        # Return average similarity
        return np.mean([s for s in similarities if s is not None])
    
    def _compare_feature_distributions(self, stats1: Dict, stats2: Dict) -> Optional[float]:
        """Compare feature distributions using Jensen-Shannon divergence."""
        if not stats1 or not stats2:
            return None
            
        feature_sims = []
        for feature in stats1:
            if feature not in stats2:
                continue
                
            # Create probability distributions from feature stats
            # Use histogram data if available, otherwise create simple distributions
            # from mean and std using normal distribution approximation
            bins = 50
            range_max = max(stats1[feature]['max_val'], stats2[feature]['max_val'])
            if range_max == 0:
                feature_sims.append(1.0)  # Both distributions are zero
                continue
                
            # Create distributions using mean and std
            x = np.linspace(0, range_max, bins)
            p1 = self._create_distribution(x, stats1[feature])
            p2 = self._create_distribution(x, stats2[feature])
            
            # Compute Jensen-Shannon divergence
            js_div = self._jensen_shannon_divergence(p1, p2)
            # Convert to similarity score (1 - normalized divergence)
            feature_sims.append(1 - js_div)
        
        return np.mean(feature_sims) if feature_sims else None
        
    def _create_distribution(self, x: np.ndarray, stats: Dict) -> np.ndarray:
        """Create a probability distribution from feature statistics."""
        # Create normal distribution using mean and std
        mean, std = stats['mean'], stats['std']
        if std == 0:
            std = mean * 0.1 if mean != 0 else 0.1  # Avoid zero std
            
        dist = np.exp(-0.5 * ((x - mean) / std) ** 2)
        # Add small epsilon to avoid division by zero
        dist_sum = np.sum(dist)
        if dist_sum == 0:
            return np.ones_like(dist) / len(dist)  # Uniform distribution if all zeros
        
        dist = dist / dist_sum  # Normalize
        
        # Incorporate density information
        density = stats['density']
        dist = dist * density + (1 - density) * (dist == 0)
        dist_sum = np.sum(dist)
        if dist_sum == 0:
            return np.ones_like(dist) / len(dist)
        return dist / dist_sum  # Renormalize
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        m = 0.5 * (p + q)
        js_div = 0.5 * (
            np.sum(p * np.log2(p / m)) + 
            np.sum(q * np.log2(q / m))
        )
        
        # Normalize to [0,1]
        return js_div / np.log2(2)
    
    def _compare_behavior_patterns(self, patterns1: Dict, patterns2: Dict) -> Optional[float]:
        """Compare behavioral patterns using cosine similarity."""
        if not patterns1 or not patterns2:
            return None
            
        # Get all unique patterns
        all_patterns = sorted(set(patterns1.keys()) | set(patterns2.keys()))
        if not all_patterns:
            return None
            
        # Create feature vectors
        vec1 = np.array([patterns1.get(p, 0) for p in all_patterns])
        vec2 = np.array([patterns2.get(p, 0) for p in all_patterns])
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 and norm2 == 0:
            return 1.0  # Both vectors are zero - consider them similar
        elif norm1 == 0 or norm2 == 0:
            return 0.0  # One vector is zero, one isn't - consider them different
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _compare_graph_characteristics(self, chars1: Dict, chars2: Dict) -> Optional[float]:
        """Compare graph characteristics using cosine similarity."""
        if not chars1 or not chars2:
            return None
            
        # Get all characteristics
        all_chars = sorted(set(chars1.keys()) | set(chars2.keys()))
        if not all_chars:
            return None
            
        # Create feature vectors
        vec1 = np.array([chars1.get(c, 0) for c in all_chars])
        vec2 = np.array([chars2.get(c, 0) for c in all_chars])
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 and norm2 == 0:
            return 1.0
        elif norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
class MalwareBehaviorAggregator:
    """Aggregate behavioral patterns across malware families."""
    
    def __init__(self, batch_dir: Path):
        self.batch_dir = Path(batch_dir)
        self.family_graphs = defaultdict(list)
        self.family_distributions = {}

    def _aggregate_family_behaviors(self, pyg_graphs: List) -> Dict:
        """
        Aggregate behavioral features directly from PyG graphs.
        
        Args:
            pyg_graphs: List of PyG graphs
            
        Returns:
            Dictionary of graph-level behavioral signatures
        """
        feature_names = [
            'mem_ops', 'calls', 'instructions', 'stack_ops', 'reg_writes',
            'external_calls', 'internal_calls', 'mem_reads', 'mem_writes',
            'in_degree', 'out_degree', 'is_conditional', 'has_jump', 'has_ret'
        ]
        
        # Initialize with regular dictionaries instead of defaultdict
        family_features = {
            'feature_distributions': {},
            'node_behavior_patterns': {
                'external_calling': 0,
                'external_with_write': 0,
                'conditional_jump': 0,
                'memory_rw': 0
            },
            'graph_signatures': []
        }
        
        # Initialize feature distributions
        for feature in feature_names:
            family_features['feature_distributions'][feature] = []
        
        for graph in pyg_graphs:
            # Work with node features directly from PyG
            node_features = graph.x.numpy()  # Convert to numpy for easier handling
            
            # 1. Calculate distributions for each feature
            for feat_idx, feature in enumerate(feature_names):
                values = node_features[:, feat_idx]
                if len(values) > 0:
                    family_features['feature_distributions'][feature].append({
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'max': float(np.max(values)),
                        'density': float(len(values[values > 0])) / len(values)
                    })
            
            # 2. Extract behavioral node patterns
            # Use column indices for efficiency
            external_calls = node_features[:, feature_names.index('external_calls')]
            mem_writes = node_features[:, feature_names.index('mem_writes')]
            mem_reads = node_features[:, feature_names.index('mem_reads')]
            is_conditional = node_features[:, feature_names.index('is_conditional')]
            has_jump = node_features[:, feature_names.index('has_jump')]
            
            # Count behavioral patterns
            family_features['node_behavior_patterns']['external_calling'] += int(np.sum(external_calls > 0))
            family_features['node_behavior_patterns']['external_with_write'] += int(np.sum(
                (external_calls > 0) & (mem_writes > 0)
            ))
            family_features['node_behavior_patterns']['conditional_jump'] += int(np.sum(
                (is_conditional > 0) & (has_jump > 0)
            ))
            family_features['node_behavior_patterns']['memory_rw'] += int(np.sum(
                (mem_reads > 0) & (mem_writes > 0)
            ))
            
            # 3. Graph-level signatures
            graph_sig = {
                'size': len(node_features),
                'active_nodes': int(np.sum(np.any(node_features > 0, axis=1))),
                'api_density': float(np.sum(external_calls)) / len(node_features),
                'control_flow_complexity': float(np.sum(is_conditional + has_jump)) / len(node_features)
            }
            family_features['graph_signatures'].append(graph_sig)
        
        # Skip families with no valid data
        if not any(family_features['feature_distributions'].values()):
            return None
        
        # Normalize behavior patterns by total number of nodes across all graphs
        total_nodes = sum(len(g.x) for g in pyg_graphs)
        normalized_patterns = {
            pattern: float(count) / total_nodes
            for pattern, count in family_features['node_behavior_patterns'].items()
        }
        
        # Aggregate final distributions
        aggregated = {
            # Average distributions for each feature
            'feature_stats': {
                feature: {
                    'mean': float(np.mean([d['mean'] for d in dists])),
                    'std': float(np.mean([d['std'] for d in dists])),
                    'max_val': float(np.max([d['max'] for d in dists])),
                    'density': float(np.mean([d['density'] for d in dists]))
                }
                for feature, dists in family_features['feature_distributions'].items()
                if dists  # Only process features that have data
            },
            
            'behavior_patterns': normalized_patterns,
            
            # Average graph signatures
            'graph_characteristics': {
                metric: float(np.mean([sig[metric] for sig in family_features['graph_signatures']]))
                for metric in ['size', 'active_nodes', 'api_density', 'control_flow_complexity']
            }
        }
        
        return aggregated

    def load_processed_batches(self, split: str = 'train'):
        """Load preprocessed PyG graphs from batches."""
        split_dir = self.batch_dir / split
        logger.info(f"Loading batches from {split_dir}")
        
        for batch_file in tqdm(list(split_dir.glob("batch_*.pt")), desc="Loading batches"):
            try:
                batch_graphs = torch.load(batch_file)
                for graph in batch_graphs:
                    self.family_graphs[graph.family].append(graph)
            except Exception as e:
                logger.error(f"Error loading {batch_file}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(self.family_graphs)} families")
        for family, graphs in self.family_graphs.items():
            logger.info(f"Family {family}: {len(graphs)} samples")

    def _convert_pyg_to_networkx(self, pyg_graph) -> nx.DiGraph:
        """Convert PyG graph to NetworkX for feature extraction."""
        G = nx.DiGraph()
        
        # Add nodes with features
        for i in range(pyg_graph.num_nodes):
            node_features = {
                'mem_ops': float(pyg_graph.x[i][0]),
                'calls': float(pyg_graph.x[i][1]),
                'instructions': float(pyg_graph.x[i][2]),
                'stack_ops': float(pyg_graph.x[i][3]),
                'reg_writes': float(pyg_graph.x[i][4]),
                'external_calls': float(pyg_graph.x[i][5]),
                'internal_calls': float(pyg_graph.x[i][6]),
                'mem_reads': float(pyg_graph.x[i][7]),
                'mem_writes': float(pyg_graph.x[i][8]),
                'in_degree': float(pyg_graph.x[i][9]),
                'out_degree': float(pyg_graph.x[i][10]),
                'is_conditional': float(pyg_graph.x[i][11]),
                'has_jump': float(pyg_graph.x[i][12]),
                'has_ret': float(pyg_graph.x[i][13])
            }
            G.add_node(i, **node_features)
        
        # Add edges - Fixed version
        edge_index = pyg_graph.edge_index.t().numpy()
        edge_attrs = pyg_graph.edge_attr.numpy() if pyg_graph.edge_attr is not None else None
        
        for idx, (src, dst) in enumerate(edge_index):
            edge_attr = {}
            if edge_attrs is not None:
                edge_attr['condition'] = bool(edge_attrs[idx][0])
            G.add_edge(int(src), int(dst), **edge_attr)
        
        return G

    def process_families(self):
        """Process all families to extract behavioral distributions."""
        logger.info("Processing family behaviors...")
        
        for family, graphs in tqdm(self.family_graphs.items(), desc="Processing families"):
            try:
                result = self._aggregate_family_behaviors(graphs)
                if result is not None:
                    self.family_distributions[family] = result
                else:
                    logger.warning(f"No valid data for family {family}")
            except Exception as e:
                logger.error(f"Error processing family {family}: {str(e)}")
                continue
                
        if not self.family_distributions:
            logger.error("No family distributions were successfully processed")
        else:
            logger.info(f"Successfully processed {len(self.family_distributions)} families")

    def create_behavioral_groups(self, n_clusters: Optional[int] = None):
        """
        Create behavioral groups from processed families.
        Args:
            n_clusters: Target number of clusters. If None, will estimate using silhouette analysis.
        """
        if not self.family_distributions:
            raise ValueError("No family distributions available. Run process_families first.")
            
        similarity_computer = BehavioralSimilarityComputer(self.family_distributions)
        
        # Compute similarity matrix
        families = list(self.family_distributions.keys())
        n_families = len(families)
        similarity_matrix = np.zeros((n_families, n_families))
        
        logger.info("Computing family similarities...")
        for i, fam1 in enumerate(tqdm(families)):
            for j, fam2 in enumerate(families):
                similarity_matrix[i,j] = similarity_computer.compute_similarity(
                    self.family_distributions[fam1],
                    self.family_distributions[fam2]
                )
        
        # Handle any NaN values
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
        distance_matrix = 1 - similarity_matrix
        
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        if n_clusters is None:
            # Try different numbers of clusters and use silhouette analysis
            candidate_n_clusters = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            scores = []
            
            for n in candidate_n_clusters:
                clustering = AgglomerativeClustering(
                    n_clusters=n,
                    metric='precomputed',
                    linkage='average'
                )
                labels = clustering.fit_predict(distance_matrix)
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                scores.append(score)
                logger.info(f"Clusters: {n}, Silhouette Score: {score:.3f}")
            
            # Pick number of clusters with best score
            best_n = candidate_n_clusters[np.argmax(scores)]
            logger.info(f"Selected {best_n} clusters based on silhouette analysis")
            n_clusters = best_n
        
        # Perform final clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Create groups
        behavioral_groups = defaultdict(list)
        for family, label in zip(families, labels):
            behavioral_groups[label].append(family)
        
        # Log grouping results with more detailed statistics
        n_groups = len(behavioral_groups)
        logger.info(f"\nFound {n_groups} behavioral groups:")
        
        # Compute group statistics
        group_sizes = [len(families) for families in behavioral_groups.values()]
        logger.info(f"Average group size: {np.mean(group_sizes):.1f}")
        logger.info(f"Group size std: {np.std(group_sizes):.1f}")
        logger.info(f"Largest group: {max(group_sizes)}")
        logger.info(f"Smallest group: {min(group_sizes)}")
        
        # Sort groups by size for better visualization
        sorted_groups = sorted(behavioral_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Log individual groups
        for group_id, group_families in sorted_groups:
            logger.info(f"\nGroup {group_id}: {len(group_families)} families")
            if len(group_families) > 10:
                logger.info(f"Sample families: {', '.join(group_families[:10])}...")
            else:
                logger.info(f"Families: {', '.join(group_families)}")
                
            # Log some behavioral characteristics of the group
            if len(group_families) > 0:
                example_family = group_families[0]
                example_dist = self.family_distributions[example_family]
                behaviors = example_dist['behavior_patterns']
                top_behaviors = sorted(behaviors.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info("Characteristic behaviors: " + 
                        ", ".join(f"{b}: {v:.2f}" for b, v in top_behaviors))
        
        # Validate clustering quality
        sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
        logger.info(f"\nFinal silhouette score: {sil_score:.3f}")
            
        return behavioral_groups, similarity_matrix
                
    def _find_optimal_threshold(self, similarity_matrix: np.ndarray) -> float:
        """Find optimal similarity threshold using distribution analysis."""
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        distance_matrix = np.nan_to_num(distance_matrix, nan=np.max(distance_matrix[~np.isnan(distance_matrix)]))
        
        # Use finer-grained thresholds and focus on higher similarity range
        thresholds = np.linspace(0.05, 0.5, 30)  # Lower thresholds = more groups
        n_clusters = []
        
        # Also track silhouette scores to help find good clustering
        from sklearn.metrics import silhouette_score
        silhouette_scores = []
        
        for threshold in thresholds:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='precomputed',
                linkage='average'  # Changed from complete to average - less conservative
            )
            labels = clustering.fit_predict(distance_matrix)
            n_clusters_i = len(set(labels))
            n_clusters.append(n_clusters_i)
            
            # Only compute silhouette score if we have more than 1 cluster
            if n_clusters_i > 1:
                try:
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    silhouette_scores.append(score)
                except:
                    silhouette_scores.append(-1)
            else:
                silhouette_scores.append(-1)
        
        # Find a good threshold that balances number of clusters and cluster quality
        scores = np.array(silhouette_scores)
        n_clusters = np.array(n_clusters)
        
        # Filter for reasonable numbers of clusters (e.g., between 10 and 100)
        valid_indices = (n_clusters >= 10) & (n_clusters <= 100)
        if not any(valid_indices):
            # If no threshold gives us desired range, pick one that gives ~50 clusters
            target_clusters = 50
            idx = np.argmin(np.abs(n_clusters - target_clusters))
            return thresholds[idx]
        
        # Among valid thresholds, pick one with good silhouette score
        valid_scores = scores[valid_indices]
        valid_thresholds = thresholds[valid_indices]
        
        if len(valid_scores) > 0:
            # Pick threshold with best score
            best_idx = np.argmax(valid_scores)
            return valid_thresholds[best_idx]
        
        return 0.3  # Fallback threshold
def main():
    # Initialize processor
    aggregator = MalwareBehaviorAggregator(
        batch_dir=Path('bodmas_batches')
    )
    
    # Load processed data
    aggregator.load_processed_batches(split='train')
    
    # Process families
    aggregator.process_families()
    
    # Create behavioral groups
    groups, similarity_matrix = aggregator.create_behavioral_groups()
    
    # Save results
    output_dir = Path('behavioral_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Save similarity matrix
    np.save(output_dir / 'similarity_matrix.npy', similarity_matrix)
    
    # Save groupings
    import json
    with open(output_dir / 'behavioral_groups.json', 'w') as f:
        json.dump({str(k): v for k, v in groups.items()}, f, indent=2)
    
    # Optionally visualize results
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, cmap='viridis')
        plt.title('Family Similarity Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_matrix.png')
        plt.close()
        
        logger.info(f"Results saved to {output_dir}")
    except ImportError:
        logger.warning("Visualization skipped: seaborn/matplotlib not available")

if __name__ == "__main__":
    main()