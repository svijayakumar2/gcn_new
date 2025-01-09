import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import networkx as nx
from tqdm import tqdm
# counter
from collections import Counter
import json
from datetime import datetime
import traceback
from collections import Counter
import logging
import numpy as np
import json
from pathlib import Path

# Set up logging similar to your temporal.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {_convert_to_native_types(k): _convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_native_types(i) for i in obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_to_native_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class BehavioralSimilarityComputer:
    """Compute similarity between malware family behavioral profiles efficiently."""
    
    def __init__(self, family_distributions: Dict):
        self.family_distributions = family_distributions
    
    def compute_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """
        Compute similarity score between two family behavioral profiles.
        Uses efficient vectorized operations where possible.
        """
        similarities = []
        
        # 1. Compare feature distributions using histogram intersection
        feat_sim = self._compare_feature_distributions(
            profile1['feature_stats'], 
            profile2['feature_stats']
        )
        if feat_sim is not None:
            similarities.append((feat_sim, 0.4))  # Weight: 0.4
        
        # 2. Compare behavior patterns
        pattern_sim = self._compare_pattern_vectors(
            profile1['behavior_patterns'],
            profile2['behavior_patterns']
        )
        if pattern_sim is not None:
            similarities.append((pattern_sim, 0.4))  # Weight: 0.4
        
        # 3. Compare local structures
        struct_sim = self._compare_pattern_vectors(
            profile1['local_structures'],
            profile2['local_structures']
        )
        if struct_sim is not None:
            similarities.append((struct_sim, 0.2))  # Weight: 0.2
        
        if not similarities:
            return 0.0
        
        # Compute weighted average
        total_weight = sum(weight for _, weight in similarities)
        weighted_sum = sum(sim * weight for sim, weight in similarities)
        
        return weighted_sum / total_weight
    
    def _compare_feature_distributions(self, stats1: Dict, stats2: Dict) -> Optional[float]:
        """Compare feature distributions using histogram intersection."""
        if not stats1 or not stats2:
            return None
            
        feature_sims = []
        for feature in stats1:
            if feature not in stats2:
                continue
                
            hist1 = np.array(stats1[feature]['histogram'])
            hist2 = np.array(stats2[feature]['histogram'])
            
            # Compute histogram intersection
            intersection = np.minimum(hist1, hist2).sum()
            union = np.maximum(hist1, hist2).sum()
            
            if union > 0:
                feature_sims.append(intersection / union)
        
        return np.mean(feature_sims) if feature_sims else None
    
    def _compare_pattern_vectors(self, patterns1: Dict, patterns2: Dict) -> Optional[float]:
        """Compare pattern frequency vectors using cosine similarity."""
        if not patterns1 or not patterns2:
            return None
        
        # Get all patterns
        all_patterns = sorted(set(patterns1.keys()) | set(patterns2.keys()))
        if not all_patterns:
            return None
        
        # Create frequency vectors
        vec1 = np.array([patterns1.get(p, 0) for p in all_patterns])
        vec2 = np.array([patterns2.get(p, 0) for p in all_patterns])
        
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
    
    def __init__(self, batch_dir: Path, window_id: Optional[int] = None):
        self.batch_dir = Path(batch_dir)
        self.window_id = window_id  # Track which window we're analyzing
        self.family_graphs = defaultdict(list)
        self.family_distributions = {}


    def _aggregate_family_behaviors(self, pyg_graphs: List) -> Dict:
        """
        Aggregate behavioral features efficiently by focusing on key patterns
        and local structures rather than expensive graph operations.
        """
        feature_names = [
            'mem_ops', 'calls', 'instructions', 'stack_ops', 'reg_writes',
            'external_calls', 'internal_calls', 'mem_reads', 'mem_writes',
            'in_degree', 'out_degree', 'is_conditional', 'has_jump', 'has_ret'
        ]
        
        # Initialize aggregated features with default values
        family_features = {
            'feature_histograms': defaultdict(list),
            'behavior_patterns': defaultdict(float),
            'local_structures': defaultdict(float)
        }
        
        total_nodes = 0
        n_bins = 20  # Fixed number of bins for all histograms
        
        # Even if we have no graphs, return a valid minimal feature set
        if not pyg_graphs:
            print("Warning: No graphs for family, using default features")
            return {
                'feature_stats': {
                    feature: {
                        'histogram': np.zeros(n_bins).tolist(),
                        'histogram_std': np.zeros(n_bins).tolist()
                    }
                    for feature in feature_names
                },
                'behavior_patterns': {
                    'ext_call': 0.0,
                    'mem_rw': 0.0,
                    'cond_jump': 0.0,
                    'ext_write': 0.0
                },
                'local_structures': {
                    'branching_nodes': 0.0,
                    'merge_nodes': 0.0,
                    'terminal_nodes': 0.0,
                    'isolated_nodes': 0.0,
                    'dense_regions': 0.0
                }
            }
        
        for graph in pyg_graphs:
            try:
                node_features = graph.x.numpy()
                edge_index = graph.edge_index.t().numpy()
                num_nodes = len(node_features)
                total_nodes += num_nodes
                
                # 1. Feature distributions using numpy operations
                for feat_idx, feature in enumerate(feature_names):
                    values = node_features[:, feat_idx]
                    if len(values) > 0:
                        hist, _ = np.histogram(values, bins=n_bins, range=(0, np.max(values) + 1e-6), density=True)
                        family_features['feature_histograms'][feature].append(hist)
                    else:
                        # Add zero histogram if no values
                        family_features['feature_histograms'][feature].append(np.zeros(n_bins))
                
                # 2. Behavior patterns with safety checks
                has_ext_calls = node_features[:, feature_names.index('external_calls')] > 0
                has_mem_write = node_features[:, feature_names.index('mem_writes')] > 0
                has_mem_read = node_features[:, feature_names.index('mem_reads')] > 0
                is_conditional = node_features[:, feature_names.index('is_conditional')] > 0
                has_jump = node_features[:, feature_names.index('has_jump')] > 0
                
                patterns = {
                    'ext_call': np.sum(has_ext_calls),
                    'mem_rw': np.sum(has_mem_read & has_mem_write),
                    'cond_jump': np.sum(is_conditional & has_jump),
                    'ext_write': np.sum(has_ext_calls & has_mem_write)
                }
                
                for pattern, count in patterns.items():
                    family_features['behavior_patterns'][pattern] += count / max(num_nodes, 1)
                
                # 3. Local structure analysis with safety checks
                if len(edge_index) > 0:
                    in_degrees = np.bincount(edge_index[:, 1], minlength=num_nodes)
                    out_degrees = np.bincount(edge_index[:, 0], minlength=num_nodes)
                    
                    structures = {
                        'branching_nodes': np.sum(out_degrees > 1),
                        'merge_nodes': np.sum(in_degrees > 1),
                        'terminal_nodes': np.sum((out_degrees == 0) & (in_degrees > 0)),
                        'isolated_nodes': np.sum((in_degrees == 0) & (out_degrees == 0)),
                        'dense_regions': np.sum((in_degrees > 2) & (out_degrees > 2))
                    }
                    
                    for struct, count in structures.items():
                        family_features['local_structures'][struct] += count / max(num_nodes, 1)
                
            except Exception as e:
                print(f"Warning: Error processing graph: {str(e)}")
                continue
        
        num_graphs = len(pyg_graphs)
        
        # Always return a valid feature set
        return {
            'feature_stats': {
                feature: {
                    'histogram': np.mean(np.array(hists), axis=0).tolist() if hists else np.zeros(n_bins).tolist(),
                    'histogram_std': np.std(np.array(hists), axis=0).tolist() if hists else np.zeros(n_bins).tolist()
                }
                for feature, hists in family_features['feature_histograms'].items()
            },
            'behavior_patterns': {
                pattern: count / max(num_graphs, 1)
                for pattern, count in family_features['behavior_patterns'].items()
            },
            'local_structures': {
                struct: count / max(num_graphs, 1)
                for struct, count in family_features['local_structures'].items()
            }
        }

    def _extract_operation_sequences(self, node_features: np.ndarray, edge_index: np.ndarray, 
                                feature_names: List[str], max_length: int = 5) -> List[str]:
        """Extract sequences of operations along paths in the graph."""
        sequences = []
        n_nodes = len(node_features)
        
        # Create adjacency list for faster traversal
        adj_list = [[] for _ in range(n_nodes)]
        for src, dst in edge_index:
            adj_list[src].append(dst)
        
        # Find entry points (nodes with no incoming edges)
        in_degree = np.zeros(n_nodes)
        for _, dst in edge_index:
            in_degree[dst] += 1
        entry_points = np.where(in_degree == 0)[0]
        
        # DFS from each entry point to extract operation sequences
        def get_node_signature(node_idx):
            features = node_features[node_idx]
            sig_parts = []
            if features[feature_names.index('external_calls')] > 0:
                sig_parts.append('EXT')
            if features[feature_names.index('mem_writes')] > 0:
                sig_parts.append('WRITE')
            if features[feature_names.index('mem_reads')] > 0:
                sig_parts.append('READ')
            if features[feature_names.index('is_conditional')] > 0:
                sig_parts.append('COND')
            return '_'.join(sig_parts) if sig_parts else 'NOP'
        
        def dfs_sequence(node, path, visited):
            if len(path) >= max_length:
                sequences.append('->'.join(path))
                return
            visited.add(node)
            for next_node in adj_list[node]:
                if next_node not in visited:
                    path.append(get_node_signature(next_node))
                    dfs_sequence(next_node, path, visited.copy())
                    path.pop()
        
        # Extract sequences from each entry point
        for entry in entry_points:
            dfs_sequence(entry, [get_node_signature(entry)], set())
        
        return sequences

    def _find_subgraph_patterns(self, node_features: np.ndarray, edge_index: np.ndarray, 
                            pattern_size: int = 3) -> Dict[str, int]:
        """Find and count common subgraph patterns."""
        patterns = defaultdict(int)
        n_nodes = len(node_features)
        
        # Create adjacency matrix for faster neighbor checking
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for src, dst in edge_index:
            adj_matrix[src, dst] = 1
        
        # Helper to create pattern signature
        def get_pattern_signature(nodes):
            # Sort nodes by their feature values for canonical representation
            node_sigs = []
            for n in nodes:
                feats = node_features[n]
                node_sigs.append(f"{int(feats[5])}{int(feats[7])}{int(feats[8])}{int(feats[11])}")
            return '-'.join(sorted(node_sigs))
        
        # Find all connected subgraphs of size pattern_size
        for start_node in range(n_nodes):
            stack = [(start_node, [start_node])]
            while stack:
                node, current_pattern = stack.pop()
                if len(current_pattern) == pattern_size:
                    pattern_sig = get_pattern_signature(current_pattern)
                    patterns[pattern_sig] += 1
                    continue
                
                # Add neighbors to pattern
                neighbors = np.where(adj_matrix[node] > 0)[0]
                for neighbor in neighbors:
                    if neighbor not in current_pattern:
                        stack.append((neighbor, current_pattern + [neighbor]))
        
        return patterns

    def _count_graph_motifs(self, node_features: np.ndarray, edge_index: np.ndarray) -> Dict[str, int]:
        """Count occurrences of important graph motifs."""
        motifs = defaultdict(int)
        n_nodes = len(node_features)
        
        # Create adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for src, dst in edge_index:
            adj_matrix[src, dst] = 1
        
        # 1. Find feedback loops
        for i in range(n_nodes):
            # Simple self-loop
            if adj_matrix[i, i]:
                motifs['self_loop'] += 1
                
            # Two-node feedback
            for j in range(i + 1, n_nodes):
                if adj_matrix[i, j] and adj_matrix[j, i]:
                    motifs['two_node_feedback'] += 1
        
        # 2. Find branching patterns
        for i in range(n_nodes):
            out_degree = np.sum(adj_matrix[i])
            if out_degree > 1:
                motifs[f'branch_{int(out_degree)}'] += 1
        
        # 3. Find converging patterns
        for i in range(n_nodes):
            in_degree = np.sum(adj_matrix[:, i])
            if in_degree > 1:
                motifs[f'merge_{int(in_degree)}'] += 1
        
        return motifs

    def _finalize_family_features(self, family_features: Dict, n_graphs: int) -> Dict:
        """Finalize and normalize family-level features."""
        # Normalize histogram counts
        feature_stats = {}
        for feature, histograms in family_features['feature_histograms'].items():
            # Compute average histogram
            all_hist = np.array([h['histogram'] for h in histograms])
            avg_hist = np.mean(all_hist, axis=0)
            feature_stats[feature] = {
                'histogram': avg_hist.tolist(),
                'bin_edges': histograms[0]['bin_edges']  # Use first bin edges
            }
        
        # Get most common sequences
        from collections import Counter
        sequence_counts = Counter(family_features['node_sequences'])
        top_sequences = dict(sequence_counts.most_common(10))
        
        # Normalize pattern counts
        subgraph_patterns = {
            k: v / n_graphs for k, v in family_features['subgraph_patterns'].items()
        }
        
        motif_patterns = {
            k: v / n_graphs for k, v in family_features['graph_motifs'].items()
        }
        
        return {
            'feature_distributions': feature_stats,
            'common_sequences': top_sequences,
            'subgraph_patterns': subgraph_patterns,
            'motif_patterns': motif_patterns
        }

    def load_processed_batches(self, split: str = 'train'):
        """Load preprocessed PyG graphs from batches considering rolling windows."""
        # Modify path based on window structure
        if self.window_id is not None:
            split_dir = self.batch_dir / f"window_{self.window_id:03d}" / split
        else:
            # For final test set
            split_dir = self.batch_dir / "test"
            
        logger.info(f"Loading batches from {split_dir}")
        
        # Track malware types
        self.malware_types = {}
        
        for batch_file in tqdm(list(split_dir.glob("batch_*.pt")), desc="Loading batches"):
            try:
                batch_graphs = torch.load(batch_file)
                for graph in batch_graphs:
                    self.family_graphs[graph.family].append(graph)
                    self.malware_types[graph.family] = graph.malware_type
            except Exception as e:
                logger.error(f"Error loading {batch_file}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(self.family_graphs)} families")
        
        # Log distribution of malware types
        type_counts = Counter(self.malware_types.values())
        logger.info("\nMalware type distribution:")
        for mtype, count in type_counts.most_common():
            logger.info(f"{mtype}: {count} families")

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
                print(family)
                continue
        # print num family_distribution
        print('Number of family_distributions')
        print(len(self.family_distributions))
        print('Number of family_graphs')
        print(len(self.family_graphs))
        if not self.family_distributions:
            logger.error("No family distributions were successfully processed")
        else:
            logger.info(f"Successfully processed {len(self.family_distributions)} families")

    def create_behavioral_groups(self, similarity_threshold: Optional[float] = None):
        """Create behavioral groups constrained by malware types, with special handling for outliers."""
        if not self.family_distributions:
            raise ValueError("No family distributions available. Run process_families first.")
                
        similarity_computer = BehavioralSimilarityComputer(self.family_distributions)
        
        # Get all families and their malware types
        families = list(self.family_distributions.keys())
        family_types = {f: self.malware_types.get(f, 'unknown') for f in families}
        
        # Group families by malware type first
        type_groups = defaultdict(list)
        for family in families:
            mtype = family_types[family]
            type_groups[mtype].append(family)
        
        print(f"\nFound {len(type_groups)} malware types")
        for mtype, type_families in type_groups.items():
            print(f"{mtype}: {len(type_families)} families")
        
        # Initialize behavioral groups
        behavioral_groups = defaultdict(list)
        next_group_id = 0
        
        # Process each malware type separately
        for mtype, type_families in type_groups.items():
            print(f"\nProcessing malware type: {mtype}")
            n_families = len(type_families)
            
            if n_families == 0:
                continue
            elif n_families == 1:
                # Single family gets its own group
                behavioral_groups[next_group_id] = type_families
                next_group_id += 1
                continue
            
            # Compute similarity matrix for this type
            similarity_matrix = np.zeros((n_families, n_families))
            for i, fam1 in enumerate(type_families):
                for j, fam2 in enumerate(type_families):
                    if i != j:
                        similarity_matrix[i,j] = similarity_computer.compute_similarity(
                            self.family_distributions[fam1],
                            self.family_distributions[fam2]
                        )
            
            # Handle any NaN values
            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
            
            # Convert similarity to distance
            distance_matrix = 1 - similarity_matrix
            
            # Cluster families of this type
            from sklearn.cluster import AgglomerativeClustering
            
            # Use fewer clusters for small groups
            n_clusters = max(1, min(int(n_families/3), 5))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # Assign families to behavioral groups
            for family, label in zip(type_families, labels):
                behavioral_groups[next_group_id + label].append(family)
            
            next_group_id += n_clusters
        
        # Verify all families are assigned
        all_families = set(self.family_distributions.keys())
        assigned_families = set(family for group in behavioral_groups.values() for family in group)
        unassigned_families = all_families - assigned_families
        
        if unassigned_families:
            print(f"\nAssigning {len(unassigned_families)} unassigned families to new groups")
            for family in unassigned_families:
                behavioral_groups[next_group_id] = [family]
                next_group_id += 1
        
        # Log final grouping results
        print(f"\nCreated {len(behavioral_groups)} behavioral groups:")
        for group_id, group_families in behavioral_groups.items():
            print(f"\nGroup {group_id}: {len(group_families)} families")
            print(f"Sample families: {', '.join(group_families[:5])}...")
        
        return behavioral_groups, similarity_matrix
                    
    def _find_optimal_clusters(self, distance_matrix: np.ndarray, malware_families: list) -> tuple:
        """Test different numbers of clusters and find the optimal based on silhouette score."""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Get unique malware types
        unique_types = set(self.malware_types[f] for f in malware_families)
        min_clusters = len(unique_types)  # At least one cluster per malware type
        
        # Test range of cluster numbers
        n_cluster_range = range(2, min(len(malware_families) - 1, min_clusters + 20), 2)
        scores = []
        clustering_results = []
        
        logger.info("\nTesting different numbers of clusters:")
        for n_clusters in n_cluster_range:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # Check if clustering maintains malware type constraint
            is_valid = True
            for cluster_id in range(n_clusters):
                cluster_families = [malware_families[i] for i, l in enumerate(labels) if l == cluster_id]
                cluster_types = set(self.malware_types[f] for f in cluster_families)
                if len(cluster_types) > 1:
                    is_valid = False
                    break
            
            if is_valid:
                sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
                scores.append(sil_score)
                clustering_results.append(labels)
                logger.info(f"n_clusters={n_clusters}, silhouette={sil_score:.3f}")
            else:
                scores.append(-1)  # Invalid clustering
                clustering_results.append(None)
                logger.info(f"n_clusters={n_clusters}, invalid - mixed malware types")
        
        # Find best valid clustering
        valid_scores = [(i, s) for i, s in enumerate(scores) if s > -1]
        if not valid_scores:
            raise ValueError("No valid clustering found that maintains malware type constraints")
            
        best_idx = max(valid_scores, key=lambda x: x[1])[0]
        best_n_clusters = list(n_cluster_range)[best_idx]
        best_score = scores[best_idx]
        best_labels = clustering_results[best_idx]
        
        logger.info(f"\nBest clustering: n_clusters={best_n_clusters}, silhouette={best_score:.3f}")
        
        return best_labels, best_score


def analyze_family_drift(all_groups: Dict, all_similarities: Dict, windows: List[int]) -> Dict:
    """Analyze how individual families drift behaviorally over time."""
    drift_metrics = defaultdict(list)
    
    # First, create a mapping of families to their group for each window
    window_families = {}  # Track which families are in which window
    family_to_group = {}  # Track which group each family is in for each window
    
    for window in windows:
        window_families[window] = set()
        family_to_group[window] = {}
        for group_id, families in all_groups[window].items():
            for family in families:
                window_families[window].add(family)
                family_to_group[window][family] = group_id
    
    # For each pair of consecutive windows
    for i in range(len(windows)-1):
        curr_window = windows[i]
        next_window = windows[i+1]
        
        # Get common families between these windows
        common_families = window_families[curr_window] & window_families[next_window]
        logger.info(f"Windows {curr_window}->{next_window}: {len(common_families)} common families")
        
        curr_sim = all_similarities[curr_window]
        next_sim = all_similarities[next_window]
        
        # For each family present in both windows
        for family in common_families:
            try:
                # Get groups for this family in both windows
                curr_group = family_to_group[curr_window][family]
                next_group = family_to_group[next_window][family]
                
                # Get all families in the same groups
                curr_group_families = all_groups[curr_window][curr_group]
                next_group_families = all_groups[next_window][next_group]
                
                # Calculate average similarity to other families in group
                curr_similarities = []
                next_similarities = []
                
                for other_family in curr_group_families:
                    if other_family != family and other_family in common_families:
                        try:
                            curr_idx1 = list(window_families[curr_window]).index(family)
                            curr_idx2 = list(window_families[curr_window]).index(other_family)
                            if curr_idx1 < curr_sim.shape[0] and curr_idx2 < curr_sim.shape[1]:
                                curr_similarities.append(curr_sim[curr_idx1, curr_idx2])
                        except ValueError:
                            continue
                
                for other_family in next_group_families:
                    if other_family != family and other_family in common_families:
                        try:
                            next_idx1 = list(window_families[next_window]).index(family)
                            next_idx2 = list(window_families[next_window]).index(other_family)
                            if next_idx1 < next_sim.shape[0] and next_idx2 < next_sim.shape[1]:
                                next_similarities.append(next_sim[next_idx1, next_idx2])
                        except ValueError:
                            continue
                
                # Calculate drift only if we have valid similarities
                if curr_similarities and next_similarities:
                    curr_mean = float(np.mean(curr_similarities))
                    next_mean = float(np.mean(next_similarities))
                    # Avoid division by zero and handle very small numbers
                    if abs(curr_mean) > 1e-10:
                        drift_rate = 1 - next_mean / curr_mean
                    else:
                        drift_rate = 0.0 if abs(next_mean) < 1e-10 else 1.0
                    
                    drift_metrics[family].append({
                        'window_transition': f"{curr_window}->{next_window}",
                        'drift_rate': float(drift_rate),
                        'group_change': curr_group != next_group,
                        'num_comparisons': len(curr_similarities),
                        'curr_group': curr_group,
                        'next_group': next_group
                    })
            except Exception as e:
                logger.debug(f"Error processing family {family} in windows {curr_window}->{next_window}: {str(e)}")
                continue
    
    return dict(drift_metrics)

def analyze_convergent_evolution(all_similarities: Dict, windows: List[int]) -> Dict:
    """Identify cases where different families evolve similar behaviors."""
    convergence_patterns = {}
    
    # Track window transitions
    for i in range(len(windows)-1):
        curr_window = windows[i]
        next_window = windows[i+1]
        
        curr_sim = all_similarities[curr_window]
        next_sim = all_similarities[next_window]
        
        if curr_sim.shape != next_sim.shape:
            logger.warning(f"Similarity matrix shape mismatch between windows {curr_window} and {next_window}")
            # Take the minimum dimensions
            min_rows = min(curr_sim.shape[0], next_sim.shape[0])
            min_cols = min(curr_sim.shape[1], next_sim.shape[1])
            # Trim matrices to same size
            curr_sim = curr_sim[:min_rows, :min_cols]
            next_sim = next_sim[:min_rows, :min_cols]
        
        # Find pairs that become more similar
        convergent_pairs = []
        for i in range(curr_sim.shape[0]):
            for j in range(i + 1, curr_sim.shape[1]):
                try:
                    curr_similarity = float(curr_sim[i, j])
                    next_similarity = float(next_sim[i, j])
                    
                    if np.isnan(curr_similarity) or np.isnan(next_similarity):
                        continue
                    
                    similarity_increase = next_similarity - curr_similarity
                    if similarity_increase > 0.2:  # Threshold for significant increase
                        convergent_pairs.append({
                            'pair_indices': (int(i), int(j)),
                            'similarity_increase': float(similarity_increase),
                            'final_similarity': float(next_similarity),
                            'initial_similarity': float(curr_similarity)
                        })
                except Exception as e:
                    logger.debug(f"Error processing similarity pair ({i}, {j}): {str(e)}")
                    continue
        
        convergence_patterns[f"{curr_window}->{next_window}"] = {
            'num_convergent_pairs': len(convergent_pairs),
            'convergent_pairs': convergent_pairs,
            'matrix_shape': {
                'rows': int(curr_sim.shape[0]),
                'cols': int(curr_sim.shape[1])
            }
        }
    
    return convergence_patterns

def analyze_group_stability(all_groups: Dict, windows: List[int]) -> Dict:
    """Analyze how stable behavioral groups remain across windows."""
    stability_metrics = {}
    
    for i in range(len(windows)-1):
        curr_window = windows[i]
        next_window = windows[i+1]
        
        try:
            curr_groups = all_groups[curr_window]
            next_groups = all_groups[next_window]
        except KeyError as e:
            logger.error(f"Missing window data: {e}")
            continue
            
        # Track group changes
        changes = {
            'split': [],    # Groups that split into multiple groups
            'merged': [],   # Groups that merged together
            'stable': [],   # Groups that remained relatively stable
            'new': [],      # Entirely new groups
            'dissolved': [] # Groups that disappeared
        }
        
        # Get sets of families for each window
        curr_families = {f for fams in curr_groups.values() for f in fams}
        next_families = {f for fams in next_groups.values() for f in fams}
        
        # Find new and dissolved families
        new_families = next_families - curr_families
        dissolved_families = curr_families - next_families
        
        # Track new and dissolved groups
        for group_id, families in next_groups.items():
            if any(f in new_families for f in families):
                changes['new'].append(group_id)
        
        for group_id, families in curr_groups.items():
            if all(f in dissolved_families for f in families):
                changes['dissolved'].append(group_id)
        
        # Analyze group transitions
        for curr_id, curr_families in curr_groups.items():
            curr_families = set(curr_families) - dissolved_families
            if not curr_families:
                continue
                
            # Find where families went
            destinations = defaultdict(set)
            for family in curr_families:
                for next_id, next_families in next_groups.items():
                    if family in next_families and next_id not in changes['new']:
                        destinations[next_id].add(family)
            
            # Classify the change
            if len(destinations) == 0:
                if curr_id not in changes['dissolved']:
                    changes['dissolved'].append(curr_id)
            elif len(destinations) == 1:
                next_id = list(destinations.keys())[0]
                overlap = len(destinations[next_id]) / len(curr_families)
                if overlap >= 0.7:
                    changes['stable'].append({
                        'from_group': curr_id,
                        'to_group': next_id,
                        'overlap_ratio': float(overlap)
                    })
                else:
                    changes['split'].append({
                        'from_group': curr_id,
                        'to_groups': [next_id],
                        'distributions': {
                            next_id: float(len(destinations[next_id]) / len(curr_families))
                        }
                    })
            else:
                total = sum(len(fams) for fams in destinations.values())
                changes['split'].append({
                    'from_group': curr_id,
                    'to_groups': list(destinations.keys()),
                    'distributions': {
                        str(k): float(len(v) / total) 
                        for k, v in destinations.items()
                    }
                })
        
        stability_metrics[f"{curr_window}->{next_window}"] = {
            'num_split': len(changes['split']),
            'num_merged': len(changes['merged']),
            'num_stable': len(changes['stable']),
            'num_new': len(changes['new']),
            'num_dissolved': len(changes['dissolved']),
            'changes': changes,
            'stats': {
                'total_families_start': len(curr_families),
                'total_families_end': len(next_families),
                'families_added': len(new_families),
                'families_removed': len(dissolved_families)
            }
        }
    
    return stability_metrics

def analyze_behavioral_evolution(all_groups: Dict, all_similarities: Dict) -> Dict:
    """Analyze behavioral evolution across windows."""
    if not all_groups or not all_similarities:
        raise ValueError("Empty groups or similarities dictionary")
        
    windows = sorted(all_groups.keys())
    if not all(w in all_similarities for w in windows):
        raise ValueError("Mismatch between windows in groups and similarities")
    
    evolution_analysis = {
        'group_stability': analyze_group_stability(all_groups, windows),
        'family_drift': analyze_family_drift(all_groups, all_similarities, windows),
        'convergent_evolution': analyze_convergent_evolution(all_similarities, windows)
    }
    return evolution_analysis

def main():
    """Modified main to handle rolling window analysis."""
    base_dir = Path('bodmas_rolling')
    output_dir = Path('behavioral_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize numpy for safe array operations
    np.seterr(all='warn')
    
    # Get list of all windows
    windows = sorted([d for d in base_dir.glob("window_*") if d.is_dir()], 
                    key=lambda x: int(x.name.split('_')[1]))
    
    logger.info(f"Found {len(windows)} windows: {[w.name for w in windows]}")
    
    # Track behavioral evolution across windows
    all_groups = {}
    all_similarities = {}
    
    # Store family mappings for each window
    window_family_maps = {}
    
    # Process each window
    for window_dir in windows:
        window_id = int(window_dir.name.split('_')[1])
        logger.info(f"\nProcessing window {window_id}")
        
        # Initialize processor for this window
        aggregator = MalwareBehaviorAggregator(
            batch_dir=base_dir,
            window_id=window_id
        )
        
        # Load and process data for this window
        aggregator.load_processed_batches(split='train')
        
        # Log family counts
        logger.info(f"Window {window_id} loaded {len(aggregator.family_graphs)} families")
        
        # Store family list for this window
        window_family_maps[window_id] = list(aggregator.family_graphs.keys())
        
        # Process families
        aggregator.process_families()
        logger.info(f"Window {window_id} processed {len(aggregator.family_distributions)} family distributions")
        
        # Log malware types distribution
        type_counts = Counter(aggregator.malware_types.values())
        logger.info("\nMalware type distribution:")
        for mtype, count in type_counts.most_common():
            logger.info(f"{mtype}: {count} families")
        
        # Create behavioral groups
        groups, similarity_matrix = aggregator.create_behavioral_groups()
        
        logger.info(f"Window {window_id} created {len(groups)} behavioral groups")
        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        # Store results for this window
        all_groups[window_id] = groups
        all_similarities[window_id] = similarity_matrix
        
        # Save window-specific results
        window_output = output_dir / f"window_{window_id:03d}"
        window_output.mkdir(exist_ok=True)
        
        # Save similarity matrix
        np.save(window_output / 'similarity_matrix.npy', similarity_matrix)
        
        # Save behavioral groups and family mapping
        with open(window_output / 'behavioral_groups.json', 'w') as f:
            json.dump({
                'groups': {str(k): v for k, v in groups.items()},
                'family_mapping': {
                    str(k): list(v) for k, v in aggregator.malware_types.items()
                }
            }, f, indent=2)
    
    # Log final collection summary
    logger.info("\nAll windows processed. Summary:")
    logger.info(f"Number of windows with groups: {len(all_groups)}")
    logger.info(f"Windows with groups: {sorted(all_groups.keys())}")
    logger.info(f"Number of windows with similarities: {len(all_similarities)}")
    logger.info(f"Windows with similarities: {sorted(all_similarities.keys())}")
    
    # Save family mappings for all windows
    with open(output_dir / 'window_family_mappings.json', 'w') as f:
        json.dump({str(k): v for k, v in window_family_maps.items()}, f, indent=2)
    
    # Analyze behavioral evolution
    logger.info("\nAnalyzing behavioral evolution across windows")
    if not all_groups or not all_similarities:
        logger.error("No data collected for analysis!")
        logger.error(f"all_groups is empty: {len(all_groups) == 0}")
        logger.error(f"all_similarities is empty: {len(all_similarities) == 0}")
        return
    
    try:
        # Create directory for analysis results
        analysis_dir = output_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        # Run evolution analysis
        evolution_results = analyze_behavioral_evolution(all_groups, all_similarities)
        
        # Convert results to JSON-serializable format
        serializable_results = _convert_to_native_types(evolution_results)
        
        # Save detailed results in separate files
        if 'group_stability' in serializable_results:
            with open(analysis_dir / 'group_stability.json', 'w') as f:
                json.dump(serializable_results['group_stability'], f, indent=2)
                
        if 'family_drift' in serializable_results:
            with open(analysis_dir / 'family_drift.json', 'w') as f:
                json.dump(serializable_results['family_drift'], f, indent=2)
                
        if 'convergent_evolution' in serializable_results:
            with open(analysis_dir / 'convergent_evolution.json', 'w') as f:
                json.dump(serializable_results['convergent_evolution'], f, indent=2)
        
        # Save summary results
        with open(output_dir / 'evolution_analysis_summary.json', 'w') as f:
            summary = {
                'windows_analyzed': len(all_groups),
                'window_ids': sorted(all_groups.keys()),
                'total_families': sum(len(families) 
                    for groups in all_groups.values() 
                    for families in groups.values()),
                'analysis_timestamp': datetime.now().isoformat()
            }
            json.dump(summary, f, indent=2)
            
        logger.info("Analysis complete and results saved.")
        
    except Exception as e:
        logger.error(f"Error during evolution analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()