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

# Set up logging similar to your temporal.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, batch_dir: Path):
        self.batch_dir = Path(batch_dir)
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
        
        # Initialize aggregated features
        family_features = {
            'feature_histograms': defaultdict(list),
            'behavior_patterns': defaultdict(float),
            'local_structures': defaultdict(float)
        }
        
        total_nodes = 0
        n_bins = 20  # Fixed number of bins for all histograms
        
        for graph in pyg_graphs:
            node_features = graph.x.numpy()
            edge_index = graph.edge_index.t().numpy()
            num_nodes = len(node_features)
            total_nodes += num_nodes
            
            # 1. Efficient feature distributions using numpy operations
            for feat_idx, feature in enumerate(feature_names):
                values = node_features[:, feat_idx]
                if len(values) > 0:
                    hist, _ = np.histogram(values, bins=n_bins, range=(0, np.max(values) + 1e-6), density=True)
                    family_features['feature_histograms'][feature].append(hist)
            
            # 2. Behavior patterns - look for significant combinations
            # Pre-compute boolean masks for efficiency
            has_ext_calls = node_features[:, feature_names.index('external_calls')] > 0
            has_mem_write = node_features[:, feature_names.index('mem_writes')] > 0
            has_mem_read = node_features[:, feature_names.index('mem_reads')] > 0
            is_conditional = node_features[:, feature_names.index('is_conditional')] > 0
            has_jump = node_features[:, feature_names.index('has_jump')] > 0
            
            # Update pattern counts efficiently using numpy
            patterns = {
                'ext_call': np.sum(has_ext_calls),
                'mem_rw': np.sum(has_mem_read & has_mem_write),
                'cond_jump': np.sum(is_conditional & has_jump),
                'ext_write': np.sum(has_ext_calls & has_mem_write)
            }
            
            for pattern, count in patterns.items():
                family_features['behavior_patterns'][pattern] += count / num_nodes
            
            # 3. Local structure analysis - focus on node neighborhood characteristics
            if len(edge_index) > 0:  # Only compute if we have edges
                in_degrees = np.bincount(edge_index[:, 1], minlength=num_nodes)
                out_degrees = np.bincount(edge_index[:, 0], minlength=num_nodes)
                
                # Analyze local structures
                structures = {
                    'branching_nodes': np.sum(out_degrees > 1) / num_nodes,
                    'merge_nodes': np.sum(in_degrees > 1) / num_nodes,
                    'terminal_nodes': np.sum((out_degrees == 0) & (in_degrees > 0)) / num_nodes,
                    'isolated_nodes': np.sum((in_degrees == 0) & (out_degrees == 0)) / num_nodes,
                    'dense_regions': np.sum((in_degrees > 2) & (out_degrees > 2)) / num_nodes
                }
                
                for struct, ratio in structures.items():
                    family_features['local_structures'][struct] += ratio
        
        num_graphs = len(pyg_graphs)
        if num_graphs == 0:
            return None
            
        # Finalize features
        return {
            'feature_stats': {
                feature: {
                    'histogram': np.mean(np.array(hists), axis=0).tolist(),
                    'histogram_std': np.std(np.array(hists), axis=0).tolist()
                }
                for feature, hists in family_features['feature_histograms'].items()
                if hists  # Only include features that have histograms
            },
            'behavior_patterns': {
                pattern: count / num_graphs
                for pattern, count in family_features['behavior_patterns'].items()
            },
            'local_structures': {
                struct: count / num_graphs
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
        """Load preprocessed PyG graphs from batches."""
        split_dir = self.batch_dir / split
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
        """Create behavioral groups constrained by malware types, with special handling for benign samples."""
        if not self.family_distributions:
            raise ValueError("No family distributions available. Run process_families first.")
                
        similarity_computer = BehavioralSimilarityComputer(self.family_distributions)
        
        # Separate benign and malware families
        families = list(self.family_distributions.keys())
        malware_families = [f for f in families if f != 'benign']
        n_families = len(malware_families)
        
        # Handle benign families separately
        benign_families = [f for f in families if f == 'benign']
        if benign_families:
            logger.info(f"\nFound {len(benign_families)} benign families - will be grouped together")
        
        # Compute similarity matrix for malware families
        similarity_matrix = np.zeros((n_families, n_families))
        
        logger.info("Computing family similarities...")
        for i, fam1 in enumerate(tqdm(malware_families)):
            for j, fam2 in enumerate(malware_families):
                # If malware types are different, set similarity to 0
                if self.malware_types[fam1] != self.malware_types[fam2]:
                    similarity_matrix[i,j] = 0
                else:
                    similarity_matrix[i,j] = similarity_computer.compute_similarity(
                        self.family_distributions[fam1],
                        self.family_distributions[fam2]
                    )
        
        # Handle any NaN values
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Find optimal clustering
        labels, sil_score = self._find_optimal_clusters(distance_matrix, malware_families)
        logger.info(f"Final Silhouette Score: {sil_score:.3f}")
        
        # Create groups - start with benign group
        behavioral_groups = defaultdict(list)
        if benign_families:
            behavioral_groups[0] = benign_families
            # Adjust other labels to start from 1
            labels = [l + 1 for l in labels]
        
        # Add malware groups
        for family, label in zip(malware_families, labels):
            behavioral_groups[label].append(family)
        
        # Log grouping results
        logger.info(f"\nFound {len(behavioral_groups)} behavioral groups:")
        
        # Log benign group first if it exists
        if benign_families:
            logger.info(f"\nGroup 0 (Benign): {len(benign_families)} families")
            if len(benign_families) > 10:
                logger.info(f"Sample families: {', '.join(benign_families[:10])}...")
            else:
                logger.info(f"Families: {', '.join(benign_families)}")
        
        # Log malware groups
        for group_id, group_families in behavioral_groups.items():
            if group_id == 0:  # Skip benign group as it's already logged
                continue
            logger.info(f"\nGroup {group_id}: {len(group_families)} families")
            mtype = self.malware_types[group_families[0]]  # All families in group should have same type
            logger.info(f"Malware type: {mtype}")
            
            if len(group_families) > 10:
                logger.info(f"Sample families: {', '.join(group_families[:10])}...")
            else:
                logger.info(f"Families: {', '.join(group_families)}")
        

        # Verify all families are assigned
        all_families = set(self.family_distributions.keys())
        assigned_families = set(family for group in behavioral_groups.values() for family in group)
        missing_families = all_families - assigned_families

        if missing_families:
            logger.warning(f"Found {len(missing_families)} unassigned families - assigning them to new groups")
            
            # Assign each missing family to its own new group
            max_group = max(behavioral_groups.keys()) if behavioral_groups else -1
            for i, family in enumerate(missing_families):
                behavioral_groups[max_group + i + 1] = [family]

        # Log final counts to verify
        total_families = len(all_families)
        total_assigned = sum(len(families) for families in behavioral_groups.values())
        logger.info(f"Total families: {total_families}, Total assigned: {total_assigned}")
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