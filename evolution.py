import torch
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class MalwareEvolutionAnalyzer:
    """Analyze malware family evolution and relationships over time."""
    
    def __init__(self, batch_dir: Path):
        self.batch_dir = Path(batch_dir)
        self.samples_by_family = defaultdict(list)
        self.sample_features = {}
        self.family_timestamps = defaultdict(list)
        
    def load_samples(self, split: str = 'train'):
        """Load all samples and their features from the specified split."""
        split_dir = self.batch_dir / split
        logging.info(f"Loading samples from {split_dir}")
        
        for batch_file in tqdm(list(split_dir.glob("batch_*.pt"))):
            try:
                batch_graphs = torch.load(batch_file)
                for graph in batch_graphs:
                    # Extract core features
                    features = self._extract_sample_features(graph)
                    
                    # Store sample information
                    sample_id = graph.sha
                    family = graph.family
                    timestamp = datetime.strptime(graph.timestamp, '%Y-%m-%d %H:%M:%S UTC')
                    
                    self.samples_by_family[family].append(sample_id)
                    self.sample_features[sample_id] = features
                    self.family_timestamps[family].append(timestamp)
                    
            except Exception as e:
                logging.error(f"Error loading {batch_file}: {str(e)}")
                continue
                
        logging.info(f"Loaded {len(self.sample_features)} samples from {len(self.samples_by_family)} families")

    def _extract_sample_features(self, graph) -> np.ndarray:
        """Extract a feature vector from a graph for similarity comparison."""
        # Aggregate node features
        node_features = graph.x.numpy()
        
        # Calculate statistical features
        features = []
        
        # Node-level statistics
        node_stats = [
            np.mean(node_features, axis=0),  # Mean of each feature
            np.std(node_features, axis=0),   # Std of each feature
            np.max(node_features, axis=0),   # Max of each feature
            np.percentile(node_features, 75, axis=0)  # 75th percentile
        ]
        features.extend(node_stats)
        
        # Graph-level features - ensure these are numpy arrays
        graph_stats = [
            np.array([float(graph.num_nodes)]),  # Total nodes
            np.array([float(graph.edge_index.shape[1])]),  # Total edges
            np.array([float(graph.edge_index.shape[1]) / max(1, graph.num_nodes)])  # Edge density
        ]
        features.extend(graph_stats)
        
        # Convert all features to numpy arrays and flatten
        flattened_features = []
        for feature in features:
            if isinstance(feature, (int, float)):
                flattened_features.append(np.array([float(feature)]))
            else:
                flattened_features.append(feature.flatten())
                
        return np.concatenate(flattened_features)
    
    def analyze_family_evolution(self, family: str, min_samples: int = 5) -> Dict:
        """Analyze how a family evolves over time."""
        if family not in self.samples_by_family:
            raise ValueError(f"Family {family} not found")
            
        samples = self.samples_by_family[family]
        if len(samples) < min_samples:
            raise ValueError(f"Family {family} has too few samples ({len(samples)})")
            
        # Sort samples by timestamp
        sample_times = [(s, self.family_timestamps[family][i]) for i, s in enumerate(samples)]
        sample_times.sort(key=lambda x: x[1])
        
        # Calculate pairwise similarities over time
        similarities = []
        for i in range(len(samples)-1):
            current = self.sample_features[sample_times[i][0]]
            next_sample = self.sample_features[sample_times[i+1][0]]
            sim = cosine_similarity(current.reshape(1, -1), next_sample.reshape(1, -1))[0,0]
            time_delta = (sample_times[i+1][1] - sample_times[i][1]).days
            similarities.append({
                'from_sample': sample_times[i][0],
                'to_sample': sample_times[i+1][0],
                'similarity': sim,
                'time_delta_days': time_delta
            })
            
        return {
            'n_samples': len(samples),
            'time_span_days': (sample_times[-1][1] - sample_times[0][1]).days,
            'evolution_path': similarities,
            'avg_similarity': np.mean([s['similarity'] for s in similarities]),
            'similarity_std': np.std([s['similarity'] for s in similarities])
        }
    
    def find_similar_families(self, target_family: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar families to a target family."""
        if target_family not in self.samples_by_family:
            raise ValueError(f"Target family {target_family} not found")
            
        # Calculate average feature vector for target family
        target_features = np.mean([
            self.sample_features[s] for s in self.samples_by_family[target_family]
        ], axis=0)
        
        # Compare with other families
        similarities = []
        for family in self.samples_by_family:
            if family == target_family:
                continue
                
            family_features = np.mean([
                self.sample_features[s] for s in self.samples_by_family[family]
            ], axis=0)
            
            sim = cosine_similarity(
                target_features.reshape(1, -1), 
                family_features.reshape(1, -1)
            )[0,0]
            
            similarities.append((family, sim))
        
        # Return top-k most similar families
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def detect_potential_variants(self, similarity_threshold: float = 0.85) -> List[Dict]:
        """Detect potential variants across different families."""
        variants = []
        processed_pairs = set()
        
        # Compare all family pairs
        families = list(self.samples_by_family.keys())
        for i, fam1 in enumerate(families):
            for fam2 in families[i+1:]:
                pair_id = tuple(sorted([fam1, fam2]))
                if pair_id in processed_pairs:
                    continue
                    
                # Calculate average similarity between families
                sim = self._calculate_family_similarity(fam1, fam2)
                
                if sim > similarity_threshold:
                    variants.append({
                        'family1': fam1,
                        'family2': fam2,
                        'similarity': sim,
                        'samples1': len(self.samples_by_family[fam1]),
                        'samples2': len(self.samples_by_family[fam2]),
                        'time_relation': self._get_time_relation(fam1, fam2)
                    })
                    
                processed_pairs.add(pair_id)
        
        return sorted(variants, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_family_similarity(self, fam1: str, fam2: str) -> float:
        """Calculate average similarity between two families."""
        feat1 = np.mean([self.sample_features[s] for s in self.samples_by_family[fam1]], axis=0)
        feat2 = np.mean([self.sample_features[s] for s in self.samples_by_family[fam2]], axis=0)
        return cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0,0]
    
    def _get_time_relation(self, fam1: str, fam2: str) -> str:
        """Determine temporal relationship between families."""
        time1 = min(self.family_timestamps[fam1])
        time2 = min(self.family_timestamps[fam2])
        
        if abs((time1 - time2).days) < 30:
            return "concurrent"
        return "first_to_second" if time1 < time2 else "second_to_first"

    def visualize_family_relationships(self, min_samples: int = 3, output_file: str = None):
        """Create a t-SNE visualization of family relationships."""
        # Filter families with enough samples
        valid_families = [f for f in self.samples_by_family 
                        if len(self.samples_by_family[f]) >= min_samples]
        
        if not valid_families:
            raise ValueError(f"No families with at least {min_samples} samples")
            
        # Calculate average feature vectors for each family
        feature_vecs = []
        family_labels = []
        
        for family in valid_families:
            avg_features = np.mean([
                self.sample_features[s] for s in self.samples_by_family[family]
            ], axis=0)
            feature_vecs.append(avg_features)
            family_labels.append(family)
            
        # Convert list to numpy array for t-SNE
        feature_vecs = np.array(feature_vecs)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(feature_vecs)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.6)
        
        # # Add labels for some points
        # for i, family in enumerate(family_labels):
        #     if len(self.samples_by_family[family]) > min_samples * 10:
        #         plt.annotate(family, (embedded[i, 0], embedded[i, 1]))
                
        plt.title("Malware Family Relationships (t-SNE)")
        
        if output_file:
            plt.savefig(output_file)
        plt.close()

    def plot_family_evolution(self, min_samples=10, similarity_threshold=0.85):
        """Plot both family evolution and variant relationships."""
        # Get large families and their stats
        large_families = []
        similarities = []
        samples = []
        
        for family in self.samples_by_family:
            if len(self.samples_by_family[family]) >= min_samples:
                try:
                    evolution = self.analyze_family_evolution(family)
                    large_families.append(family)
                    similarities.append(evolution['avg_similarity'])
                    samples.append(evolution['n_samples'])
                except Exception as e:
                    continue
                    
        # Get variant relationships
        variants = self.detect_potential_variants(similarity_threshold=similarity_threshold)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Scatter plot of family size vs internal similarity
        ax1.scatter(samples, similarities, alpha=0.6)
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Average Internal Similarity')
        ax1.set_title('Family Size vs Internal Similarity')
        
        # Add some family labels
        for i, family in enumerate(large_families):
            if samples[i] > np.mean(samples)*100:  # Label only larger families
                ax1.annotate(family, (samples[i], similarities[i]))
        
        # Plot 2: Histogram of variant similarities
        variant_sims = [v['similarity'] for v in variants]
        ax2.hist(variant_sims, bins=30, alpha=0.7)
        ax2.set_xlabel('Similarity Score')
        ax2.set_ylabel('Number of Variant Pairs')
        ax2.set_title('Distribution of Variant Similarities')
        
        plt.tight_layout()
        return fig

    def visualize_family_samples(self, num_families: int = 20, min_samples: int = 10, output_file: str = None):
        """Create a t-SNE visualization of individual samples from top families."""
        
        # Get the top N families by number of samples
        family_sizes = [(f, len(samples)) for f, samples in self.samples_by_family.items() 
                        if len(samples) >= min_samples]
        top_families = sorted(family_sizes, key=lambda x: x[1], reverse=True)[:num_families]
        
        # Collect samples and labels
        feature_vecs = []
        family_labels = []
        
        for family, _ in top_families:
            for sample_id in self.samples_by_family[family]:
                feature_vecs.append(self.sample_features[sample_id])
                family_labels.append(family)
        
        # Convert to numpy array
        feature_vecs = np.array(feature_vecs)
        
        # Apply t-SNE
        print(f"Running t-SNE on {len(feature_vecs)} samples from {num_families} families...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(feature_vecs)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Create color map for families
        unique_families = list(set(family_labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_families)))
        color_map = dict(zip(unique_families, colors))
        
        # Plot each family's samples
        for family in unique_families:
            mask = [f == family for f in family_labels]
            points = embedded[mask]
            color = color_map[family]
            
            plt.scatter(points[:, 0], points[:, 1], 
                    c=[color], 
                    label=f"{family} ({sum(mask)} samples)",
                    alpha=0.6, 
                    s=50)
        
        plt.title(f"Malware Sample Relationships (t-SNE)\nTop {num_families} Families")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        plt.close()

def main():
    # Initialize analyzer
    analyzer = MalwareEvolutionAnalyzer(
        batch_dir=Path('bodmas_batches')
    )
    
    # Load samples
    analyzer.load_samples()

    fig = analyzer.plot_family_evolution(min_samples=10, similarity_threshold=0.85)
    fig.savefig('family_evolution.png')

    
    # Print top variant pairs for reference
    variants = analyzer.detect_potential_variants(similarity_threshold=0.85)
    print("\nTop variant pairs:")
    for var in variants[:10]:
        print(f"\n{var['family1']} <-> {var['family2']}")
        print(f"Similarity: {var['similarity']:.3f}")
        print(f"Time relation: {var['time_relation']}")

    # Analyze large families
    large_families = [f for f in analyzer.samples_by_family 
                     if len(analyzer.samples_by_family[f]) >= 10]
    
    print(f"\nAnalyzing {len(large_families)} large families...")
    for family in large_families:
        try:
            evolution = analyzer.analyze_family_evolution(family)
            print(f"\nFamily: {family}")
            print(f"Samples: {evolution['n_samples']}")
            print(f"Time span: {evolution['time_span_days']} days")
            print(f"Average similarity: {evolution['avg_similarity']:.3f}")
        except Exception as e:
            print(f"Error analyzing {family}: {str(e)}")
            continue
    
    # Detect variants
    print("\nDetecting potential variants...")
    variants = analyzer.detect_potential_variants(similarity_threshold=0.85)
    
    print(f"\nFound {len(variants)} potential variant relationships:")
    for var in variants[:10]:  # Show top 10
        print(f"\n{var['family1']} <-> {var['family2']}")
        print(f"Similarity: {var['similarity']:.3f}")
        print(f"Time relation: {var['time_relation']}")
    
    # Visualize relationships
    print("\nGenerating visualization...")
    analyzer.visualize_family_relationships(
        min_samples=3,
        output_file='family_relationships.png'
    )

    analyzer.visualize_family_samples(
        num_families=20,  # Number of top families to include
        min_samples=10,   # Minimum samples needed for a family to be included
        output_file='malware_samples_tsne.png'
    )

if __name__ == "__main__":
    main()