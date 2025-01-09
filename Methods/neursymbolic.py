import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

# Import your existing components
from gcn import GCN, CentroidLayer

import sys
sys.path.append('./Processing')
from family_aggregator import (
    BehavioralSimilarityComputer,
    MalwareBehaviorAggregator
)
from FamilyLabels import FamilyLabelEncoder


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeurosymbolicEvolutionaryAnalyzer:
    """Analyze malware evolution using neural embeddings and symbolic patterns."""
    
    def __init__(self, 
                 model_path: Path,
                 batch_dir: Path,
                 metadata_path: Path,
                 malware_types_path: Path,
                 embedding_dim: int = 256,
                 n_behavioral_patterns: int = 64):
        self.embedding_dim = embedding_dim
        self.n_behavioral_patterns = n_behavioral_patterns
        
        # Initialize neural components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_model = self._load_gnn_model(model_path)
        self.centroid_layer = CentroidLayer(
            input_dim=embedding_dim,
            n_classes=n_behavioral_patterns,
            n_centroids_per_class=2,
            reject_input=True
        ).to(self.device)
        
        # Initialize behavioral components
        self.behavior_aggregator = MalwareBehaviorAggregator(batch_dir)
        self.similarity_computer = BehavioralSimilarityComputer({})
        
        # Initialize family encoder with metadata
        self.family_encoder = self._initialize_encoder(
            metadata_path, 
            malware_types_path
        )
        
        # Evolution tracking
        self.family_trajectories = defaultdict(list)
        self.behavioral_patterns = defaultdict(dict)
        self.mutation_patterns = defaultdict(list)
        
        # Temporal analysis
        self.temporal_clusters = defaultdict(list)
        self.coevolution_patterns = defaultdict(list)
        
    def _load_gnn_model(self, model_path: Path) -> GCN:
        """Load pretrained GNN model."""
        model = GCN(hidden_channels=self.embedding_dim)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
        
    def _initialize_encoder(self, metadata_path: Path, 
                          malware_types_path: Path) -> FamilyLabelEncoder:
        """Initialize FamilyLabelEncoder with metadata."""
        # Load and merge metadata
        metadata_df = pd.read_csv(metadata_path)
        malware_types_df = pd.read_csv(malware_types_path)
        
        # Merge using your format
        metadata_df['filename'] = metadata_df['sha'].apply(lambda x: x)
        malware_types_df['filename'] = malware_types_df['sha256'].apply(
            lambda x: Path(x).stem
        )
        
        merged_df = pd.merge(
            metadata_df,
            malware_types_df[['filename', 'category']].rename(
                columns={'category': 'malware_type'}
            ),
            on='filename',
            how='left'
        )
        
        # Initialize and fit encoder
        encoder = FamilyLabelEncoder()
        encoder.fit_from_metadata(merged_df)
        
        return encoder
        
    @torch.no_grad()
    def process_sample(self, graph_data, family: str, timestamp: datetime, sha: str):
        """Process a single malware sample."""
        # 1. Get neural embedding
        graph_data = graph_data.to(self.device)
        embedding = self.gnn_model(
            graph_data.x, 
            graph_data.edge_index, 
            graph_data.batch
        )
        
        # 2. Get behavioral features
        behavioral_features = self.behavior_aggregator._aggregate_family_behaviors([graph_data])
        
        # 3. Update evolutionary state
        self.update_family_state(
            family=family,
            embedding=embedding,
            behavioral_features=behavioral_features,
            timestamp=timestamp,
            sha=sha
        )
        
        # 4. Analyze temporal relationships
        self._analyze_temporal_relationships(family, timestamp)
        
    def update_family_state(self, 
                          family: str,
                          embedding: torch.Tensor,
                          behavioral_features: Dict,
                          timestamp: datetime,
                          sha: str):
        """Update evolutionary state with enhanced tracking."""
        # Get neural patterns
        embedding_features = self._process_embedding(embedding)
        
        # Combine with behavioral patterns
        combined_state = self._combine_neural_symbolic(
            embedding_features,
            behavioral_features
        )
        
        # Track trajectory with sample identification
        self.family_trajectories[family].append({
            'state': combined_state,
            'timestamp': timestamp,
            'embedding': embedding.detach().cpu(),
            'behavioral': behavioral_features,
            'sha': sha,
            'malware_type': self.family_encoder.malware_types.get(family, 'unknown')
        })
        
        # Sort trajectory by timestamp
        self.family_trajectories[family].sort(key=lambda x: x['timestamp'])
        
        # Analyze mutations if we have history
        if len(self.family_trajectories[family]) > 1:
            self._analyze_mutations(family)
            
        # Update temporal clusters
        self._update_temporal_clusters(family, timestamp, combined_state)
        
    def _process_embedding(self, embedding: torch.Tensor) -> Dict:
        """Extract features from neural embedding using centroid layer."""
        with torch.no_grad():
            centroid_output = self.centroid_layer(embedding.unsqueeze(0))
            
        pattern_activations = centroid_output[0, :-1]
        acceptance_score = centroid_output[0, -1]
        
        return {
            'pattern_activations': pattern_activations.cpu().numpy(),
            'acceptance_score': float(acceptance_score)
        }
        
    def _combine_neural_symbolic(self, 
                               neural_features: Dict,
                               symbolic_features: Dict) -> Dict:
        """Combine neural and symbolic features with behavioral context."""
        combined_features = {
            'neural_patterns': neural_features['pattern_activations'],
            'acceptance': neural_features['acceptance_score'],
            
            # Symbolic features from your structure
            'api_patterns': symbolic_features.get('behavior_patterns', {}),
            'structural_patterns': symbolic_features.get('local_structures', {}),
            'feature_stats': symbolic_features.get('feature_stats', {})
        }
        
        # Add behavioral group context if available
        behavioral_group = self.family_encoder.get_group(
            combined_features.get('family', 'unknown')
        )
        if behavioral_group >= 0:
            combined_features['behavioral_group'] = behavioral_group
            
        return combined_features
        
    def _analyze_mutations(self, family: str):
        """Analyze behavioral mutations with temporal context."""
        trajectory = self.family_trajectories[family]
        current = trajectory[-1]
        previous = trajectory[-2]
        
        # Compute behavioral distance
        behavioral_dist = self._compute_behavioral_distance(
            current['state'],
            previous['state']
        )
        
        # Identify mutations
        mutations = self._identify_mutations(
            current['state'],
            previous['state']
        )
        
        if mutations['significant']:
            # Add temporal context
            time_delta = (current['timestamp'] - previous['timestamp']).total_seconds()
            
            mutation_record = {
                'timestamp': current['timestamp'],
                'mutations': mutations['changes'],
                'magnitude': behavioral_dist,
                'time_delta': time_delta,
                'previous_sha': previous['sha'],
                'current_sha': current['sha']
            }
            
            # Check for coexisting families during mutation
            coexisting = self.family_encoder.get_coexisting_families(
                family,
                time_window_days=30
            )
            if coexisting:
                mutation_record['coexisting_families'] = coexisting
            
            self.mutation_patterns[family].append(mutation_record)
            
    def _update_temporal_clusters(self, family: str, 
                                timestamp: datetime,
                                state: Dict):
        """Update temporal clustering of behavioral patterns."""
        # Find temporal cluster
        cluster_found = False
        for cluster in self.temporal_clusters[family]:
            # Check if timestamp fits in existing cluster
            cluster_start = min(s['timestamp'] for s in cluster['states'])
            cluster_end = max(s['timestamp'] for s in cluster['states'])
            
            if (timestamp >= cluster_start and 
                timestamp <= cluster_end):
                # Add to existing cluster
                cluster['states'].append({
                    'timestamp': timestamp,
                    'state': state
                })
                cluster_found = True
                break
                
        if not cluster_found:
            # Create new cluster
            self.temporal_clusters[family].append({
                'states': [{
                    'timestamp': timestamp,
                    'state': state
                }],
                'start_time': timestamp
            })
            
    def _analyze_temporal_relationships(self, family: str, timestamp: datetime):
        """Analyze temporal relationships between families."""
        # Get coexisting families
        coexisting = self.family_encoder.get_coexisting_families(
            family,
            time_window_days=30
        )
        
        for other_family in coexisting:
            if other_family == family:
                continue
                
            # Look for concurrent mutations
            family_mutations = [m for m in self.mutation_patterns[family]
                              if abs((m['timestamp'] - timestamp).total_seconds()) 
                              <= 30 * 24 * 3600]  # 30 days
            
            other_mutations = [m for m in self.mutation_patterns[other_family]
                             if abs((m['timestamp'] - timestamp).total_seconds())
                             <= 30 * 24 * 3600]
                             
            if family_mutations and other_mutations:
                self.coevolution_patterns[family].append({
                    'timestamp': timestamp,
                    'related_family': other_family,
                    'family_mutations': family_mutations,
                    'other_mutations': other_mutations
                })
                
    def get_evolution_summary(self, family: str) -> Optional[Dict]:
        """Get comprehensive evolution summary."""
        if family not in self.family_trajectories:
            return None
            
        trajectory = self.family_trajectories[family]
        if len(trajectory) < 2:
            return None
            
        # Get metadata from encoder
        metadata = self.family_encoder.get_family_metadata(family)
        
        # Compute basic evolution metrics
        total_time = (trajectory[-1]['timestamp'] - 
                     trajectory[0]['timestamp']).total_seconds()
        mutation_records = self.mutation_patterns[family]
        
        # Get behavioral distances over time
        distances = []
        for i in range(1, len(trajectory)):
            dist = self._compute_behavioral_distance(
                trajectory[i]['state'],
                trajectory[i-1]['state']
            )
            distances.append(dist)
            
        # Analyze temporal clustering
        temporal_stats = {
            'num_clusters': len(self.temporal_clusters[family]),
            'avg_cluster_size': np.mean([
                len(cluster['states']) 
                for cluster in self.temporal_clusters[family]
            ]) if self.temporal_clusters[family] else 0
        }
        
        # Analyze coevolution
        coevolution_stats = {
            'num_coevolution_events': len(self.coevolution_patterns[family]),
            'related_families': list(set(
                event['related_family'] 
                for event in self.coevolution_patterns[family]
            ))
        }
        
        return {
            'metadata': metadata,
            'evolution_metrics': {
                'first_seen': trajectory[0]['timestamp'],
                'last_seen': trajectory[-1]['timestamp'],
                'total_mutations': len(mutation_records),
                'mutation_rate': len(mutation_records) / total_time if total_time > 0 else 0,
                'avg_mutation_magnitude': np.mean(distances) if distances else 0,
                'std_mutation_magnitude': np.std(distances) if distances else 0
            },
            'temporal_dynamics': temporal_stats,
            'coevolution': coevolution_stats
        }
        
    def save_analysis(self, output_dir: Path):
        """Save comprehensive analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evolution summaries
        summaries = {}
        for family in self.family_trajectories:
            summary = self.get_evolution_summary(family)
            if summary:
                summaries[family] = summary
                
        with open(output_dir / 'evolution_summaries.json', 'w') as f:
            json.dump(summaries, f, indent=2, default=str)
            
        # Save mutation patterns
        with open(output_dir / 'mutation_patterns.json', 'w') as f:
            json.dump(self.mutation_patterns, f, indent=2, default=str)
            
        # Save coevolution patterns
        with open(output_dir / 'coevolution_patterns.json', 'w') as f:
            json.dump(self.coevolution_patterns, f, indent=2, default=str)
            
        logger.info(f"Analysis results saved to {output_dir}")

def main():
    """Main function demonstrating the integrated pipeline."""
    config = {
        'model_path': Path('best_model.pt'),
        'batch_dir': Path('/data/saranyav/gcn_new/bodmas_batches'),
        'metadata_path': Path('bodmas_metadata_cleaned.csv'),
        'malware_types_path': Path('bodmas_malware_category.csv'),
        'behavioral_groups_path': Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        'output_dir': Path('results/evolutionary_analysis'),
        'embedding_dim': 256,
        'n_behavioral_patterns': 64
    }
    
    try:
        # Load behavioral groups
        with open(config['behavioral_groups_path']) as f:
            behavioral_groups = json.load(f)
            
        # Create family to group mapping
        family_to_group = {}
        for group_id, families in behavioral_groups.items():
            for family in families:
                family_to_group[family.lower()] = int(group_id)
        
        # Initialize analyzer
        analyzer = NeurosymbolicEvolutionaryAnalyzer(
            model_path=config['model_path'],
            batch_dir=config['batch_dir'],
            metadata_path=config['metadata_path'],
            malware_types_path=config['malware_types_path'],
            embedding_dim=config['embedding_dim'],
            n_behavioral_patterns=config['n_behavioral_patterns']
        )
        
        # Update family encoder with behavioral groups
        analyzer.family_encoder.update_group_mappings(family_to_group)
        
        # Load and process batches
        analyzer.behavior_aggregator.load_processed_batches(split='train')
        
        # Process samples chronologically
        families_processed = set()
        mutations_detected = 0
        coevolutions_detected = 0
        
        logger.info("Processing samples chronologically...")
        for family, graphs in tqdm(analyzer.behavior_aggregator.family_graphs.items()):
            # Sort graphs by timestamp
            sorted_graphs = sorted(graphs, key=lambda g: g.timestamp)
            
            # Process each sample
            for graph in sorted_graphs:
                analyzer.process_sample(
                    graph_data=graph,
                    family=family,
                    timestamp=pd.to_datetime(graph.timestamp),
                    sha=graph.sha
                )
            
            families_processed.add(family)
            mutations_detected += len(analyzer.mutation_patterns[family])
            coevolutions_detected += len(analyzer.coevolution_patterns[family])
            
        # Save analysis results
        analyzer.save_analysis(config['output_dir'])
        
        # Print summary statistics
        logger.info("\nAnalysis Summary:")
        logger.info(f"Total families processed: {len(families_processed)}")
        logger.info(f"Total mutations detected: {mutations_detected}")
        logger.info(f"Total coevolution events: {coevolutions_detected}")
        
        # Analyze behavioral group transitions
        group_transitions = defaultdict(int)
        for family, mutations in analyzer.mutation_patterns.items():
            if not mutations:
                continue
                
            current_group = analyzer.family_encoder.get_group(family)
            for mutation in mutations:
                # Check if mutation led to group change
                coexisting = mutation.get('coexisting_families', [])
                for other_family in coexisting:
                    other_group = analyzer.family_encoder.get_group(other_family)
                    if other_group != current_group:
                        group_transitions[(current_group, other_group)] += 1
        
        # Print group transition statistics
        logger.info("\nBehavioral Group Transitions:")
        for (src, dst), count in sorted(group_transitions.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"Group {src} -> Group {dst}: {count} transitions")
            # Print some example families involved
            src_families = analyzer.family_encoder.get_families_in_group(src)[:3]
            dst_families = analyzer.family_encoder.get_families_in_group(dst)[:3]
            logger.info(f"  Example families: {', '.join(src_families)} -> {', '.join(dst_families)}")
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()