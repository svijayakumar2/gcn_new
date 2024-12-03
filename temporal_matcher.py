# once we have the json of features, we can use the following code to extract the features
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict


from torch_geometric.loader import DataLoader
import torch
from torch_geometric.data import Data
import gzip
import torch
import glob 


def convert_to_pytorch_geometric(graph_structure):
    """Convert saved graph structure to PyTorch Geometric format with enhanced features."""
    
    # Convert node features to tensor
    node_feats = []
    for node in graph_structure['node_features']:
        feats = [
            # Basic operation counts
            node['mem_ops'],
            node['calls'],
            node['instructions'],
            
            # Control flow features
            float(node['is_conditional']),  # Convert boolean to float
            float(node['has_jump']),
            float(node['has_ret']),
            
            # Operation types
            node['stack_ops'],
            node['reg_writes'],
            
            # Call types
            node['external_calls'],
            node['internal_calls'],
            
            # Memory operations
            node['mem_reads'],
            node['mem_writes'],
            
            # Structural features
            node['in_degree'],
            node['out_degree']
        ]
        node_feats.append(feats)
    
    # Convert to tensor and handle empty graphs
    if node_feats:
        x = torch.tensor(node_feats, dtype=torch.float)
    else:
        # Handle empty graphs by creating a single zero node
        x = torch.zeros((1, 14), dtype=torch.float)  # 14 is the number of features
    
    # Convert edge index to tensor
    if graph_structure['edge_index']:
        edge_index = torch.tensor(graph_structure['edge_index'], dtype=torch.long).t()
    else:
        # Handle graphs with no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Convert edge features if needed
    edge_attr = None
    if graph_structure['edge_features']:
        edge_feats = []
        for edge in graph_structure['edge_features']:
            # Enhanced edge features
            edge_feat = [
                1 if edge['condition'] else 0,
                1 if edge.get('is_call') else 0,
                1 if edge.get('is_conditional_branch') else 0
            ]
            edge_feats.append(edge_feat)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    # Add number of nodes as a graph-level feature
    data.num_nodes = x.size(0)
    
    return data

def load_for_gnn(json_file):
    """Load a single graph JSON.gz and convert to PyG format."""
    with gzip.open(json_file, 'rt') as f:  # Changed to gzip.open
        data = json.load(f)
        
    # Convert to PyG format
    pyg_graph = convert_to_pytorch_geometric(data['graph_structure'])
    
    # Add metadata
    pyg_graph.sha = os.path.basename(data['file']).split('_')[0]
    pyg_graph.timestamp = data['timestamp']
    if 'family' in data:
        pyg_graph.family = data['family']
    
    return pyg_graph

def load_dataset(results_dir):
    """Load all graphs in directory."""
    graphs = []
    failed_files = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json.gz'):  # Changed file extension check
            try:
                graph = load_for_gnn(os.path.join(results_dir, filename))
                graphs.append(graph)
                if len(graphs) % 100 == 0:  # Progress reporting
                    print(f"Loaded {len(graphs)} graphs...")
            except Exception as e:
                failed_files.append((filename, str(e)))
    
    if failed_files:
        print(f"Failed to load {len(failed_files)} files:")
        for fname, error in failed_files:
            print(f"- {fname}: {error}")
    
    print(f"Successfully loaded {len(graphs)} graphs")
    return graphs

def get_dataset_stats(graphs):
    """Get statistics about the loaded dataset."""
    stats = {
        'num_graphs': len(graphs),
        'avg_nodes': sum(g.num_nodes for g in graphs) / len(graphs),
        'avg_edges': sum(g.edge_index.size(1) for g in graphs) / len(graphs),
        'num_node_features': graphs[0].x.size(1),
        'num_edge_features': graphs[0].edge_attr.size(1) if graphs[0].edge_attr is not None else 0,
        'families': set(g.family for g in graphs if hasattr(g, 'family')),
    }
    return stats


def process_in_temporal_batches(metadata_csv, results_dir, batch_size=100, train_ratio=0.7, val_ratio=0.15):
    """Process the dataset in temporal batches with train/val/test splits."""
    
    # First get temporal ordering from metadata
    df = pd.read_csv(metadata_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate split indices
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split dataframe
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f'bodmas_batches/{split}', exist_ok=True)
    
    # Process each split
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} split")
        num_batches = (len(split_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            print(f"Processing {split_name} batch {batch_idx + 1}/{num_batches}")
            
            # Get SHAs for this batch
            batch_df = split_df.iloc[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_shas = set(batch_df['sha'].values)
            
            # Load corresponding graphs
            batch_graphs = []
            for filename in os.listdir(results_dir):
                if not filename.endswith('.json.gz'):
                    continue
                    
                sha = filename.split('_')[0]
                if sha in batch_shas:
                    try:
                        filepath = os.path.join(results_dir, filename)
                        graph = load_for_gnn(filepath)
                        
                        # Add family label if available
                        sha_row = batch_df[batch_df['sha'] == sha].iloc[0]
                        if pd.notna(sha_row['family']):
                            graph.family = sha_row['family']
                        
                        batch_graphs.append(graph)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
            
            if batch_graphs:
                # Save this batch
                batch_file = f'bodmas_batches/{split_name}/batch_{batch_idx:04d}.pt'
                torch.save(batch_graphs, batch_file)
                
                print(f"Saved {len(batch_graphs)} graphs to {batch_file}")
                print(f"Time period: {batch_df['timestamp'].min()} to {batch_df['timestamp'].max()}")



def main():
    # Process dataset into batches
    process_in_temporal_batches(
        metadata_csv='bodmas_metadata.csv',
        results_dir='cfg_analysis_results',
        batch_size=100  # Adjust based on your memory constraints
    )
    
    # Example of how to use batches for training
    print("\nExample of batch iteration:")
    batch_files = sorted(glob.glob('bodmas_batches/batch_*.pt'))
    print(f"Found {len(batch_files)} batch files")
    

if __name__ == "__main__":
    batch_files = main()



# def main():
#     # First, match metadata and create temporal dataset
#     matcher = BodmasMetadataMatcher(
#         metadata_csv='bodmas_metadata.csv',
#         analysis_dir='cfg_analysis_results'
#     )
#     matcher.save_temporal_dataset('bodmas_dataset')
    
#     # Then load and convert graphs to PyG format
#     print("Loading graphs (this might take a while)...")
#     graphs = load_dataset('cfg_analysis_results')
    
#     # Print dataset statistics
#     stats = get_dataset_stats(graphs)
#     print("\nDataset Statistics:")
#     print(f"Number of graphs: {stats['num_graphs']}")
#     print(f"Average nodes per graph: {stats['avg_nodes']:.2f}")
#     print(f"Average edges per graph: {stats['avg_edges']:.2f}")
#     print(f"Number of node features: {stats['num_node_features']}")
#     print(f"Number of edge features: {stats['num_edge_features']}")
#     print(f"Number of families: {len(stats['families'])}")
    
#     # Create PyG DataLoader for training
#     loader = DataLoader(graphs, batch_size=32, shuffle=False)
    
#     # Save processed PyG dataset
#     print("\nSaving PyG dataset...")
#     torch.save(graphs, 'bodmas_pyg_dataset.pt')
    
#     print("\nProcessing complete!")
#     print("Saved:")
#     print("- Temporal metadata: bodmas_dataset_temporal.json")
#     print("- PyG dataset: bodmas_pyg_dataset.pt")
    
#     return graphs, loader

# if __name__ == "__main__":
#     graphs, loader = main()

