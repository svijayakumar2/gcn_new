import os
import gzip
import json
import torch
import hashlib
import pandas as pd
from torch_geometric.data import Data

def clean_timestamps(metadata_csv, output_csv='bodmas_metadata_cleaned.csv'):
    """Clean and standardize timestamps in metadata."""
    print(f"Reading from {metadata_csv}...")
    df = pd.read_csv(metadata_csv)
    print("\nSample of original timestamps:")
    print(df['timestamp'].head())
    
    # Convert mixed format timestamps to standard UTC format
    def convert_timestamp(ts):
        try:
            # Try MM/DD/YY format first
            dt = pd.to_datetime(ts, format='%m/%d/%y %H:%M', utc=True)
        except:
            try:
                # Try ISO format with timezone
                dt = pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S%z', utc=True)
            except:
                # Last resort: try mixed format
                dt = pd.to_datetime(ts, format='mixed', utc=True)
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    df['timestamp'] = df['timestamp'].apply(convert_timestamp)
    print("\nSample of standardized timestamps:")
    print(df['timestamp'].head())
    
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned metadata to {output_csv}")
    return output_csv

def convert_to_pytorch_geometric(graph_structure):
    """Convert graph structure to PyG format."""
    # Get all unique feature keys
    feature_keys = sorted(list({
        key for node in graph_structure['node_features'] 
        for key in node.keys()
    }))
    
    # Create feature vectors
    node_feats = []
    for node in graph_structure['node_features']:
        feats = [
            float(node.get(key, 0)) if not isinstance(node.get(key), bool)
            else float(node.get(key, False))
            for key in feature_keys
        ]
        node_feats.append(feats)
    
    # Convert to tensors
    x = torch.tensor(node_feats, dtype=torch.float) if node_feats else \
        torch.zeros((1, len(feature_keys)), dtype=torch.float)
        
    edge_index = torch.tensor(graph_structure['edge_index'], dtype=torch.long).t() \
        if graph_structure['edge_index'] else torch.zeros((2, 0), dtype=torch.long)
        
    edge_attr = None
    if graph_structure['edge_features']:
        edge_feats = [[float(edge['condition'] is not None)] 
                     for edge in graph_structure['edge_features']]
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = x.size(0)
    return data

def get_md5_filename(sha):
    """Generate MD5 filename from SHA using original path structure."""
    original_path = f"/large/bodmas/exe_cfg/{sha}_refang_cfg.exe"
    return hashlib.md5(original_path.encode()).hexdigest() + '.json.gz'

def process_dataset(metadata_csv, results_dir, batch_size=100, train_ratio=0.7, val_ratio=0.15):
    """Process the dataset into temporal batches using cleaned metadata."""
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(metadata_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Generate MD5 filenames
    print("Generating MD5 filenames...")
    df['filename'] = df['sha'].apply(get_md5_filename)
    df = df.sort_values('timestamp')
    
    # Create splits
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)
    splits = {
        'train': df.iloc[:n_train],
        'val': df.iloc[n_train:n_train + n_val],
        'test': df.iloc[n_train + n_val:]
    }
    
    # Process each split
    total_graphs = 0
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} split...")
        os.makedirs(f'bodmas_batches/{split_name}', exist_ok=True)
        split_graphs = 0
        
        for batch_idx, batch_start in enumerate(range(0, len(split_df), batch_size)):
            batch_df = split_df.iloc[batch_start:batch_start + batch_size]
            batch_graphs = []
            files_found = 0
            files_not_found = 0
            
            for _, row in batch_df.iterrows():
                filepath = os.path.join(results_dir, row['filename'])
                if os.path.exists(filepath):
                    try:
                        with gzip.open(filepath, 'rt') as f:
                            data = json.load(f)
                            graph = convert_to_pytorch_geometric(data['graph_structure'])
                            graph.sha = row['sha']
                            graph.timestamp = row['timestamp']
                            if pd.notna(row['family']):
                                graph.family = row['family']
                            batch_graphs.append(graph)
                            files_found += 1
                    except Exception as e:
                        print(f"Error processing {row['filename']}: {str(e)}")
                else:
                    files_not_found += 1
                    if files_not_found <= 5:  # Only show first 5 missing files
                        print(f"File not found: {row['filename']} (SHA: {row['sha']})")
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"Files found: {files_found}")
            print(f"Files not found: {files_not_found}")
            
            if batch_graphs:
                batch_file = f'bodmas_batches/{split_name}/batch_{batch_idx:04d}.pt'
                torch.save(batch_graphs, batch_file)
                split_graphs += len(batch_graphs)
                print(f"Saved {len(batch_graphs)} graphs to {batch_file}")
                print(f"Time range: {batch_df['timestamp'].min()} to {batch_df['timestamp'].max()}")
        
        print(f"Total graphs saved for {split_name}: {split_graphs}")
        total_graphs += split_graphs
    
    print(f"\nTotal graphs saved: {total_graphs}")

def main():
    # First clean timestamps
    cleaned_csv = clean_timestamps('bodmas_metadata.csv')
    
    # Then process using cleaned metadata
    process_dataset(
        metadata_csv=cleaned_csv,
        results_dir='cfg_analysis_results'
    )

if __name__ == "__main__":
    main()