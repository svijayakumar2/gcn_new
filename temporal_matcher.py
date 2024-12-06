import os
import gzip
import json
import torch
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from torch_geometric.data import Data
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimestampCleaner:
    """Handle timestamp cleaning and standardization."""
    
    @staticmethod
    def standardize_timestamp(ts: str) -> str:
        """Convert various timestamp formats to standard UTC format."""
        timestamp_formats = [
            '%m/%d/%y %H:%M',
            '%Y-%m-%d %H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M'
        ]
        
        for fmt in timestamp_formats:
            try:
                dt = pd.to_datetime(ts, format=fmt)
                return dt.tz_localize('UTC' if dt.tz is None else None).strftime('%Y-%m-%d %H:%M:%S UTC')
            except ValueError:
                continue
        
        try:
            # Last resort: try pandas' flexible parser
            dt = pd.to_datetime(ts)
            return dt.tz_localize('UTC' if dt.tz is None else None).strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception as e:
            logger.error(f"Failed to parse timestamp {ts}: {str(e)}")
            return None

    @classmethod
    def clean_metadata_timestamps(cls, metadata_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """Clean and standardize timestamps in metadata file."""
        logger.info(f"Reading metadata from {metadata_path}")
        df = pd.read_csv(metadata_path)
        
        if 'timestamp' not in df.columns:
            raise ValueError("Metadata file must contain a 'timestamp' column")
            
        logger.info("Original timestamp samples:")
        logger.info(df['timestamp'].head())
        
        df['timestamp'] = df['timestamp'].apply(cls.standardize_timestamp)
        df = df.dropna(subset=['timestamp'])
        
        logger.info("Cleaned timestamp samples:")
        logger.info(df['timestamp'].head())
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned metadata to {output_path}")
            
        return df

class GraphConverter:
    """Convert graph structures to PyTorch Geometric format."""
    
    @staticmethod
    def convert_to_pytorch_geometric(graph_structure: Dict) -> Data:
        """Convert graph structure to PyG format with error handling."""
        try:
            # Extract feature keys
            feature_keys = sorted(list({
                key for node in graph_structure['node_features'] 
                for key in node.keys() if key != 'id'  # Exclude id from features
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
            
            # Convert to tensors with proper error handling
            x = torch.tensor(node_feats, dtype=torch.float) if node_feats else \
                torch.zeros((len(graph_structure['node_features']), len(feature_keys)), dtype=torch.float)
            
            edge_index = torch.tensor(graph_structure['edge_index'], dtype=torch.long).t() \
                if graph_structure['edge_index'] else torch.zeros((2, 0), dtype=torch.long)
            
            # Handle edge features if present
            edge_attr = None
            if graph_structure.get('edge_features'):
                edge_feats = [[float(edge.get('condition', False) is not None)] 
                             for edge in graph_structure['edge_features']]
                edge_attr = torch.tensor(edge_feats, dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.num_nodes = x.size(0)
            return data
            
        except Exception as e:
            logger.error(f"Error converting graph structure: {str(e)}")
            raise

class DatasetProcessor:
    """Process and batch the dataset."""
    
    def __init__(self, 
                 metadata_path: str,
                 data_dir: str,
                 output_dir: str,
                 batch_size: int = 100,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15):
        self.metadata_path = metadata_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.graph_converter = GraphConverter()
    
    def process(self):
        """Process the complete dataset."""
        # Clean metadata
        df = TimestampCleaner.clean_metadata_timestamps(self.metadata_path)
        
        # Sort by timestamp -  we want temporal splits, not random
        df = df.sort_values('timestamp')
        logger.info(f"Processing {len(df)} samples")
        
        # Create splits
        n_train = int(len(df) * self.train_ratio)
        n_val = int(len(df) * self.val_ratio)
        splits = {
            'train': df.iloc[:n_train],
            'val': df.iloc[n_train:n_train + n_val],
            'test': df.iloc[n_train + n_val:]
        }
        
        # Process each split
        total_processed = 0
        for split_name, split_df in splits.items():
            logger.info(f"\nProcessing {split_name} split ({len(split_df)} samples)")
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            processed = self._process_split(split_df, split_dir)
            total_processed += processed
            
        logger.info(f"\nTotal processed graphs: {total_processed}")
        return total_processed
    
    def _process_split(self, split_df: pd.DataFrame, split_dir: Path) -> int:
        """Process a single data split."""
        processed_count = 0
        
        # Process in batches
        for batch_idx, batch_start in enumerate(range(0, len(split_df), self.batch_size)):
            batch_df = split_df.iloc[batch_start:batch_start + self.batch_size]
            batch_graphs = []
            
            # Process each file in the batch
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), 
                             desc=f"Processing batch {batch_idx + 1}"):
                try:
                    filepath = self.data_dir / f"{row['sha']}.json.gz"
                    
                    if not filepath.exists():
                        logger.warning(f"File not found: {filepath}")
                        continue
                    
                    with gzip.open(filepath, 'rt') as f:
                        data = json.load(f)
                        graph = self.graph_converter.convert_to_pytorch_geometric(data['graph_structure'])
                        
                        # Add metadata
                        graph.sha = row['sha']
                        graph.timestamp = row['timestamp']
                        if pd.notna(row.get('family')):
                            graph.family = row['family']
                            
                        batch_graphs.append(graph)
                        
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {str(e)}")
                    continue
            
            # Save batch if not empty
            if batch_graphs:
                batch_file = split_dir / f"batch_{batch_idx:04d}.pt"
                torch.save(batch_graphs, batch_file)
                processed_count += len(batch_graphs)
                
                logger.info(f"Saved {len(batch_graphs)} graphs to {batch_file}")
                logger.info(f"Time range: {batch_df['timestamp'].min()} to {batch_df['timestamp'].max()}")
            
        return processed_count

def main():
    processor = DatasetProcessor(
        metadata_path='bodmas_metadata_cleaned.csv',
        data_dir='cfg_analysis_results',#'cfg_analysis_results',
        output_dir='bodmas_batches',
        batch_size=100,
        train_ratio=0.7,
        val_ratio=0.15
    )           
    
    processor.process()

if __name__ == "__main__":
    main()