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
from dateutil.relativedelta import relativedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetadataManager:
    """Handle metadata loading and merging."""
    
    @staticmethod
    def load_and_merge_metadata(primary_metadata_path: str, malware_types_path: str) -> pd.DataFrame:
        """Load and merge primary metadata with malware types."""
        logger.info(f"Reading primary metadata from {primary_metadata_path}")
        primary_df = pd.read_csv(primary_metadata_path)
        
        logger.info(f"Reading malware types from {malware_types_path}")
        malware_types_df = pd.read_csv(malware_types_path)
        
        # Extract filename without extension for joining
        primary_df['filename'] = primary_df['sha'].apply(lambda x: x)
        malware_types_df['filename'] = malware_types_df['sha256'].apply(lambda x: Path(x).stem)
        # drop duplicate column 256
        malware_types_df.drop('sha256', axis=1, inplace=True)
        
        # Merge dataframes
        merged_df = pd.merge(
            primary_df,
            malware_types_df[['filename', 'category']],
            on='filename',
            how='left'
        )
        
        # Clean up
        merged_df.drop('filename', axis=1, inplace=True)
        
        # Fill missing malware types
        merged_df['malware_type'] = merged_df['category'].fillna('unknown')
        
        logger.info(f"Merged metadata shape: {merged_df.shape}")
        logger.info("\nMalware type distribution:")
        logger.info(merged_df['malware_type'].value_counts())
        
        return merged_df

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
            dt = pd.to_datetime(ts)
            return dt.tz_localize('UTC' if dt.tz is None else None).strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception as e:
            logger.error(f"Failed to parse timestamp {ts}: {str(e)}")
            return None

    @classmethod
    def clean_metadata_timestamps(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize timestamps in metadata DataFrame."""
        if 'timestamp' not in df.columns:
            raise ValueError("Metadata must contain a 'timestamp' column")
            
        logger.info("Original timestamp samples:")
        logger.info(df['timestamp'].head())
        
        df['timestamp'] = df['timestamp'].apply(cls.standardize_timestamp)
        df = df.dropna(subset=['timestamp'])
        
        logger.info("Cleaned timestamp samples:")
        logger.info(df['timestamp'].head())
            
        return df

class GraphConverter:
    """Convert graph structures to PyTorch Geometric format."""
    @staticmethod
    def convert_to_pytorch_geometric(graph_structure: Dict) -> Data:
        """Convert graph structure to PyG format with error handling."""
        try:
            # Handle node features directly as a list
            node_feats = torch.tensor(graph_structure['node_features'], dtype=torch.float)
            
            # Convert edge indices
            edge_index = torch.tensor(graph_structure['edge_index'], dtype=torch.long).t() \
                if graph_structure['edge_index'] else torch.zeros((2, 0), dtype=torch.long)
            
            # Handle edge features if present
            if graph_structure.get('edge_features'):
                edge_feats = [[float(edge.get('condition', False) is not None)] 
                             for edge in graph_structure['edge_features']]
                edge_attr = torch.tensor(edge_feats, dtype=torch.float)
            else:
                edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.float)
            
            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
            data.num_nodes = node_feats.size(0)
            return data
            
        except Exception as e:
            logger.error(f"Error converting graph structure: {str(e)}")
            raise

class DatasetProcessor:
    def __init__(self, 
                 primary_metadata_path: str,
                 malware_types_path: str,
                 data_dir: str,
                 output_dir: str,
                 batch_size: int = 100,
                 window_size: str = '6M',  # Training window size (e.g., '6M' for 6 months)
                 eval_size: str = '1M',    # Evaluation window size
                 test_hold_out: str = '3M'): # Final testing period to hold out
        self.primary_metadata_path = primary_metadata_path
        self.malware_types_path = malware_types_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.window_size = self.parse_relative_delta(window_size)
        self.eval_size = self.parse_relative_delta(eval_size)
        self.test_hold_out = self.parse_relative_delta(test_hold_out)
        self.graph_converter = GraphConverter()  # Retained, assuming it's still needed


    def _process_split(self, split_df: pd.DataFrame, split_dir: Path) -> int:
        """Process a single data split with feature validation."""
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
                        
                        # Validate feature dimensions
                        expected_features = 14  # Hardcode for now as we know the exact number
                        actual_features = graph.x.size(1)
                        if actual_features != expected_features:
                            logger.error(f"Feature dimension mismatch in {row['sha']}: "
                                    f"expected {expected_features}, got {actual_features}")
                            logger.error(f"Feature tensor shape: {graph.x.shape}")
                            continue

                        # Add metadata
                        graph.sha = row['sha']
                        graph.timestamp = row['timestamp']
                        graph.family = row['family'] if pd.notna(row.get('family')) else 'benign'
                        graph.malware_type = row['malware_type']

                        batch_graphs.append(graph)
                        
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {str(e)}")
                    continue
            
            # Validate batch feature consistency before saving
            if batch_graphs:
                feature_dims = [g.x.size(1) for g in batch_graphs]
                if len(set(feature_dims)) > 1:
                    logger.error(f"Inconsistent feature dimensions in batch {batch_idx}:")
                    for i, g in enumerate(batch_graphs):
                        logger.error(f"Graph {i} ({g.sha}): {g.x.size(1)} features")
                    # Filter out graphs with wrong dimensions
                    batch_graphs = [g for g in batch_graphs if g.x.size(1) == expected_features]
                
                if batch_graphs:  # Only save if we still have valid graphs
                    batch_file = split_dir / f"batch_{batch_idx:04d}.pt"
                    torch.save(batch_graphs, batch_file)
                    processed_count += len(batch_graphs)
                    
                    logger.info(f"Saved {len(batch_graphs)} graphs to {batch_file}")
                    logger.info(f"Time range: {batch_df['timestamp'].min()} to {batch_df['timestamp'].max()}")
            
        return processed_count

    @staticmethod
    def parse_relative_delta(period: str) -> relativedelta:
        """Convert a string like '6M' or '1Y' to a relativedelta object."""
        if period.endswith('M'):
            return relativedelta(months=int(period[:-1]))
        elif period.endswith('Y'):
            return relativedelta(years=int(period[:-1]))
        else:
            raise ValueError(f"Unsupported period format: {period}")
            
    def process(self):
        """Process the dataset using rolling windows."""
        # Load and merge metadata
        df = MetadataManager.load_and_merge_metadata(
            self.primary_metadata_path,
            self.malware_types_path
        )
        
        # Convert timestamps to datetime and sort by time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Filter metadata to include only files present in the data directory
        available_files = list(self.data_dir.glob('*.json.gz'))
        available_shas = {f.stem.replace('.json', '') for f in available_files}
        df = df[df['sha'].isin(available_shas)]

        if df.empty:
            logger.warning("No matching files found in the dataset. Exiting process.")
            return

        logger.info(f"Filtered dataset size: {len(df)} records with available files.")

        # Hold out the final test period
        test_start = df['timestamp'].max() - self.test_hold_out
        test_df = df[df['timestamp'] >= test_start]
        train_val_df = df[df['timestamp'] < test_start]

        # Create rolling windows
        windows = []
        current_start = train_val_df['timestamp'].min()
        final_time = train_val_df['timestamp'].max()

        while current_start + self.window_size + self.eval_size <= final_time:
            train_end = current_start + self.window_size
            eval_end = train_end + self.eval_size

            train_split = train_val_df[
                (train_val_df['timestamp'] >= current_start) &
                (train_val_df['timestamp'] < train_end)
            ]
            val_split = train_val_df[
                (train_val_df['timestamp'] >= train_end) &
                (train_val_df['timestamp'] < eval_end)
            ]

            # Skip creating windows if train or val splits are empty
            if train_split.empty or val_split.empty:
                logger.info(f"Skipping empty rolling window: "
                            f"Train ({len(train_split)}), Val ({len(val_split)})")
            else:
                windows.append({'train': train_split, 'val': val_split})
                logger.info(f"Added rolling window: Train ({len(train_split)}), Val ({len(val_split)})")

            # Move window forward by evaluation period
            current_start += self.eval_size

        # Process each rolling window
        for window_idx, window in enumerate(windows):
            logger.info(f"Processing window {window_idx + 1}/{len(windows)}")
            window_dir = self.output_dir / f"window_{window_idx:03d}"
            for split_name, split_df in window.items():
                split_dir = window_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                processed_count = self._process_split(split_df, split_dir)
                logger.info(f"Processed {processed_count} graphs in {split_name} split.")

        # Process the held-out test set
        test_dir = self.output_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_count = self._process_split(test_df, test_dir)
        logger.info(f"Processed {test_count} graphs in test split.")
    
def main():
    processor = DatasetProcessor(
        primary_metadata_path='bodmas_metadata_cleaned.csv',
        malware_types_path='bodmas_malware_category.csv',
        data_dir='cfg_features',
        output_dir='bodmas_rolling',
        batch_size=100,
        window_size='6M',
        eval_size='1M',
        test_hold_out='3M'
    )
    processor.process()

if __name__ == "__main__":
    main()



# class DatasetProcessor:
#     """Process and batch the dataset."""
    
#     def __init__(self, 
#                  primary_metadata_path: str,
#                  malware_types_path: str,
#                  data_dir: str,
#                  output_dir: str,
#                  batch_size: int = 100,
#                  train_ratio: float = 0.7,
#                  val_ratio: float = 0.15):
#         self.primary_metadata_path = primary_metadata_path
#         self.malware_types_path = malware_types_path
#         self.data_dir = Path(data_dir)
#         self.output_dir = Path(output_dir)
#         self.batch_size = batch_size
#         self.train_ratio = train_ratio
#         self.val_ratio = val_ratio
#         self.graph_converter = GraphConverter()
    
#     def process(self):
#         """Process the complete dataset."""
#         # Load and merge metadata
#         df = MetadataManager.load_and_merge_metadata(
#             self.primary_metadata_path,
#             self.malware_types_path
#         )
        
#         # Clean metadata timestamps
#         df = TimestampCleaner.clean_metadata_timestamps(df)
        
#         # Sort by timestamp - we want temporal splits, not random
#         df = df.sort_values('timestamp')
#         logger.info(f"Processing {len(df)} samples")
        
#         # Create splits
#         n_train = int(len(df) * self.train_ratio)
#         n_val = int(len(df) * self.val_ratio)
#         splits = {
#             'train': df.iloc[:n_train],
#             'val': df.iloc[n_train:n_train + n_val],
#             'test': df.iloc[n_train + n_val:]
#         }
        
#         # Process each split
#         total_processed = 0
#         for split_name, split_df in splits.items():
#             logger.info(f"\nProcessing {split_name} split ({len(split_df)} samples)")
#             split_dir = self.output_dir / split_name
#             split_dir.mkdir(parents=True, exist_ok=True)
            
#             processed = self._process_split(split_df, split_dir)
#             total_processed += processed
            
#         logger.info(f"\nTotal processed graphs: {total_processed}")
#         return total_processed

#     def _process_split(self, split_df: pd.DataFrame, split_dir: Path) -> int:
#         """Process a single data split with feature validation."""
#         processed_count = 0
        
#         # Process in batches
#         for batch_idx, batch_start in enumerate(range(0, len(split_df), self.batch_size)):
#             batch_df = split_df.iloc[batch_start:batch_start + self.batch_size]
#             batch_graphs = []
            
#             # Process each file in the batch
#             for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), 
#                             desc=f"Processing batch {batch_idx + 1}"):
#                 try:
#                     filepath = self.data_dir / f"{row['sha']}.json.gz"
                    
#                     if not filepath.exists():
#                         logger.warning(f"File not found: {filepath}")
#                         continue
                    
#                     with gzip.open(filepath, 'rt') as f:
#                         data = json.load(f)
#                         graph = self.graph_converter.convert_to_pytorch_geometric(data['graph_structure'])
                        
#                         # Validate feature dimensions
#                         expected_features = 14  # Hardcode for now as we know the exact number
#                         actual_features = graph.x.size(1)
#                         if actual_features != expected_features:
#                             logger.error(f"Feature dimension mismatch in {row['sha']}: "
#                                     f"expected {expected_features}, got {actual_features}")
#                             logger.error(f"Feature tensor shape: {graph.x.shape}")
#                             continue

#                         # Add metadata
#                         graph.sha = row['sha']
#                         graph.timestamp = row['timestamp']
#                         graph.family = row['family'] if pd.notna(row.get('family')) else 'benign'
#                         graph.malware_type = row['malware_type']

#                         batch_graphs.append(graph)
                        
#                 except Exception as e:
#                     logger.error(f"Error processing {filepath}: {str(e)}")
#                     continue
            
#             # Validate batch feature consistency before saving
#             if batch_graphs:
#                 feature_dims = [g.x.size(1) for g in batch_graphs]
#                 if len(set(feature_dims)) > 1:
#                     logger.error(f"Inconsistent feature dimensions in batch {batch_idx}:")
#                     for i, g in enumerate(batch_graphs):
#                         logger.error(f"Graph {i} ({g.sha}): {g.x.size(1)} features")
#                     # Filter out graphs with wrong dimensions
#                     batch_graphs = [g for g in batch_graphs if g.x.size(1) == expected_features]
                
#                 if batch_graphs:  # Only save if we still have valid graphs
#                     batch_file = split_dir / f"batch_{batch_idx:04d}.pt"
#                     torch.save(batch_graphs, batch_file)
#                     processed_count += len(batch_graphs)
                    
#                     logger.info(f"Saved {len(batch_graphs)} graphs to {batch_file}")
#                     logger.info(f"Time range: {batch_df['timestamp'].min()} to {batch_df['timestamp'].max()}")
            
#         return processed_count
