import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import pandas as pd
from torch_geometric.data import Data, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import pandas as pd
from torch_geometric.data import Data, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import pandas as pd
from torch_geometric.data import Data, DataLoader
from gcn import CentroidLayer, MalwareGNN, MalwareTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TemporalMalwareDataLoader:
    """Load and process temporal malware data with both family and behavioral group labels."""
    
    def __init__(self, 
                 batch_dir: Path,
                 behavioral_groups_path: Path,
                 metadata_path: Path,
                 malware_types_path: Path):
        self.batch_dir = Path(batch_dir)
        
        # Load behavioral groups
        with open(behavioral_groups_path) as f:
            behavioral_groups = json.load(f)
            
        # Create mappings
        self.family_to_group = {}
        for group_id, families in behavioral_groups.items():
            for family in families:
                self.family_to_group[family.lower()] = int(group_id)
                
        # Load metadata
        self.metadata_df = self._load_metadata(metadata_path, malware_types_path)
        
        # Initialize with timezone-aware timestamps
        default_min_ts = pd.Timestamp.min.tz_localize('UTC')
        default_max_ts = pd.Timestamp.max.tz_localize('UTC')
        
        # Track statistics with timezone-aware timestamps
        self.family_first_seen = defaultdict(lambda: default_max_ts)
        self.family_last_seen = defaultdict(lambda: default_min_ts)
        self.family_counts = defaultdict(int)
        self.group_counts = defaultdict(int)

    def _standardize_timestamp(self, ts: str) -> pd.Timestamp:
        """Convert timestamp string to pandas Timestamp with UTC timezone."""
        try:
            # Parse timestamp with pandas
            dt = pd.to_datetime(ts)
            # Ensure UTC timezone
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            else:
                dt = dt.tz_convert('UTC')
            return dt
        except Exception as e:
            logger.error(f"Error parsing timestamp {ts}: {str(e)}")
            return None

    def _process_graph(self, graph: Data, use_groups: bool = False) -> Optional[Data]:
        """Process a single graph, adding family and group labels."""
        try:
            # Validate required attributes
            if not all(hasattr(graph, attr) for attr in ['family', 'timestamp']):
                return None
            
            # Clean family name
            family = graph.family.lower() if hasattr(graph, 'family') else 'unknown'
            
            # Standardize timestamp to UTC
            timestamp = self._standardize_timestamp(graph.timestamp)
            if timestamp is None:
                logger.warning(f"Invalid timestamp for family {family}")
                return None

            # Update temporal statistics with standardized timestamps
            self.family_first_seen[family] = min(
                self.family_first_seen[family],
                timestamp
            )
            self.family_last_seen[family] = max(
                self.family_last_seen[family],
                timestamp
            )
            self.family_counts[family] += 1
            
            # Add behavioral group
            group = self.family_to_group.get(family, -1)
            if group >= 0:
                self.group_counts[group] += 1
                graph.group = torch.tensor(group, dtype=torch.long)
            else:
                logger.warning(f"Unknown family: {family}")
                return None
            
            # Set target based on mode
            graph.y = graph.group if use_groups else torch.tensor(
                list(self.family_to_group.keys()).index(family), 
                dtype=torch.long
            )
            
            # Store standardized timestamp
            graph.timestamp = timestamp
            
            return graph
            
        except Exception as e:
            logger.error(f"Error processing graph: {str(e)}")
            logger.error(f"Family: {family if 'family' in locals() else 'unknown'}")
            logger.error(f"Timestamp: {graph.timestamp if hasattr(graph, 'timestamp') else 'missing'}")
            return None
      
    def _load_metadata(self, metadata_path: Path, malware_types_path: Path) -> pd.DataFrame:
        """Load and merge metadata."""
        metadata_df = pd.read_csv(metadata_path)
        malware_types_df = pd.read_csv(malware_types_path)
        
        # Prepare for merge
        metadata_df['filename'] = metadata_df['sha']
        malware_types_df['filename'] = malware_types_df['sha256'].apply(lambda x: Path(x).stem)
        
        # Merge
        merged_df = pd.merge(
            metadata_df,
            malware_types_df[['filename', 'category']].rename(columns={'category': 'malware_type'}),
            on='filename',
            how='left'
        )
        
        return merged_df
        
    def load_split(self, split: str = 'train', batch_size: int = 32, 
                  use_groups: bool = False) -> DataLoader:
        """Load a data split maintaining temporal order."""
        split_dir = self.batch_dir / split
        batch_files = sorted(list(split_dir.glob('batch_*.pt')))
        
        logger.info(f"Loading {len(batch_files)} batch files from {split_dir}")
        
        all_graphs = []
        skipped = 0
        
        for batch_file in batch_files:
            try:
                batch_data = torch.load(batch_file)
                
                # Process each graph
                for graph in batch_data:
                    processed = self._process_graph(graph, use_groups)
                    if processed is not None:
                        all_graphs.append(processed)
                    else:
                        skipped += 1
                        
            except Exception as e:
                logger.error(f"Error loading {batch_file}: {str(e)}")
                continue
        
        # Sort by timestamp
        all_graphs.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Loaded {len(all_graphs)} graphs from {split} split "
                   f"({skipped} skipped)")
        
        # Create loader
        loader = DataLoader(all_graphs, batch_size=batch_size, shuffle=False)
        return loader
        
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'families': {
                'total': len(self.family_counts),
                'counts': dict(self.family_counts),
                'temporal_range': {
                    family: {
                        'first_seen': self.family_first_seen[family],
                        'last_seen': self.family_last_seen[family]
                    }
                    for family in self.family_counts
                }
            },
            'groups': {
                'total': len(self.group_counts),
                'counts': dict(self.group_counts)
            }
        }
        return stats

def main():
    """Test data loading."""
    loader = TemporalMalwareDataLoader(
        batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches'),
        behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        metadata_path=Path('bodmas_metadata_cleaned.csv'),
        malware_types_path=Path('bodmas_malware_category.csv')
    )
    
    # Test family-level loading
    train_loader = loader.load_split('train', use_groups=False)
    logger.info("Family-level statistics:")
    stats = loader.get_statistics()
    logger.info(f"Total families: {stats['families']['total']}")
    logger.info(f"Total groups: {stats['groups']['total']}")
    
    # Test group-level loading
    train_loader_groups = loader.load_split('train', use_groups=True)
    
    # Print example batch
    batch = next(iter(train_loader_groups))
    logger.info(f"\nExample batch:")
    logger.info(f"Batch size: {batch.num_graphs}")
    logger.info(f"Node features: {batch.x.shape}")
    logger.info(f"Labels: {batch.y}")

if __name__ == "__main__":
    main()