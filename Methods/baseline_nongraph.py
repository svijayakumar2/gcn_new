import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MalConv(nn.Module):
    """MalConv model for binary classification"""
    def __init__(self, input_length=2**20, num_classes=2):
        super(MalConv, self).__init__()
        self.embedding = nn.Embedding(257, 8, padding_idx=0)
        
        self.conv1 = nn.Conv1d(8, 128, kernel_size=500, stride=500)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=500, stride=500)
        
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class MalwareDataset(Dataset):
    def __init__(self, file_paths: List[Path], labels: List[int], max_length: int = 2**20):
        self.file_paths = file_paths
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Read binary file
        with open(file_path, 'rb') as f:
            binary = f.read()
            
        # Convert to numpy array of uint8
        binary = np.frombuffer(binary, dtype=np.uint8)
        
        # Add padding byte (256) to distinguish padding from null bytes
        if len(binary) > self.max_length:
            binary = binary[:self.max_length]
        else:
            padding_length = self.max_length - len(binary)
            binary = np.pad(binary, (0, padding_length), 
                          constant_values=(0, 256))  # Use 256 for padding
        
        # Convert to torch tensor - MalConv expects long tensor for embedding
        return torch.from_numpy(binary).long(), torch.tensor(label)

class TemporalDatasetManager:
    """Handle temporal splitting and dataset creation"""
    def __init__(self, 
                 exe_dir: Path,
                 metadata_path: Path,
                 behavioral_groups_path: Path,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15):
        self.exe_dir = Path(exe_dir)
        self.metadata_path = Path(metadata_path)
        self.behavioral_groups_path = Path(behavioral_groups_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Load behavioral groups
        with open(self.behavioral_groups_path) as f:
            self.behavioral_groups = json.load(f)
            
        # Create family to group mapping
        self.family_to_group = {}
        for group_id, families in self.behavioral_groups.items():
            for family in families:
                self.family_to_group[family.lower()] = int(group_id)
                
        # Load and process metadata
        self.metadata_df = self._load_metadata()
        
        # Create splits
        self.train_data, self.val_data, self.test_data = self._create_temporal_splits()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load and process metadata"""
        df = pd.read_csv(self.metadata_path)
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Handle missing values in family column and convert to lowercase
        df['family'] = df['family'].fillna('unknown').str.lower()
        
        # Get unique families
        families = sorted(df['family'].unique())
        self.family_to_idx = {family: idx for idx, family in enumerate(families)}
        
        # Add behavioral group mapping
        df['behavioral_group'] = df['family'].apply(
            lambda x: self.family_to_group.get(x, -1)  # -1 for unknown group
        )
        
        # Create group mapping
        unique_groups = sorted(set(self.family_to_group.values()))
        self.group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
        self.group_to_idx[-1] = len(self.group_to_idx)  # Add unknown group
        
        return df
    
    def _create_temporal_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create temporal train/val/test splits"""
        n = len(self.metadata_df)
        train_idx = int(n * self.train_ratio)
        val_idx = int(n * (self.train_ratio + self.val_ratio))
        
        return (
            self.metadata_df.iloc[:train_idx],
            self.metadata_df.iloc[train_idx:val_idx],
            self.metadata_df.iloc[val_idx:]
        )
    
    def create_datasets(self, batch_size: int = 32) -> Dict[str, DataLoader]:
        """Create DataLoader objects for each split"""
        loaders = {}
        
        for split_name, split_data in [
            ('train', self.train_data),
            ('val', self.val_data),
            ('test', self.test_data)
        ]:
            # Get file paths and labels
            file_paths = []
            family_labels = []
            group_labels = []
            
            for _, row in split_data.iterrows():
                exe_path = self.exe_dir / f"{row['sha']}_refang.exe"
                if exe_path.exists():
                    file_paths.append(exe_path)
                    family_labels.append(self.family_to_idx[row['family']])
                    group_labels.append(self.group_to_idx[row['behavioral_group']])
            
            # Create datasets
            family_dataset = MalwareDataset(file_paths, family_labels)
            group_dataset = MalwareDataset(file_paths, group_labels)
            
            # Create dataloaders
            loaders[f'{split_name}_family'] = DataLoader(
                family_dataset, 
                batch_size=batch_size,
                shuffle=(split_name == 'train')
            )
            
            loaders[f'{split_name}_group'] = DataLoader(
                group_dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train')
            )
            
        return loaders

class MalwareTrainer:
    """Handle training and evaluation"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
            
        metrics = self._compute_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy())
        
        metrics = self._compute_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def _compute_metrics(self, true_labels, predictions):
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset manager
    data_manager = TemporalDatasetManager(
        exe_dir=Path('/data/datasets/bodmas_exes/refanged_exes/'),
        metadata_path=Path('/data/saranyav/gcn_new/bodmas_metadata_cleaned.csv'),
        behavioral_groups_path=Path('behavioral_groups.json')
    )
    
    # Initialize models
    family_model = MalConv(num_classes=len(data_manager.family_to_idx)).to(device)
    group_model = MalConv(num_classes=len(data_manager.group_to_idx)).to(device)
    
    # Train and evaluate
    loaders = data_manager.create_datasets(batch_size=32)
    
    family_trainer = MalwareTrainer(family_model, device)
    group_trainer = MalwareTrainer(group_model, device)
    
    # Training loop
    num_epochs = 100
    early_stopping_patience = 5
    best_family_f1 = 0
    best_group_f1 = 0
    family_patience = 0
    group_patience = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train family model
        family_metrics = family_trainer.train_epoch(loaders['train_family'])
        family_val_metrics = family_trainer.evaluate(loaders['val_family'])
        
        # Train group model
        group_metrics = group_trainer.train_epoch(loaders['train_group'])
        group_val_metrics = group_trainer.evaluate(loaders['val_group'])
        
        # Early stopping checks
        if family_val_metrics['f1'] > best_family_f1:
            best_family_f1 = family_val_metrics['f1']
            family_patience = 0
            torch.save(family_model.state_dict(), 'best_family_model_nongraph.pt')
        else:
            family_patience += 1
            
        if group_val_metrics['f1'] > best_group_f1:
            best_group_f1 = group_val_metrics['f1']
            group_patience = 0
            torch.save(group_model.state_dict(), 'best_group_model_nongraph.pt')
        else:
            group_patience += 1
            
        # Check early stopping
        if family_patience >= early_stopping_patience and group_patience >= early_stopping_patience:
            logger.info("Early stopping triggered")
            break
    
    # Final evaluation
    logger.info("Loading best models for final evaluation...")
    
    family_model.load_state_dict(torch.load('best_family_model_nongraph.pt'))
    group_model.load_state_dict(torch.load('best_group_model_nongraph.pt'))
    
    test_family_metrics = family_trainer.evaluate(loaders['test_family'])
    test_group_metrics = group_trainer.evaluate(loaders['test_group'])
    
    logger.info("\nFinal Test Results:")
    logger.info("Family Classification:")
    for metric, value in test_family_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
        
    logger.info("\nBehavioral Group Classification:")
    for metric, value in test_group_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()