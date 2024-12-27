import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

class ContrastiveGNN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_families: int,
        num_groups: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.unknown_group_id = -999  # Match your unknown group ID
        
        # Encoder
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        
        # Classification heads
        self.group_classifier = torch.nn.Linear(embedding_dim, num_groups)
        self.family_classifiers = torch.nn.ModuleDict({
            str(i): torch.nn.Linear(embedding_dim, num_families)
            for i in range(num_groups)
        })

    def get_embeddings(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        # Handle batch index
        if not hasattr(data, 'batch'):
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            batch = data.batch

        # GNN layers
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = self.conv2(h1, edge_index)
        
        # Global pooling
        pooled = global_mean_pool(h2, batch)
        return pooled

    def contrastive_loss(self, embeddings: torch.Tensor, families: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss handling unknown groups.
        
        Args:
            embeddings: Graph embeddings [batch_size, embedding_dim]
            families: Family labels [batch_size]
            groups: Group labels [batch_size]
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Convert labels to tensor if they aren't already
        if not isinstance(families, torch.Tensor):
            families = torch.tensor(families, device=device)
        if not isinstance(groups, torch.Tensor):
            groups = torch.tensor(groups, device=device)

        # Reshape labels
        families = families.view(-1, 1)
        groups = groups.view(-1, 1)

        # Create mask for known samples (not in unknown group)
        known_mask = (groups != self.unknown_group_id).float()
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()) / self.temperature

        # Create positive pair mask (same family AND same group, excluding unknowns)
        family_mask = torch.eq(families, families.t()).float()
        group_mask = torch.eq(groups, groups.t()).float()
        valid_mask = torch.matmul(known_mask, known_mask.t())
        pos_mask = family_mask * group_mask * valid_mask
        
        # Remove self-contrast
        pos_mask.fill_diagonal_(0)

        # Handle case where there are no positive pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Compute loss only for known samples
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        loss = -(log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        
        # Only consider loss for known samples
        return (loss * known_mask.squeeze()).sum() / known_mask.sum().clamp(min=1)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        embeddings = self.get_embeddings(data)
        
        # Group classification
        group_logits = self.group_classifier(embeddings)
        
        # Family classification per group
        family_logits = {}
        pred_groups = group_logits.argmax(dim=1)
        
        for group_id in range(len(self.family_classifiers)):
            group_mask = (pred_groups == group_id)
            if group_mask.any():
                group_embeddings = embeddings[group_mask]
                family_logits[str(group_id)] = self.family_classifiers[str(group_id)](group_embeddings)

        return embeddings, group_logits, family_logits

class ContrastiveTrainer:
    def __init__(self, model: ContrastiveGNN, device: torch.device, lr: float = 0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, train_loader: DataLoader, family_to_group: Dict[str, int]) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            try:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                embeddings, group_logits, family_logits = self.model(batch)

                # Ensure we have family and group labels
                if not hasattr(batch, 'y') or not hasattr(batch, 'group'):
                    continue

                # Contrastive loss
                contrastive_loss = self.model.contrastive_loss(
                    embeddings,
                    batch.y,  # family labels
                    batch.group  # group labels
                )

                # Group classification loss (exclude unknown groups)
                known_mask = (batch.group != -999)
                if known_mask.any():
                    group_loss = F.cross_entropy(
                        group_logits[known_mask],
                        batch.group[known_mask]
                    )
                else:
                    group_loss = torch.tensor(0.0, device=self.device)

                # Family classification loss
                family_loss = 0
                num_groups = 0
                for group_id, logits in family_logits.items():
                    group_mask = (batch.group == int(group_id))
                    if group_mask.any():
                        family_loss += F.cross_entropy(logits, batch.y[group_mask])
                        num_groups += 1

                if num_groups > 0:
                    family_loss /= num_groups
                else:
                    family_loss = torch.tensor(0.0, device=self.device)

                # Combined loss
                loss = group_loss + family_loss + contrastive_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logging.error(f"Error in batch: {str(e)}")
                continue

        return total_loss / max(1, num_batches)

def prepare_data(data_dir: str, family_to_group: Dict[str, int], batch_size: int = 32) -> Dict[str, DataLoader]:
    """
    Prepare data loaders with proper label handling.
    """
    loaders = {}
    data_dir = Path(data_dir)

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        processed_graphs = []
        for batch_file in split_dir.glob('batch_*.pt'):
            try:
                batch_data = torch.load(batch_file)
                for graph in batch_data:
                    # Skip invalid graphs
                    if not isinstance(graph, Data) or not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                        continue

                    # Get family and assign group
                    family = getattr(graph, 'family', None)
                    if family is None:
                        continue

                    # Assign group ID (unknown_group_id if family not in mapping)
                    group_id = family_to_group.get(family, -999)
                    
                    # Create new graph with proper labels
                    new_graph = Data(
                        x=graph.x,
                        edge_index=graph.edge_index,
                        y=torch.tensor([family_to_group[family]] if family in family_to_group else [-999]),
                        group=torch.tensor([group_id])
                    )
                    
                    processed_graphs.append(new_graph)

            except Exception as e:
                logging.error(f"Error processing {batch_file}: {str(e)}")
                continue

        if processed_graphs:
            loaders[split] = DataLoader(
                processed_graphs,
                batch_size=batch_size,
                shuffle=(split == 'train')
            )

    return loaders

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Configuration
    config = {
        'data_dir': '/data/saranyav/gcn_new/bodmas_batches',  # Updated to your path
        'groups_file': '/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json',  # Updated to your path
        'num_epochs': 100,
        'batch_size': 32,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'learning_rate': 0.001
    }

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        # Load group mappings
        with open(config['groups_file']) as f:
            group_data = json.load(f)
            
        family_to_group = {}
        for group_id, families in group_data.items():
            for family in families:
                family_to_group[family] = int(group_id)

        # Prepare data
        loaders = prepare_data(
            config['data_dir'],
            family_to_group,
            config['batch_size']
        )

        if not loaders:
            raise ValueError("No data loaded!")

        # Get dimensions from first batch
        first_batch = next(iter(loaders['train']))
        num_features = first_batch.x.size(1)
        num_families = len(set(family_to_group.keys()))
        num_groups = len(set(family_to_group.values()))

        # Initialize model
        model = ContrastiveGNN(
            num_node_features=num_features,
            num_families=num_families,
            num_groups=num_groups,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim']
        )

        # Initialize trainer
        trainer = ContrastiveTrainer(
            model,
            device,
            lr=config['learning_rate']
        )

        # Training loop
        for epoch in range(config['num_epochs']):
            train_loss = trainer.train_epoch(loaders['train'], family_to_group)
            logging.info(f"Epoch {epoch}: Loss = {train_loss:.4f}")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()