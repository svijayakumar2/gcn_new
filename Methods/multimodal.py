import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import os
import glob
from tqdm import tqdm
from collections import defaultdict
from typing import Optional
import numpy as np


# GNN model definition
class MalwareGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        # GCN layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        # Output for binary classification
        self.classifier = nn.Linear(hidden_dim, 2)  # 2 classes: benign and malware

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        # Global mean pooling
        x = global_mean_pool(x, batch)
        # Classification output
        return self.classifier(x)

def load_batch(batch_file: str, batch_size: int = 32, device: torch.device = torch.device('cpu')) -> Optional[DataLoader]:
    """Load a batch file, ensure labels are set, and return a DataLoader."""
    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}")
        return None

    batch = torch.load(batch_file)
    processed = []

    for graph in batch:
        try:
            # Assign binary label (malware = 1, benign = 0) based on 'family'
            family = getattr(graph, 'family', None)
            graph.y = torch.tensor(1 if family else 0, dtype=torch.long).to(device)

            # Move graph attributes to the target device
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            graph.edge_attr = graph.edge_attr.to(device)
            processed.append(graph)
        except Exception as e:
            print(f"Error processing graph: {str(e)}")
            continue

    if not processed:
        return None
    return DataLoader(processed, batch_size=batch_size, shuffle=True)


def filter_new_family_batches(val_loader, new_families):
    """Filter validation loader to include only graphs with new families."""
    new_family_batches = []

    for batch in val_loader:
        # Convert batch to list of graphs to check individual families
        graphs = batch.to_data_list()
        new_family_graphs = []
        
        for graph in graphs:
            # Check if this graph belongs to a new family
            if hasattr(graph, 'family') and graph.family in new_families:
                new_family_graphs.append(graph)
        
        # If we found any graphs from new families, create a new batch
        if new_family_graphs:
            from torch_geometric.data import Batch
            new_batch = Batch.from_data_list(new_family_graphs)
            new_family_batches.append(new_batch)

    print(f"Created {len(new_family_batches)} batches containing samples from new families")
    total_samples = sum(len(batch.y) for batch in new_family_batches) if new_family_batches else 0
    print(f"Total samples from new families: {total_samples}")
    
    return new_family_batches

def identify_new_families(train_loader, val_loader):
    """Identify malware family IDs in validation that weren't present in training."""
    train_families = set()
    val_families = set()

    # Collect malware family IDs from training
    for batch in train_loader:
        for graph in batch.to_data_list():
            if hasattr(graph, 'family') and graph.family not in (None, 0):  # Exclude benign (0) and None
                train_families.add(graph.family)

    # Collect malware family IDs from validation
    for batch in val_loader:
        for graph in batch.to_data_list():
            if hasattr(graph, 'family') and graph.family not in (None, 0):  # Exclude benign (0) and None
                val_families.add(graph.family)

    # Find new malware families (in val but not in train)
    new_families = val_families - train_families
    print(f"Found {len(new_families)} new malware families in validation")
    print(f"New family IDs: {sorted(new_families)}")
    return new_families

def evaluate_new_families(model, new_family_batches, criterion, device):
    """Evaluate model on validation graphs with new malware families."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for batch in new_family_batches:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            # For new families, correct means predicting 1 (malware)
            correct += (pred == 1).sum().item()  # These should all be malware
            total += batch.y.size(0)
            predictions.extend(pred.cpu().tolist())
    
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(new_family_batches) if new_family_batches else 0
    
    print("\nEvaluation of new malware families:")
    print(f"Total samples from new families: {total}")
    print(f"Correctly identified as malware: {correct}")
    print(f"Detection rate: {accuracy:.2f}%")
    print(f"Predictions distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    
    return accuracy, avg_loss

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    """Train and validate the binary classification model."""
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.2f}%")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dir = '/data/saranyav/gcn_new/bodmas_batches'
    batch_files = {split: sorted(glob.glob(os.path.join(base_dir, split, 'batch_*.pt'))) 
                  for split in ['train', 'val']}
    print(f"Found {len(batch_files['train'])} training batches and {len(batch_files['val'])} validation batches.")

    # Use load_batch instead of directly loading
    train_loader = load_batch(batch_files['train'][0], batch_size=32, device=device)
    val_loader = load_batch(batch_files['val'][0], batch_size=32, device=device)
    
    if train_loader is None or val_loader is None:
        print("Error: Could not load data")
        return

    # Get number of features from the first batch
    first_batch = next(iter(train_loader))
    num_features = first_batch.x.size(1)

    model = MalwareGNN(num_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    new_families = identify_new_families(train_loader, val_loader)
    print(f"New families in validation: {new_families}")

    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10)

    new_family_batches = filter_new_family_batches(val_loader, new_families)
    new_accuracy, new_loss = evaluate_new_families(model, new_family_batches, criterion, device)
    print(f"Validation Accuracy on New Families: {new_accuracy:.2f}%")

if __name__ == '__main__':
    main()