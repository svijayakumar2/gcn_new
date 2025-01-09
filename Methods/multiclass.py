import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import os
import glob
from collections import defaultdict
from typing import Optional
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import copy


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

##########

class MalwareGNN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class FamilyLabelEncoder:
    def __init__(self):
        self.family_to_id = {'benign': 0}
        self.id_to_family = {0: 'benign'}
        self.next_id = 1

    def get_num_classes(self):
        return len(self.family_to_id)

    def fit(self, families):
        for family in families:
            if family not in self.family_to_id:
                self.family_to_id[family] = self.next_id
                self.id_to_family[self.next_id] = family
                self.next_id += 1
        return self

    def transform(self, family):
        if family in self.family_to_id:
            return self.family_to_id[family]
        raise ValueError(f"Unknown family: {family}")

def load_batch(batch_file: str, label_encoder, batch_size: int = 32, device: torch.device = torch.device('cpu')) -> Optional[DataLoader]:
    """Load a batch file with multiclass labels."""
    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}")
        return None

    batch = torch.load(batch_file)
    processed = []

    for graph in batch:
        try:
            # Convert family to numeric label using encoder
            family = getattr(graph, 'family', None)
            if family is None or family not in label_encoder.family_to_id:
                # Assign benign (0) if label is missing or unknown
                label = label_encoder.family_to_id.get('benign', 0)
            else:
                label = label_encoder.transform(family)
            graph.y = torch.tensor(label, dtype=torch.long).to(device)

            # Move graph attributes to device
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


from sklearn.metrics import precision_score, recall_score

def calculate_metrics(preds, labels, num_classes):
    """Calculate precision and recall."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    return precision, recall

def train_model(model, train_loader, val_loader, optimizer, criterion, label_encoder, device, epochs=10):
    # Identify new families before training
    new_families = identify_new_families(train_loader, val_loader)
    new_family_batches = filter_new_family_batches(val_loader, new_families)

    num_classes = label_encoder.get_num_classes()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        all_train_preds, all_train_labels = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            if batch.y is None or batch.y.nelement() == 0:
                print("Warning: Found a batch with no valid labels. Skipping...")
                continue

            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)

            all_train_preds.append(pred)
            all_train_labels.append(batch.y)

        # Aggregate training metrics
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        train_precision, train_recall = calculate_metrics(torch.cat(all_train_preds), torch.cat(all_train_labels), num_classes)
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}, "
              f"Accuracy: {train_accuracy:.2f}%, Precision: {train_precision:.2f}, Recall: {train_recall:.2f}")

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                if batch.y is None or batch.y.nelement() == 0:
                    continue

                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()

                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

                all_val_preds.append(pred)
                all_val_labels.append(batch.y)

        # Aggregate validation metrics
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        val_precision, val_recall = calculate_metrics(torch.cat(all_val_preds), torch.cat(all_val_labels), num_classes)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {avg_val_loss:.4f}, "
              f"Accuracy: {val_accuracy:.2f}%, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}")

        # New family evaluation
        new_accuracy, new_loss, new_precision, new_recall = evaluate_new_families(
            model, new_family_batches, criterion, label_encoder, device
        )
        print(f"New Family Validation - Epoch {epoch + 1}/{epochs}: "
              f"Accuracy: {new_accuracy:.2f}%, Precision: {new_precision:.2f}, Recall: {new_recall:.2f}")


def evaluate_new_families(model, new_family_batches, criterion, label_encoder, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in new_family_batches:
            if batch.y is None or batch.y.nelement() == 0:
                continue

            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

            all_preds.append(pred)
            all_labels.append(batch.y)

    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(new_family_batches) if len(new_family_batches) > 0 else 0
    precision, recall = calculate_metrics(torch.cat(all_preds), torch.cat(all_labels), label_encoder.get_num_classes())

    return accuracy, avg_loss, precision, recall

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dir = '/data/saranyav/gcn_new/bodmas_batches'
    batch_files = {split: sorted(glob.glob(os.path.join(base_dir, split, 'batch_*.pt')))
                   for split in ['train', 'val']}

    label_encoder = FamilyLabelEncoder()
    all_families = set()
    for split in ['train', 'val']:
        for batch_file in batch_files[split]:
            batch = torch.load(batch_file)
            for graph in batch:
                if hasattr(graph, 'family'):
                    all_families.add(graph.family)
    label_encoder.fit(all_families)

    num_classes = label_encoder.get_num_classes()
    print(f"Total classes: {num_classes}")

    train_loader = load_batch(batch_files['train'][0], label_encoder, device=device)
    val_loader = load_batch(batch_files['val'][0], label_encoder, device=device)

    if train_loader is None or val_loader is None:
        print("Error: Could not load data")
        return

    first_batch = next(iter(train_loader))
    num_features = first_batch.x.size(1)
    model = MalwareGNN(num_features, num_classes, hidden_dim=128).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Pass the `device` explicitly here
    train_model(model, train_loader, val_loader, optimizer, criterion, label_encoder, device, epochs=20)


if __name__ == '__main__':
    main()
    