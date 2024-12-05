
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import os
import glob
from collections import defaultdict
import pandas as pd
from architectures import CentroidLayer

class MalwareGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64):
        super().__init__()
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Centroid layer for classification
        self.centroid = CentroidLayer(
            input_dim=hidden_dim,
            n_classes=num_classes,
            n_centroids_per_class=3,
            reject_input=True
        )
    
    def get_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Third GNN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        return x
    
    def forward(self, data):
        embedding = self.get_embedding(data)
        return self.centroid(embedding)

def prepare_data(base_dir='bodmas_batches_test'):
    """Prepare train/val/test datasets using existing batch files."""
    split_files = defaultdict(list)
    families = set()
    
    # Collect files and families
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            files = glob.glob(os.path.join(split_dir, 'batch_*.pt'))
            
            # Extract families from the batch files
            for file in files:
                batch = torch.load(file)
                for graph in batch:
                    if hasattr(graph, 'family') and graph.family:
                        families.add(graph.family)
    
            split_files[split].extend(files)
            print(f"{split} files found: {len(files)}")
    
    # Create family mappings
    families = sorted(list(families | {'none'}))  # Add 'none' class
    family_to_idx = {family: idx for idx, family in enumerate(families)}
    
    print(f"\nFound {len(families)} unique families:")
    for family in families:
        print(f"- {family}")
    
    # Update graphs with numeric labels
    for split in split_files:
        for file in split_files[split]:
            batch = torch.load(file)
            for graph in batch:
                if not hasattr(graph, 'family') or graph.family is None:
                    graph.family = 'none'
                graph.y = torch.tensor(family_to_idx[graph.family])
                
                # Handle empty edge cases
                if graph.edge_index.size(1) == 0:
                    graph.edge_attr = torch.zeros((0, 1))
                elif not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
                    graph.edge_attr = torch.ones((graph.edge_index.size(1), 1))
            
            torch.save(batch, file)
    
    return split_files, len(families), family_to_idx

def load_and_process_batch(batch_file, batch_size=32):
    """Load a batch file and ensure proper edge handling."""
    try:
        if not os.path.exists(batch_file):
            print(f"Batch file not found: {batch_file}")
            return None
        
        batch = torch.load(batch_file)
        processed_batch = []
        
        for graph in batch:
            # Ensure proper edge attributes
            if graph.edge_index.size(1) == 0:
                graph.edge_attr = torch.zeros((0, 1))
            elif not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
                graph.edge_attr = torch.ones((graph.edge_index.size(1), 1))
            processed_batch.append(graph)
        
        return DataLoader(processed_batch, batch_size=batch_size, shuffle=True)
        
    except Exception as e:
        print(f"Error processing {batch_file}: {e}")
        return None

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        
        if model.centroid.reject_input:
            # Cross entropy loss excluding rejection class for known samples
            loss = F.cross_entropy(out[:, :-1], data.y)
        else:
            loss = F.cross_entropy(out, data.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        valid_batches += 1
    
    return total_loss / max(1, valid_batches)

def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            
            if model.centroid.reject_input:
                out = out[:, :-1]  # Exclude rejection score for accuracy calculation
            
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.y.size(0)
    
    return correct / max(1, total)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    split_files, num_classes, family_to_idx = prepare_data(csv_path='families.csv')
    
    # Load first batch to get feature dimensions
    first_batch = torch.load(split_files['train'][0])
    num_features = first_batch[0].x.size(1)
    
    # Initialize model
    model = MalwareGNN(
        num_node_features=num_features,
        num_classes=num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    best_val_acc = 0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training
        epoch_losses = []
        for batch_file in split_files['train']:
            train_loader = load_and_process_batch(batch_file)
            if train_loader:
                loss = train_epoch(model, train_loader, optimizer, device)
                epoch_losses.append(loss)
        
        if not epoch_losses:
            continue
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Validation
        val_accs = []
        for batch_file in split_files['val']:
            val_loader = load_and_process_batch(batch_file)
            if val_loader:
                acc = evaluate(model, val_loader, device)
                val_accs.append(acc)
        
        if not val_accs:
            continue
            
        val_acc = sum(val_accs) / len(val_accs)
        
        print(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Test evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    
    test_accs = []
    for batch_file in split_files['test']:
        test_loader = load_and_process_batch(batch_file)
        if test_loader:
            acc = evaluate(model, test_loader, device)
            test_accs.append(acc)
    
    if test_accs:
        final_acc = sum(test_accs) / len(test_accs)
        print(f'\nFinal Test Accuracy: {final_acc:.4f}')

if __name__ == "__main__":
    main()


sys.exit()


class ClusterMetrics:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        
    def compute_cluster_stats(self, embeddings, labels, centroids):
        """Compute various clustering metrics."""
        # Convert to numpy for easier computation
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        centroids = centroids.cpu().numpy()
        
        # Calculate distances to nearest centroids
        distances = cdist(embeddings, centroids.reshape(-1, centroids.shape[-1]))
        min_distances = np.min(distances, axis=1)
        
        # Identify potential new families (points far from all centroids)
        potential_new = min_distances > self.epsilon
        
        # Calculate intra-class distances
        intra_class_distances = defaultdict(list)
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            if not potential_new[i]:
                class_centroids = centroids[label]
                dist_to_class = np.min(np.linalg.norm(emb - class_centroids, axis=1))
                intra_class_distances[label].append(dist_to_class)
        
        stats = {
            'potential_new_families': np.sum(potential_new),
            'avg_min_distance': np.mean(min_distances),
            'intra_class_distances': {k: np.mean(v) for k, v in intra_class_distances.items()},
            'silhouette': silhouette_score(embeddings, labels) if len(np.unique(labels)) > 1 else 0,
        }
        
        return stats, potential_new

def get_family_info(batch_files):
    """Extract unique families and create mappings from batch files."""
    families = set()
    
    # First pass: collect all unique families
    for batch_file in batch_files:
        if not os.path.exists(batch_file):
            print(f"Warning: Batch file not found: {batch_file}")
            continue
            
        try:
            batch_graphs = torch.load(batch_file)
            for graph in batch_graphs:
                if hasattr(graph, 'family') and graph.family:
                    families.add(graph.family)
        except Exception as e:
            print(f"Error loading {batch_file}: {e}")
            continue
    
    # Create mappings
    families = sorted(list(families))
    family_to_idx = {family: idx for idx, family in enumerate(families)}
    idx_to_family = {idx: family for family, idx in family_to_idx.items()}
    
    print(f"\nFound {len(families)} unique families:")
    for family in families:
        print(f"- {family}")
    
    return family_to_idx, idx_to_family

def process_batch_files(batch_files, family_to_idx):
    """Process batch files and add numeric labels."""
    processed_files = []
    
    for batch_file in batch_files:
        if not os.path.exists(batch_file):
            continue
            
        try:
            batch_graphs = torch.load(batch_file)
            for graph in batch_graphs:
                if hasattr(graph, 'family') and graph.family in family_to_idx:
                    graph.y = torch.tensor(family_to_idx[graph.family])
                else:
                    graph.y = torch.tensor(-1)  # For unlabeled samples
            
            # Save processed batch with same filename
            torch.save(batch_graphs, batch_file)
            processed_files.append(batch_file)
            
        except Exception as e:
            print(f"Error processing {batch_file}: {e}")
            continue
    
    return processed_files

def get_split_files(base_dir='bodmas_batches_test'):
    """Get the batch files for each split and process them."""
    split_files = {}
    all_batch_files = []
    
    # Collect all batch files
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir, exist_ok=True)
            print(f"Created directory: {split_dir}")
            continue
            
        batch_files = sorted(glob.glob(os.path.join(split_dir, 'batch_*.pt')))
        split_files[split] = batch_files
        all_batch_files.extend(batch_files)
        print(f"{split} files found: {len(batch_files)}")
    
    if not all_batch_files:
        raise ValueError(f"No batch files found in {base_dir}")
    
    # Extract family information
    family_to_idx, idx_to_family = get_family_info(all_batch_files)
    
    # Save family mappings
    mapping_dict = {
        'idx_to_family': idx_to_family,
        'family_to_idx': family_to_idx
    }
    with open(os.path.join(base_dir, 'family_mapping.json'), 'w') as f:
        json.dump(mapping_dict, f, indent=2)
    
    # Process each split
    processed_splits = {}
    for split, files in split_files.items():
        processed_splits[split] = process_batch_files(files, family_to_idx)
        print(f"\nProcessed {split} files: {len(processed_splits[split])}")
    
    return processed_splits, len(family_to_idx)


def normalize_features(graphs, target_dim=None):
    """Normalize feature dimensions across all graphs."""
    if not graphs:
        return []
    
    # If target_dim not specified, use the most common feature dimension
    if target_dim is None:
        dimensions = [g.x.size(1) for g in graphs]
        target_dim = max(set(dimensions), key=dimensions.count)
    
    normalized_graphs = []
    for graph in graphs:
        current_dim = graph.x.size(1)
        
        if current_dim == target_dim:
            normalized_graphs.append(graph)
        elif current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(graph.x.size(0), target_dim - current_dim)
            graph.x = torch.cat([graph.x, padding], dim=1)
            normalized_graphs.append(graph)
        else:
            # Truncate to target dimension
            graph.x = graph.x[:, :target_dim]
            normalized_graphs.append(graph)
    
    return normalized_graphs

def load_and_process_batch(batch_file, batch_size=32, target_dim=None):
    """Load a single batch file and return its dataloader with normalized features."""
    try:
        if not os.path.exists(batch_file):
            print(f"Batch file not found: {batch_file}")
            return None
            
        graphs = torch.load(batch_file)
        
        if not graphs:
            print(f"No graphs found in {batch_file}")
            return None
            
        # Filter out graphs with no features
        valid_graphs = [g for g in graphs if hasattr(g, 'x') and g.x is not None]
        
        if not valid_graphs:
            print(f"No valid graphs with features found in {batch_file}")
            return None
        
        # Normalize feature dimensions
        normalized_graphs = normalize_features(valid_graphs, target_dim)
        
        if not normalized_graphs:
            print(f"No graphs remained after normalization in {batch_file}")
            return None
        
        return DataLoader(normalized_graphs, batch_size=batch_size, shuffle=False)
        
    except Exception as e:
        print(f"Error loading batch {batch_file}: {e}")
        return None

def get_target_dimension(split_files):
    """Determine the target feature dimension from all training files."""
    all_dimensions = []
    
    for batch_file in split_files['train']:
        try:
            graphs = torch.load(batch_file)
            dimensions = [g.x.size(1) for g in graphs if hasattr(g, 'x') and g.x is not None]
            all_dimensions.extend(dimensions)
        except Exception as e:
            print(f"Error reading dimensions from {batch_file}: {e}")
            continue
    
    if not all_dimensions:
        raise ValueError("Could not determine target dimension from training files")
    
    # Use the most common dimension
    target_dim = max(set(all_dimensions), key=all_dimensions.count)
    print(f"Selected target dimension: {target_dim}")
    return target_dim


class MalwareGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64):
        super().__init__()
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Centroid layer for classification
        self.centroid = CentroidLayer(
            input_dim=hidden_dim,
            n_classes=num_classes,
            n_centroids_per_class=3,
            reject_input=True
        )
    
    def get_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Third GNN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        return x
    
    def forward(self, data):
        embedding = self.get_embedding(data)
        return self.centroid(embedding)

class MalwareTrainer:
    def __init__(self, model, device, lr=0.001, epsilon=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.cluster_metrics = ClusterMetrics(epsilon=epsilon)
    
    def train_epoch(self, loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data)
            
            if self.model.centroid.reject_input:
                # Cross entropy loss excluding rejection class for known samples
                loss = F.cross_entropy(out[:, :-1], data.y)
            else:
                loss = F.cross_entropy(out, data.y)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Get processed batch files and number of classes
        split_files, num_classes = get_split_files()
        
        if not split_files['train']:
            raise ValueError("No training files found after processing")
        
        # Determine target feature dimension from training data
        target_dim = get_target_dimension(split_files)
        
        # Initialize model with target dimension
        model = MalwareGNN(
            num_node_features=target_dim,
            num_classes=num_classes,
            hidden_dim=128
        )
        trainer = MalwareTrainer(model, device)
        
        # Training loop
        num_epochs = 100
        best_val_acc = 0
        train_losses = []
        val_accs = []
        
        print("\nStarting training...")
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Training
            for batch_file in split_files['train']:
                train_loader = load_and_process_batch(batch_file, batch_size=32, target_dim=target_dim)
                if train_loader is not None and len(train_loader) > 0:
                    loss = trainer.train_epoch(train_loader)
                    epoch_losses.append(loss)
            
            if not epoch_losses:
                print(f"Warning: No valid batches in epoch {epoch}")
                continue
                
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_epoch_loss)
            
            # Validation
            val_results = []
            for batch_file in split_files['val']:
                val_loader = load_and_process_batch(batch_file, batch_size=32, target_dim=target_dim)
                if val_loader is not None and len(val_loader) > 0:
                    batch_results = trainer.evaluate(val_loader)
                    val_results.append(batch_results)
            
            if val_results:
                val_acc = sum(r['accuracy'] for r in val_results) / len(val_results)
                val_accs.append(val_acc)
                
                print(f'Epoch {epoch:03d}, Loss: {avg_epoch_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), 'best_model.pt')
            else:
                print(f"Warning: No valid validation results in epoch {epoch}")
        
    except Exception as e:
        print(f"Error in main(): {e}")
        raise

    # # Load best model for testing
    # model.load_state_dict(torch.load('best_model.pt'))
    # trainer = MalwareTrainer(model, device)
    
    # # Test evaluation
    # test_results = []
    # for batch_file in split_files['test']:
    #     test_loader = load_and_process_batch(batch_file, batch_size=32)
    #     if test_loader:
    #         batch_results = trainer.evaluate(test_loader)
    #         test_results.append(batch_results)
    
    # Aggregate test results
    # final_results = {
    #     'accuracy': sum(r['accuracy'] for r in test_results) / len(test_results),
    #     'embeddings': np.concatenate([r['embeddings'] for r in test_results]),
    #     'labels': np.concatenate([r['labels'] for r in test_results]),
    #     'potential_new': np.concatenate([r['potential_new'] for r in test_results])
    # }
    
    # # Save results
    # np.save('embeddings.npy', final_results['embeddings'])
    # np.save('labels.npy', final_results['labels'])
    # np.save('potential_new.npy', final_results['potential_new'])
    
    # # Plot training curves
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses)
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(val_accs)
    # plt.title('Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    
    # plt.tight_layout()
    # plt.savefig('training_curves.png')
    # plt.close()

    

if __name__ == "__main__":
    main()














sys.exit()



def save_results(results, output_file, family_mapping):
    """Save evaluation results with actual family names and detailed metrics.
    
    Args:
        results: Dictionary containing evaluation results
        output_file: Path to save JSON results
        family_mapping: Dictionary containing family name mappings
    """
    labeled_results = {
        'metrics': {
            'accuracy': float(results['accuracy']),
            'cluster_stats': {
                'potential_new_families': int(results['cluster_stats']['potential_new_families']),
                'avg_min_distance': float(results['cluster_stats']['avg_min_distance']),
                'silhouette': float(results['cluster_stats']['silhouette']),
                'intra_class_distances': {
                    family_mapping['idx_to_family'][str(k)]: float(v) 
                    for k, v in results['cluster_stats']['intra_class_distances'].items()
                }
            }
        },
        'predictions': {
            'true_labels': [family_mapping['idx_to_family'].get(str(x), 'unknown') for x in results['labels']],
            'predicted_labels': [family_mapping['idx_to_family'].get(str(x), 'unknown') for x in results['predictions']],
            'potential_new': results['potential_new'].tolist()
        },
        'embeddings': results['embeddings'].tolist()  # Convert numpy array to list for JSON
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(labeled_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Accuracy: {labeled_results['metrics']['accuracy']:.4f}")
    print(f"Potential new families detected: {labeled_results['metrics']['cluster_stats']['potential_new_families']}")

# Update the call in main():


class MalwareGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.centroid = CentroidLayer(
            input_dim=hidden_dim,
            n_classes=num_classes,
            n_centroids_per_class=3,
            reject_input=True
        )
    
    def get_embedding(self, data):
        """Extract internal embedding before centroid layer."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        
        # Get graph-level embedding
        embedding = global_mean_pool(x, batch)
        return embedding
    
    def forward(self, data):
        embedding = self.get_embedding(data)
        return self.centroid(embedding)

class MalwareTrainer:
    def __init__(self, model, device, lr=0.001, epsilon=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.cluster_metrics = ClusterMetrics(epsilon=epsilon)
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_embeddings = []
        all_predictions = []
        all_labels = []
        all_scores = []
        
        for data in loader:
            data = data.to(self.device)
            
            # Get embeddings and predictions
            embeddings = self.model.get_embedding(data)
            out = self.model(data)
            
            if self.model.centroid.reject_input:
                class_scores = out[:, :-1]
                rejection_scores = out[:, -1]
                pred = class_scores.argmax(dim=1)
            else:
                pred = out.argmax(dim=1)
            
            all_embeddings.append(embeddings)
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_scores.extend(rejection_scores.cpu().numpy() if self.model.centroid.reject_input else [])
        
        # Combine embeddings and compute cluster metrics
        embeddings = torch.cat(all_embeddings, dim=0)
        cluster_stats, potential_new = self.cluster_metrics.compute_cluster_stats(
            embeddings,
            torch.tensor(all_labels),
            self.model.centroid.centroids
        )
        
        # Compute accuracy excluding potential new families
        known_mask = ~potential_new
        known_acc = np.mean(np.array(all_predictions)[known_mask] == np.array(all_labels)[known_mask])
        
        results = {
            'accuracy': known_acc,
            'cluster_stats': cluster_stats,
            'predictions': all_predictions,
            'labels': all_labels,
            'potential_new': potential_new,
            'embeddings': embeddings.cpu().numpy()
        }
        
        return results

def analyze_temporal_clusters(trainer, test_loader, epsilons=[0.5, 1.0, 2.0]):
    """Analyze cluster evolution over time."""
    results = {}
    
    for epsilon in epsilons:
        trainer.cluster_metrics.epsilon = epsilon
        eval_results = trainer.evaluate(test_loader)
        
        results[epsilon] = {
            'accuracy': eval_results['accuracy'],
            'num_new_families': eval_results['cluster_stats']['potential_new_families'],
            'avg_distance': eval_results['cluster_stats']['avg_min_distance'],
            'silhouette': eval_results['cluster_stats']['silhouette']
        }
    
    return results



def prepare_dataset():
    """Prepare dataset with proper family mapping and temporal ordering."""
    # Load the saved PyG dataset
    graphs = torch.load('bodmas_pyg_dataset.pt')
    
    # Create family mapping (only from labeled samples)
    labeled_families = sorted(list(set(g.family for g in graphs if hasattr(g, 'family') and g.family)))
    family_to_idx = {family: idx for idx, family in enumerate(labeled_families)}
    
    # Save the mapping for later reference
    mapping_dict = {
        'idx_to_family': {idx: family for family, idx in family_to_idx.items()},
        'family_to_idx': family_to_idx
    }
    with open('family_mapping.json', 'w') as f:
        json.dump(mapping_dict, f, indent=2)
    
    # Add numeric labels to graphs
    for g in graphs:
        if hasattr(g, 'family') and g.family and g.family in family_to_idx:
            g.y = torch.tensor(family_to_idx[g.family])
        else:
            g.y = torch.tensor(-1)  # For unlabeled samples
            
    # Sort by timestamp to maintain temporal order
    graphs = sorted(graphs, key=lambda x: x.timestamp)
    
    # Split dataset temporally
    train_ratio = 0.7
    val_ratio = 0.15
    n = len(graphs)
    
    # Filter out unlabeled samples for training
    labeled_graphs = [g for g in graphs if g.y != -1]
    unlabeled_graphs = [g for g in graphs if g.y == -1]
    
    n_labeled = len(labeled_graphs)
    train_graphs = labeled_graphs[:int(n_labeled * train_ratio)]
    val_graphs = labeled_graphs[int(n_labeled * train_ratio):int(n_labeled * (train_ratio + val_ratio))]
    test_graphs = labeled_graphs[int(n_labeled * (train_ratio + val_ratio)):]
    
    print("\nDataset Statistics:")
    print(f"Total graphs: {len(graphs)}")
    print(f"Labeled graphs: {len(labeled_graphs)}")
    print(f"Unlabeled graphs: {len(unlabeled_graphs)}")
    print(f"Number of families: {len(family_to_idx)}")
    print(f"Train samples: {len(train_graphs)}")
    print(f"Val samples: {len(val_graphs)}")
    print(f"Test samples: {len(test_graphs)}")
    print("\nFamily distribution:")
    family_counts = defaultdict(int)
    for g in labeled_graphs:
        family_counts[g.family] += 1
    for family, count in sorted(family_counts.items()):
        print(f"{family}: {count}")
        
    return train_graphs, val_graphs, test_graphs, unlabeled_graphs, len(family_to_idx), family_to_idx

def get_dataloaders(batch_size=32):
    """Get train, val, and test dataloaders."""
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        batch_files = sorted(glob.glob(f'bodmas_batches/{split}/batch_*.pt'))
        if not batch_files:
            print(f"No batch files found for {split} split")
            continue
            
        # Concatenate all batches for validation and test
        if split in ['val', 'test']:
            all_graphs = []
            for batch_file in batch_files:
                all_graphs.extend(torch.load(batch_file))
            loaders[split] = DataLoader(all_graphs, batch_size=batch_size, shuffle=False)
            print(f"{split} loader: {len(all_graphs)} graphs in {len(loaders[split])} batches")
        else:
            # For training, just store the file paths
            loaders[split] = batch_files
            print(f"{split} files: {len(batch_files)} batches")
    
    return loaders

def get_batch_dataloader(batch_file, batch_size=32):
    """Load a single batch and return its dataloader."""
    batch_graphs = torch.load(batch_file)
    return DataLoader(batch_graphs, batch_size=batch_size, shuffle=False)

def load_and_process_batch(batch_file, batch_size=32):
    """Load a single batch file and return its dataloader."""
    try:
        graphs = torch.load(batch_file)
        return DataLoader(graphs, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error loading batch {batch_file}: {e}")
        return None

def get_split_files():
    """Get the batch files for each split."""
    split_files = {}
    for split in ['train', 'val', 'test']:
        batch_files = sorted(glob.glob(f'bodmas_batches/{split}/batch_*.pt'))
        if not batch_files:
            print(f"No batch files found for {split} split")
            continue
        split_files[split] = batch_files
        print(f"{split} files: {len(batch_files)} batches")
    return split_files

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get paths to batch files
    split_files = get_split_files()
    
    # Initialize model (we'll get num_features from first batch)
    first_batch = torch.load(split_files['train'][0])
    num_node_features = first_batch[0].x.size(1)
    num_classes = len(set(g.family for g in first_batch if hasattr(g, 'family')))
    
    model = MalwareGNN(num_node_features, num_classes)
    trainer = MalwareTrainer(model, device)
    
    # Training loop
    num_epochs = 100
    best_val_acc = 0
    train_losses = []
    val_accs = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Training
        for batch_file in split_files['train']:
            train_loader = load_and_process_batch(batch_file, batch_size=32)
            if train_loader:
                loss = trainer.train_epoch(train_loader)
                epoch_losses.append(loss)
        
        # Calculate average loss for epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_epoch_loss)
        
        # Validation
        val_results = []
        for batch_file in split_files['val']:
            val_loader = load_and_process_batch(batch_file, batch_size=32)
            if val_loader:
                batch_results = trainer.evaluate(val_loader)
                val_results.append(batch_results)
        
        # Calculate average validation accuracy
        val_acc = sum(r['accuracy'] for r in val_results) / len(val_results)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch:03d}, Loss: {avg_epoch_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pt'))
    trainer = MalwareTrainer(model, device)
    
    # Test evaluation
    test_results = []
    for batch_file in split_files['test']:
        test_loader = load_and_process_batch(batch_file, batch_size=32)
        if test_loader:
            batch_results = trainer.evaluate(test_loader)
            test_results.append(batch_results)
    
    # Aggregate test results
    final_results = {
        'accuracy': sum(r['accuracy'] for r in test_results) / len(test_results),
        'embeddings': np.concatenate([r['embeddings'] for r in test_results]),
        'labels': np.concatenate([r['labels'] for r in test_results]),
        'potential_new': np.concatenate([r['potential_new'] for r in test_results])
    }
    
    # Save results
    np.save('embeddings.npy', final_results['embeddings'])
    np.save('labels.npy', final_results['labels'])
    np.save('potential_new.npy', final_results['potential_new'])
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    main()
    # This can be called after evaluation:
    # final_results = trainer.evaluate(test_loader)
    # save_results(final_results, 'final_results.json')
