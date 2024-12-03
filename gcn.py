import json 
import os
from collections import defaultdict
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import classification_report, silhouette_score
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
import glob 

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
