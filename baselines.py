import json
import os
from datetime import datetime
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from grakel import GraphKernel
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import glob
from sklearn.ensemble import IsolationForest

# Weisfeiler-Lehman kernel is common for graph classification
wl_kernel = GraphKernel(kernel="weisfeiler_lehman", normalize=True)


# Simple GCN without centroid layer
class BaselineGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
# K-nearest neighbors on graph embeddings
class KNNBaseline:
    def __init__(self, k=3):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        
    def fit(self, graphs):
        # Get graph embeddings using mean pooling
        embeddings = []
        labels = []
        for graph in graphs:
            emb = torch.mean(graph.x, dim=0)
            embeddings.append(emb.numpy())
            labels.append(graph.family)
        self.knn.fit(embeddings, labels)


class IsolationBaseline:
    def __init__(self):
        self.iso = IsolationForest(contamination='auto')
        
    def fit_predict(self, graphs):
        embeddings = []
        for graph in graphs:
            emb = torch.mean(graph.x, dim=0)
            embeddings.append(emb.numpy())
        return self.iso.fit_predict(embeddings)
    
class BaselineEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.baselines = {
            'gcn': BaselineGCN,
            'knn': KNNBaseline(),
            'isolation': IsolationBaseline(),
        }
        
    def prepare_traditional_features(self, graph):
        """Convert a PyG graph to features for traditional ML."""
        # Graph-level features
        features = {
            'num_nodes': graph.num_nodes,
            'num_edges': graph.edge_index.size(1),
            # Mean of node features
            'mean_features': torch.mean(graph.x, dim=0).cpu().numpy(),
            # Max of node features
            'max_features': torch.max(graph.x, dim=0)[0].cpu().numpy(),
            # Other graph statistics
            'density': graph.edge_index.size(1) / (graph.num_nodes * (graph.num_nodes - 1))
        }
        return np.concatenate([
            [features['num_nodes'], features['num_edges'], features['density']],
            features['mean_features'],
            features['max_features']
        ])

    def evaluate_traditional_ml(self, split_files):
        """Evaluate traditional ML baselines using batched data."""
        # Collect all features and labels
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        # Load training data
        print("Loading training data...")
        for batch_file in split_files['train']:
            batch_graphs = torch.load(batch_file)
            for graph in batch_graphs:
                features = self.prepare_traditional_features(graph)
                X_train.append(features)
                if hasattr(graph, 'family'):
                    y_train.append(graph.family)

        # Load validation data
        print("Loading validation data...")
        for batch_file in split_files['val']:
            batch_graphs = torch.load(batch_file)
            for graph in batch_graphs:
                features = self.prepare_traditional_features(graph)
                X_val.append(features)
                if hasattr(graph, 'family'):
                    y_val.append(graph.family)

        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Train and evaluate models
        results = {}
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        results['random_forest'] = {
            'val_acc': rf.score(X_val, y_val)
        }
        
        # SVM
        print("Training SVM...")
        svm = SVC()
        svm.fit(X_train, y_train)
        results['svm'] = {
            'val_acc': svm.score(X_val, y_val)
        }
        
        return results

    def evaluate_baseline_gcn(self, split_files, num_classes):
        """Evaluate baseline GCN using batched data."""
        # Load first batch to get feature dimensions
        first_batch = torch.load(split_files['train'][0])
        num_features = first_batch[0].x.size(1)
        
        # Initialize model
        model = BaselineGCN(num_features, num_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Training loop
        best_val_acc = 0
        for epoch in range(100):
            model.train()
            # Train on batches
            for batch_file in split_files['train']:
                batch_graphs = torch.load(batch_file)
                loader = DataLoader(batch_graphs, batch_size=32, shuffle=True)
                
                for batch in loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = F.nll_loss(out, batch.y)
                    loss.backward()
                    optimizer.step()
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            for batch_file in split_files['val']:
                batch_graphs = torch.load(batch_file)
                loader = DataLoader(batch_graphs, batch_size=32)
                
                for batch in loader:
                    batch = batch.to(self.device)
                    pred = model(batch).argmax(dim=1)
                    val_correct += (pred == batch.y).sum().item()
                    val_total += batch.y.size(0)
            
            val_acc = val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_baseline_gcn.pt')

        return {'val_acc': best_val_acc}

    def evaluate_novelty_detection(self, split_files):
        """Evaluate novelty detection using Isolation Forest."""
        # Extract features from training data
        print("Extracting features for novelty detection...")
        train_features = []
        for batch_file in split_files['train']:
            batch_graphs = torch.load(batch_file)
            for graph in batch_graphs:
                features = self.prepare_traditional_features(graph)
                train_features.append(features)

        # Train Isolation Forest
        iso = IsolationForest(contamination='auto')
        iso.fit(train_features)
        
        # Evaluate on validation set
        val_predictions = []
        val_labels = []
        for batch_file in split_files['val']:
            batch_graphs = torch.load(batch_file)
            for graph in batch_graphs:
                features = self.prepare_traditional_features(graph)
                pred = iso.predict([features])[0]
                val_predictions.append(pred)
                if hasattr(graph, 'family'):
                    val_labels.append(graph.family)
        
        return {
            'novelty_scores': val_predictions,
            'true_labels': val_labels
        }

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
    # Get split files
    split_files = get_split_files()
    
    # Initialize evaluator
    evaluator = BaselineEvaluator()
    
    # Run evaluations
    print("\nEvaluating traditional ML baselines...")
    trad_results = evaluator.evaluate_traditional_ml(split_files)
    
    print("\nEvaluating baseline GCN...")
    gcn_results = evaluator.evaluate_baseline_gcn(split_files, num_classes=10)  # adjust num_classes
    
    print("\nEvaluating novelty detection...")
    novelty_results = evaluator.evaluate_novelty_detection(split_files)
    
    # Save results
    all_results = {
        'traditional_ml': trad_results,
        'gcn': gcn_results,
        'novelty': novelty_results
    }
    
    with open('baseline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()