import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
import sys
# append('/data/saranyav/gcn_new/Methods')
sys.path.append('/data/saranyav/gcn_new/Methods')
from TemporalGNN import TemporalMalwareDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineGNN(torch.nn.Module):
    """Basic GNN without centroid layer for baseline comparison."""
    def __init__(self, num_node_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        
        # GNN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        logits = self.classifier(x)
        return logits

class BaselineAnalyzer:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device
        self.results = defaultdict(dict)
        

    def extract_graph_features(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from graphs for traditional ML methods."""
        features_list = []
        labels_list = []
        
        for batch in loader:
            batch_size = batch.y.size(0)
            for i in range(batch_size):
                # Global graph features for each graph in batch
                start_idx = batch.ptr[i].item()
                end_idx = batch.ptr[i + 1].item()
                
                num_nodes = end_idx - start_idx
                num_edges = batch.edge_index[:, (batch.edge_index[0] >= start_idx) & 
                                            (batch.edge_index[0] < end_idx)].size(1)
                avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
                
                # Node feature statistics for this graph
                node_features = batch.x[start_idx:end_idx].numpy()
                node_stats = np.concatenate([
                    np.mean(node_features, axis=0),
                    np.std(node_features, axis=0),
                    np.max(node_features, axis=0),
                    np.min(node_features, axis=0)
                ])
                
                # Combine features
                graph_features = np.concatenate([
                    [num_nodes, num_edges, avg_degree],
                    node_stats
                ])
                
                features_list.append(graph_features)
                labels_list.append(batch.y[i].item())
        
        return np.array(features_list), np.array(labels_list)

    
    def evaluate_baseline_gnn(self, train_loader, val_loader, test_loader):
        """Evaluate baseline GNN model."""
        logger.info("Evaluating Baseline GNN...")
        
        model = BaselineGNN(
            num_node_features=14,  # From your data
            hidden_dim=128,
            num_classes=self.data_loader.get_num_classes()
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training
        best_val_f1 = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                output = model(batch)
                loss = criterion(output, batch.y)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    output = model(batch)
                    preds = output.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy())
            
            val_f1 = self._compute_f1(val_labels, val_preds)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_baseline_gnn.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Test evaluation
        model.load_state_dict(torch.load('best_baseline_gnn.pt'))
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                output = model(batch)
                preds = output.argmax(dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(batch.y.cpu().numpy())
        
        self.results['baseline_gnn'] = self._compute_metrics(test_labels, test_preds)
    
    def evaluate_traditional_ml(self, train_loader, val_loader, test_loader):
        """Evaluate traditional ML baselines (RF, SVM)."""
        logger.info("Evaluating Traditional ML Methods...")
        
        # Extract features
        X_train, y_train = self.extract_graph_features(train_loader)
        X_val, y_val = self.extract_graph_features(val_loader)
        X_test, y_test = self.extract_graph_features(test_loader)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_preds = rf.predict(X_test_scaled)
        self.results['random_forest'] = self._compute_metrics(y_test, rf_preds)
        
        # Evaluate novel detection methods
        self._evaluate_novelty_detectors(X_train_scaled, X_test_scaled, y_test)
    
    def _evaluate_novelty_detectors(self, X_train, X_test, y_test):
        """Evaluate various novelty detection methods."""
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_train)
        iso_preds = iso_forest.predict(X_test)
        # Convert predictions to binary (1: normal, -1: novel)
        iso_preds = (iso_preds == 1)
        
        # One-class SVM
        ocsvm = OneClassSVM(kernel='rbf', nu=0.1)
        ocsvm.fit(X_train)
        svm_preds = ocsvm.predict(X_test)
        # Convert predictions to binary (1: normal, -1: novel)
        svm_preds = (svm_preds == 1)
        
        # Store results
        self.results['isolation_forest'] = self._compute_novelty_metrics(y_test, iso_preds)
        self.results['one_class_svm'] = self._compute_novelty_metrics(y_test, svm_preds)
    
    def _compute_metrics(self, y_true, y_pred) -> Dict:
        """Compute classification metrics."""
        metrics = {
            'accuracy': np.mean(y_true == y_pred),
            'per_class': defaultdict(dict)
        }
        
        # Per-class metrics
        classes = np.unique(y_true)
        for c in classes:
            mask = y_true == c
            if mask.any():
                tp = np.sum((y_true == c) & (y_pred == c))
                fp = np.sum((y_true != c) & (y_pred == c))
                fn = np.sum((y_true == c) & (y_pred != c))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics['per_class'][int(c)] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        # Overall metrics
        metrics['overall'] = {
            'precision': np.mean([m['precision'] for m in metrics['per_class'].values()]),
            'recall': np.mean([m['recall'] for m in metrics['per_class'].values()]),
            'f1': np.mean([m['f1'] for m in metrics['per_class'].values()])
        }
        
        return metrics
    
    def _compute_novelty_metrics(self, y_true, y_pred) -> Dict:
        """Compute novelty detection metrics."""
        # Consider samples with unseen classes as novel
        novel_mask = y_true >= self.data_loader.num_known_families
        
        tp = np.sum(~y_pred & novel_mask)
        fp = np.sum(~y_pred & ~novel_mask)
        fn = np.sum(y_pred & novel_mask)
        tn = np.sum(y_pred & ~novel_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        }
    
    def _compute_f1(self, y_true, y_pred) -> float:
        """Compute overall F1 score."""
        metrics = self._compute_metrics(y_true, y_pred)
        return metrics['overall']['f1']
        
    def run_all_baselines(self):
        """Run all baseline evaluations."""
        # Load data
        train_loader, _ = self.data_loader.load_split('train', batch_size=32)
        val_loader, _ = self.data_loader.load_split('val', batch_size=32)
        test_loader, _ = self.data_loader.load_split('test', batch_size=32)
        
        # Check if GNN model exists
        if not Path('best_baseline_gnn.pt').exists():
            self.evaluate_baseline_gnn(train_loader, val_loader, test_loader)
        else:
            logger.info("Loading existing GNN model...")
            model = BaselineGNN(
                num_node_features=14,
                hidden_dim=128, 
                num_classes=self.data_loader.get_num_classes()
            ).to(self.device)
            model.load_state_dict(torch.load('best_baseline_gnn.pt'))
            
            # Evaluate on test set
            model.eval()
            test_preds = []
            test_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    output = model(batch)
                    preds = output.argmax(dim=1)
                    test_preds.extend(preds.cpu().numpy())
                    test_labels.extend(batch.y.cpu().numpy())
            
            self.results['baseline_gnn'] = self._compute_metrics(test_labels, test_preds)
        
        # Run traditional ML evaluation
        self.evaluate_traditional_ml(train_loader, val_loader, test_loader)
        
        # Save results
        with open('baseline_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return self.results

def main():
    # Initialize data loader (using your existing TemporalMalwareDataLoader)
    data_loader = TemporalMalwareDataLoader(
        batch_dir=Path('/data/saranyav/gcn_new/bodmas_batches_new'),
        behavioral_groups_path=Path('/data/saranyav/gcn_new/behavioral_analysis/behavioral_groups.json'),
        metadata_path=Path('bodmas_metadata_cleaned.csv'),
        malware_types_path=Path('bodmas_malware_category.csv')
    )
    
    # Initialize analyzer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    analyzer = BaselineAnalyzer(data_loader, device)
    
    # Run baseline analysis
    results = analyzer.run_all_baselines()
    
    # Print summary
    logger.info("\nBaseline Results Summary:")
    for method, metrics in results.items():
        logger.info(f"\n{method.upper()}:")
        if 'overall' in metrics:
            logger.info(f"Overall F1: {metrics['overall']['f1']:.4f}")
        else:
            logger.info(f"F1: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()