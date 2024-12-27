import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

class HierarchicalMalwareGNN(torch.nn.Module):
    def __init__(self, num_features, num_groups=16, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Base Graph Neural Network layers
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, embedding_dim)
        
        # Group classification head
        self.group_classifier = torch.nn.Linear(embedding_dim, num_groups)
        
        # Separate classifiers for each behavioral group
        self.family_classifiers = torch.nn.ModuleDict()
    
    def add_family_classifier(self, group_id: str, num_families: int):
        """Dynamically add a family classifier for a behavioral group"""
        self.family_classifiers[str(group_id)] = torch.nn.Linear(
            self.embedding_dim, num_families
        )
    
    def get_embeddings(self, data):
        """Extract graph embeddings"""
        device = next(self.parameters()).device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        embeddings = global_mean_pool(x, batch)
        return embeddings
        
    def forward(self, data):
            device = next(self.parameters()).device
            
            embeddings = self.get_embeddings(data)
            # Ensure embeddings are on the correct device
            embeddings = embeddings.to(device)
            group_logits = self.group_classifier(embeddings)
            
            family_logits = {}
            for group_id in self.family_classifiers:
                family_logits[group_id] = self.family_classifiers[group_id](embeddings)
                
            return embeddings, group_logits, family_logits
    
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_batch(batch_file, family_to_group, batch_size=32):
    """Load and preprocess batch file"""
    try:
        batch_data = torch.load(batch_file)
        return DataLoader(batch_data, batch_size=min(batch_size, len(batch_data)), shuffle=False)
    except Exception as e:
        print(f"Error loading batch {batch_file}: {str(e)}")
        return None

def analyze_predictions(model, test_files, family_to_group, group_to_families, device='cuda'):
    """Analyze model predictions with detailed metrics"""
    model.eval()
    results = {
        'group_true': [],
        'group_pred': [],
        'family_true': [],
        'family_pred': [],
        'embeddings': []
    }
    
    sample_details = []  # Store detailed info per sample
    
    with torch.no_grad():
        for batch_file in test_files:
            batch_loader = load_batch(batch_file, family_to_group)
            if not batch_loader:
                continue
                
            for batch in batch_loader:
                batch = batch.to(device)
                
                # Get predictions
                embeddings, group_logits, family_logits = model(batch)
                pred_groups = group_logits.argmax(dim=1)
                
                # Store true and predicted values
                true_groups = [family_to_group.get(fam, -999) for fam in batch.family]
                results['group_true'].extend(true_groups)
                results['group_pred'].extend(pred_groups.cpu().numpy())
                results['family_true'].extend(batch.family)
                results['embeddings'].extend(embeddings.cpu().numpy())
                
                # Get family predictions for each sample
                for i, (pred_group, true_family) in enumerate(zip(pred_groups, batch.family)):
                    group_id = str(pred_group.item())
                    if group_id in family_logits:
                        family_logits_group = family_logits[group_id][i]
                        pred_family_idx = family_logits_group.argmax().item()
                        # Map predicted family based on group's family list
                        pred_family = group_to_families[pred_group.item()][pred_family_idx]
                        results['family_pred'].append(pred_family)
                    else:
                        results['family_pred'].append('unknown')
                        
                    # Store detailed info for each sample
                    sample_details.append({
                        'true_family': true_family,
                        'pred_family': results['family_pred'][-1],
                        'true_group': true_groups[-1],
                        'pred_group': pred_groups[i].item(),
                        'group_confidence': torch.softmax(group_logits[i], dim=0).max().item(),
                        'family_confidence': torch.softmax(family_logits[group_id][i], dim=0).max().item() 
                        if group_id in family_logits else 0
                    })
    
    return results, sample_details

def evaluate_group_metrics(results):
    """Evaluate behavioral group classification metrics"""
    true_groups = np.array(results['group_true'])
    pred_groups = np.array(results['group_pred'])
    
    # Overall metrics
    accuracy = np.mean(true_groups == pred_groups)
    balanced_acc = balanced_accuracy_score(true_groups, pred_groups)
    
    # Per-group metrics
    group_metrics = {}
    for group in set(true_groups):
        if group == -999:  # Skip unknown group
            continue
        mask = (true_groups == group)
        group_metrics[group] = {
            'accuracy': np.mean(pred_groups[mask] == group),
            'samples': mask.sum(),
            'confusion': {
                int(pred): np.sum((true_groups == group) & (pred_groups == pred))
                for pred in set(pred_groups)
            }
        }
    
    return {
        'overall_accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'per_group': group_metrics
    }

def evaluate_family_metrics(results):
    """Evaluate family-level classification metrics"""
    family_metrics = classification_report(
        results['family_true'],
        results['family_pred'],
        output_dict=True
    )
    
    # Add sample counts
    family_counts = Counter(results['family_true'])
    for family in family_metrics:
        if family in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        family_metrics[family]['samples'] = family_counts[family]
    
    return family_metrics

def analyze_errors(sample_details):
    """Analyze error patterns"""
    error_patterns = defaultdict(list)
    
    for sample in sample_details:
        if sample['true_group'] != sample['pred_group']:
            error_patterns['group_errors'].append({
                'true_group': sample['true_group'],
                'pred_group': sample['pred_group'],
                'confidence': sample['group_confidence'],
                'true_family': sample['true_family']
            })
        
        if sample['true_family'] != sample['pred_family']:
            error_patterns['family_errors'].append({
                'true_family': sample['true_family'],
                'pred_family': sample['pred_family'],
                'confidence': sample['family_confidence']
            })
    
    # Analyze common misclassification patterns
    group_confusion = Counter([
        (e['true_group'], e['pred_group']) 
        for e in error_patterns['group_errors']
    ])
    
    family_confusion = Counter([
        (e['true_family'], e['pred_family']) 
        for e in error_patterns['family_errors']
    ])
    
    return {
        'group_confusion': dict(group_confusion.most_common(10)),
        'family_confusion': dict(family_confusion.most_common(10)),
        'low_confidence': [
            s for s in sample_details 
            if s['group_confidence'] < 0.5 or s['family_confidence'] < 0.5
        ]
    }

def plot_metrics(output_dir, group_metrics, family_metrics, error_analysis):
    """Generate visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group accuracy distribution
    plt.figure(figsize=(10, 6))
    group_accs = [m['accuracy'] for m in group_metrics['per_group'].values()]
    group_sizes = [m['samples'] for m in group_metrics['per_group'].values()]
    plt.scatter(group_sizes, group_accs, alpha=0.6)
    plt.xlabel('Number of Samples')
    plt.ylabel('Group Accuracy')
    plt.title('Group Accuracy vs Sample Size')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'group_accuracy_dist.png')
    plt.close()
    
    # Confidence distribution for errors
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    group_conf = [e['confidence'] for e in error_analysis['low_confidence']]
    plt.hist(group_conf, bins=20)
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Group Prediction Confidence\nfor Errors')
    
    plt.subplot(1, 2, 2)
    family_conf = [e['family_confidence'] for e in error_analysis['low_confidence']]
    plt.hist(family_conf, bins=20)
    plt.xlabel('Confidence')
    plt.title('Family Prediction Confidence\nfor Errors')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_confidence_dist.png')
    plt.close()

def evaluate():
    """Main evaluation function"""
    # Default paths
    model_path = 'best_model.pt'
    test_dir = 'bodmas_batches/test'
    groups_path = 'behavioral_analysis/behavioral_groups.json'
    output_dir = 'evaluation_results'
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # First, get test files
    test_files = list(Path(test_dir).glob('batch_*.pt'))
    if not test_files:
        print(f"Error: No test files found in {test_dir}")
        return

    # Load behavioral groups
    try:
        with open(groups_path, 'r') as f:
            group_data = json.load(f)
            
        # Create group_to_families mapping
        group_to_families = {}
        for group_id, families in group_data.items():
            group_to_families[int(group_id)] = sorted(families)
        
        # Create reverse mapping: family -> group
        family_to_group = {}
        for group_id, families in group_to_families.items():
            for family in families:
                family_to_group[family] = group_id
                
        print(f"Loaded {len(family_to_group)} families in {len(group_to_families)} groups")
    except FileNotFoundError:
        print(f"Error: {groups_path} not found!")
        return
    
    # Now load model
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get number of features from first test batch
        first_batch = torch.load(test_files[0])
        num_features = first_batch[0].x.size(1)
        
        # Initialize model architecture
        model = HierarchicalMalwareGNN(
            num_features=num_features,
            num_groups=len(group_to_families),
            embedding_dim=256  # Use same as training
        ).to(device)
        
        # Add family classifiers for each group including unknown group (-999)
        group_to_families[-999] = []  # Add unknown group
        for group_id, families in group_to_families.items():
            if group_id == -999:  # For unknown group
                model.add_family_classifier(str(group_id), 0)  # Empty classifier as in saved model
            else:
                model.add_family_classifier(str(group_id), len(families))
        
        # Load the trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except FileNotFoundError:
        print(f"Error: {model_path} not found!")
        return
        
    # Load and create behavioral group mappings
    try:
        # Load behavioral groups JSON - should be in format:
        # {"0": ["family1", "family2"], "1": ["family3", "family4"], ...}
        with open(groups_path, 'r') as f:
            group_data = json.load(f)
            
        # Create group_to_families mapping
        group_to_families = {}
        for group_id, families in group_data.items():
            group_to_families[int(group_id)] = sorted(families)  # Convert to int and sort families
        
        # Create reverse mapping: family -> group
        family_to_group = {}
        for group_id, families in group_to_families.items():
            for family in families:
                family_to_group[family] = group_id
                
        print(f"Loaded {len(family_to_group)} families in {len(group_to_families)} groups")
    except FileNotFoundError:
        print(f"Error: {groups_path} not found!")
        return
    
    # Get test files
    test_files = list(Path(test_dir).glob('batch_*.pt'))
    if not test_files:
        print(f"Error: No test files found in {test_dir}")
        return
    
    # Run analysis
    print("Analyzing model predictions...")
    results, sample_details = analyze_predictions(model, test_files, family_to_group, group_to_families, device)
    
    # Compute metrics
    group_metrics = evaluate_group_metrics(results)
    family_metrics = evaluate_family_metrics(results)
    error_analysis = analyze_errors(sample_details)
    
    # Print summary results
    print("\n=== Behavioral Group Classification ===")
    print(f"Overall Accuracy: {group_metrics['overall_accuracy']:.4f}")
    print(f"Balanced Accuracy: {group_metrics['balanced_accuracy']:.4f}")
    
    print("\n=== Group Performance Summary ===")
    for group_id, metrics in group_metrics['per_group'].items():
        print(f"\nGroup {group_id}:")
        print(f"  Samples: {metrics['samples']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n=== Most Common Group Misclassifications ===")
    for (true_group, pred_group), count in error_analysis['group_confusion'].items():
        print(f"True Group {true_group} → Predicted Group {pred_group}: {count} samples")
    
    print("\n=== Family Classification Performance ===")
    weighted_avg = family_metrics['weighted avg']
    print(f"Weighted Precision: {weighted_avg['precision']:.4f}")
    print(f"Weighted Recall: {weighted_avg['recall']:.4f}")
    print(f"Weighted F1: {weighted_avg['f1-score']:.4f}")
    
    # Generate plots
    plot_metrics(output_dir, group_metrics, family_metrics, error_analysis)
    
    # Save detailed results
    output_dir = Path(output_dir)
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump({
            'group_metrics': group_metrics,
            'family_metrics': family_metrics,
            'error_analysis': {
                'group_confusion': error_analysis['group_confusion'],
                'family_confusion': error_analysis['family_confusion']
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_dir}/")

if __name__ == "__main__":
    evaluate()