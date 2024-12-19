import torch
import json
import numpy as np
from architectures import PhasedGNN
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
import logging
import glob
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    def __init__(self, model_dir="trained_model"):
        # Load model
        checkpoint = torch.load(os.path.join(model_dir, "phased_gnn_final.pt"))
        
        self.model = PhasedGNN(
            num_node_features=checkpoint['num_features'],
            num_families=checkpoint['num_families']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load mapping
        with open(os.path.join(model_dir, "family_mapping.json"), 'r') as f:
            self.mapping = json.load(f)
        
        self.idx_to_family = {int(k): v for k, v in self.mapping['idx_to_family'].items()}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def analyze_batch(self, batch_file, family_to_idx):
        """Analyze a single batch file for all phases."""
        results = {
            'family': defaultdict(list),
            'goodware': defaultdict(list),
            'novelty': defaultdict(list)
        }
        
        try:
            batch = torch.load(batch_file)
            loader = DataLoader([g.to(self.device) for g in batch], batch_size=32)
            
            with torch.no_grad():
                for data in loader:
                    # Family classification
                    self.model.set_phase('family')
                    logits = self.model(data)
                    preds = logits.argmax(dim=1)
                    
                    for true, pred in zip(data.y.cpu(), preds.cpu()):
                        true_family = self.idx_to_family[true.item()]
                        pred_family = self.idx_to_family[pred.item()]
                        results['family']['true'].append(true_family)
                        results['family']['pred'].append(pred_family)
                    
                    # Goodware detection
                    self.model.set_phase('goodware')
                    logits = self.model(data)
                    is_malware = (logits[:, 1] > logits[:, 0]).float()
                    
                    for true, pred in zip(data.y.cpu(), is_malware.cpu()):
                        true_type = "goodware" if true.item() == 0 else "malware"
                        pred_type = "malware" if pred.item() == 1 else "goodware"
                        results['goodware']['true'].append(true_type)
                        results['goodware']['pred'].append(pred_type)
                    
                    # Novelty detection
                    self.model.set_phase('novelty')
                    logits, novelty_scores = self.model(data)
                    is_novel = novelty_scores > 0.5
                    
                    for true, score in zip(data.y.cpu(), novelty_scores.cpu()):
                        family = self.idx_to_family[true.item()]
                        results['novelty']['families'].append(family)
                        results['novelty']['scores'].append(score.item())
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_file}: {str(e)}")
            
        return results

    def plot_confusion_matrix(self, true_labels, pred_labels, phase):
        """Plot confusion matrix for given predictions."""
        plt.figure(figsize=(12, 8))
        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {phase} Phase')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{phase}.png')
        plt.close()

    def plot_novelty_distribution(self, families, scores):
        """Plot novelty score distribution by family."""
        plt.figure(figsize=(12, 6))
        
        # Box plot for each family
        family_scores = defaultdict(list)
        for f, s in zip(families, scores):
            family_scores[f].append(s)
            
        plt.boxplot([scores for scores in family_scores.values()], 
                   labels=family_scores.keys())
        plt.xticks(rotation=45)
        plt.title('Novelty Score Distribution by Family')
        plt.ylabel('Novelty Score')
        plt.tight_layout()
        plt.savefig('novelty_distribution.png')
        plt.close()

    def analyze_dataset(self, data_dir='bodmas_batches', split='test'):
        """Analyze full test dataset."""
        results = {
            'family': defaultdict(list),
            'goodware': defaultdict(list),
            'novelty': defaultdict(list)
        }
        
        split_dir = os.path.join(data_dir, split)
        batch_files = glob.glob(os.path.join(split_dir, 'batch_*.pt'))
        
        for batch_file in batch_files:
            batch_results = self.analyze_batch(batch_file, self.mapping['family_to_idx'])
            
            # Aggregate results
            for phase in results:
                for key in batch_results[phase]:
                    results[phase][key].extend(batch_results[phase][key])
        
        # Generate reports
        logger.info("\nFamily Classification Report:")
        print(classification_report(results['family']['true'], 
                                 results['family']['pred']))
        
        logger.info("\nGoodware Detection Report:")
        print(classification_report(results['goodware']['true'], 
                                 results['goodware']['pred']))
        
        # Plot results
        self.plot_confusion_matrix(results['family']['true'],
                                 results['family']['pred'],
                                 'Family')
        
        self.plot_confusion_matrix(results['goodware']['true'],
                                 results['goodware']['pred'],
                                 'Goodware')
        
        self.plot_novelty_distribution(results['novelty']['families'],
                                     results['novelty']['scores'])

def main():
    analyzer = ModelAnalyzer()
    analyzer.analyze_dataset()

if __name__ == "__main__":
    main()