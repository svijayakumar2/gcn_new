import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import os
from pathlib import Path
import random
from PIL import Image
import logging
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalwareImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, novel_families=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.novel_families = novel_families or set()
        
        # Get all image files
        self.samples = []
        self.family_to_idx = {}
        self.idx_to_family = {}
        self.load_samples()
        
    def load_samples(self):
        """Load all image samples and create family mappings."""
        idx = 0
        for img_path in self.img_dir.glob("**/*.png"):
            family = img_path.name.split('_')[0]
            
            if family not in self.family_to_idx:
                self.family_to_idx[family] = idx
                self.idx_to_family[idx] = family
                idx += 1
                
            self.samples.append({
                'path': img_path,
                'family': family,
                'is_novel': family in self.novel_families
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.family_to_idx[sample['family']]
        is_novel = sample['is_novel']
        
        return {
            'image': image,
            'label': label,
            'is_novel': is_novel,
            'family': sample['family']
        }

class MalwareCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Novelty detection head
        self.novelty_detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        novelty_score = self.novelty_detector(features)
        return logits, novelty_score

class MalwareTrainer:
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        self.criterion = nn.CrossEntropyLoss()
        self.novelty_criterion = nn.BCELoss()
        
        # Thresholds for novelty detection
        self.confidence_threshold = 0.8
        self.novelty_threshold = 0.7
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            is_novel = batch['is_novel'].float().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, novelty_scores = self.model(images)
            
            # Classification loss for known samples
            known_mask = ~is_novel#.bool()
            if known_mask.any():
                cls_loss = self.criterion(logits[known_mask], labels[known_mask])
            else:
                cls_loss = 0
                
            # Novelty detection loss
            novelty_loss = self.novelty_criterion(novelty_scores.squeeze(), is_novel)
            
            # Combined loss
            loss = cls_loss + novelty_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred[known_mask].eq(labels[known_mask]).sum().item()
            total += known_mask.sum().item()
            
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total if total > 0 else 0
        }
        
    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_novelty_scores = []
        all_is_novel = []
        all_families = []
        
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                is_novel = batch['is_novel']
                
                # Forward pass
                logits, novelty_scores = self.model(images)
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                confidences, preds = probs.max(dim=1)
                
                # Track metrics
                known_mask = ~is_novel
                if known_mask.any():
                    correct += preds[known_mask].eq(labels[known_mask]).sum().item()
                    total += known_mask.sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_novelty_scores.extend(novelty_scores.cpu().numpy())
                all_is_novel.extend(is_novel.numpy())
                all_families.extend(batch['family'])
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_novelty_scores = np.array(all_novelty_scores).squeeze()
        all_is_novel = np.array(all_is_novel)
        
        # Calculate metrics
        metrics = {
            'accuracy': correct / total if total > 0 else 0,
            'known': self.compute_known_metrics(all_preds, all_labels, all_is_novel),
            'novel': self.compute_novel_metrics(all_novelty_scores, all_is_novel)
        }
        
        return metrics
    
    def compute_known_metrics(self, preds, labels, is_novel):
        known_mask = ~is_novel.bool() # TODO just added this 
        if not known_mask.any():
            return {'precision': 0, 'recall': 0, 'f1': 0}
            
        known_preds = preds[known_mask]
        known_labels = labels[known_mask]
        
        return {
            'precision': classification_report(known_labels, known_preds, output_dict=True)['macro avg']['precision'],
            'recall': classification_report(known_labels, known_preds, output_dict=True)['macro avg']['recall'],
            'f1': classification_report(known_labels, known_preds, output_dict=True)['macro avg']['f1-score']
        }
        
    def compute_novel_metrics(self, novelty_scores, is_novel):
        novel_preds = novelty_scores > self.novelty_threshold
        
        return {
            'precision': sum((novel_preds == 1) & (is_novel == 1)) / sum(novel_preds == 1) if sum(novel_preds == 1) > 0 else 0,
            'recall': sum((novel_preds == 1) & (is_novel == 1)) / sum(is_novel == 1) if sum(is_novel == 1) > 0 else 0
        }

def prepare_data(img_dir, novel_ratio=0.1, val_ratio=0.15, test_ratio=0.15):
    """Split data into train/val/test and select novel families."""
    img_dir = Path(img_dir)
    
    # Get all unique families
    families = set()
    for img_path in img_dir.glob("**/*.png"):
        family = img_path.name.split('_')[0]
        families.add(family)
    
    # Select novel families
    num_novel = max(1, int(len(families) * novel_ratio))
    novel_families = set(random.sample(list(families), num_novel))
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = MalwareImageDataset(img_dir, transform=transform, novel_families=novel_families)
    
    # Split indices
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    
    test_size = int(len(indices) * test_ratio)
    val_size = int(len(indices) * val_ratio)
    train_size = len(indices) - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_indices),
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, val_indices),
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, test_indices),
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader, full_dataset.family_to_idx

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data
    img_dir = "/data/datasets/malimg/img_files"
    train_loader, val_loader, test_loader, family_to_idx = prepare_data(img_dir)
    
    # Initialize model
    model = MalwareCNN(num_classes=len(family_to_idx)).to(device)
    
    # Initialize trainer
    trainer = MalwareTrainer(model, device)
    
    # Training loop
    num_epochs = 50
    best_f1 = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Log metrics
        logger.info(f"\nEpoch {epoch + 1}:")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val Known F1: {val_metrics['known']['f1']:.4f}")
        logger.info(f"Val Known Precision: {val_metrics['known']['precision']:.4f}")
        logger.info(f"Val Known Recall: {val_metrics['known']['recall']:.4f}")
        logger.info(f"Val Novel Precision: {val_metrics['novel']['precision']:.4f}")
        logger.info(f"Val Novel Recall: {val_metrics['novel']['recall']:.4f}")
        
        # Early stopping
        current_f1 = val_metrics['known']['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'metrics': val_metrics
            }, 'best_model_CNN.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
        # Update learning rate
        trainer.scheduler.step(current_f1)
    
    # Final evaluation
    logger.info("\nLoading best model for final evaluation...")
    checkpoint = torch.load('best_model_CNN.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = trainer.evaluate(test_loader)
    logger.info("\nFinal Test Results:")
    logger.info(f"Known Families F1: {test_metrics['known']['f1']:.4f}")
    logger.info(f"Known Detection Precision: {test_metrics['known']['precision']:.4f}")
    logger.info(f"Known Detection Recall: {test_metrics['known']['recall']:.4f}")
    logger.info(f"Novel Detection Precision: {test_metrics['novel']['precision']:.4f}")
    logger.info(f"Novel Detection Recall: {test_metrics['novel']['recall']:.4f}")
    logger.info(f"Novel F1: {test_metrics['novel']['f1']:.4f}")

if __name__ == "__main__":
    main()