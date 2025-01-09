import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import json
import logging
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
import cv2
from PIL import Image
import random
from CNN_Malimg import MalwareCNN, prepare_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class CNNEvasionAnalyzer:
    """Analyze CNN model robustness against image-based evasion techniques."""
    
    def __init__(self, model, device, transform=None):
        self.model = model
        self.device = device
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _get_prediction(self, image_tensor):
        """Get model prediction and confidence."""
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
                
            logits, novelty_score = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs.max(dim=1)[0].item()
            return pred, conf, float(novelty_score.squeeze())

    def test_evasion_techniques(self, image_tensor):
        """Test various evasion techniques on a single image."""
        techniques = {
            'noise_injection': self._test_noise_injection,
            'rotation_transform': self._test_rotation,
            'blur_transform': self._test_blur,
            'contrast_adjustment': self._test_contrast,
            'pixel_perturbation': self._test_pixel_perturbation
        }

        results = {}
        original_pred, original_conf, original_novelty = self._get_prediction(image_tensor)

        for technique_name, technique_func in techniques.items():
            try:
                perturbed_image = technique_func(image_tensor.clone())
                new_pred, new_conf, new_novelty = self._get_prediction(perturbed_image)
                
                results[technique_name] = {
                    'evasion_success': int(new_pred != original_pred),
                    'confidence_drop': float(original_conf - new_conf),
                    'detection_score': float(new_conf),
                    'novelty_score_change': float(new_novelty - original_novelty)
                }
            except Exception as e:
                logger.error(f"Error in {technique_name}: {str(e)}")
                results[technique_name] = {
                    'evasion_success': 0,
                    'confidence_drop': 0,
                    'detection_score': 0,
                    'novelty_score_change': 0
                }

        return results

    def _test_noise_injection(self, image_tensor):
        """Add Gaussian noise to the image."""
        noise = torch.randn_like(image_tensor) * 0.1
        noisy_image = image_tensor + noise
        return torch.clamp(noisy_image, 0, 1)

    def _test_rotation(self, image_tensor):
        """Apply small random rotation."""
        angle = random.uniform(-15, 15)
        rotated_image = transforms.functional.rotate(image_tensor, angle)
        return rotated_image

    def _test_blur(self, image_tensor):
        """Apply Gaussian blur."""
        # Convert to numpy for OpenCV operations
        np_image = image_tensor.cpu().numpy().transpose(1, 2, 0)
        blurred = cv2.GaussianBlur(np_image, (5, 5), 0)
        # Convert back to tensor
        tensor_image = torch.from_numpy(blurred.transpose(2, 0, 1))
        return tensor_image

    def _test_contrast(self, image_tensor):
        """Adjust image contrast."""
        factor = random.uniform(0.8, 1.2)
        adjusted = transforms.functional.adjust_contrast(image_tensor, factor)
        return adjusted

    def _test_pixel_perturbation(self, image_tensor):
        """Randomly modify pixel values."""
        mask = torch.rand_like(image_tensor) > 0.95  # Modify 5% of pixels
        perturbation = torch.rand_like(image_tensor) * 0.1
        perturbed_image = image_tensor.clone()
        perturbed_image[mask] += perturbation[mask]
        return torch.clamp(perturbed_image, 0, 1)


def evaluate_cnn_security(model, test_loader, device):
    """Evaluate CNN model security against various evasion techniques."""
    analyzer = CNNEvasionAnalyzer(model, device)
    security_metrics = defaultdict(list)
    
    for batch in tqdm(test_loader, desc="Evaluating security"):
        images = batch['image']
        for image in images:
            try:
                results = analyzer.test_evasion_techniques(image)
                for technique, metrics in results.items():
                    security_metrics[technique].append(metrics)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                continue

    # Aggregate results
    aggregated_metrics = {}
    for technique, results in security_metrics.items():
        aggregated_metrics[technique] = {
            'evasion_success_rate': np.mean([r['evasion_success'] for r in results]),
            'avg_confidence_drop': np.mean([r['confidence_drop'] for r in results]),
            'avg_detection_score': np.mean([r['detection_score'] for r in results]),
            'avg_novelty_score_change': np.mean([r['novelty_score_change'] for r in results]),
            'std_evasion_success': np.std([r['evasion_success'] for r in results]),
            'std_confidence_drop': np.std([r['confidence_drop'] for r in results]),
            'std_detection_score': np.std([r['detection_score'] for r in results]),
            'std_novelty_score_change': np.std([r['novelty_score_change'] for r in results])
        }

    return aggregated_metrics, security_metrics


def main():
    """Main function to run CNN evasion analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load the trained model
    model_path = 'best_model_CNN_malimg.pt'
    checkpoint = torch.load(model_path)
    
    # Initialize model (use your MalwareCNN class)
    model = MalwareCNN(num_classes=25).to(device)  # Adjust num_classes as needed
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare test data
    img_dir = "/data/datasets/malimg/img_files"
    _, _, test_loader, _ = prepare_data(img_dir)

    # Run security evaluation
    logger.info("Starting security evaluation...")
    aggregated_metrics, per_sample_metrics = evaluate_cnn_security(model, test_loader, device)

    # Save results
    output_dir = Path('security_analysis')
    output_dir.mkdir(exist_ok=True)

    # Save detailed results
    with open(output_dir / 'cnn_evasion_analysis.json', 'w') as f:
        json.dump({
            'aggregate_results': aggregated_metrics,
            'per_sample_results': per_sample_metrics
        }, f, indent=2, cls=NumpyEncoder)

    # Generate and save summary
    summary = {
        'overall_robustness': {
            technique: {
                'evasion_resistance': 1 - metrics['evasion_success_rate'],
                'confidence_stability': 1 - metrics['avg_confidence_drop'],
                'detection_reliability': metrics['avg_detection_score'],
                'novelty_stability': 1 - abs(metrics['avg_novelty_score_change'])
            }
            for technique, metrics in aggregated_metrics.items()
        },
        'vulnerability_ranking': sorted(
            aggregated_metrics.items(),
            key=lambda x: x[1]['evasion_success_rate'],
            reverse=True
        )
    }

    with open(output_dir / 'cnn_security_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Log final results
    logger.info("\nFinal Security Analysis Results:")
    for technique, metrics in aggregated_metrics.items():
        logger.info(f"\n{technique}:")
        logger.info(f"  Evasion Success Rate: {metrics['evasion_success_rate']:.3f} (±{metrics['std_evasion_success']:.3f})")
        logger.info(f"  Average Confidence Drop: {metrics['avg_confidence_drop']:.3f} (±{metrics['std_confidence_drop']:.3f})")
        logger.info(f"  Average Detection Score: {metrics['avg_detection_score']:.3f} (±{metrics['std_detection_score']:.3f})")
        logger.info(f"  Average Novelty Score Change: {metrics['avg_novelty_score_change']:.3f} (±{metrics['std_novelty_score_change']:.3f})")


if __name__ == "__main__":
    main()