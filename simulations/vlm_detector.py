"""
VLM Detector Module
Provides a flexible interface for different Vision-Language Models
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class VLMDetector(ABC):
    """
    Abstract base class for Vision-Language Model object detectors
    """
    
    def __init__(self):
        """Initialize the detector"""
        self.model = None
        self.processor = None
    
    @abstractmethod
    def detect_object(self, image: np.ndarray) -> Dict:
        """
        Detect and classify object in the image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Dictionary with detection results:
            {
                'name': str,  # Object name/class
                'confidence': float,  # Detection confidence (0-1)
                'shape': str,  # Object shape if detectable
                'color': str,  # Object color if detectable
            }
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the model into memory"""
        pass


class OWLViTDetector(VLMDetector):
    """
    OWL-ViT-B/16 (MPS) detector implementation
    """
    
    def __init__(self, device: str = "mps"):
        """
        Initialize OWL-ViT detector
        
        Args:
            device: Device to run on ('mps', 'cuda', or 'cpu')
        """
        super().__init__()
        self.device = device
        # Use 2D shape names since we're viewing from top-down
        # circle = sphere, square = cube, rectangle = box
        self.candidate_labels = [
            "square", "circle", "rectangle"
        ]
        self.loaded = False
    
    def load_model(self):
        """Load OWL-ViT model"""
        if self.loaded:
            return
        
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            import torch
            
            print("📦 Loading OWL-ViT-B/16 model...")
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
            
            # Move to device
            if self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            self.loaded = True
            print("✅ OWL-ViT model loaded")
            
        except ImportError as e:
            print(f"❌ Failed to load OWL-ViT: {e}")
            print("   Install with: pip install transformers torch")
            raise
    
    def detect_object(self, image: np.ndarray, confidence_threshold: float = 0.1) -> Dict:
        """
        Detect object using OWL-ViT
        
        Args:
            image: RGB image as numpy array
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Detection result dictionary
        """
        if not self.loaded:
            self.load_model()
        
        import torch
        from PIL import Image
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Prepare inputs
        inputs = self.processor(
            text=[self.candidate_labels], 
            images=pil_image, 
            return_tensors="pt"
        )
        
        # Move to device
        if self.device == "mps" and torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        target_sizes = torch.Tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )[0]
        
        # Print all predictions for debugging
        print(f"\n🔍 VLM Predictions (threshold={confidence_threshold}):")
        print(f"   Image size: {pil_image.size}")
        print(f"   Total detections: {len(results['scores'])}")
        
        if len(results["scores"]) > 0:
            # Print all predictions sorted by score
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            boxes = results["boxes"].cpu().numpy()
            
            # Sort by score descending
            sorted_indices = np.argsort(scores)[::-1]
            
            print(f"\n   All predictions:")
            for i, idx in enumerate(sorted_indices[:10]):  # Show top 10
                label_name = self.candidate_labels[labels[idx]]
                score = scores[idx]
                box = boxes[idx]
                print(f"   {i+1}. {label_name}: {score:.4f} (box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}])")
            
            # Get best prediction
            best_idx = results["scores"].argmax()
            max_score = results["scores"][best_idx]
            best_label = self.candidate_labels[results["labels"][best_idx]]
            print(f"\n   ✅ Best prediction: {best_label} ({float(max_score):.4f})")
        else:
            print(f"   ⚠️ No detections above threshold!")
            best_label = "rectangle"
            max_score = 0.0
        
        shape = self._extract_shape(best_label)
        
        return {
            "name": best_label,
            "confidence": float(max_score),
            "shape": shape,
            "color": "unknown"
        }
    
    def _extract_shape(self, label: str) -> str:
        """Extract 3D shape from 2D label (mapping back from top-down view)"""
        label_lower = label.lower()
        if 'square' in label_lower:
            return 'cube'  # square → cube
        elif 'circle' in label_lower:
            return 'sphere'  # circle → sphere
        elif 'rectangle' in label_lower:
            return 'box'  # rectangle → box
        return 'unknown'
    
    


class NanoOWLDetector(VLMDetector):
    """
    NanoOWL/GroundingDino-T detector implementation (placeholder for future)
    """
    
    def __init__(self):
        """Initialize NanoOWL detector"""
        super().__init__()
        self.loaded = False
    
    def load_model(self):
        """Load NanoOWL model"""
        if self.loaded:
            return
        
        # TODO: Implement NanoOWL loading
        print("⚠️ NanoOWL not yet implemented")
        raise NotImplementedError("NanoOWL detector coming soon")
    
    def detect_object(self, image: np.ndarray) -> Dict:
        """Detect object using NanoOWL"""
        if not self.loaded:
            self.load_model()
        
        # TODO: Implement NanoOWL detection
        return {
            'name': 'unknown',
            'confidence': 0.0,
            'shape': 'unknown',
            'color': 'unknown'
        }


def create_detector(model_name: str = "owlvit") -> VLMDetector:
    """
    Factory function to create a detector instance
    
    Args:
        model_name: Name of the model ('owlvit', 'nanoowl', 'groundingdino')
        
    Returns:
        VLMDetector instance
    """
    model_name = model_name.lower()
    
    if model_name in ["owlvit", "owl-vit", "owlvit-b"]:
        return OWLViTDetector()
    elif model_name in ["nanoowl", "nano-owl"]:
        return NanoOWLDetector()
    elif model_name in ["groundingdino", "grounding-dino"]:
        # Can add GroundingDino as separate class or reuse NanoOWL
        return NanoOWLDetector()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: owlvit, nanoowl, groundingdino")


if __name__ == "__main__":
    # Test the detector
    print("Testing VLM Detector...")
    
    # Create a dummy image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test OWL-ViT
    try:
        detector = create_detector("owlvit")
        detector.load_model()
        result = detector.detect_object(test_image)
        print(f"Detection result: {result}")
    except Exception as e:
        print(f"Test failed: {e}")
