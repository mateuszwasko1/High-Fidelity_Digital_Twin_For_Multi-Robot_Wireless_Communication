"""
VLM Detector Module
Provides interface for Vision-Language Models with CLIP implementation
"""

from abc import ABC, abstractmethod
from typing import Dict
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


class CLIPDetector(VLMDetector):
    """
    CLIP (OpenAI) detector for shape classification
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize CLIP detector
        
        Args:
            device: Device to run on ('cpu', 'cuda', or 'mps')
        
        Note: Default is 'cpu' for better reliability on Mac.
        """
        super().__init__()
        self.device = device
        # CLIP works best with clear, descriptive phrases
        # Describe what the model sees from TOP-DOWN perspective
        # Use very distinct descriptions to improve classification
        self.candidate_labels = [
            "a perfect circle or sphere",
            "a triangle with three corners and three sides", 
            "a rectangle with four corners and four sides"
        ]
        self.loaded = False
    
    def load_model(self):
        """Load CLIP model"""
        if self.loaded:
            return
        
        try:
            import torch
            import clip
            
            # Use ViT-B/32 - Much faster than ViT-L/14, still accurate enough
            # ViT-B/32: 338MB, ~3x faster inference than ViT-L/14 (890MB)
            print("Loading CLIP model (ViT-B/32 - Fast)...")
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Pre-encode text labels once (major speedup!)
            text_inputs = torch.cat([clip.tokenize(label) for label in self.candidate_labels]).to(self.device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(text_inputs)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
            self.loaded = True
            print("CLIP model loaded successfully (optimized)")
            
        except ImportError as e:
            print(f"Failed to load CLIP: {e}")
            print("   Install with: pip install git+https://github.com/openai/CLIP.git")
            raise
    
    def detect_object(self, image: np.ndarray, confidence_threshold: float = 0.1) -> Dict:
        """
        Detect object using CLIP
        
        Args:
            image: RGB image as numpy array
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Detection result dictionary
        """
        if not self.loaded:
            self.load_model()
        
        import torch
        import clip
        from PIL import Image
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Preprocess image (faster without debug window)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Get predictions (using pre-encoded text features - major speedup!)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity using pre-computed text features
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(len(self.candidate_labels))
        
        # Get best prediction
        best_idx = indices[0]
        best_label = self.candidate_labels[best_idx]
        confidence = values[0].item()
        
        shape = self._extract_shape(best_label)
        
        return {
            "name": best_label,
            "confidence": float(confidence),
            "shape": shape,
            "color": "unknown"
        }
    
    def _extract_shape(self, label: str) -> str:
        """Extract 3D shape from CLIP label"""
        label_lower = label.lower()
        
        if 'sphere' in label_lower or 'ball' in label_lower or 'round' in label_lower:
            return 'sphere'
        elif 'triangle' in label_lower or 'triangular' in label_lower:
            return 'triangle'
        elif 'rectangular' in label_lower or 'rectangle' in label_lower or 'box' in label_lower:
            return 'rectangle'
        
        return 'unknown'


def create_detector(model_name: str = "clip") -> VLMDetector:
    """
    Factory function to create a detector instance
    
    Args:
        model_name: Name of the model (only 'clip' supported)
        
    Returns:
        VLMDetector instance
    """
    if model_name.lower() == "clip":
        return CLIPDetector()
    else:
        raise ValueError(f"Unknown model: {model_name}. Only 'clip' is supported.")


if __name__ == "__main__":
    # Test the detector
    print("Testing CLIP Detector...")
    
    # Create a dummy image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test CLIP
    try:
        detector = create_detector("clip")
        detector.load_model()
        result = detector.detect_object(test_image)
        print(f"Detection result: {result}")
    except Exception as e:
        print(f"Test failed: {e}")
