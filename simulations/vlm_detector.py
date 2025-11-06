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


class CLIPDetector(VLMDetector):
    """
    CLIP (OpenAI) detector implementation - Much better for shape classification
    """
    
    def __init__(self, device: str = "mps"):
        """
        Initialize CLIP detector
        
        Args:
            device: Device to run on ('mps', 'cuda', or 'cpu')
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
            print("🔄 Loading CLIP model (ViT-B/32 - Fast)...")
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Pre-encode text labels once (major speedup!)
            text_inputs = torch.cat([clip.tokenize(label) for label in self.candidate_labels]).to(self.device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(text_inputs)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
            self.loaded = True
            print("✅ CLIP model loaded successfully (optimized)")
            
        except ImportError as e:
            print(f"❌ Failed to load CLIP: {e}")
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
        
        # Print all predictions for debugging
        print(f"\n   🔍 CLIP predictions:")
        for i in range(len(self.candidate_labels)):
            conf = values[i].item()
            label = self.candidate_labels[indices[i]]
            print(f"      {label}: {conf*100:.1f}%")
        
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


class OWLViTDetector(VLMDetector):
    """
    OWL-ViT-B/16 (MPS) detector implementation (BACKUP - CLIP is better)
    """
    
    def __init__(self, device: str = "mps"):
        """
        Initialize OWL-ViT detector
        
        Args:
            device: Device to run on ('mps', 'cuda', or 'cpu')
        """
        super().__init__()
        self.device = device
        # Use simple geometric shape descriptions
        # OWL-ViT works better with natural language descriptions
        self.candidate_labels = [
            "a circular object",        # Spheres appear circular from top
            "a square-shaped box",      # Cubes appear square from top
            "an elongated rectangular box"  # Rectangles are elongated
        ]
        self.loaded = False
    
    def load_model(self):
        """Load OWL-ViT model"""
        if self.loaded:
            return
        
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            import torch
            
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
            
            # Move to device
            if self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            self.loaded = True
            
        except ImportError as e:
            print(f"❌ Failed to load OWL-ViT: {e}")
            print("   Install with: pip install transformers torch")
            raise
    
    def detect_object(self, image: np.ndarray, confidence_threshold: float = 0.1) -> Dict:
        """
        Detect object using classical CV + OWL-ViT hybrid approach
        
        Args:
            image: RGB image as numpy array
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Detection result dictionary
        """
        # Use classical computer vision to analyze shape geometry
        # This is much more reliable than VLM for simple geometric shapes from top-down
        import cv2
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour (the object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Circularity: 4π*area/perimeter²  (1.0 = perfect circle, <0.8 = elongated)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Bounding rectangle aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 1.0
            
            # Make aspect ratio symmetric (width/height or height/width, whichever > 1)
            if aspect_ratio < 1.0:
                aspect_ratio = 1.0 / aspect_ratio
            
            print(f"\n   📐 Shape analysis:")
            print(f"      Circularity: {circularity:.3f} (1.0=circle, <0.8=elongated)")
            print(f"      Aspect ratio: {aspect_ratio:.3f} (1.0=square, >1.5=rectangle)")
            
            # Classify based on geometry
            if circularity > 0.85:
                # High circularity = sphere
                shape = "sphere"
                label = "a circular object"
                confidence = circularity
            elif aspect_ratio > 1.5:
                # Elongated = rectangle
                shape = "rectangle"
                label = "an elongated rectangular box"
                confidence = min(1.0 - circularity, aspect_ratio / 2.0)
            else:
                # Square-ish = cube
                shape = "cube"
                label = "a square-shaped box"
                confidence = 1.0 - abs(aspect_ratio - 1.0)
            
            print(f"   ✅ Classified as: {shape.upper()} (confidence: {confidence*100:.1f}%)")
            
            return {
                "name": label,
                "confidence": float(confidence),
                "shape": shape,
                "color": "unknown"
            }
        
        # Fallback if no contours found
        return {
            "name": "unknown",
            "confidence": 0.0,
            "shape": "unknown",
            "color": "unknown"
        }
    
    def _extract_shape(self, label: str) -> str:
        """Extract 3D shape from descriptive label"""
        label_lower = label.lower()
        
        # Check for shape keywords in descriptive labels
        if 'circular' in label_lower or 'round' in label_lower or 'circle' in label_lower:
            return 'sphere'  # circular → sphere
        elif 'square' in label_lower and 'box' in label_lower:
            return 'cube'  # square-shaped box → cube
        elif 'elongated' in label_lower or 'rectangular' in label_lower:
            return 'rectangle'  # elongated rectangular → rectangle
        elif 'square' in label_lower:
            return 'cube'  # square → cube (fallback)
        elif 'rectangle' in label_lower or 'box' in label_lower:
            return 'rectangle'  # rectangle/box → rectangle (fallback)
        
        return 'unknown'


class FastShapeDetector(VLMDetector):
    """
    Fast shape detector using basic CV operations
    Optimized for real-time performance with accurate shape detection
    """
    
    def __init__(self):
        super().__init__()
        self.loaded = True
    
    def load_model(self):
        """No model to load"""
        pass
    
    def detect_object(self, image: np.ndarray, confidence_threshold: float = 0.1) -> Dict:
        """
        Detect object shape using basic CV operations
        """
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "name": "unknown",
                "confidence": 0.0,
                "shape": "unknown",
                "color": "unknown"
            }
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon with tighter epsilon for better accuracy
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Calculate circularity
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate aspect ratio for rectangles vs squares
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Classify based on vertices and circularity
        num_vertices = len(approx)
        
        # More robust classification
        if circularity > 0.8:
            # Very circular - definitely a sphere
            shape = "sphere"
            confidence = min(circularity, 0.95)
        elif num_vertices <= 4 and circularity > 0.7:
            # Somewhat circular with few vertices - likely sphere
            shape = "sphere"
            confidence = 0.85
        elif num_vertices == 3:
            # Triangle
            shape = "triangle"
            confidence = 0.92
        elif num_vertices == 4:
            # Rectangle (includes squares)
            shape = "rectangle"
            confidence = 0.90
        elif 5 <= num_vertices <= 8 and circularity > 0.75:
            # Many vertices and circular - sphere
            shape = "sphere"
            confidence = 0.88
        else:
            # Unknown
            shape = "unknown"
            confidence = 0.5
        
        # Print debug info
        print(f"      CV detector: vertices={num_vertices}, circ={circularity:.2f}, aspect={aspect_ratio:.2f} -> {shape} ({confidence:.0%})")
        
        return {
            "name": f"a {shape}",
            "confidence": float(confidence),
            "shape": shape,
            "color": "unknown"
        }


def create_detector(model_name: str = "clip") -> VLMDetector:
    """
    Factory function to create a detector instance
    
    Args:
        model_name: Name of the model ('clip', 'owlvit', or 'fast')
        
    Returns:
        VLMDetector instance
    """
    model_name = model_name.lower()
    
    if model_name in ["clip"]:
        return CLIPDetector()
    elif model_name in ["owlvit", "owl-vit", "owlvit-b"]:
        return OWLViTDetector()
    elif model_name in ["fast", "debug"]:
        return FastShapeDetector()
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: 'clip', 'owlvit', 'fast'")


if __name__ == "__main__":
    # Test the detector
    print("Testing VLM Detector...")
    
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
