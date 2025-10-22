"""
VLM (Vision Language Model) Analysis Module
Handles object detection and analysis using Google Gemini Vision API
"""

import google.generativeai as genai
import json
import base64
from PIL import Image
import io
import numpy as np
from typing import List, Dict, Tuple, Optional


class VLMAnalyzer:
    """Vision Language Model analyzer for object detection and positioning"""
    
    def __init__(self, api_key: str):
        """
        Initialize VLM analyzer with API key
        
        Args:
            api_key: Google AI Studio API key
        """
        genai.configure(api_key=api_key)
        
        # Try different model names in order of preference (fastest first)
        model_names = [
            'models/gemini-2.5-flash',  # Latest and fast
            'models/gemini-robotics-er-1.5-preview',  # Robotics-specific
            'models/gemini-2.5-computer-use-preview-10-2025',  # Computer vision
        ]
        
        self.model = None
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                print(f"✅ Using model: {model_name}")
                break
            except Exception as e:
                print(f"⚠️ Model {model_name} not available: {e}")
                continue
        
        if self.model is None:
            raise RuntimeError("No available Gemini models found")
        
        # Simple scene caching to avoid repeated VLM calls
        self._scene_cache = {}
        self._cache_threshold = 0.95  # Similarity threshold for cache hits
        
    def analyze_scene(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Analyze the scene to detect objects and their positions
        
        Args:
            rgb_image: RGB image array from camera
            depth_image: Optional depth image for better position estimation
            
        Returns:
            List of detected objects with their properties
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(rgb_image.astype('uint8'))
            
            # Check cache first
            image_hash = self._hash_image(rgb_image)
            if image_hash in self._scene_cache:
                print("🚀 Using cached VLM result")
                return self._scene_cache[image_hash]
            
            # OPTIMIZATION: Resize image to reduce processing time
            # Smaller image = faster VLM processing
            original_size = pil_image.size
            target_size = (160, 120)  # Even smaller for fastest processing
            pil_image = pil_image.resize(target_size, Image.LANCZOS)
            
            # OPTIMIZATION: Compress to JPEG format to reduce data transfer
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=60)  # Lower quality = faster
            img_buffer.seek(0)
            pil_image = Image.open(img_buffer)
            
            # Create the prompt for object detection
            prompt = self._create_detection_prompt()
            
            # Send image and prompt to Gemini
            response = self.model.generate_content([prompt, pil_image])
            
            # Parse the response and scale coordinates back to original size
            objects = self._parse_response(response.text, original_size, target_size)
            
            # Cache the result for future use
            self._scene_cache[image_hash] = objects
            
            return objects
            
        except Exception as e:
            print(f"❌ VLM analysis failed: {e}")
            # Return a mock detection for testing purposes
            return [{
                "name": "mock_object",
                "color": "unknown", 
                "shape": "unknown",
                "position_description": "center",
                "confidence": "low"
            }]
    
    def _hash_image(self, image: np.ndarray) -> str:
        """Create a simple hash of the image for caching"""
        import hashlib
        # Downsample image for hashing
        small = Image.fromarray(image).resize((32, 24))
        img_bytes = np.array(small).tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    
    def _create_detection_prompt(self) -> str:
        """Create a minimal prompt for fastest object detection"""
        prompt = """JSON only:
{"objects":[{"name":"type","color":"color","shape":"shape","center_x":80,"center_y":60}]}

Find objects (cubes/spheres/cylinders) in 160x120 image. Give pixel centers. Skip robot arm."""
        return prompt
    
    def _parse_response(self, response_text: str, original_size: tuple = None, target_size: tuple = None) -> List[Dict]:
        """
        Parse the VLM response into structured object data
        
        Args:
            response_text: Raw response from VLM
            original_size: Original image size (width, height)
            target_size: Resized image size (width, height)
            
        Returns:
            List of detected objects
        """
        try:
            # Clean up the response text
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            data = json.loads(cleaned_text)
            
            # Extract objects list
            objects = []
            if 'objects' in data:
                objects = data['objects']
                
                # Scale coordinates back to original size if needed
                if original_size and target_size:
                    scale_x = original_size[0] / target_size[0]
                    scale_y = original_size[1] / target_size[1]
                    
                    for obj in objects:
                        if 'center_x' in obj:
                            obj['center_x'] = int(obj['center_x'] * scale_x)
                        if 'center_y' in obj:
                            obj['center_y'] = int(obj['center_y'] * scale_y)
                            
            return objects
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text}")
            return []
        except Exception as e:
            print(f"Error processing response: {e}")
            return []
    
    def get_object_world_coordinates(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                                   camera_params: Optional[Dict] = None) -> List[Dict]:
        """
        Complete pipeline: Detect objects and return their world coordinates
        
        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera
            camera_params: Camera intrinsic parameters
            
        Returns:
            List of objects with world coordinates
        """
        # Default camera parameters for PyBullet overhead camera
        if camera_params is None:
            camera_params = {
                'fx': 640,  # Focal length X for 640px width  
                'fy': 640,  # Focal length Y (same for square pixels)
                'cx': 320,  # Image center X (width/2)
                'cy': 240,  # Image center Y (height/2)
                'near': 0.01,  # Near clipping plane
                'far': 3.0,   # Far clipping plane
                'height': 1.5  # Camera height above table
            }
        
        # Step 1: Detect objects with VLM
        detected_objects = self.analyze_scene(rgb_image, depth_image)
        
        # Step 2: Convert to world coordinates
        objects_with_coords = []
        
        for obj in detected_objects:
            # Get pixel coordinates from VLM
            if 'center_x' in obj and 'center_y' in obj:
                pixel_x = int(obj['center_x'])
                pixel_y = int(obj['center_y'])
                
                print(f"🔍 Debug - Object {obj.get('name', 'Unknown')} detected at pixel ({pixel_x}, {pixel_y})")
                
                # Get depth at that pixel
                if 0 <= pixel_y < depth_image.shape[0] and 0 <= pixel_x < depth_image.shape[1]:
                    # PyBullet depth buffer - let's try different interpretations
                    raw_depth = depth_image[pixel_y, pixel_x]
                    
                    # Use method 2 (linearized depth buffer) 
                    near = camera_params['near']
                    far = camera_params['far']
                    if raw_depth < 1.0:
                        actual_depth = near / (1.0 - raw_depth * (1.0 - near/far))
                    else:
                        actual_depth = far
                    
                    print(f"🔍 Debug - Distance from camera: {actual_depth:.4f}m, Raw depth: {raw_depth:.4f}")
                    
                    # Convert to world coordinates
                    world_x, world_y, world_z = self.pixel_to_world_coordinates(
                        (pixel_x, pixel_y), actual_depth, camera_params)
                    
                    print(f"🔍 Debug - World coordinates: ({world_x:.4f}, {world_y:.4f}, {world_z:.4f})")
                    
                    # Add world coordinates to object
                    obj['world_coordinates'] = {
                        'x': float(world_x),
                        'y': float(world_y), 
                        'z': float(world_z)
                    }
                    obj['pixel_coordinates'] = (pixel_x, pixel_y)
                    obj['reachable'] = self._is_reachable(world_x, world_y, world_z)
                else:
                    obj['world_coordinates'] = None
                    obj['pixel_coordinates'] = None
                    obj['reachable'] = False
            else:
                # Fallback to old position description method
                obj = self._fallback_position_estimation(obj, depth_image, camera_params)
            
            objects_with_coords.append(obj)
        
        return objects_with_coords
    
    def _is_reachable(self, x: float, y: float, z: float) -> bool:
        """Check if coordinates are within robot arm reach"""
        # Simple reachability check for Franka Panda (adjust as needed)
        distance_from_base = (x**2 + y**2)**0.5
        return (distance_from_base < 0.8 and  # Within 80cm radius
                z > 0.0 and z < 0.5)        # Between table and reasonable height
    
    def _fallback_position_estimation(self, obj: Dict, depth_image: np.ndarray, 
                                    camera_params: Dict) -> Dict:
        """Fallback method using position description approach"""
        # Simple fallback - assume center position
        height, width = depth_image.shape[:2]
        pixel_coords = (width // 2, height // 2)
        
        depth_value = depth_image[pixel_coords[1], pixel_coords[0]] * camera_params['scale']
        
        world_x, world_y, world_z = self.pixel_to_world_coordinates(
            pixel_coords, depth_value, camera_params)
        
        obj['world_coordinates'] = {'x': world_x, 'y': world_y, 'z': world_z}
        obj['pixel_coordinates'] = pixel_coords
        obj['reachable'] = self._is_reachable(world_x, world_y, world_z)
        
        return obj
    
    def pixel_to_world_coordinates(self, pixel_coords: Tuple[int, int], 
                                 distance_from_camera: float, 
                                 camera_params: Dict) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to world coordinates for overhead camera
        
        Args:
            pixel_coords: (x, y) pixel coordinates
            distance_from_camera: Distance from camera to object in meters
            camera_params: Camera intrinsic parameters
            
        Returns:
            (x, y, z) world coordinates
        """
        px, py = pixel_coords
        
        # Camera is 1.5m above table, looking at center [0.5, 0.0, 0]
        camera_height = camera_params.get('height', 1.5)
        
        # Calculate object height above table (camera height - distance from camera)
        object_height = camera_height - distance_from_camera
        
        # Camera viewing area matches object spawn area:
        # X: 0.0 to 1.0 meters
        # Y: -0.5 to 0.5 meters  
        # Image is 640x480 pixels
        
        image_width = 640
        image_height = 480
        
        # Convert pixel coordinates to world coordinates
        # Pixel (0,0) = World (0.0, 0.5)  [top-left]
        # Pixel (640,0) = World (1.0, 0.5)  [top-right]  
        # Pixel (0,480) = World (0.0, -0.5)  [bottom-left]
        # Pixel (640,480) = World (1.0, -0.5)  [bottom-right]
        
        world_x = (px / image_width) * 1.0  # 0 to 1 meter
        world_y = 0.5 - (py / image_height) * 1.0  # 0.5 to -0.5 meter (inverted Y)
        world_z = object_height
        
        return world_x, world_y, world_z


def test_vlm_analyzer():
    """Test function for VLM analyzer"""
    # This is just a placeholder test
    print("VLM Analyzer module loaded successfully!")
    print("To use:")
    print("1. Get your Google AI Studio API key")
    print("2. analyzer = VLMAnalyzer(api_key='your_key')")
    print("3. objects = analyzer.analyze_scene(rgb_image)")


if __name__ == "__main__":
    test_vlm_analyzer()