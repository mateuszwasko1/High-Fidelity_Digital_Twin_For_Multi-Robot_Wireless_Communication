"""
Object Finder Module
Manages detected objects in the simulation workspace
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple


class Object:
    """
    Represents a detected object with its metadata
    """
    
    def __init__(self, color: str, shape: str, location: Tuple[float, float, float], 
                 name: Optional[str] = None, reachable: bool = True, image: Optional[np.ndarray] = None):
        """
        Initialize an object
        
        Args:
            color: Object color (e.g., "red", "blue", "green")
            shape: Object shape (e.g., "cube", "sphere", "cylinder")
            location: 3D coordinates (x, y, z) in world space
            name: Optional descriptive name
            reachable: Whether the object is reachable by the robot
            image: RGB image of the object (numpy array)
        """
        self.color = color
        self.shape = shape
        self.location = location  # (x, y, z)
        self.name = name if name else f"{color} {shape}"
        self.reachable = reachable
        self.image = image  # Store RGB image for VLM analysis later
    
    def __repr__(self) -> str:
        """String representation of the object"""
        x, y, z = self.location
        has_image = "with image" if self.image is not None else "no image"
        return f"Object(name='{self.name}', color='{self.color}', shape='{self.shape}', location=({x:.3f}, {y:.3f}, {z:.3f}), reachable={self.reachable}, {has_image})"


def find_closest_object(camera, robot_id: int, camera_height: float = 1.5, 
                        exclude_ids: List[int] = None, debug: bool = False) -> Optional[Object]:
    """
    Find the closest object to the robot using depth clustering
    
    Args:
        camera: OverheadCamera instance
        robot_id: PyBullet robot ID to mask out
        camera_height: Height of camera above ground (meters)
        exclude_ids: List of object IDs to exclude (e.g., containers)
        debug: If True, show debug visualizations
        
    Returns:
        Closest Object instance, or None if no objects detected
    """
    # Get masked RGB and depth images
    rgb, depth = camera.get_cropped_image(robot_id=robot_id, exclude_ids=exclude_ids)
    
    if debug:
        print(f"\n📊 Debug Info:")
        print(f"   Depth range: {np.nanmin(depth):.3f}m to {np.nanmax(depth):.3f}m")
        print(f"   Camera height: {camera_height}m")
        print(f"   Non-NaN pixels: {(~np.isnan(depth)).sum()}")
    
    # Filter out floor and invalid pixels
    # Camera is at 1.5m, table is at 0m, objects are ~0.05-0.15m above table
    # So objects should be at distance ~1.35-1.45m from camera
    floor_distance = camera_height  # ~1.5m
    object_height_range = (0.05, 0.20)  # Objects are 5-20cm above table
    
    # Create mask for object pixels (closer than floor, not NaN)
    valid_mask = ~np.isnan(depth)
    object_mask = valid_mask & (depth < floor_distance - object_height_range[0])
    
    if debug:
        print(f"   Object pixels: {object_mask.sum()}")
        cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # Visualize depth (normalize for display)
        depth_vis = depth.copy()
        depth_vis[np.isnan(depth_vis)] = depth_vis[~np.isnan(depth_vis)].max()
        depth_vis = ((depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min()) * 255).astype(np.uint8)
        cv2.imshow("Depth", depth_vis)
        
        cv2.imshow("Object Mask", (object_mask * 255).astype(np.uint8))
        cv2.waitKey(1000)  # Show for 1 second
    
    # Convert boolean mask to uint8 for OpenCV
    object_mask_uint8 = (object_mask * 255).astype(np.uint8)
    
    # Clean up noise with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    object_mask_uint8 = cv2.morphologyEx(object_mask_uint8, cv2.MORPH_CLOSE, kernel)
    object_mask_uint8 = cv2.morphologyEx(object_mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Find connected components (separate objects)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask_uint8, connectivity=8)
    
    closest_object = None
    min_distance = float('inf')
    
    # Robot is at origin (0, 0)
    robot_pos = (0.0, 0.0)
    
    # Skip label 0 (background)
    for label_id in range(1, num_labels):
        # Get stats for this object
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        # Filter out tiny noise (minimum 50 pixels)
        if area < 50:
            continue
        
        # Get centroid in image coordinates
        cx, cy = centroids[label_id]
        cx, cy = int(cx), int(cy)
        
        # Get average depth at this object
        object_pixels = (labels == label_id)
        avg_depth = np.nanmean(depth[object_pixels])
        
        # Convert image coordinates to world coordinates
        world_pos = _image_to_world(cx, cy, avg_depth, camera)
        
        # Calculate distance to robot (only X-Y, ignore Z)
        distance = np.sqrt((world_pos[0] - robot_pos[0])**2 + (world_pos[1] - robot_pos[1])**2)
        
        # Check if this is the closest object
        if distance < min_distance:
            min_distance = distance
            
            # Extract cropped image of the object
            object_image = _extract_object_image(rgb, labels, label_id)
            
            # Create Object instance (color and shape will be set by VLM)
            closest_object = Object(
                color="unknown",
                shape="unknown",
                location=world_pos,
                name="detected_object",
                reachable=True,
                image=object_image
            )
    
    if debug and closest_object:
        print(f"   Closest object: {closest_object.name} at distance {min_distance:.3f}m")
        # Show the cropped object image
        if closest_object.image is not None:
            cv2.imshow("Closest Object Image", cv2.cvtColor(closest_object.image, cv2.COLOR_RGB2BGR))
            print(f"   Image shape: {closest_object.image.shape}")
            cv2.waitKey(2000)  # Show for 2 seconds
    
    return closest_object


def _image_to_world(px: int, py: int, depth: float, camera) -> Tuple[float, float, float]:
    """
    Convert pixel coordinates + depth to world coordinates
    
    Args:
        px: Pixel x coordinate
        py: Pixel y coordinate  
        depth: Depth in meters
        camera: OverheadCamera instance
        
    Returns:
        (x, y, z) world coordinates
    """
    # Get image dimensions
    width, height = camera.image_size
    
    # Normalize pixel coordinates to [-1, 1]
    # OpenCV/PyBullet: origin at top-left
    nx = (px / width) * 2 - 1
    ny = (py / height) * 2 - 1
    
    # Compute FOV angle in radians
    fov_rad = np.deg2rad(camera.fov)
    
    # At the given depth, compute the size of the viewing plane
    aspect = width / height
    view_height = 2 * depth * np.tan(fov_rad / 2)
    view_width = view_height * aspect
    
    # Convert normalized coords to view-plane offsets  
    # After testing: the correct mapping for yaw=270, pitch=-90 camera
    dx = -nx * (view_width / 2)   # Image right/left → World forward/back (inverted)
    dy = -ny * (view_height / 2)  # Image up/down → World left/right (inverted)  
    
    # Camera target is the world position we're looking at
    target_x, target_y, target_z = camera.camera_target
    
    # Apply offsets to camera target position
    world_x = target_x + dx
    world_y = target_y + dy
    
    # Calibration offsets (empirically determined)
    # Fine-tuned to match actual object positions
    world_x += 0.05  # Move right by 5cm
    world_y += 0.02  # Move up by 2cm (reduced from 5cm)
    
    # Use fixed Z height instead of depth-based calculation for consistency
    # Objects spawn between 0.05-0.15m, so use middle value
    world_z = 0.10  # Fixed height at 10cm above ground
    
    print(f"   DEBUG TRANSFORM: pixel=({px},{py}), norm=({nx:.2f},{ny:.2f}), offset=({dx:.2f},{dy:.2f}), world=({world_x:.2f},{world_y:.2f},{world_z:.2f})")
    
    return (world_x, world_y, world_z)


def _extract_object_image(rgb: np.ndarray, labels: np.ndarray, label_id: int, 
                         padding: int = 10) -> np.ndarray:
    """
    Extract a cropped RGB image of a specific object with background removed
    
    Args:
        rgb: Full RGB image array
        labels: Label array from connected components
        label_id: ID of the object to extract
        padding: Extra pixels to include around the object
        
    Returns:
        Cropped RGB image of the object with white background
    """
    # Create mask for this specific object
    object_mask = (labels == label_id)
    
    # Find bounding box
    rows, cols = np.where(object_mask)
    
    if len(rows) == 0 or len(cols) == 0:
        # Return empty image if no pixels found
        return np.zeros((1, 1, 3), dtype=np.uint8)
    
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Add padding
    height, width = rgb.shape[:2]
    min_row = max(0, min_row - padding)
    max_row = min(height - 1, max_row + padding)
    min_col = max(0, min_col - padding)
    max_col = min(width - 1, max_col + padding)
    
    # Crop the image and mask
    cropped = rgb[min_row:max_row+1, min_col:max_col+1].copy()
    cropped_mask = object_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Set background pixels to white
    cropped[~cropped_mask] = [255, 255, 255]
    
    # Resize to 224x224 for better VLM performance
    target_size = 224
    cropped_resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Display the resized cropped image
    cv2.imshow("Detected Object (Cropped)", cv2.cvtColor(cropped_resized, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # Allow window to update
    
    return cropped_resized

