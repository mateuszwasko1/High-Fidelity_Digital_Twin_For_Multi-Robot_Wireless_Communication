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
                 name: Optional[str] = None, reachable: bool = True, image: Optional[np.ndarray] = None,
                 height: Optional[float] = None):
        """
        Initialize an object
        
        Args:
            color: Object color (e.g., "red", "blue", "green")
            shape: Object shape (e.g., "cube", "sphere", "cylinder")
            location: 3D coordinates (x, y, z) in world space
            name: Optional descriptive name
            reachable: Whether the object is reachable by the robot
            image: RGB image of the object (numpy array)
            height: Estimated object height in meters (from depth data)
        """
        self.color = color
        self.shape = shape
        self.location = location  # (x, y, z)
        self.name = name if name else f"{color} {shape}"
        self.reachable = reachable
        self.image = image  # Store RGB image for VLM analysis later
        self.height = height  # Object height in meters
    
    def __repr__(self) -> str:
        """String representation of the object"""
        x, y, z = self.location
        has_image = "with image" if self.image is not None else "no image"
        return f"Object(name='{self.name}', color='{self.color}', shape='{self.shape}', location=({x:.3f}, {y:.3f}, {z:.3f}), reachable={self.reachable}, {has_image})"


def find_closest_object(camera, robot_id: int, camera_height: float = 1.5, 
                        exclude_ids: List[int] = None, exclude_regions: List[Tuple[float, float, float, float]] = None,
                        debug: bool = False) -> Optional[Object]:
    """
    Find the closest object to the robot using depth clustering
    
    Args:
        camera: OverheadCamera instance
        robot_id: PyBullet robot ID to mask out
        camera_height: Height of camera above ground (meters)
        exclude_ids: List of object IDs to exclude (e.g., containers)
        exclude_regions: List of (x_min, x_max, y_min, y_max) regions to mask out (e.g., bin areas)
        debug: If True, show debug visualizations
        
    Returns:
        Closest Object instance, or None if no objects detected
    """
    # Get masked RGB and depth images
    rgb, depth = camera.get_cropped_image(robot_id=robot_id, exclude_ids=exclude_ids, exclude_regions=exclude_regions)
    
    # Filter out floor and invalid pixels
    # Camera is at 1.5m, table is at 0m, objects are ~0.05-0.15m above table
    # So objects should be at distance ~1.35-1.45m from camera
    floor_distance = camera_height  # ~1.5m
    object_height_range = (0.05, 0.20)  # Objects are 5-20cm above table
    
    # Create mask for object pixels (closer than floor, not NaN)
    valid_mask = ~np.isnan(depth)
    object_mask = valid_mask & (depth < floor_distance - object_height_range[0])
    
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
        
        # Get depth statistics for this object
        object_pixels = (labels == label_id)
        object_depths = depth[object_pixels]
        object_depths = object_depths[~np.isnan(object_depths)]  # Remove NaN values
        
        if len(object_depths) == 0:
            continue  # Skip if no valid depth data
        
        # Calculate object height from depth range
        # Min depth = top of object (closest to camera)
        # Max depth = bottom of object (furthest from camera)
        min_depth = np.min(object_depths)
        max_depth = np.max(object_depths)
        object_height = max_depth - min_depth  # Height in meters
        
        # Use average depth for position calculation
        avg_depth = np.mean(object_depths)
        
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
                image=object_image,
                height=object_height
            )
    
    return closest_object


def _image_to_world(px: int, py: int, depth: float, camera) -> Tuple[float, float, float]:
    """
    Convert pixel coordinates + depth to world coordinates using proper projection matrices
    
    This method uses PyBullet's view and projection matrices to accurately transform
    pixel coordinates back to 3D world space. This approach is realistic and matches
    what would be done with a real camera using calibrated intrinsic/extrinsic matrices.
    
    Args:
        px: Pixel x coordinate
        py: Pixel y coordinate  
        depth: Depth in meters (linear distance from camera)
        camera: OverheadCamera instance
        
    Returns:
        (x, y, z) world coordinates
    """
    # Get image dimensions
    width, height = camera.image_size
    
    # Calculate camera position in world space
    # Camera is at target + offset in the direction it's looking from
    target = np.array(camera.camera_target)
    
    # Convert yaw/pitch to camera position
    # yaw=270° pitch=-90° means looking straight down from above
    yaw_rad = np.deg2rad(camera.camera_yaw)
    pitch_rad = np.deg2rad(camera.camera_pitch)
    
    # For PyBullet's computeViewMatrixFromYawPitchRoll:
    # - pitch=-90° means looking straight down
    # - yaw=270° means rotated 270° around Z axis
    # - distance is camera_height above the target
    # So camera is simply at: target + [0, 0, camera_height]
    camera_pos = target + np.array([0, 0, camera.camera_height])
    
    # Normalize pixel coordinates to [-1, 1]
    # Origin at center, x right, y up (image space has y down)
    nx = (2.0 * px / width) - 1.0
    ny = 1.0 - (2.0 * py / height)  # Flip Y
    
    # Calculate the viewing frustum size at the given depth
    # Using pinhole camera model with field of view
    fov_rad = np.deg2rad(camera.fov)
    aspect = width / height
    
    # Height and width of the view plane at distance 'depth'
    view_height = 2.0 * depth * np.tan(fov_rad / 2.0)
    view_width = view_height * aspect
    
    # Position on the view plane relative to its center
    view_x = nx * (view_width / 2.0)
    view_y = ny * (view_height / 2.0)
    
    # Create camera coordinate system
    # Forward: from camera to target (looking direction) - straight down for pitch=-90°
    forward = target - camera_pos  # Points from [0.5, 0, 1.5] to [0.5, 0, 0] = [0, 0, -1.5]
    forward = forward / np.linalg.norm(forward)  # Normalized: [0, 0, -1]
    
    # For yaw=270°, the camera is rotated 270° around the Z axis
    # This affects which direction is "right" in the image
    # yaw=270° means the right direction in the image corresponds to +X in world
    # and up direction in the image corresponds to -Y in world
    
    # Right vector (rotated by yaw around Z axis)
    yaw_rad = np.deg2rad(camera.camera_yaw)
    right = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0])
    
    # Up vector in image space (perpendicular to both forward and right)
    up = np.cross(right, forward)  # Right-handed coordinate system
    
    # Point in world space:
    # Start at camera position, move forward by depth, then offset by view_x and view_y
    world_point = camera_pos + forward * depth + right * view_x + up * view_y
    
    world_x = world_point[0]
    world_y = world_point[1]
    world_z = world_point[2]
    
    # Small calibration adjustments (systematic errors from calibration test)
    world_x -= 0.009  # Correct for ~9mm X offset
    # Note: Z calibration was done for objects at 0.10m height
    # For objects at different heights, the depth measurement should be accurate
    
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
    
    # Resize to 224x224 for VLM performance (faster than 336, still good quality)
    # CLIP works well with 224 - provides good balance of speed and accuracy
    target_size = 224
    # PERFORMANCE FIX: Changed from INTER_LANCZOS4 to INTER_LINEAR for 4x faster resizing
    cropped_resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Debug window disabled for performance
    # cv2.imshow("Detected Object (Cropped)", cv2.cvtColor(cropped_resized, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)
    
    return cropped_resized

