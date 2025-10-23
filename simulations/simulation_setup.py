"""
Simulation Setup Module
Handles PyBullet environment initialization and camera configuration
"""

import pybullet as p
import pybullet_data
import numpy as np
import random
import cv2
from typing import Tuple, List


def initialize_physics(gui_mode: bool = True) -> int:
    """
    Initialize PyBullet physics engine
    
    Args:
        gui_mode: Whether to run with GUI or headless
        
    Returns:
        Physics client ID
    """
    connection_mode = p.GUI if gui_mode else p.DIRECT
    physics_client = p.connect(connection_mode)
    
    # Set up physics environment
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    return physics_client


def load_environment() -> Tuple[int, int]:
    """
    Load the robot and ground plane
    
    Returns:
        Tuple of (plane_id, robot_id)
    """
    # Load plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load robot arm
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    print(f"✅ Environment loaded - Robot ID: {robot_id}")
    
    return plane_id, robot_id


def initialize_robot_gripper(robot_id: int, gripper_joints: list):
    """
    Initialize robot gripper to open position
    
    Args:
        robot_id: PyBullet robot ID
        gripper_joints: List of gripper joint indices
    """
    for joint in gripper_joints:
        p.setJointMotorControl2(
            robot_id,
            joint,
            p.POSITION_CONTROL,
            targetPosition=0.04,  # Open position
            force=50
        )


class OverheadCamera:
    """Camera system for capturing overhead workspace images"""
    
    def __init__(self, camera_height: float = 1.5, fov: float = 60, 
                 image_size: Tuple[int, int] = (640, 480)):
        """
        Initialize overhead camera
        
        Args:
            camera_height: Height above workspace
            fov: Field of view in degrees
            image_size: (width, height) of captured images
        """
        self.camera_height = camera_height
        self.fov = fov
        self.image_size = image_size
        
        # Camera parameters - moved to the right and rotated to align with robot
        self.camera_target = [0.5, 0.0, 0]  # Center of workspace
        self.camera_yaw = 270  # Robot at bottom of screen
        self.camera_pitch = -90  # Looking straight down
        
        # Compute matrices
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            self.camera_target, self.camera_height, 
            self.camera_yaw, self.camera_pitch, 0, 2)
        
        # Store near/far for depth conversion later
        self.near = 0.01  # 1cm - close plane
        self.far = 3.0

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.image_size[0] / self.image_size[1],
            nearVal=self.near,
            farVal=self.far
        )
    
    def capture_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture image from overhead camera
        
        Returns:
            Tuple of (rgb_array, depth_array, segmentation_array)
        """
        width, height = self.image_size
        
        # Get camera image
        _, _, rgb_img, depth_img, seg_img = p.getCameraImage(
            width, height, self.view_matrix, self.projection_matrix)
        
        # Convert to numpy arrays
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]  # Remove alpha
        depth_array = np.array(depth_img).reshape(height, width)
        seg_array = np.array(seg_img).reshape(height, width)
        
        return rgb_array, depth_array, seg_array
    
    def get_cropped_image(self, robot_id: int = None, exclude_ids: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture image and mask out robot arm and other objects using segmentation
        
        Args:
            robot_id: PyBullet robot ID to mask out (if None, no masking is applied)
            exclude_ids: List of additional object IDs to mask out (e.g., containers)
            
        Returns:
            Tuple of (RGB image, linear depth in meters with masked pixels as NaN)
        """
        rgb, depth, seg = self.capture_image()
        # Convert depth buffer to linear depth (meters)
        depth_m = self._depth_buffer_to_distance(depth)

        # Combine all IDs to mask
        mask_ids = []
        if robot_id is not None:
            mask_ids.append(robot_id)
        if exclude_ids is not None:
            mask_ids.extend(exclude_ids)
        
        if mask_ids:
            # Create mask where any of the excluded objects are
            combined_mask = np.zeros_like(seg, dtype=bool)
            for obj_id in mask_ids:
                combined_mask |= (seg == obj_id)

            # Mask out pixels in depth by setting them to NaN so they won't be counted
            depth_m = depth_m.astype(float)
            depth_m[combined_mask] = np.nan

        return rgb, depth_m

    def _depth_buffer_to_distance(self, depth_buffer: np.ndarray) -> np.ndarray:
        """
        Convert PyBullet depth buffer values to linear distances (meters).

        PyBullet returns a non-linear depth buffer in the range [0, 1]. The
        conversion to real-world depth z (meters) for a perspective projection
        can be done with the formula:

            z = (near * far) / (far - (far - near) * depth_buffer)

        Args:
            depth_buffer: 2D numpy array from getCameraImage()

        Returns:
            2D numpy array of distances in meters (same shape as depth_buffer)
        """
        # Protect against divide-by-zero
        near = float(self.near)
        far = float(self.far)
        db = depth_buffer.astype(float)
        # Formula derived from projection math used by PyBullet/OpenGL
        z = (near * far) / (far - (far - near) * db)
        return z


def generate_random_position(min_radius: float = 0.2, max_radius: float = 0.5, 
                            height_range: Tuple[float, float] = (0.05, 0.15)) -> List[float]:
    """
    Generate a random position within robot arm reach AND camera field of view
    
    Args:
        min_radius: Minimum distance from robot base (meters)
        max_radius: Maximum distance from robot base (meters)  
        height_range: (min_height, max_height) above table (meters)
        
    Returns:
        [x, y, z] position coordinates
    """
    # Camera is positioned at [0.5, 0.0, 1.5] looking down at [0.5, 0.0, 0]
    # With 60° FOV, the viewing area is roughly a rectangle around the target
    
    # Define camera viewing area (approximate bounds for 60° FOV at 1.5m height)
    camera_fov_bounds = {
        'x_min': 0.0,   # Left edge of camera view
        'x_max': 1.0,   # Right edge of camera view  
        'y_min': -0.5,  # Bottom edge of camera view
        'y_max': 0.5    # Top edge of camera view
    }
    
    # Generate position within camera bounds AND robot reach
    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts:
        # Generate random position within camera bounds
        x = random.uniform(camera_fov_bounds['x_min'], camera_fov_bounds['x_max'])
        y = random.uniform(camera_fov_bounds['y_min'], camera_fov_bounds['y_max'])
        z = random.uniform(height_range[0], height_range[1])
        
        # Check if position is within robot reach
        distance_from_robot = (x**2 + y**2)**0.5
        
        if min_radius <= distance_from_robot <= max_radius:
            return [x, y, z]
        
        attempts += 1
    
    # Fallback: generate a safe position if we can't find one in bounds
    print(f"⚠️ Could not find position in camera view after {max_attempts} attempts, using fallback")
    angle = random.uniform(0, 2 * np.pi)
    radius = random.uniform(min_radius, max_radius)
    x = radius * np.cos(angle) + 0.5  # Offset to center of camera view
    y = radius * np.sin(angle)
    z = random.uniform(height_range[0], height_range[1])
    
    # Clamp to camera bounds
    x = max(camera_fov_bounds['x_min'], min(camera_fov_bounds['x_max'], x))
    y = max(camera_fov_bounds['y_min'], min(camera_fov_bounds['y_max'], y))
    
    return [x, y, z]


def spawn_object(obj_type: str, position: List[float], color: List[float], 
                size: float = 0.05) -> int:
    """
    Spawn a single object in the simulation
    
    Args:
        obj_type: 'cube', 'sphere', or 'cylinder'
        position: [x, y, z] position
        color: [r, g, b, a] color values (0-1)
        size: Object size/radius
        
    Returns:
        PyBullet object ID
    """
    if obj_type == 'cube':
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=color)
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[size, size, size])
    elif obj_type == 'sphere':
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=size, rgbaColor=color)
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=size)
    elif obj_type == 'cylinder':
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=size, length=size*2, rgbaColor=color)
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=size, height=size*2)
    else:
        raise ValueError(f"Unsupported object type: {obj_type}")
    
    object_id = p.createMultiBody(
        baseMass=0.5,  # Increased from 0.1 to 0.5kg for better stability
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )
    
    # Add friction and restitution for better gripping
    p.changeDynamics(
        object_id, 
        -1,  # -1 means base link
        lateralFriction=1.0,
        spinningFriction=0.1,
        rollingFriction=0.01,
        restitution=0.1
    )
    
    return object_id


def spawn_random_objects(num_objects: int = 3):
    """
    Spawn multiple random objects within camera view
    
    Args:
        num_objects: Number of objects to create
        
    Returns:
        Dictionary mapping object names to their info (id, type, position, color)
    """
    object_types = ['cube', 'sphere', 'cylinder']
    colors = [
        [1, 0, 0, 1],    # Red
        [0, 1, 0, 1],    # Green  
        [0, 0, 1, 1],    # Blue
        [1, 1, 0, 1],    # Yellow
        [1, 0, 1, 1],    # Magenta
        [0, 1, 1, 1],    # Cyan
        [1, 0.5, 0, 1],  # Orange
        [0.5, 0, 1, 1],  # Purple
    ]
    color_names = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
    
    objects_dict = {}
    
    print(f"📷 Generating {num_objects} objects within camera field of view...")
    print(f"   Camera viewing area: X(0.0-1.0), Y(-0.5-0.5), Z(0.05-0.15)")
    
    for i in range(num_objects):
        # Random object type and color
        obj_type = random.choice(object_types)
        color_idx = random.randint(0, len(colors) - 1)
        color = colors[color_idx]
        color_name = color_names[color_idx]
        
        # Generate random position within camera view
        position = generate_random_position()
        
        # Create unique name
        obj_name = f"{color_name}_{obj_type}_{i+1}"
        
        # Spawn object
        try:
            object_id = spawn_object(obj_type, position, color)
            objects_dict[obj_name] = {
                'id': object_id,
                'type': obj_type,
                'position': position,
                'color': color
            }
            print(f"   ✅ {obj_name} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        except Exception as e:
            print(f"   ⚠️ Failed to create {obj_name}: {e}")
    
    return objects_dict


def create_sorting_containers():
    """
    Create colored sorting containers/bins
    
    Returns:
        Dictionary mapping container names to their info (id, position, color)
    """
    # Containers in a straight line along Y-axis with spacing
    container_positions = {
        'red_bin': [-0.5, -0.45, 0.15],
        'blue_bin': [-0.5, -0.15, 0.15],
        'green_bin': [-0.5, 0.15, 0.15],
        'yellow_bin': [-0.5, 0.45, 0.15]
    }
    
    container_colors = {
        'red_bin': [1, 0, 0, 0.5],
        'blue_bin': [0, 0, 1, 0.5],
        'green_bin': [0, 1, 0, 0.5],
        'yellow_bin': [1, 1, 0, 0.5]
    }
    
    containers_dict = {}
    
    print("📦 Adding sorting containers...")
    for name, pos in container_positions.items():
        # Create semi-transparent box as container (taller now)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.1, 0.1, 0.15],  # Made taller (0.15 height instead of 0.05)
            rgbaColor=container_colors[name]
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.15]
        )
        
        container_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos
        )
        
        containers_dict[name] = {
            'id': container_id,
            'position': pos,
            'color': container_colors[name][:3]
        }
        print(f"   ✅ {name} at {pos}")
    
    return containers_dict


if __name__ == "__main__":
    import time
    
    # Initialize and run basic simulation
    physics_client = initialize_physics(gui_mode=True)
    plane_id, robot_id = load_environment()
    initialize_robot_gripper(robot_id, [9, 10])
    camera = OverheadCamera()
    containers = create_sorting_containers()
    objects = spawn_random_objects(num_objects=4)
    
    # Settle physics
    for _ in range(240):
        p.stepSimulation()
    
    # Run simulation
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()
