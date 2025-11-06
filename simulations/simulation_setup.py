"""
Simulation Setup Module
Handles PyBullet environment initialization and camera configuration
"""

import pybullet as p
import pybullet_data
import numpy as np
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
    Load the robot and ground plane in normal orientation
    
    Returns:
        Tuple of (plane_id, robot_id)
    """
    # Load plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load robot arm in normal orientation (bins are at negative X behind robot)
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    print(f"✅ Environment loaded - Robot ID: {robot_id}")
    
    return plane_id, robot_id


def initialize_robot_gripper(robot_id: int, gripper_joints: list):
    """
    Initialize robot gripper to maximum open position
    
    Args:
        robot_id: PyBullet robot ID
        gripper_joints: List of gripper joint indices
    """
    for joint in gripper_joints:
        p.setJointMotorControl2(
            robot_id,
            joint,
            p.POSITION_CONTROL,
            targetPosition=0.04,  # Maximum open position (8cm total opening)
            force=100  # Strong force to ensure full opening
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
            image_size: (width, height) of captured images (default 640x480 for performance)
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
    
    def get_cropped_image(self, robot_id: int = None, exclude_ids: List[int] = None, 
                          exclude_regions: List[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture image and mask out robot arm and other objects using segmentation
        
        Args:
            robot_id: PyBullet robot ID to mask out (if None, no masking is applied)
            exclude_ids: List of additional object IDs to mask out (e.g., containers)
            exclude_regions: List of (x_min, x_max, y_min, y_max) regions in world coordinates to mask out
            
        Returns:
            Tuple of (RGB image with masked regions as white, linear depth in meters with masked pixels as NaN)
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
        
        # Start with ID-based mask
        combined_mask = np.zeros_like(seg, dtype=bool)
        
        if mask_ids:
            # Create mask where any of the excluded objects are
            for obj_id in mask_ids:
                combined_mask |= (seg == obj_id)
        
        # Add spatial region masking for bin areas
        if exclude_regions:
            height, width = depth_m.shape
            
            # Create coordinate grids for the image
            for row in range(height):
                for col in range(width):
                    if combined_mask[row, col]:
                        continue  # Already masked
                    
                    # Convert pixel to world coordinates using depth
                    d = depth_m[row, col]
                    if np.isnan(d):
                        continue
                    
                    # Get world position for this pixel
                    world_pos = self._pixel_to_world(col, row, d)
                    x, y = world_pos[0], world_pos[1]
                    
                    # Check if point is in any excluded region
                    for (x_min, x_max, y_min, y_max) in exclude_regions:
                        if x_min <= x <= x_max and y_min <= y <= y_max:
                            combined_mask[row, col] = True
                            break
        
        # Apply mask to depth
        depth_m = depth_m.astype(float)
        depth_m[combined_mask] = np.nan
        
        # Apply mask to RGB (set to white background)
        rgb = rgb.copy()  # Don't modify original
        rgb[combined_mask] = [255, 255, 255]

        return rgb, depth_m
    
    def _pixel_to_world(self, px: int, py: int, depth: float) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates + depth to world coordinates
        
        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate
            depth: Depth in meters
            
        Returns:
            (x, y, z) world coordinates
        """
        width, height = self.image_size
        
        # Camera position
        target = np.array(self.camera_target)
        camera_pos = target + np.array([0, 0, self.camera_height])
        
        # Normalize pixel coordinates
        nx = (2.0 * px / width) - 1.0
        ny = 1.0 - (2.0 * py / height)
        
        # View frustum size at depth
        fov_rad = np.deg2rad(self.fov)
        aspect = width / height
        view_height = 2.0 * depth * np.tan(fov_rad / 2.0)
        view_width = view_height * aspect
        
        view_x = nx * (view_width / 2.0)
        view_y = ny * (view_height / 2.0)
        
        # Camera coordinate system
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        yaw_rad = np.deg2rad(self.camera_yaw)
        right = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0])
        up = np.cross(right, forward)
        
        # World point
        world_point = camera_pos + forward * depth + right * view_x + up * view_y
        
        return (world_point[0], world_point[1], world_point[2])

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


def spawn_object(obj_type: str, position: List[float], color: List[float], 
                size: float = 0.035) -> int:
    """
    Spawn a single object in the simulation
    
    Args:
        obj_type: 'cube', 'sphere', or 'rectangle'
        position: [x, y, z] position
        color: [r, g, b, a] color values (0-1)
        size: Object size/radius (default 0.035m = 3.5cm for easier gripping)
        
    Returns:
        PyBullet object ID
    """
    if obj_type == 'triangle':
        # Create triangular prism using convex hull (better depth rendering)
        # Triangle will be visible from top-down view
        import numpy as np
        height = size * 2.0  # Height of the prism (taller for easier detection)
        base_size = size * 2.0  # Size of triangle base (much larger for visibility)
        
        # Create equilateral triangle vertices (flat on XY plane)
        # Triangle pointing UP (toward +Y axis)
        h = base_size * np.sqrt(3) / 2  # Height of equilateral triangle
        
        # Bottom triangle (Z = 0)
        vertices = [
            [0, h * 2/3, 0],              # Top vertex
            [-base_size/2, -h * 1/3, 0],  # Bottom-left vertex
            [base_size/2, -h * 1/3, 0],   # Bottom-right vertex
        ]
        
        # Top triangle (Z = height)
        vertices.extend([
            [0, h * 2/3, height],              # Top vertex
            [-base_size/2, -h * 1/3, height],  # Bottom-left vertex
            [base_size/2, -h * 1/3, height],   # Bottom-right vertex
        ])
        
        # Define triangular faces (indices into vertices array)
        # Each face is defined by 3 vertex indices (counter-clockwise winding)
        indices = [
            # Bottom face (looking from below)
            0, 2, 1,
            # Top face (looking from above)
            3, 4, 5,
            # Side faces
            0, 1, 4,  0, 4, 3,  # Side 1
            1, 2, 5,  1, 5, 4,  # Side 2
            2, 0, 3,  2, 3, 5   # Side 3
        ]
        
        # Create collision shape with proper mesh
        collision_shape = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=vertices,
            indices=indices
        )
        
        # Create visual shape with proper mesh
        visual_shape = p.createVisualShape(
            p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=color
        )
    elif obj_type == 'sphere':
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=size, rgbaColor=color)
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=size)
    elif obj_type == 'rectangle':
        # Rectangle: 2x wider in one dimension
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[size*2, size, size], rgbaColor=color)
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[size*2, size, size])
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


def spawn_debug_grid_objects():
    """
    Spawn objects randomly with proper spacing to avoid overlaps
    Accounts for bins on left (Y=-0.55) and right (Y=+0.55)
    
    Returns:
        Dictionary mapping object names to their info (id, type, position, color)
    """
    import random
    
    # Define workspace bounds accounting for bins
    # X: 0.2 to 0.6 (safe reachable area)
    # Y: -0.4 to +0.4 (avoiding bins at ±0.55)
    # Z: 0.08 (table height)
    x_min, x_max = 0.25, 0.55
    y_min, y_max = -0.35, 0.35
    z_height = 0.08
    min_spacing = 0.12  # Minimum distance between object centers (prevents overlap)
    
    # Define 5 objects with different shapes and colors
    objects_to_spawn = [
        ('sphere', [0, 0, 1, 1], 'blue'),
        ('triangle', [1, 0, 0, 1], 'red'),
        ('rectangle', [0, 1, 0, 1], 'green'),
        ('sphere', [1, 1, 0, 1], 'yellow'),
        ('triangle', [1, 0, 1, 1], 'magenta'),
    ]
    
    objects_dict = {}
    placed_positions = []
    
    print(f"🎯 Spawning test objects with random placement...")
    print(f"   Workspace area: X({x_min:.2f}-{x_max:.2f}), Y({y_min:.2f}-{y_max:.2f})")
    print(f"   Minimum spacing: {min_spacing:.2f}m")
    
    for idx, (shape, color, color_name) in enumerate(objects_to_spawn):
        # Try to find a valid position
        max_attempts = 100
        position_found = False
        
        for attempt in range(max_attempts):
            # Generate random position
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            candidate_pos = [x, y, z_height]
            
            # Check if position is far enough from all existing objects
            valid = True
            for existing_pos in placed_positions:
                distance = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if distance < min_spacing:
                    valid = False
                    break
            
            if valid:
                position_found = True
                placed_positions.append(candidate_pos)
                break
        
        if not position_found:
            print(f"   ⚠️ Could not find valid position for {color_name}_{shape} after {max_attempts} attempts")
            continue
        
        obj_name = f"{color_name}_{shape}_{idx}"
        
        # Spawn object with specified shape
        try:
            object_id = spawn_object(shape, candidate_pos, color)
            objects_dict[obj_name] = {
                'id': object_id,
                'type': shape,
                'position': candidate_pos,
                'color': color
            }
            print(f"   ✅ {obj_name} at ({candidate_pos[0]:.2f}, {candidate_pos[1]:.2f}, {candidate_pos[2]:.2f})")
        except Exception as e:
            print(f"   ⚠️ Failed to create {obj_name}: {e}")
    
    return objects_dict


def create_sorting_containers():
    """
    Create colored sorting containers/bins
    
    Returns:
        Dictionary mapping container names to their info (id, position, color)
    """
    # Containers in VERTICAL lines on left and right sides
    # Further from robot: Y=±0.55, near edge of reach envelope
    # Arranged front-to-back (close to far)
    # Shape-based sorting: spheres, triangles, rectangles, mixed
    container_positions = {
        'sphere_bin': [0.15, -0.55, 0.15],      # Left side, CLOSE (front) - for SPHERES
        'triangle_bin': [0.45, -0.55, 0.15],    # Left side, FAR (back) - for TRIANGLES
        'rectangle_bin': [0.15, 0.55, 0.15],    # Right side, CLOSE (front) - for RECTANGLES
        'mixed_bin': [0.45, 0.55, 0.15]         # Right side, FAR (back) - for UNKNOWN shapes
    }
    
    container_colors = {
        'sphere_bin': [0.2, 0.6, 1, 0.5],       # Cyan/blue for spheres
        'triangle_bin': [1, 0.3, 0, 0.5],       # Orange/red for triangles
        'rectangle_bin': [0.2, 1, 0.3, 0.5],    # Bright green for rectangles
        'mixed_bin': [1, 1, 0.2, 0.5]           # Yellow for mixed/unknown
    }
    
    containers_dict = {}
    
    print("📦 Adding sorting containers...")
    for name, pos in container_positions.items():
        # Create OPEN-TOP bin with 4 walls (no top, no bottom collision)
        # Each bin is 20cm x 20cm with 30cm tall walls
        wall_thickness = 0.01  # 1cm thick walls
        wall_height = 0.15  # 15cm half-height (30cm total)
        bin_size = 0.1  # 10cm half-size (20cm total)
        
        color = container_colors[name]
        container_ids = []
        
        # Create 4 walls: front, back, left, right
        walls = [
            # [relative_x, relative_y, half_x, half_y]
            [bin_size, 0, wall_thickness, bin_size],  # Right wall
            [-bin_size, 0, wall_thickness, bin_size], # Left wall
            [0, bin_size, bin_size, wall_thickness],  # Back wall
            [0, -bin_size, bin_size, wall_thickness], # Front wall
        ]
        
        for wall_offset in walls:
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[wall_offset[2], wall_offset[3], wall_height],
                rgbaColor=color
            )
            wall_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[wall_offset[2], wall_offset[3], wall_height]
            )
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=[pos[0] + wall_offset[0], pos[1] + wall_offset[1], pos[2]]
            )
            container_ids.append(wall_id)
        
        containers_dict[name] = {
            'id': container_ids,  # List of wall IDs
            'position': pos,
            'color': color[:3]
        }
        print(f"   ✅ {name} at {pos} (open-top bin)")
    
    return containers_dict


if __name__ == "__main__":
    import time
    
    # Initialize and run basic simulation
    physics_client = initialize_physics(gui_mode=True)
    plane_id, robot_id = load_environment()
    initialize_robot_gripper(robot_id, [9, 10])
    camera = OverheadCamera()
    containers = create_sorting_containers()
    objects = spawn_debug_grid_objects()  # Use debug grid instead
    
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
