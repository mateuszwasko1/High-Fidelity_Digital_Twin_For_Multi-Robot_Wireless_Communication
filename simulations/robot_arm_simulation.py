"""
Object-Oriented PyBullet Robot Arm Simulation with VLM Integration
This module provides a clean, modular robot simulation with vision language model analysis.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import random
from typing import List, Dict, Tuple, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use regular os.getenv()

# Try to import VLM analyzer (optional)
try:
    # Try relative import first
    from .vlm_analysis import VLMAnalyzer
    VLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        # Try absolute import from simulations folder
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from vlm_analysis import VLMAnalyzer
        VLM_AVAILABLE = True
    except ImportError:
        print("⚠️ VLM analysis not available - missing dependencies")
        VLMAnalyzer = None
        VLM_AVAILABLE = False


class RobotArmSimulation:
    """
    Main simulation class for robot arm with VLM-based object detection
    """
    
    def __init__(self, gui_mode: bool = True, api_key: Optional[str] = None):
        """
        Initialize the robot arm simulation
        
        Args:
            gui_mode: Whether to run with GUI or headless
            api_key: Google AI Studio API key for VLM analysis
        """
        self.gui_mode = gui_mode
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self.objects = {}
        self.containers = {}  # Sorting bins
        self.camera = None
        self.vlm_analyzer = None
        
        # Robot control parameters
        self.num_joints = 7  # Franka Panda has 7 arm joints
        self.end_effector_link = 11  # Franka Panda end-effector
        self.gripper_joints = [9, 10]  # Franka Panda gripper finger joints
        
        # Initialize VLM if API key provided and VLM is available
        if api_key and VLM_AVAILABLE and VLMAnalyzer:
            try:
                self.vlm_analyzer = VLMAnalyzer(api_key)
                print("✅ VLM Analyzer initialized successfully")
                
                # Warm up the VLM with a dummy prediction to speed up future ones
                print("🔥 Warming up VLM (first prediction is always slower)...")
                self._warmup_vlm()
                print("✅ VLM warm-up completed - future predictions will be faster")
                
            except Exception as e:
                print(f"⚠️ VLM initialization failed: {e}")
        elif not VLM_AVAILABLE:
            print("⚠️ VLM not available - install dependencies with:")
            print("   conda env update -n bullet39 --file environment.yml --prune")
        
        self._setup_simulation()
    
    def _warmup_vlm(self):
        """Warm up VLM with a simple dummy prediction to speed up future ones"""
        if not self.vlm_analyzer:
            return
            
        try:
            # Create a simple 1x1 white image
            dummy_image = np.ones((1, 1, 3), dtype=np.uint8) * 255  # White pixel
            
            # Create a simple warm-up prompt
            warmup_prompt = """
            This is a warm-up test. Please respond with a simple JSON:
            {
                "objects": [
                    {
                        "name": "test",
                        "color": "white",
                        "shape": "pixel",
                        "center_x": 0,
                        "center_y": 0,
                        "confidence": "high"
                    }
                ]
            }
            """
            
            # Send warm-up request (this will be slow but subsequent ones will be fast)
            import time
            start_time = time.time()
            
            # Use a simpler method for warm-up to avoid complex processing
            try:
                from PIL import Image
                pil_image = Image.fromarray(dummy_image.astype('uint8'))
                response = self.vlm_analyzer.model.generate_content([warmup_prompt, pil_image])
                
                warmup_time = time.time() - start_time
                print(f"   ⏱️ Warm-up took {warmup_time:.1f} seconds")
                
            except Exception as e:
                print(f"   ⚠️ Warm-up had issues but VLM should still work: {e}")
                
        except Exception as e:
            print(f"   ⚠️ Could not warm up VLM: {e}")
    
    def _setup_simulation(self):
        """Initialize PyBullet simulation environment"""
        # Connect to physics server
        connection_mode = p.GUI if self.gui_mode else p.DIRECT
        self.physics_client = p.connect(connection_mode)
        
        # Set up physics environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Load environment
        self._load_environment()
        self._setup_camera()
    
    def _load_environment(self):
        """Load the robot, plane, and objects"""
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot arm
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        
        print(f"✅ Environment loaded - Robot ID: {self.robot_id}")
    
    def _setup_camera(self):
        """Initialize camera systems"""
        # Overhead camera for scene survey
        self.camera = OverheadCamera()
        print("✅ Overhead camera initialized")
    
    def add_sorting_containers(self):
        """Add labeled containers for sorting objects"""
        container_positions = {
            'red_bin': [-0.3, 0.5, 0.1],
            'blue_bin': [-0.3, -0.5, 0.1],
            'green_bin': [-0.6, 0.5, 0.1],
            'yellow_bin': [-0.6, -0.5, 0.1]
        }
        
        container_colors = {
            'red_bin': [1, 0, 0, 0.5],
            'blue_bin': [0, 0, 1, 0.5],
            'green_bin': [0, 1, 0, 0.5],
            'yellow_bin': [1, 1, 0, 0.5]
        }
        
        print("📦 Adding sorting containers...")
        for name, pos in container_positions.items():
            # Create semi-transparent box as container
            visual_shape = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=[0.15, 0.15, 0.05],
                rgbaColor=container_colors[name]
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.15, 0.15, 0.05]
            )
            
            container_id = p.createMultiBody(
                baseMass=0,  # Static
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos
            )
            
            self.containers[name] = {
                'id': container_id,
                'position': pos,
                'color': container_colors[name][:3]
            }
            print(f"   ✅ {name} at {pos}")
        
        return list(self.containers.keys())
    
    def generate_random_position(self, min_radius: float = 0.2, max_radius: float = 0.5, 
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
    
    def add_random_objects(self, num_objects: int = 3) -> List[str]:
        """
        Add multiple objects at random positions within camera view
        
        Args:
            num_objects: Number of objects to create
            
        Returns:
            List of object names created
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
        
        created_objects = []
        
        print(f"📷 Generating {num_objects} objects within camera field of view...")
        print(f"   Camera viewing area: X(0.0-1.0), Y(-0.5-0.5), Z(0.05-0.15)")
        
        for i in range(num_objects):
            # Random object type and color
            obj_type = random.choice(object_types)
            color_idx = random.randint(0, len(colors) - 1)
            color = colors[color_idx]
            color_name = color_names[color_idx]
            
            # Generate random position within camera view
            position = self.generate_random_position()
            
            # Create unique name
            obj_name = f"{color_name}_{obj_type}_{i+1}"
            
            # Add object
            try:
                self.add_object(obj_name, obj_type, position, color)
                created_objects.append(obj_name)
                print(f"   ✅ {obj_name} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
            except Exception as e:
                print(f"   ⚠️ Failed to create {obj_name}: {e}")
        
        return created_objects
    
    def add_object(self, name: str, obj_type: str, position: List[float], 
                   color: List[float], size: float = 0.05) -> int:
        """
        Add an object to the simulation
        
        Args:
            name: Object identifier
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
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.objects[name] = {
            'id': object_id,
            'type': obj_type,
            'position': position,
            'color': color
        }
        
        print(f"✅ Added {obj_type} '{name}' at {position}")
        return object_id
    
    def capture_scene(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture current scene from overhead camera
        
        Returns:
            Tuple of (rgb_image, depth_image, segmentation_image)
        """
        return self.camera.capture_image()
    

    

    

    
    def determine_target_container(self, classification: Dict) -> str:
        """
        Decide which container to use based on object classification
        
        Args:
            classification: Object classification result
            
        Returns:
            Container name
        """
        color = classification.get('color', 'unknown').lower()
        
        # Simple color-based sorting
        if 'red' in color:
            return 'red_bin'
        elif 'blue' in color:
            return 'blue_bin'
        elif 'green' in color:
            return 'green_bin'
        elif 'yellow' in color:
            return 'yellow_bin'
        else:
            # Default to red bin for unknown colors
            return 'red_bin'
    
    def move_to_position(self, target_pos: List[float], duration: float = 2.0):
        """
        Move end-effector to target position using inverse kinematics
        
        Args:
            target_pos: [x, y, z] target position
            duration: Time to complete motion (seconds)
        """
        print(f"🤖 Moving to position ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})...")
        
        # Compute inverse kinematics
        target_orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # Gripper pointing down
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link,
            target_pos,
            target_orientation,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        
        # Set joint positions for arm joints only (first 7 joints)
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
                maxVelocity=1.0
            )
        
        # Simulate motion with physics
        steps = int(duration * 240)  # 240 Hz physics
        for step in range(steps):
            self.step_simulation()
            time.sleep(1./240.)
        
        # Get final position
        link_state = p.getLinkState(self.robot_id, self.end_effector_link)
        final_pos = link_state[0]
        error = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
        print(f"✅ Reached position (error: {error*100:.1f}cm)")
    
    def grasp_object(self):
        """Close gripper to grasp object"""
        print("✋ Closing gripper to grasp object...")
        
        # Close gripper fingers
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=0.0,  # Closed position
                force=50
            )
        
        # Simulate gripper closing
        for _ in range(60):  # 0.25 seconds at 240 Hz
            self.step_simulation()
            time.sleep(1./240.)
        
        print("✅ Object grasped")
    
    def release_object(self):
        """Open gripper to release object"""
        print("✋ Opening gripper to release object...")
        
        # Open gripper fingers
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=0.04,  # Open position (4cm)
                force=50
            )
        
        # Simulate gripper opening
        for _ in range(60):  # 0.25 seconds at 240 Hz
            self.step_simulation()
            time.sleep(1./240.)
        
        print("✅ Object released")
    
    def analyze_scene_with_vlm(self) -> List[Dict]:
        """
        Analyze current scene using VLM and get world coordinates
        
        Returns:
            List of detected objects with world coordinates
        """
        if not self.vlm_analyzer:
            print("⚠️ VLM not initialized. Skipping analysis.")
            return []

        try:
            import time
            
            # Time the image capture
            start_time = time.time()
            rgb_img, depth_img, seg_img = self.capture_scene()
            capture_time = time.time() - start_time
            
            # Time the VLM analysis
            vlm_start = time.time()
            detected_objects = self.vlm_analyzer.get_object_world_coordinates(rgb_img, depth_img)
            vlm_time = time.time() - vlm_start
            
            print(f"⏱️ Timing: Capture={capture_time:.2f}s, VLM={vlm_time:.2f}s, Total={capture_time+vlm_time:.2f}s")
            
            if detected_objects:
                print(f"🔍 VLM detected {len(detected_objects)} objects with coordinates:")
                for i, obj in enumerate(detected_objects):
                    name = obj.get('name', 'Unknown')
                    color = obj.get('color', 'Unknown')
                    coords = obj.get('world_coordinates')
                    reachable = obj.get('reachable', False)
                    
                    if coords:
                        print(f"  {i+1}. {name} ({color}) - Position: ({coords['x']:.3f}, {coords['y']:.3f}, {coords['z']:.3f}) - {'✅ Reachable' if reachable else '❌ Out of reach'}")
                    else:
                        print(f"  {i+1}. {name} ({color}) - Position: Unknown")
            else:
                print("🔍 No objects detected")
            
            return detected_objects
            
        except Exception as e:
            print(f"❌ VLM analysis error: {e}")
            # Disable VLM for rest of simulation
            print("🔇 Disabling VLM for remainder of simulation")
            self.vlm_analyzer = None
            return []
    
    def step_simulation(self, steps: int = 1):
        """Advance simulation by specified steps"""
        for _ in range(steps):
            if self.physics_client is not None:
                try:
                    # Check if still connected
                    if p.isConnected(self.physics_client):
                        p.stepSimulation()
                    else:
                        print("❌ Physics server disconnected unexpectedly")
                        self.physics_client = None
                        raise RuntimeError("Physics server disconnected")
                except Exception as e:
                    print(f"❌ Error stepping simulation: {e}")
                    self.physics_client = None
                    raise
            else:
                raise RuntimeError("Simulation not connected")
    
    def get_object_coordinates(self, object_name: Optional[str] = None) -> Dict:
        """
        Easy method to get world coordinates of detected objects
        
        Args:
            object_name: Specific object to find, or None for all objects
            
        Returns:
            Dictionary with object coordinates
        """
        detected_objects = self.analyze_scene_with_vlm()
        
        if object_name:
            # Find specific object
            for obj in detected_objects:
                if object_name.lower() in obj.get('name', '').lower():
                    coords = obj.get('world_coordinates')
                    if coords:
                        return {
                            'name': obj.get('name'),
                            'coordinates': (coords['x'], coords['y'], coords['z']),
                            'reachable': obj.get('reachable', False)
                        }
            return {'error': f'Object "{object_name}" not found'}
        else:
            # Return all objects
            result = {}
            for i, obj in enumerate(detected_objects):
                coords = obj.get('world_coordinates')
                name = obj.get('name', f'object_{i}')
                if coords:
                    result[name] = {
                        'coordinates': (coords['x'], coords['y'], coords['z']),
                        'reachable': obj.get('reachable', False),
                        'color': obj.get('color', 'unknown')
                    }
            return result
    
    def pick_and_sort_workflow(self):
        """
        Complete pick-and-sort workflow:
        1. Scan workspace with overhead camera
        2. Detect objects
        3. For each object:
           a. Move arm above object
           b. Take close-up photo with wrist camera
           c. Classify object
           d. Pick up object
           e. Move to appropriate container
           f. Release object
        """
        if not self.vlm_analyzer:
            print("❌ Cannot run pick-and-sort without VLM")
            return
        
        print("\n" + "="*60)
        print("🤖 STARTING PICK-AND-SORT WORKFLOW")
        print("="*60)
        
        # Step 1: Scan workspace with overhead camera
        print("\n📸 Step 1: Scanning workspace...")
        detected_objects = self.analyze_scene_with_vlm()
        
        if not detected_objects:
            print("❌ No objects detected in workspace")
            return
        
        # Step 2: Process each object
        for i, obj in enumerate(detected_objects):
            print(f"\n{'='*60}")
            print(f"🎯 Processing object {i+1}/{len(detected_objects)}")
            print(f"{'='*60}")
            
            coords = obj.get('world_coordinates')
            if not coords or not obj.get('reachable', False):
                print(f"⚠️ Object not reachable, skipping...")
                continue
            
            obj_pos = [coords['x'], coords['y'], coords['z']]
            print(f"📍 Target position: ({coords['x']:.2f}, {coords['y']:.2f}, {coords['z']:.2f})")
            
            # Step 3a: Move arm above object
            approach_pos = [coords['x'], coords['y'], coords['z'] + 0.2]
            self.move_to_position(approach_pos, duration=2.0)
            
            # Step 3b: Take close-up photo with wrist camera
            print("📸 Taking close-up photo...")
            classification = self.classify_object_with_wrist_camera()
            
            # Step 3c: Determine target container
            target_container = self.determine_target_container(classification)
            print(f"🎯 Target container: {target_container}")
            
            # Step 3d: Move down and grasp
            self.move_to_position(obj_pos, duration=1.0)
            self.grasp_object()
            
            # Step 3e: Lift and move to container
            lift_pos = [coords['x'], coords['y'], coords['z'] + 0.3]
            self.move_to_position(lift_pos, duration=1.0)
            
            container_pos = self.containers[target_container]['position']
            drop_pos = [container_pos[0], container_pos[1], container_pos[2] + 0.3]
            self.move_to_position(drop_pos, duration=2.0)
            
            # Step 3f: Release object
            self.release_object()
            
            print(f"✅ Object {i+1} sorted successfully!")
        
        print(f"\n{'='*60}")
        print(f"✅ PICK-AND-SORT WORKFLOW COMPLETED")
        print(f"   Processed {len(detected_objects)} objects")
        print(f"{'='*60}\n")
    
    def run_simulation(self, duration: float = 10.0, vlm_analysis_interval: float = 2.0):
        """
        Run the simulation for specified duration
        
        Args:
            duration: Simulation duration in seconds (use float('inf') for infinite)
            vlm_analysis_interval: How often to run VLM analysis (seconds)
        """
        start_time = time.time()
        last_vlm_time = 0
        frame_count = 0
        
        if duration == float('inf'):
            print(f"🚀 Starting simulation (infinite duration - press Ctrl+C to stop)...")
        else:
            print(f"🚀 Starting simulation for {duration} seconds...")
        
        try:
            while True:
                current_time = time.time() - start_time
                
                # Check duration limit
                if duration != float('inf') and current_time >= duration:
                    break
                
                # Step simulation
                self.step_simulation()
                frame_count += 1
                
                # Run VLM analysis at specified intervals
                if (self.vlm_analyzer and 
                    current_time - last_vlm_time >= vlm_analysis_interval):
                    print(f"\n📸 Frame {frame_count}: Running VLM analysis...")
                    self.analyze_scene_with_vlm()
                    last_vlm_time = current_time
                
                time.sleep(1./240.)  # 240 FPS target
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Simulation interrupted by user after {frame_count} frames")
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
            print(f"Completed {frame_count} frames before error")
        
        print(f"✅ Simulation completed after {frame_count} frames")
    
    def close(self):
        """Clean up and close simulation"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
                print("🔌 Simulation disconnected")
            except:
                print("🔌 Simulation already disconnected")
            finally:
                self.physics_client = None


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
        
        # Camera parameters
        self.camera_target = [0.5, 0.0, 0]  # Center of workspace
        self.camera_yaw = 0
        self.camera_pitch = -90  # Looking straight down
        
        # Compute matrices
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            self.camera_target, self.camera_height, 
            self.camera_yaw, self.camera_pitch, 0, 2)
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.image_size[0] / self.image_size[1],
            nearVal=0.01,  # 1cm - much closer to capture objects on table
            farVal=3.0
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


def main():
    """Main function to run the pick-and-sort simulation"""
    print("="*70)
    print("🤖 ROBOT ARM PICK-AND-SORT SIMULATION")
    print("="*70)
    
    # Load API key from environment variable
    API_KEY = os.getenv('GOOGLE_AI_API_KEY')
    
    if API_KEY and VLM_AVAILABLE:
        print("🔑 API key found - VLM analysis will be enabled")
    else:
        print("⚠️ VLM analysis will be disabled")
        if not API_KEY:
            print("   Missing API key: Set GOOGLE_AI_API_KEY in your .env file")
        if not VLM_AVAILABLE:
            print("   Missing dependencies: conda env update -n bullet39 --file environment.yml --prune")
    
    print("🤖 Initializing Robot Arm Simulation for Pick-and-Sort...")
    
    # Create simulation with VLM if available
    sim = RobotArmSimulation(gui_mode=True, api_key=API_KEY if VLM_AVAILABLE else None)
    
    # Add sorting containers
    print("\n📦 Setting up sorting system...")
    containers = sim.add_sorting_containers()
    print(f"✅ Created {len(containers)} sorting bins")
    
    # Add objects at random positions
    print("\n🎲 Generating random objects around robot arm...")
    created_objects = sim.add_random_objects(num_objects=4)  # Create 4 random objects
    
    if created_objects:
        print(f"✅ Successfully created {len(created_objects)} random objects:")
        for obj_name in created_objects:
            print(f"   - {obj_name}")
    
    # Initialize gripper to open position
    print("\n🤖 Initializing robot gripper...")
    for joint in sim.gripper_joints:
        p.setJointMotorControl2(
            sim.robot_id,
            joint,
            p.POSITION_CONTROL,
            targetPosition=0.04,  # Open position
            force=50
        )
    
    # Allow physics to settle
    print("⏳ Settling physics...")
    for _ in range(240):  # 1 second at 240 Hz
        sim.step_simulation()
    
    try:
        if API_KEY and VLM_AVAILABLE:
            # Run pick-and-sort workflow
            sim.pick_and_sort_workflow()
            
            # Keep simulation running to view results
            print("\n👀 Keeping simulation open to view results...")
            print("   Press Ctrl+C to exit")
            sim.run_simulation(duration=float('inf'), vlm_analysis_interval=999999)
        else:
            print("\n⚠️ Cannot run pick-and-sort without VLM")
            print("   Running standard simulation instead...")
            sim.run_simulation(duration=float(30), vlm_analysis_interval=5.0)
            
    except KeyboardInterrupt:
        print("\n⏹️ Simulation interrupted by user")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
