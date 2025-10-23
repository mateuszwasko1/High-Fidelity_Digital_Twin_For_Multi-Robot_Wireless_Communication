"""
Main Pipeline
Connects all simulation components together
"""

import os
import sys
import time

# Add simulations directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulations'))

from simulation_setup import (
    initialize_physics,
    load_environment,
    initialize_robot_gripper,
    OverheadCamera,
    spawn_random_objects,
    create_sorting_containers
)
from object_finder import find_closest_object
from vlm_detector import create_detector
from arm_movement import ArmController
import pybullet as p


def main():
    # Initialize simulation
    physics_client = initialize_physics(gui_mode=True)
    plane_id, robot_id = load_environment()
    initialize_robot_gripper(robot_id, gripper_joints=[9, 10])
    camera = OverheadCamera()

    # Create containers and spawn objects
    containers = create_sorting_containers()
    objects = spawn_random_objects(num_objects=5)

    # Initialize VLM detector
    print("\n🤖 Initializing VLM detector...")
    vlm_detector = create_detector("owlvit")  # Change to "nanoowl" later when implemented
    vlm_detector.load_model()
    
    # Initialize arm controller
    print("\n🦾 Initializing arm controller...")
    # Franka Panda standard config: 7 arm joints (0-6), link 11 is end effector, joints 9-10 are gripper
    arm_controller = ArmController(robot_id, end_effector_link_index=11, gripper_joints=[9, 10])

    # Let physics settle
    for _ in range(240):
        p.stepSimulation()
    
    # Find closest object using depth clustering
    print("\n🔍 Detecting closest object...")
    # Extract container IDs to exclude from detection
    container_ids = [container['id'] for container in containers.values()]
    closest_object = find_closest_object(camera, robot_id, camera_height=camera.camera_height, 
                                        exclude_ids=container_ids, debug=True)
    
    if closest_object:
        print(f"\n✅ Found closest object (depth detection):")
        print(f"   Color: {closest_object.color}")
        print(f"   Location: {closest_object.location}")
        
        # Use VLM to identify the object from its image
        if closest_object.image is not None:
            print(f"\n🔬 Analyzing object with VLM...")
            # Use much lower threshold since we have small cropped images
            vlm_result = vlm_detector.detect_object(closest_object.image, confidence_threshold=0.001)
            
            # Update object with VLM detection results
            closest_object.name = vlm_result['name']
            closest_object.shape = vlm_result['shape']
            closest_object.color = vlm_result['color']  # Use VLM color instead of depth-based color
            
            print(f"   VLM Detection:")
            print(f"      Name: {vlm_result['name']}")
            print(f"      Shape: {vlm_result['shape']}")
            print(f"      Color: {vlm_result['color']}")
            print(f"      Confidence: {vlm_result['confidence']:.2%}")
            
            print(f"\n📦 Final object details:")
            print(f"   {closest_object}")
            
            # Now pick up the object!
            print(f"\n🤖 Attempting to pick up object...")
            object_pos = closest_object.location
            
            # Convert numpy float64 to regular float for arm controller
            object_pos = (float(object_pos[0]), float(object_pos[1]), float(object_pos[2]))
            
            # DEBUG: Add small marker to verify detection accuracy
            p.addUserDebugLine(
                [object_pos[0], object_pos[1], 0], 
                [object_pos[0], object_pos[1], 0.2],
                lineColorRGB=[1, 0, 0], lineWidth=3, lifeTime=10
            )
            print(f"   🎯 Detected object at position: {object_pos}")
            
            # Use higher approach height for better reachability
            # Franka Panda needs higher Z positions when reaching forward
            success = arm_controller.pick_object(object_pos, approach_height=0.30)
            
            if success:
                print(f"\n✅ Successfully picked up {closest_object.name}!")
                
                # Determine target container based on shape
                # For now, let's just place in the first container
                target_container = list(containers.values())[0]
                target_pos = target_container['position']
                
                print(f"\n📥 Placing object in {target_container['name']}...")
                if arm_controller.place_object(target_pos, approach_height=0.15):
                    print(f"✅ Successfully placed object!")
                    
                    # Move back to home position
                    arm_controller.move_to_home_position()
                else:
                    print(f"❌ Failed to place object")
            else:
                print(f"❌ Failed to pick up object")
        else:
            print(f"   ⚠️ No image captured for VLM analysis")
    else:
        print("❌ No objects detected")
    
    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        pass
    
    p.disconnect()


if __name__ == "__main__":
    main()
