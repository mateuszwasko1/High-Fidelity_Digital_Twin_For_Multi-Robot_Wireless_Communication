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
    spawn_debug_grid_objects,
    create_sorting_containers
)
from object_finder import find_closest_object
from vlm_detector import create_detector
from arm_movement import ArmController
import pybullet as p
import numpy as np


def main():
    # Initialize simulation
    physics_client = initialize_physics(gui_mode=True)
    plane_id, robot_id = load_environment()
    initialize_robot_gripper(robot_id, gripper_joints=[9, 10])
    camera = OverheadCamera()

    # Create containers and spawn objects
    containers = create_sorting_containers()
    objects = spawn_debug_grid_objects()

    # Initialize VLM detector (using optimized CLIP)
    vlm_detector = create_detector("clip")  # Using ViT-B/32 with timing diagnostics
    vlm_detector.load_model()
    
    # Initialize arm controller (silent)
    arm_controller = ArmController(robot_id, end_effector_link_index=11, gripper_joints=[9, 10], verbose=False)

    # Let physics settle
    for _ in range(240):
        p.stepSimulation()
    
    # Extract container IDs to exclude from detection (flatten lists since each bin has 4 walls)
    container_ids = []
    for container in containers.values():
        if isinstance(container['id'], list):
            container_ids.extend(container['id'])  # Add all wall IDs
        else:
            container_ids.append(container['id'])  # Single ID
    
    # Define spatial regions to exclude (bin areas + margin)
    # Format: (x_min, x_max, y_min, y_max) in world coordinates
    bin_regions = [
        # Left column bins (Y = -0.55) - SPHERES and TRIANGLES
        (0.05, 0.25, -0.65, -0.45),  # sphere_bin at [0.15, -0.55]
        (0.35, 0.55, -0.65, -0.45),  # triangle_bin at [0.45, -0.55]
        # Right column bins (Y = +0.55) - RECTANGLES and MIXED
        (0.05, 0.25, 0.45, 0.65),    # rectangle_bin at [0.15, 0.55]
        (0.35, 0.55, 0.45, 0.65),    # mixed_bin at [0.45, 0.55]
    ]
    
    # Process all objects - try to pick up and sort each one
    print("\n" + "="*60)
    print("🎯 STARTING OBJECT SORTING")
    print("="*60)
    picked_objects = 0
    failed_objects = []
    failed_positions = []  # Track positions we've already tried and failed
    attempt = 0
    max_consecutive_failures = 3  # Stop after 3 consecutive failures to find objects
    consecutive_failures = 0
    
    while True:  # Continue until no more objects detected
        attempt += 1
        
        # Keep trying to find a NEW object (not one we've already attempted)
        max_detection_attempts = 10
        closest_object = None
        
        for detection_attempt in range(max_detection_attempts):
            temp_object = find_closest_object(camera, robot_id, camera_height=camera.camera_height, 
                                                exclude_ids=container_ids, exclude_regions=bin_regions,
                                                debug=(detection_attempt==0))
            
            if not temp_object:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n   ℹ️  No more objects detected after {consecutive_failures} attempts")
                break
            
            # Check if this position is too close to any previously failed position
            is_duplicate = False
            for failed_pos in failed_positions:
                distance = ((temp_object.location[0] - failed_pos[0])**2 + 
                           (temp_object.location[1] - failed_pos[1])**2)**0.5
                if distance < 0.05:  # Within 5cm of a failed position
                    is_duplicate = True
                    break
            
            # Also filter out objects behind the robot (X < 0) - these are dropped balls
            if temp_object.location[0] < 0:
                is_duplicate = True
            
            if not is_duplicate:
                closest_object = temp_object
                break
            else:
                # This is a duplicate, try to exclude it by adding a temporary marker
                # We'll just try again - the depth detection might pick a different cluster
                time.sleep(0.1)
                continue
        
        # Break outer loop if we've had too many consecutive failures
        if consecutive_failures >= max_consecutive_failures:
            break
        
        if not closest_object:
            continue  # Try again in next iteration
        
        # Reset consecutive failures since we found an object
        consecutive_failures = 0
        
        # Use VLM to identify the object from its image
        if closest_object.image is not None:
            import time as time_module
            
            # Time VLM detection
            t0 = time_module.time()
            vlm_result = vlm_detector.detect_object(closest_object.image, confidence_threshold=0.001)
            t1 = time_module.time()
            print(f"   ⏱️  VLM detection took: {(t1-t0)*1000:.1f}ms")
            
            # Update object with VLM detection results
            closest_object.name = vlm_result['name']
            closest_object.shape = vlm_result['shape']
            closest_object.color = vlm_result['color']  # Use VLM color instead of depth-based color
            
            # Print detected object info
            print(f"\n🔍 Object #{attempt + 1}: {vlm_result['shape'].upper()} (confidence: {vlm_result['confidence']:.1%})")
            print(f"   📍 Object location: {closest_object.location}")
            
            # Pick up the object
            object_pos = closest_object.location
            object_pos = (float(object_pos[0]), float(object_pos[1]), float(object_pos[2]))
            print(f"   🎯 Attempting to pick object at: ({object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f})")
            
            t2 = time_module.time()
            print(f"   ⏱️  Setup took: {(t2-t1)*1000:.1f}ms")
            
            # Adaptive grasp height based on lateral distance
            robot_pos = p.getBasePositionAndOrientation(robot_id)[0]
            lateral_distance = abs(object_pos[1] - robot_pos[1])
            
            if lateral_distance > 0.25:
                offset = 0.015
                grasp_z = max(0.02, object_pos[2] - offset)
            elif lateral_distance > 0.15:
                offset = 0.015
                grasp_z = max(0.02, object_pos[2] - offset)
            else:
                offset = 0.025
                grasp_z = max(0.02, object_pos[2] - offset)
            
            adjusted_pos = (object_pos[0], object_pos[1], grasp_z)
            print(f"   ⚙️  Calling pick_object with adjusted position: ({adjusted_pos[0]:.3f}, {adjusted_pos[1]:.3f}, {adjusted_pos[2]:.3f})")
            
            t3 = time_module.time()
            success = arm_controller.pick_object(
                adjusted_pos, 
                approach_height=0.10,
                object_shape=closest_object.shape,
                object_image=closest_object.image
            )
            t4 = time_module.time()
            print(f"   ⏱️  pick_object took: {(t4-t3):.2f}s")
            print(f"   {'✅' if success else '❌'} pick_object returned: {success}")
            
            if success:
                time.sleep(0.5)
                
                # Lift and verify grip
                lift_pos = (object_pos[0], object_pos[1], 0.40)
                arm_controller.move_to_position(lift_pos)
                time.sleep(0.5)
                
                # Check if holding object
                object_is_held = False
                for obj_name, obj_info in objects.items():
                    obj_pos = p.getBasePositionAndOrientation(obj_info['id'])[0]
                    if obj_pos[2] > 0.15:
                        object_is_held = True
                        break
                
                if object_is_held:
                    picked_objects += 1
                    
                    # PRE-PROGRAMMED DROP LOCATIONS for each bin (shape-based)
                    # Vertical arrangement: left column (Y=-0.55) and right column (Y=+0.55)
                    drop_locations = {
                        'sphere_bin': [0.15, -0.55, 0.35],      # Left close - SPHERES
                        'triangle_bin': [0.45, -0.55, 0.35],    # Left far - TRIANGLES
                        'rectangle_bin': [0.15, 0.55, 0.35],    # Right close - RECTANGLES
                        'mixed_bin': [0.45, 0.55, 0.35]         # Right far - MIXED/UNKNOWN
                    }
                    
                    # Shape-based bin selection using VLM detected shape
                    shape_to_bin = {
                        'sphere': 'sphere_bin',
                        'triangle': 'triangle_bin',
                        'rectangle': 'rectangle_bin',
                        'unknown': 'mixed_bin'  # Default: unknown goes to mixed bin
                    }
                    
                    # Get detected shape from VLM
                    detected_shape = closest_object.shape.lower()
                    target_box_name = shape_to_bin.get(detected_shape, 'mixed_bin')
                    drop_pos = drop_locations[target_box_name]
                    
                    # Move to high transit position first (0.45m height) to prevent dropping
                    # This ensures object stays secure during horizontal movement
                    transit_height = 0.45
                    transit_pos = (drop_pos[0], drop_pos[1], transit_height)
                    print(f"   🚁 Moving to transit position at height {transit_height}m...")
                    arm_controller.move_to_position(transit_pos)
                    
                    # Now descend to drop position
                    print(f"   ⬇️  Descending to drop position...")
                    arm_controller.move_to_position(drop_pos)
                    
                    # Release object
                    arm_controller.open_gripper()
                    time.sleep(1.0)
                    
                    # Return to ready
                    arm_controller.move_to_ready_position()
                else:
                    success = False
            
            if not success:
                failed_objects.append({
                    'name': closest_object.name,
                    'location': closest_object.location,
                    'lateral_distance': lateral_distance
                })
                failed_positions.append((object_pos[0], object_pos[1], object_pos[2]))
                
                # Move away from the failed object
                arm_controller.move_to_ready_position()
        else:
            print(f"   ⚠️ No image captured for VLM analysis")
            break
    
    # Print summary
    total_objects = picked_objects + len(failed_objects)
    print(f"\n{'='*60}")
    print(f"📊 SORTING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully sorted: {picked_objects} objects")
    if failed_objects:
        print(f"❌ Failed: {len(failed_objects)} objects")
    print(f"📦 Total objects processed: {total_objects}")
    
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
