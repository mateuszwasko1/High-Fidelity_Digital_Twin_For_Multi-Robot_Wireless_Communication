"""
Main Pipeline
Connects all simulation components together
"""

import os
import sys
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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
    # ========================================
    # CONFIGURATION: Change these to switch components
    # ========================================
    MOTION_PLANNER_TYPE = "ik"  # Options: "ik", "vlm"
    VLM_DETECTOR_TYPE = "clip"   # Options: "clip"
    
    # Initialize simulation
    physics_client = initialize_physics(gui_mode=True)
    plane_id, robot_id = load_environment()
    initialize_robot_gripper(robot_id, gripper_joints=[9, 10])
    camera = OverheadCamera()

    # Create containers and spawn objects
    containers = create_sorting_containers()
    objects = spawn_debug_grid_objects()

    # Initialize VLM detector (using optimized CLIP)
    vlm_detector = create_detector(VLM_DETECTOR_TYPE)
    vlm_detector.load_model()
    
    # Initialize arm controller with selected motion planner
    arm_controller = ArmController(
        robot_id, 
        end_effector_link_index=11, 
        gripper_joints=[9, 10],
        planner_type=MOTION_PLANNER_TYPE,  # MODULAR: Easy to switch planners!
        verbose=True
    )

    # Let physics settle
    # PERFORMANCE FIX: Reduced from 240 to 120 steps for faster initialization
    for _ in range(120):
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
    print("STARTING OBJECT SORTING")
    print("="*60)
    picked_objects = 0
    failed_objects = []
    failed_positions = []  # Track positions we've already tried and failed
    attempt = 0
    max_consecutive_failures = 3  # Stop after 3 consecutive failures to find objects
    consecutive_failures = 0
    max_total_attempts = 20  # Safety limit: stop after 20 total attempts
    
    while attempt < max_total_attempts:  # Safety limit to prevent infinite loops
        attempt += 1
        print(f"\n{'='*60}")
        print(f"Attempt {attempt}/{max_total_attempts}")
        print(f"{'='*60}")
        
        # Keep trying to find a NEW object (not one we've already attempted)
        # PERFORMANCE FIX: Reduced from 10 to 2 attempts to eliminate 80% of redundant detection calls
        max_detection_attempts = 2
        closest_object = None
        
        for detection_attempt in range(max_detection_attempts):
            temp_object = find_closest_object(camera, robot_id, camera_height=camera.camera_height, 
                                                exclude_ids=container_ids, exclude_regions=bin_regions,
                                                exclude_positions=failed_positions,
                                                debug=(detection_attempt==0))
            
            if not temp_object:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n   No more objects detected after {consecutive_failures} attempts")
                break
            
            # Filter out objects behind the robot (X < 0) - these are dropped balls
            if temp_object.location[0] < 0:
                if detection_attempt < max_detection_attempts - 1:
                    print(f"      Object behind robot, trying again...")
                    # Brief pause to let physics settle
                    for _ in range(12):
                        p.stepSimulation()
                        time.sleep(1./240.)
                continue
            
            # Valid object found
            closest_object = temp_object
            break
        
        # Break outer loop if we've had too many consecutive failures
        if consecutive_failures >= max_consecutive_failures:
            print(f"\n   Stopping: {consecutive_failures} consecutive detection failures")
            break
        
        if not closest_object:
            print(f"   No valid object found in this attempt, continuing...")
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
            print(f"   VLM detection took: {(t1-t0)*1000:.1f}ms")
            
            # Update object with VLM detection results
            closest_object.name = vlm_result['name']
            closest_object.shape = vlm_result['shape']
            closest_object.color = vlm_result['color']  # Use VLM color instead of depth-based color
            
            # Print detected object info
            print(f"\nObject #{attempt + 1}: {vlm_result['shape'].upper()} (confidence: {vlm_result['confidence']:.1%})")
            
            # Pick up the object
            object_pos = closest_object.location
            object_pos = (float(object_pos[0]), float(object_pos[1]), float(object_pos[2]))
            
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
            
            success = arm_controller.pick_object(
                adjusted_pos, 
                approach_height=0.10,
                object_shape=closest_object.shape,
                object_image=closest_object.image
            )
            
            if success:
                print(f"   Object picked up")
                # Lift and verify grip (reduced delays for faster operation)
                lift_pos = (object_pos[0], object_pos[1], 0.40)
                arm_controller.move_to_position(lift_pos)
                
                # Give physics time to settle (reduced from 0.5 to 0.2 seconds)
                for _ in range(48):  # 0.2 seconds at 240 Hz
                    p.stepSimulation()
                    time.sleep(1./240.)
                
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
                    
                    # Move to transit position and drop
                    transit_height = 0.45
                    transit_pos = (drop_pos[0], drop_pos[1], transit_height)
                    arm_controller.move_to_position(transit_pos)
                    arm_controller.move_to_position(drop_pos)
                    
                    # Release object
                    arm_controller.open_gripper()
                    print(f"   Object dropped into {target_box_name.replace('_', ' ')}")
                    
                    # Brief settling time (reduced from 1.0 to 0.3 seconds)
                    for _ in range(72):  # 0.3 seconds at 240 Hz
                        p.stepSimulation()
                        time.sleep(1./240.)
                    
                    # Return to ready
                    arm_controller.move_to_ready_position()
                else:
                    success = False
            
            if not success:
                print(f"   Failed to pick object, marking position and moving on...")
                failed_objects.append({
                    'name': closest_object.name,
                    'location': closest_object.location,
                    'lateral_distance': lateral_distance
                })
                failed_positions.append((object_pos[0], object_pos[1], object_pos[2]))
                
                # Move away from the failed object
                print(f"   Returning to ready position...")
                arm_controller.move_to_ready_position()
                print(f"   Ready for next attempt")
        else:
            print(f"   No image captured for VLM analysis")
            break
    
    # Print summary
    total_objects = picked_objects + len(failed_objects)
    print(f"\n{'='*60}")
    print(f"SORTING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully sorted: {picked_objects} objects")
    if failed_objects:
        print(f"Failed: {len(failed_objects)} objects")
    print(f"Total objects processed: {total_objects}")
    
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
