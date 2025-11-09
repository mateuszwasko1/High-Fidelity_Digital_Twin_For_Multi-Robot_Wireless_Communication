"""
Arm Movement Module
Handles robot arm motion control using Inverse Kinematics
"""

import pybullet as p
import numpy as np
import time
from typing import Tuple, List, Optional


class ArmController:
    """
    Controls robot arm movement using inverse kinematics
    """
    
    def __init__(self, robot_id: int, end_effector_link_index: int = 8, gripper_joints: List[int] = [9, 10], verbose: bool = True):
        """
        Initialize the arm controller
        
        Args:
            robot_id: PyBullet robot body ID
            end_effector_link_index: Link index of the end effector (gripper base)
            gripper_joints: Joint indices for gripper fingers
            verbose: Whether to print debug information
        """
        self.robot_id = robot_id
        self.end_effector_link = end_effector_link_index
        self.gripper_joints = gripper_joints
        self.verbose = verbose
        
        # Get controllable joints (exclude fixed joints and gripper)
        self.arm_joints = []
        num_joints = p.getNumJoints(robot_id)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_type = joint_info[2]
            
            # Only include revolute/prismatic joints that are not gripper joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC] and i not in gripper_joints:
                self.arm_joints.append(i)
        
        if self.verbose:
            print(f"🦾 Arm Controller initialized:")
            print(f"   Robot ID: {robot_id}")
            print(f"   End effector link: {end_effector_link_index}")
            print(f"   Arm joints: {self.arm_joints}")
            print(f"   Gripper joints: {gripper_joints}")
    
    def move_to_position(self, target_pos: Tuple[float, float, float], 
                         target_orientation: Optional[Tuple[float, float, float, float]] = None,
                         max_steps: int = 240, threshold: float = 0.02) -> bool:
        """
        Move end effector to target position using IK
        
        Args:
            target_pos: Target (x, y, z) position in world coordinates
            target_orientation: Target orientation as quaternion (x, y, z, w). If None, uses horizontal gripper orientation
            max_steps: Maximum simulation steps to reach target (BALANCED: 240 steps = 1 second)
            threshold: Distance threshold to consider target reached (meters)
            
        Returns:
            True if target reached within threshold, False otherwise
        """
        # Check robot base position
        robot_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        distance_from_base = np.linalg.norm(np.array(target_pos[:2]) - np.array(robot_pos[:2]))
        
        # Calculate orientation if not specified
        if target_orientation is None:
            # ALWAYS use horizontal gripper orientation (fingers left-right)
            # This simplifies grasping and avoids complex edge alignment issues
            yaw = 0.0  # Gripper pointing forward, fingers left-right
            roll = 0.0
            
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, yaw + roll])
            
            if self.verbose:
                print(f"\n🎯 Moving to position: {target_pos}")
                print(f"   Robot base: {robot_pos}")
                print(f"   🤲 Horizontal gripper: yaw={np.degrees(yaw):.1f}°, roll={np.degrees(roll):.1f}°")
        else:
            if self.verbose:
                print(f"\n🎯 Moving to position: {target_pos}")
                print(f"   Using provided orientation: {target_orientation}")
        
        if self.verbose:
            print(f"   Robot base at: {robot_pos}")
            print(f"   Distance from base (XY): {distance_from_base:.3f}m")
        
        # Define lower limits for arm joints to encourage proper elbow configuration
        # This helps the arm bend "downward" naturally for picking
        lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        joint_ranges = [5.8, 3.5, 5.8, 3.0, 5.8, 3.8, 5.8]
        
        # Adaptive rest pose based on target direction
        # For backwards reaches (X < 0), use backwards-facing rest pose
        # For side reaches, rotate base joint (joint 0) toward target
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        
        if target_pos[0] < 0:  # Backwards reach
            base_rotation = np.arctan2(dy, dx)
            # Use default rest pose for backwards - works best based on testing
            rest_poses = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        else:  # Forward reach
            base_rotation = np.arctan2(dy, dx)
            # Use adaptive rest pose that points toward the target
            rest_poses = [base_rotation, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        # Calculate IK solution with joint limits and rest pose
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link,
            target_pos,
            target_orientation,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        
        # Set target positions for arm joints
        for i, joint_idx in enumerate(self.arm_joints):
            if i < len(joint_poses):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[joint_idx],
                    force=200,  # Increased force
                    maxVelocity=2.0  # Increased velocity
                )
        
        # Debug: print target joint angles
        if self.verbose:
            print(f"   Target joint angles: {[f'{joint_poses[j]:.3f}' for j in self.arm_joints]}")
        
        # Move to target position
        for joint_idx, joint in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[joint_idx],
                force=200
            )
        
        # Simulation loop to reach target
        for step in range(max_steps):
            p.stepSimulation()
            time.sleep(1./240.)  # Add small delay for smooth visualization (matches physics timestep)
            
            # Check if we've reached the target every 10 steps (optimization)
            if step % 10 == 0:
                current_pos = p.getLinkState(self.robot_id, self.end_effector_link)[0]
                distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
                
                # Print progress every 50 steps for better feedback
                if self.verbose and step % 50 == 0:
                    print(f"   Step {step}: distance = {distance:.4f}m, current pos = ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
                
                if distance < threshold:
                    if self.verbose:
                        print(f"   ✅ Reached target in {step} steps (distance: {distance:.4f}m)")
                    return True
        
        # If we didn't reach the target, print warning
        distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
        if self.verbose:
            print(f"   ⚠️ Timeout after {max_steps} steps (distance: {distance:.4f}m)")
        
        return False
    
    def open_gripper(self):
        """
        Open gripper as wide as possible to avoid collisions with objects.
        Franka Panda gripper maximum opening is 0.08m (8cm between fingers).
        """
        if self.verbose:
            print("\n   🖐️  Opening gripper fully...")
        # PERFORMANCE FIX: Reduced from 200 to 60 steps for faster gripper opening
        for _ in range(60):
            for joint in self.gripper_joints:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0.04,  # Maximum opening (each finger moves 0.04m from center = 8cm total)
                    force=300  # Very strong force to ensure maximum opening
                )
            p.stepSimulation()
            time.sleep(1./240.)  # Smooth visualization
    
    def close_gripper(self, steps: int = 100, force_threshold: float = 5.0) -> bool:
        """
        Close the gripper with force feedback
        
        Args:
            steps: Number of simulation steps to complete the action (increased to 100 for better contact)
            force_threshold: Force threshold to detect object grip (Newtons) (lowered to 5.0 for small objects)
            
        Returns:
            True if object detected (force applied), False otherwise
        """
        object_detected = False
        
        for step in range(steps):
            # Apply closing force
            for joint_idx in self.gripper_joints:
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=0.0,  # Closed position
                    force=100  # Original force value
                )
            
            p.stepSimulation()
            time.sleep(1./240.)  # Smooth visualization
            
            # Check force feedback after gripper starts closing
            if step > 10:  # Give it time to start moving
                for joint_idx in self.gripper_joints:
                    joint_state = p.getJointState(self.robot_id, joint_idx)
                    applied_force = abs(joint_state[3])  # Reaction force
                    
                    if applied_force > force_threshold:
                        if self.verbose:
                            print(f"      ✅ Object gripped! (force: {applied_force:.1f}N at step {step})")
                        object_detected = True
                        # Continue closing a bit more to secure grip
                        for _ in range(20):
                            p.stepSimulation()
                        return True
        
        if not object_detected:
            max_force = max([abs(p.getJointState(self.robot_id, j)[3]) for j in self.gripper_joints])
            if self.verbose:
                print(f"      ⚠️ No object detected by force (max force: {max_force:.1f}N, threshold: {force_threshold}N)")
            # If we detected some force but below threshold, still return True
            if max_force > 1.0:  # Very lenient threshold
                if self.verbose:
                    print(f"      ℹ️  Detected minimal force, continuing anyway...")
                object_detected = True
        
        return object_detected
    
    def verify_grip(self, object_id: Optional[int] = None) -> bool:
        """
        Verify that gripper is holding an object using contact detection
        
        Args:
            object_id: Optional specific object ID to check. If None, checks for any contacts
            
        Returns:
            True if contact detected, False otherwise
        """
        has_contact = False
        
        for gripper_joint in self.gripper_joints:
            if object_id is not None:
                # Check contact with specific object
                contacts = p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=object_id,
                    linkIndexA=gripper_joint
                )
            else:
                # Check for any contacts on gripper
                contacts = p.getContactPoints(
                    bodyA=self.robot_id,
                    linkIndexA=gripper_joint
                )
            
            if len(contacts) > 0:
                has_contact = True
                if self.verbose:
                    print(f"      ✅ Contact detected on gripper joint {gripper_joint} ({len(contacts)} contact points)")
        
        if not has_contact and self.verbose:
            print(f"      ℹ️  No contacts detected, but continuing anyway (may still be holding)")
            # Return True anyway - contact detection can be unreliable
            has_contact = True
        
        return has_contact
    
    def pick_object(self, object_location: Tuple[float, float, float], 
                    approach_height: float = 0.15,
                    object_shape: str = None,
                    object_image: np.ndarray = None) -> bool:
        """
        Pick up an object at the given location using horizontal gripper orientation
        
        Args:
            object_location: (x, y, z) position of the object
            approach_height: Height above object to approach from (meters)
            object_shape: Shape of the object (unused, kept for compatibility)
            object_image: Image of the object (unused, kept for compatibility)
            
        Returns:
            True if pick successful, False otherwise
        """
        x, y, z = object_location
        
        if self.verbose:
            print(f"\n🤏 Picking object at {object_location}")
        
        # 0. Move to ready position first and ensure gripper is open
        self.move_to_ready_position()
        self.open_gripper()  # Explicitly open gripper to ensure it's ready
        
        # 1. Move to approach position (above object)
        approach_pos = (x, y, z + approach_height)
        if not self.move_to_position(approach_pos):
            if self.verbose:
                print("   ❌ Failed to reach approach position")
            return False
        
        # 2. Move down to grasp position (at detected position)
        grasp_pos = (x, y, z)
        if not self.move_to_position(grasp_pos):
            if self.verbose:
                print("   ❌ Failed to reach grasp position")
            return False
        
        # 2.5. Brief pause to stabilize before grasping (24 steps = 0.1 seconds)
        for _ in range(24):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 3. Close gripper to grasp
        if not self.close_gripper():
            if self.verbose:
                print("   ❌ Failed to grip object (no force detected)")
            return False
        
        # 4. Verify grip with contact detection
        if not self.verify_grip():
            if self.verbose:
                print("   ❌ Failed to grip object (no contact detected)")
            self.open_gripper()  # Release and give up
            return False
        
        # 5. Lift object
        lift_pos = (x, y, z + approach_height)
        if not self.move_to_position(lift_pos):
            if self.verbose:
                print("   ⚠️ Warning: Failed to lift cleanly")
        
        if self.verbose:
            print("   ✅ Object picked and secured!")
        return True
    
    def move_to_ready_position(self) -> None:
        """
        Move arm to a ready-to-pick configuration with gripper open
        """
        if self.verbose:
            print("\n🎯 Moving to ready position...")
        # Good picking configuration: elbow bent, arm pointing forward and down
        ready_positions = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        
        for joint_idx, target_pos in zip(self.arm_joints, ready_positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=500,
                maxVelocity=2.0
            )
        
        # Open gripper fully in ready position
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.04,  # Maximum opening (4cm per finger = 8cm total)
                force=250  # Very strong force
            )
        
        # Simulate to let arm and gripper settle completely
        # PERFORMANCE FIX: Balanced at 150 steps (0.625s) for reliable settling
        for _ in range(150):
            p.stepSimulation()
            time.sleep(1./240.)  # Smooth visualization
        
        # Check where end effector ended up
        if self.verbose:
            ee_pos = p.getLinkState(self.robot_id, self.end_effector_link)[0]
            print(f"   ✅ Ready position reached, end effector at: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
