"""
Arm Movement Module
Handles robot arm motion control using Inverse Kinematics
"""

import pybullet as p
import numpy as np
from typing import Tuple, List, Optional


class ArmController:
    """
    Controls robot arm movement using inverse kinematics
    """
    
    def __init__(self, robot_id: int, end_effector_link_index: int = 8, gripper_joints: List[int] = [9, 10]):
        """
        Initialize the arm controller
        
        Args:
            robot_id: PyBullet robot body ID
            end_effector_link_index: Link index of the end effector (gripper base)
            gripper_joints: Joint indices for gripper fingers
        """
        self.robot_id = robot_id
        self.end_effector_link = end_effector_link_index
        self.gripper_joints = gripper_joints
        
        # Get controllable joints (exclude fixed joints and gripper)
        self.arm_joints = []
        num_joints = p.getNumJoints(robot_id)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_type = joint_info[2]
            
            # Only include revolute/prismatic joints that are not gripper joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC] and i not in gripper_joints:
                self.arm_joints.append(i)
        
        print(f"🦾 Arm Controller initialized:")
        print(f"   Robot ID: {robot_id}")
        print(f"   End effector link: {end_effector_link_index}")
        print(f"   Arm joints: {self.arm_joints}")
        print(f"   Gripper joints: {gripper_joints}")
    
    def move_to_position(self, target_pos: Tuple[float, float, float], 
                         target_orientation: Optional[Tuple[float, float, float, float]] = None,
                         max_steps: int = 480, threshold: float = 0.02) -> bool:
        """
        Move end effector to target position using IK
        
        Args:
            target_pos: Target (x, y, z) position in world coordinates
            target_orientation: Target orientation as quaternion (x, y, z, w). If None, uses downward orientation
            max_steps: Maximum simulation steps to reach target
            threshold: Distance threshold to consider target reached (meters)
            
        Returns:
            True if target reached within threshold, False otherwise
        """
        # Default to pointing downward if no orientation specified
        if target_orientation is None:
            # Point straight down - this is critical!
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])  # Rotated 180° around X to point down
        
        print(f"\n🎯 Moving to position: {target_pos}")
        print(f"   Target orientation: {target_orientation}")
        
        # Check robot base position
        robot_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        print(f"   Robot base at: {robot_pos}")
        distance_from_base = np.linalg.norm(np.array(target_pos[:2]) - np.array(robot_pos[:2]))
        print(f"   Distance from base (XY): {distance_from_base:.3f}m")
        
        # Define lower limits for arm joints to encourage proper elbow configuration
        # This helps the arm bend "downward" naturally for picking
        lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        joint_ranges = [5.8, 3.5, 5.8, 3.0, 5.8, 3.8, 5.8]
        rest_poses = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # Good picking configuration
        
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
        print(f"   Target joint angles: {[f'{joint_poses[j]:.3f}' for j in self.arm_joints]}")
        
        # Simulate until target reached or max steps
        for step in range(max_steps):
            p.stepSimulation()
            
            # Check current end effector position every 20 steps
            if step % 20 == 0:
                current_pos = p.getLinkState(self.robot_id, self.end_effector_link)[0]
                distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                
                if step % 100 == 0:
                    print(f"   Step {step}: distance = {distance:.4f}m, current pos = ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
            
            if distance < threshold:
                print(f"   ✅ Reached target in {step} steps (distance: {distance:.4f}m)")
                return True
        
        # Didn't reach target in time
        current_pos = p.getLinkState(self.robot_id, self.end_effector_link)[0]
        distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        print(f"   ⚠️ Timeout after {max_steps} steps (distance: {distance:.4f}m)")
        return False
    
    def open_gripper(self, steps: int = 60) -> None:
        """
        Open the gripper
        
        Args:
            steps: Number of simulation steps to complete the action
        """
        print("   🖐️  Opening gripper...")
        for joint_idx in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.08,  # Increased from 0.04 to 0.08 - open MUCH wider
                force=100
            )
        
        for _ in range(steps):
            p.stepSimulation()
    
    def close_gripper(self, steps: int = 60, force_threshold: float = 10.0) -> bool:
        """
        Close the gripper with force feedback
        
        Args:
            steps: Number of simulation steps to complete the action
            force_threshold: Force threshold to detect object grip (Newtons)
            
        Returns:
            True if object detected (force applied), False otherwise
        """
        print("   ✊ Closing gripper...")
        object_detected = False
        
        for step in range(steps):
            # Apply closing force
            for joint_idx in self.gripper_joints:
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=0.0,  # Closed position
                    force=100
                )
            
            p.stepSimulation()
            
            # Check force feedback after gripper starts closing
            if step > 10:  # Give it time to start moving
                for joint_idx in self.gripper_joints:
                    joint_state = p.getJointState(self.robot_id, joint_idx)
                    applied_force = abs(joint_state[3])  # Reaction force
                    
                    if applied_force > force_threshold:
                        print(f"      ✅ Object gripped! (force: {applied_force:.1f}N at step {step})")
                        object_detected = True
                        # Continue closing a bit more to secure grip
                        for _ in range(20):
                            p.stepSimulation()
                        return True
        
        if not object_detected:
            print(f"      ⚠️ No object detected (max force: {max([abs(p.getJointState(self.robot_id, j)[3]) for j in self.gripper_joints]):.1f}N)")
        
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
                print(f"      ✅ Contact detected on gripper joint {gripper_joint} ({len(contacts)} contact points)")
        
        return has_contact
    
    def pick_object(self, object_location: Tuple[float, float, float], 
                    approach_height: float = 0.15) -> bool:
        """
        Pick up an object at the given location
        
        Args:
            object_location: (x, y, z) position of the object
            approach_height: Height above object to approach from (meters)
            
        Returns:
            True if pick successful, False otherwise
        """
        x, y, z = object_location
        
        print(f"\n🤏 Picking object at {object_location}")
        
        # 0. Move to ready position first
        self.move_to_ready_position()
        
        # 1. Open gripper
        self.open_gripper()
        
        # 2. Move to approach position (above object)
        approach_pos = (x, y, z + approach_height)
        if not self.move_to_position(approach_pos):
            print("   ❌ Failed to reach approach position")
            return False
        
        # 3. Move down to object
        if not self.move_to_position(object_location):
            print("   ❌ Failed to reach object")
            return False
        
        # 4. Close gripper to grasp
        if not self.close_gripper():
            print("   ❌ Failed to grip object (no force detected)")
            return False
        
        # 5. Verify grip with contact detection
        if not self.verify_grip():
            print("   ❌ Failed to grip object (no contact detected)")
            self.open_gripper()  # Release and give up
            return False
        
        # 6. Lift object
        lift_pos = (x, y, z + approach_height)
        if not self.move_to_position(lift_pos):
            print("   ⚠️ Warning: Failed to lift cleanly")
        
        print("   ✅ Object picked and secured!")
        return True
    
    def place_object(self, target_location: Tuple[float, float, float], 
                     approach_height: float = 0.15) -> bool:
        """
        Place the held object at the target location
        
        Args:
            target_location: (x, y, z) position to place the object
            approach_height: Height above target to approach from (meters)
            
        Returns:
            True if place successful, False otherwise
        """
        x, y, z = target_location
        
        print(f"\n📥 Placing object at {target_location}")
        
        # 1. Move to approach position (above target)
        approach_pos = (x, y, z + approach_height)
        if not self.move_to_position(approach_pos):
            print("   ❌ Failed to reach approach position")
            return False
        
        # 2. Move down to target
        if not self.move_to_position(target_location):
            print("   ❌ Failed to reach target")
            return False
        
        # 3. Open gripper to release
        self.open_gripper()
        
        # 4. Retract
        retract_pos = (x, y, z + approach_height)
        self.move_to_position(retract_pos)
        
        print("   ✅ Object placed!")
        return True
    
    def move_to_home_position(self) -> None:
        """
        Move arm to a safe home/rest position
        """
        print("\n🏠 Moving to home position...")
        # Define a safe home configuration (all joints at 0 or safe angles)
        home_joint_positions = [0.0] * len(self.arm_joints)
        
        for joint_idx, target_pos in zip(self.arm_joints, home_joint_positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=500
            )
        
        # Simulate to let arm settle
        for _ in range(240):
            p.stepSimulation()
        
        print("   ✅ Home position reached")
    
    def move_to_ready_position(self) -> None:
        """
        Move arm to a ready-to-pick configuration (arm bent down and forward)
        """
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
        
        # Simulate to let arm settle
        for _ in range(240):
            p.stepSimulation()
        
        # Check where end effector ended up
        ee_pos = p.getLinkState(self.robot_id, self.end_effector_link)[0]
        print(f"   ✅ Ready position reached, end effector at: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")


def pick_and_place(arm_controller: ArmController, 
                   object_location: Tuple[float, float, float],
                   target_location: Tuple[float, float, float]) -> bool:
    """
    Convenience function to pick an object and place it at a target
    
    Args:
        arm_controller: ArmController instance
        object_location: Where to pick from
        target_location: Where to place
        
    Returns:
        True if both pick and place succeeded
    """
    if not arm_controller.pick_object(object_location):
        return False
    
    if not arm_controller.place_object(target_location):
        return False
    
    return True
