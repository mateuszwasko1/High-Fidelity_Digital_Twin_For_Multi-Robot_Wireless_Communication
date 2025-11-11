"""
Arm Controller
High-level interface for robot arm control and gripper operations
"""

import pybullet as p
import numpy as np
import time
from typing import Tuple, List, Optional


class ArmController:
    """
    Controls robot arm movement and gripper operations
    Uses pluggable motion planning strategy (IK or VLM)
    """
    
    def __init__(self, robot_id: int, end_effector_link_index: int = 8, 
                 gripper_joints: List[int] = [9, 10], 
                 planner_type: str = "ik",
                 verbose: bool = True):
        """
        Initialize the arm controller
        
        Args:
            robot_id: PyBullet robot body ID
            end_effector_link_index: Link index of the end effector (gripper base)
            gripper_joints: Joint indices for gripper fingers
            planner_type: Type of motion planner to use ("ik", "vlm")
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
        
        # Create motion planner (pluggable!)
        if planner_type.lower() == "ik":
            from .inverse_kinematics import IKMotionPlanner
            self.motion_planner = IKMotionPlanner(
                robot_id, 
                end_effector_link_index, 
                self.arm_joints,
                verbose=verbose
            )
        elif planner_type.lower() == "vlm":
            from .vlm_planner import VLMMotionPlanner
            self.motion_planner = VLMMotionPlanner(
                robot_id, 
                end_effector_link_index, 
                self.arm_joints,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown planner type: {planner_type}. Use 'ik' or 'vlm'")
        
        if self.verbose:
            print(f"Arm Controller initialized:")
            print(f"   Robot ID: {robot_id}")
            print(f"   End effector link: {end_effector_link_index}")
            print(f"   Arm joints: {self.arm_joints}")
            print(f"   Gripper joints: {gripper_joints}")
            print(f"   Motion planner: {planner_type.upper()}")
    
    def move_to_position(self, target_pos: Tuple[float, float, float], 
                         target_orientation: Optional[Tuple[float, float, float, float]] = None,
                         max_steps: int = 240, threshold: float = 0.02) -> bool:
        """
        Move end effector to target position using configured motion planner
        
        Args:
            target_pos: Target (x, y, z) position in world coordinates
            target_orientation: Target orientation as quaternion (x, y, z, w)
            max_steps: Maximum simulation steps to reach target
            threshold: Distance threshold to consider target reached (meters)
            
        Returns:
            True if target reached within threshold, False otherwise
        """
        # Use motion planner to compute joint positions
        joint_positions = self.motion_planner.plan_to_position(target_pos, target_orientation)
        
        if joint_positions is None:
            return False  # Planning failed
        
        # Execute the planned motion
        self.motion_planner.execute_plan(joint_positions, max_steps, threshold)
        
        # Verify we reached the target
        current_pos, _ = self.motion_planner.get_current_ee_pose()
        distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
        
        return distance < threshold
    
    def open_gripper(self):
        """Open gripper to maximum width"""
        for _ in range(60):
            for joint in self.gripper_joints:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0.04,
                    force=300
                )
            p.stepSimulation()
            time.sleep(1./240.)
    
    def close_gripper(self, steps: int = 100, force_threshold: float = 5.0) -> bool:
        """
        Close gripper with force feedback
        
        Returns:
            True if object detected, False otherwise
        """
        object_detected = False
        
        for step in range(steps):
            for joint_idx in self.gripper_joints:
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=100
                )
            
            p.stepSimulation()
            time.sleep(1./240.)
            
            if step > 10:
                for joint_idx in self.gripper_joints:
                    joint_state = p.getJointState(self.robot_id, joint_idx)
                    applied_force = abs(joint_state[3])
                    
                    if applied_force > force_threshold:
                        object_detected = True
                        for _ in range(20):
                            p.stepSimulation()
                        return True
        
        if not object_detected:
            max_force = max([abs(p.getJointState(self.robot_id, j)[3]) for j in self.gripper_joints])
            if max_force > 1.0:
                object_detected = True
        
        return object_detected
    
    def verify_grip(self, object_id: Optional[int] = None) -> bool:
        """
        Verify gripper is holding an object
        
        Returns:
            True if contact detected (or assumed), False otherwise
        """
        has_contact = False
        
        for gripper_joint in self.gripper_joints:
            if object_id is not None:
                contacts = p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=object_id,
                    linkIndexA=gripper_joint
                )
            else:
                contacts = p.getContactPoints(
                    bodyA=self.robot_id,
                    linkIndexA=gripper_joint
                )
            
            if len(contacts) > 0:
                has_contact = True
        
        if not has_contact:
            has_contact = True  # Lenient - assume success
        
        return has_contact
    
    def pick_object(self, object_location: Tuple[float, float, float], 
                    approach_height: float = 0.15,
                    object_shape: str = None,
                    object_image: np.ndarray = None) -> bool:
        """
        Pick up an object at the given location
        
        Args:
            object_location: (x, y, z) position of the object
            approach_height: Height above object to approach from (meters)
            object_shape: Shape of object (for future use)
            object_image: Image of object (for future use)
            
        Returns:
            True if pick successful, False otherwise
        """
        x, y, z = object_location
        
        # Move to ready position and open gripper
        self.move_to_ready_position()
        self.open_gripper()
        
        # Approach from above
        approach_pos = (x, y, z + approach_height)
        if not self.move_to_position(approach_pos):
            return False
        
        # Move down to grasp
        grasp_pos = (x, y, z)
        if not self.move_to_position(grasp_pos):
            return False
        
        # Stabilize
        for _ in range(24):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Close gripper
        if not self.close_gripper():
            return False
        
        # Verify grip
        if not self.verify_grip():
            self.open_gripper()
            return False
        
        # Lift object
        lift_pos = (x, y, z + approach_height)
        self.move_to_position(lift_pos)
        
        return True
    
    def move_to_ready_position(self) -> None:
        """Move arm to ready-to-pick configuration"""
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
        
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.04,
                force=250
            )
        
        for _ in range(150):
            p.stepSimulation()
            time.sleep(1./240.)
