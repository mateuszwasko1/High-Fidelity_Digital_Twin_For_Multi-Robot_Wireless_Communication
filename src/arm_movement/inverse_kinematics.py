"""
Inverse Kinematics Motion Planner
Uses PyBullet's IK solver for motion planning
"""

import pybullet as p
import numpy as np
import time
from typing import Tuple, List, Optional


class IKMotionPlanner:
    """
    Motion planner using PyBullet's Inverse Kinematics solver
    """
    
    def __init__(self, robot_id: int, end_effector_link: int, arm_joints: List[int], verbose: bool = False):
        """
        Initialize IK motion planner
        
        Args:
            robot_id: PyBullet robot body ID
            end_effector_link: Link index of the end effector
            arm_joints: List of controllable arm joint indices
            verbose: Whether to print debug information
        """
        self.robot_id = robot_id
        self.end_effector_link = end_effector_link
        self.arm_joints = arm_joints
        self.verbose = verbose
        
        # Franka Panda joint limits
        self.lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        self.joint_ranges = [5.8, 3.5, 5.8, 3.0, 5.8, 3.8, 5.8]
    
    def plan_to_position(self, target_pos: Tuple[float, float, float], 
                         target_orientation: Optional[Tuple[float, float, float, float]] = None) -> Optional[List[float]]:
        """
        Use IK to calculate joint positions for target pose
        
        Args:
            target_pos: Target (x, y, z) position in world coordinates
            target_orientation: Target orientation as quaternion (x, y, z, w)
            
        Returns:
            List of joint positions, or None if IK fails
        """
        # Calculate orientation if not specified
        if target_orientation is None:
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Get robot base position for adaptive rest pose
        robot_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        
        # Adaptive rest pose based on target direction
        if target_pos[0] < 0:  # Backwards reach
            rest_poses = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        else:  # Forward reach
            base_rotation = np.arctan2(dy, dx)
            rest_poses = [base_rotation, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        # Calculate IK solution
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link,
            target_pos,
            target_orientation,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Return only arm joint positions (exclude gripper joints)
        return list(joint_poses[:len(self.arm_joints)])
    
    def execute_plan(self, joint_positions: List[float], max_steps: int = 240, threshold: float = 0.02) -> bool:
        """
        Execute IK plan by setting joint targets and stepping simulation
        
        Args:
            joint_positions: Target positions for arm joints
            max_steps: Maximum simulation steps to reach target
            threshold: Distance threshold to consider target reached (meters)
            
        Returns:
            True if execution completed
        """
        # Set joint positions
        for i, joint_idx in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500,
                maxVelocity=2.0
            )
        
        # Step simulation
        for step in range(max_steps):
            p.stepSimulation()
            time.sleep(1./240.)
        
        return True
    
    def get_current_ee_pose(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Get current end-effector position and orientation
        
        Returns:
            Tuple of (position, orientation_quaternion)
        """
        state = p.getLinkState(self.robot_id, self.end_effector_link)
        return state[0], state[1]  # Position, orientation
