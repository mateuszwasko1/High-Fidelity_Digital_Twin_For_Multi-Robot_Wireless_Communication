"""
Debug script to inspect robot configuration
"""
import pybullet as p
import pybullet_data
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'simulations'))

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load plane and robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

print("\n" + "="*60)
print("ROBOT CONFIGURATION ANALYSIS")
print("="*60)

# Get robot base position
base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
print(f"\n📍 Robot Base Position: {base_pos}")
print(f"   Robot Base Orientation: {base_orn}")

# Get number of joints
num_joints = p.getNumJoints(robot_id)
print(f"\n🔗 Total Joints: {num_joints}")

print("\n" + "-"*60)
print("DETAILED JOINT INFORMATION:")
print("-"*60)

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    link_name = joint_info[12].decode('utf-8')
    
    type_names = {
        p.JOINT_REVOLUTE: "REVOLUTE",
        p.JOINT_PRISMATIC: "PRISMATIC", 
        p.JOINT_FIXED: "FIXED",
        p.JOINT_SPHERICAL: "SPHERICAL",
        p.JOINT_PLANAR: "PLANAR"
    }
    
    joint_type_name = type_names.get(joint_type, f"UNKNOWN({joint_type})")
    
    print(f"\nJoint {i}:")
    print(f"  Name: {joint_name}")
    print(f"  Type: {joint_type_name}")
    print(f"  Link Name: {link_name}")
    
    # Get current state
    joint_state = p.getJointState(robot_id, i)
    print(f"  Position: {joint_state[0]:.4f}")
    print(f"  Velocity: {joint_state[1]:.4f}")

# Get end effector link states
print("\n" + "-"*60)
print("END EFFECTOR CANDIDATES:")
print("-"*60)

# Common end effector links for Franka Panda
candidate_links = [7, 8, 9, 10, 11]

for link_idx in candidate_links:
    if link_idx < num_joints:
        link_state = p.getLinkState(robot_id, link_idx)
        link_pos = link_state[0]
        link_orn = link_state[1]
        joint_info = p.getJointInfo(robot_id, link_idx)
        link_name = joint_info[12].decode('utf-8')
        
        print(f"\nLink {link_idx} ({link_name}):")
        print(f"  Position: ({link_pos[0]:.3f}, {link_pos[1]:.3f}, {link_pos[2]:.3f})")
        print(f"  Distance from base (XY): {((link_pos[0]-base_pos[0])**2 + (link_pos[1]-base_pos[1])**2)**0.5:.3f}m")

print("\n" + "-"*60)
print("RECOMMENDED CONFIGURATION:")
print("-"*60)

# Find arm joints (revolute/prismatic, not gripper)
arm_joints = []
gripper_joints = []

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_type = joint_info[2]
    joint_name = joint_info[1].decode('utf-8')
    
    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        if 'finger' in joint_name.lower() or 'gripper' in joint_name.lower():
            gripper_joints.append(i)
        else:
            arm_joints.append(i)

print(f"\n🦾 Suggested arm joints: {arm_joints}")
print(f"✋ Suggested gripper joints: {gripper_joints}")
print(f"🎯 Suggested end effector link: {arm_joints[-1] if arm_joints else 'unknown'}")

print("\n" + "="*60)
print("Press Ctrl+C to exit...")
print("="*60)

# Keep simulation running
try:
    while True:
        p.stepSimulation()
except KeyboardInterrupt:
    pass

p.disconnect()
