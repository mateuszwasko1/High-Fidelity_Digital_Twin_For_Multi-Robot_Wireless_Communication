"""
Basic PyBullet simulation setup for a robot arm.
This script loads a simple robot arm URDF and starts a simulation.
"""
import pybullet as p
import pybullet_data
import time

# Connect to PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load plane and robot arm (replace with your URDF as needed)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# Run simulation for a few seconds
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect
#Test
p.disconnect()
