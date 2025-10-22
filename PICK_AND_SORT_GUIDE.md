# 🤖 Pick-and-Sort Robot Simulation Guide

## Overview
This simulation implements a complete **pick-and-sort workflow** where a robot arm:
1. **Scans** the workspace with an overhead camera
2. **Detects** objects and their positions
3. **Approaches** each object
4. **Classifies** objects using a wrist-mounted camera + VLM
5. **Picks up** the object
6. **Sorts** into appropriate containers based on classification

## 📹 Dual Camera System

### 1. Overhead Camera (Scene Survey)
- **Location**: Fixed at 1.5m above workspace
- **Purpose**: Find all objects and get initial positions
- **FOV**: 60° covering workspace area X(0-1m), Y(-0.5-0.5m)
- **Advantages**:
  - Wide field of view
  - Sees multiple objects simultaneously
  - Good for spatial planning

### 2. Wrist Camera (Pre-Grasp Inspection)
- **Location**: Mounted on robot gripper (end-effector link 11)
- **Purpose**: Close-up classification before grasping
- **FOV**: 60° focused view
- **Advantages**:
  - High-resolution close-ups
  - Can verify object identity
  - Detailed classification for sorting decisions

## 🎯 Workflow Steps

```
┌─────────────────────────────────────────────────────┐
│ 1. SCAN WORKSPACE (Overhead Camera)                │
│    ├─ Capture scene image                          │
│    ├─ Detect all objects with VLM                  │
│    └─ Get world coordinates for each object        │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ 2. FOR EACH DETECTED OBJECT:                       │
│    ├─ Move arm above object (+20cm)                │
│    ├─ Capture close-up (Wrist Camera)              │
│    ├─ Classify with VLM                            │
│    │  └─ Get: type (cube/sphere/cylinder)          │
│    │       color (red/blue/green/yellow)           │
│    ├─ Determine target container                   │
│    ├─ Move down and grasp                          │
│    ├─ Lift object (+30cm)                          │
│    ├─ Move to target container                     │
│    └─ Release object                               │
└─────────────────────────────────────────────────────┘
```

## 📦 Sorting Containers

Four color-coded bins are placed at fixed positions:

| Container | Position (X, Y, Z) | Sorts |
|-----------|-------------------|-------|
| `red_bin` | (-0.3, 0.5, 0.1) | Red objects |
| `blue_bin` | (-0.3, -0.5, 0.1) | Blue objects |
| `green_bin` | (-0.6, 0.5, 0.1) | Green objects |
| `yellow_bin` | (-0.6, -0.5, 0.1) | Yellow objects |

## 🚀 Running the Simulation

### Prerequisites
```bash
# Activate environment
conda activate bullet39

# Ensure you have the API key set
export GOOGLE_AI_API_KEY="your-api-key-here"
# or add to .env file
```

### Run Pick-and-Sort
```bash
cd /Users/mateuszwasko/Desktop/iUROP/High-Fidelity_Digital_Twin_For_Multi-Robot_Wireless_Communication
python main.py
```

### What to Expect
1. **Initialization**: Creates robot, cameras, containers, and random objects
2. **Physics Settling**: 1 second for objects to settle
3. **Overhead Scan**: Detects all objects in workspace
4. **Pick-and-Sort Loop**: Processes each object:
   - Approaches object
   - Takes wrist camera photo
   - Classifies with VLM
   - Picks and sorts to appropriate bin
5. **Results View**: Simulation stays open to inspect sorted objects

## 🎛️ Key Classes

### `RobotArmSimulation`
Main simulation controller with new methods:
- `add_sorting_containers()` - Creates colored bins
- `capture_wrist_view()` - Gets wrist camera image
- `classify_object_with_wrist_camera()` - VLM classification
- `determine_target_container()` - Sorting logic
- `pick_and_sort_workflow()` - Complete workflow orchestration

### `OverheadCamera`
Fixed camera for workspace survey
- Position: [0.5, 0.0, 1.5]
- Target: [0.5, 0.0, 0]
- FOV: 60°

### `WristCamera`
Mobile camera on robot gripper
- Attached to: End-effector link 11
- Offset: 5cm above gripper
- Viewing distance: 15cm

## 🔧 Motion Control

Current implementation uses **placeholder motion** (simulated delays). For production:

### Recommended Improvements
1. **Inverse Kinematics**: Use PyBullet IK solver
   ```python
   joint_poses = p.calculateInverseKinematics(robot_id, end_effector_link, target_pos)
   ```

2. **Trajectory Planning**: Use RRT/PRM for collision-free paths

3. **Gripper Control**: Implement actual joint control
   ```python
   p.setJointMotorControl2(robot_id, finger_joint, p.POSITION_CONTROL, targetPosition=0.04)
   ```

## 📊 Performance Considerations

### VLM Processing Times
- **Overhead scan**: ~10-15 seconds (4 objects)
- **Wrist classification**: ~8-12 seconds per object
- **Total workflow**: ~1-2 minutes for 4 objects

### Optimization Options
1. **Reduce image resolution**: 640x480 → 320x240
2. **Batch processing**: Classify multiple objects in one VLM call
3. **Local detector**: Use OWL-ViT for bounding boxes (see earlier discussion)
4. **Cache results**: Skip re-classification if object hasn't moved

## 🐛 Troubleshooting

### "Wrist camera not initialized"
- Ensure robot is loaded before camera setup
- Check `end_effector_link=11` is correct for Franka Panda

### "VLM classification failed"
- Verify API key is set correctly
- Check internet connection
- Ensure wrist camera has clear view of object

### "Object not reachable"
- Objects spawn in X(0-1), Y(-0.5-0.5) - check robot workspace overlap
- Adjust `min_radius` and `max_radius` in `generate_random_position()`

### Coordinate Accuracy Issues
- Y-coordinates should be accurate within ~2cm
- X-coordinates may have offset - see calibration notes in `vlm_analysis.py`
- Z-coordinates (height) require proper depth buffer processing

## 🔮 Future Enhancements

### Immediate
- [ ] Add real IK-based motion planning
- [ ] Implement actual gripper control
- [ ] Add collision avoidance

### Advanced
- [ ] Replace VLM with OWL-ViT + CLIP for faster detection
- [ ] Add multi-view classification (combine overhead + wrist views)
- [ ] Implement object tracking across frames
- [ ] Add force sensing for grasp verification
- [ ] Support complex object shapes (non-primitive)

### Production
- [ ] Real-world camera calibration
- [ ] ROS integration for real robot control
- [ ] Error recovery and retry logic
- [ ] Performance monitoring and logging
- [ ] Multi-robot coordination

## 📝 Configuration

### Customize Sorting Rules
Edit `determine_target_container()` in `robot_arm_simulation.py`:
```python
def determine_target_container(self, classification: Dict) -> str:
    color = classification.get('color', 'unknown').lower()
    obj_type = classification.get('type', 'unknown').lower()
    
    # Custom sorting logic
    if 'fragile' in classification.get('attributes', []):
        return 'fragile_bin'
    elif obj_type == 'sphere':
        return 'round_objects_bin'
    # ... more rules
```

### Add More Containers
Edit `add_sorting_containers()`:
```python
container_positions = {
    'fragile_bin': [-0.9, 0.0, 0.1],
    'heavy_bin': [-0.9, 0.5, 0.1],
    # ... more bins
}
```

## 📚 Related Documentation
- `README.md` - Main project overview
- `vlm_analysis.py` - VLM integration details
- `environment.yml` - Dependencies

## 🤝 Contributing
When adding features:
1. Keep dual-camera architecture
2. Maintain workflow modularity
3. Add timing measurements for performance tracking
4. Update this guide with new capabilities

---

**Status**: ✅ Functional prototype  
**Next Priority**: Real IK-based motion control  
**Performance**: ~1-2 min for 4 objects
