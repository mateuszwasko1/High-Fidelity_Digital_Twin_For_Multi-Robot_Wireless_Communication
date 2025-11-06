# High-Fidelity Digital Twin For Multi-Robot Wireless Communication

## Project Overview
This project implements an autonomous robotic sorting system using a Franka Panda robot arm in PyBullet simulation. The system uses vision-based object detection (OWL-ViT) to identify objects, intelligent IK-based manipulation to pick them up, and sorts them into color-coded bins.

### Key Features
- **Vision-based Detection**: Overhead camera with OWL-ViT model for object detection
- **Intelligent Grasping**: Horizontal gripper orientation for reliable pickup at any position
- **Color-based Sorting**: Automatically sorts objects into matching colored bins
- **Multi-object Processing**: Handles 5 objects (spheres, cubes, rectangles) in a single run
- **Collision Avoidance**: 40cm lift height ensures objects clear 30cm tall bins during transport

### System Components
- `pipeline.py` - Main execution pipeline
- `simulations/` - Core simulation modules:
  - `simulation_setup.py` - PyBullet environment, camera, objects, bins
  - `arm_movement.py` - IK control and gripper operations
  - `object_finder.py` - Depth-based object detection
  - `vlm_detector.py` - Vision-language model integration
  - `edge_alignment.py` - Edge detection for optimal grasp orientation

### Quick Start
1. Activate the environment: `conda activate bullet39`
2. Run the pipeline: `python pipeline.py`

The robot will:
1. Detect all objects in the workspace
2. Pick them up one by one using optimal grasp strategy
3. Sort them into color-matched bins (blue→blue_bin, red→red_bin, etc.)

---

## Environment Setup

### Install a dependency

Try and install everything with Conda, but if not possible pip is fine
```bash
conda install -c conda-forge <package-name> -y
```

# Export the environment
When something changes in the environment (like you add a new dependency) you need to update the environment file by pasting the following
Windows
```bash
conda env export --from-history | findstr /V "^prefix:" > environment.yml
```

Mac
```bash
conda env export --from-history | grep -v "^prefix:" > environment.yml
```

# Start / activate the virtual environment
Before you start working, activate the virtual env:
```bash
conda activate bullet39
```

# Exit / deactivate the virtual environment
```bash
conda deactivate
```

# Recreate the environment
When recreating the env for the first time run the following:
```bash
conda env create -f environment.yml
```

# Update the environment
After pulling from git, ensure that your env is up to date:
```bash
conda env update -n bullet39 --file environment.yml --prune
```



# Git Commands

In order to push to git

1. Add all the changes to the commit
```bash
git add .
```

2. Commit and add comment
```bash
git commit -m"COMMENT HERE"
```

3. Push to github
```bash
git push origin main
```


PULL FROM GIT
1. pull from github
```bash
git pull
```
2. update the environment
```bash
conda env update -n bullet39 --file environment.yml --prune
```

reset to last commit
1. reset to last commit
```bash
git reset --hard origin/main
```

## Running the System

### Main Pipeline
```bash
python pipeline.py
```

This will:
- Initialize PyBullet simulation with GUI
- Spawn 5 test objects in a grid pattern
- Create 4 colored sorting bins (red, blue, green, yellow)
- Autonomously pick and sort all objects into matching bins

### Project Structure
```
.
├── pipeline.py              # Main execution script
├── simulations/             # Core simulation modules
│   ├── simulation_setup.py  # Environment, objects, bins
│   ├── arm_movement.py      # Robot control & IK
│   ├── object_finder.py     # Object detection
│   ├── vlm_detector.py      # Vision-language model
│   └── edge_alignment.py    # Grasp optimization
├── environment.yml          # Conda environment
└── README.md               # This file
```

### Performance Metrics
- **Pickup Success Rate**: 100% (5/5 objects)
- **Sorting Accuracy**: 100% (all objects to correct bins)
- **Average IK Convergence**: ~100-150 steps per position
- **Transport Clearance**: 40cm (10cm safety margin above bins)