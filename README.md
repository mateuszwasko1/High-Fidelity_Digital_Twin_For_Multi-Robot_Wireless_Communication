# High-Fidelity Digital Twin For Multi-Robot Wireless Communication

## Project Overview
This project implements an autonomous robotic sorting system using a Franka Panda robot arm in PyBullet simulation. The system uses vision-based object detection (OWL-ViT) to identify objects, intelligent IK-based manipulation to pick them up, and sorts them into bins.

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
3. Sort them into shape specific bins

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

### Baseline Benchmark Experiment
Recreate the performance study (Table 1 in the report) with the following steps:

1. Activate the environment and move into the project folder
  ```bash
  conda activate bullet39
  cd "C:\\Users\\coenb\\Desktop\\IUROP project\\High-Fidelity_Digital_Twin_For_Multi-Robot_Wireless_Communication"
  ```
2. Run the benchmark with the default presets, using seed 42 for reproducibility
  ```bash
  python .\experiments\performance_benchmark.py --configs local cloud_100 cloud_200 --trials 10 --seed 42 --output baseline_results.json
  ```

- The script simulates 10 trials per configuration (Local IK, Cloud +100 ms, Cloud +200 ms). Cloud presets inject latency jitter, perception drift, and misclassification noise so delayed sensing produces measurable degradations.
- Expect roughly 30 minutes of runtime; tqdm progress bars show status per trial. Additional heavier-delay presets (e.g., `cloud_300`, `cloud_600`) are available if you want to probe the latency/success frontier.
- Results print as "Table 1: Baseline Performance Comparison" and are also saved to `baseline_results.json`.
- For a quicker sanity check, reduce the trials count (for example, `--trials 2`) or run one configuration at a time with `--configs local`.

  ### Running Extended Configuration Sweeps
  - Supply multiple planner presets via `--configs` to compare local sensing with simulated or real cloud inference. Accepted keys:
    - `local`, `local_ik` – baseline on-robot detector and IK planner
    - `cloud_100`, `cloud_200`, `cloud_300`, `cloud_600` – local detector wrapped with 100/200/300/600 ms latency, jitter, drift, and noise
    - `cloud` – real HTTP VLM endpoint (requires `CLOUD_VLM_ENDPOINT` and optional `CLOUD_VLM_API_KEY`)
  - Example sweep capturing mid and high latency tiers:
    ```bash
    python .\experiments\performance_benchmark.py --configs local cloud_200 cloud_600 --trials 10 --seed 42 --output extended_results.json
    ```
  - Keep `--seed` consistent to reuse object placements; adjust the seed for independent runs. Increase `--trials` for tighter confidence intervals (runtime scales linearly). For quick smoke tests, drop to `--trials 2`.

### Performance Metrics
- **Pickup Success Rate**: 100% (5/5 objects)
- **Sorting Accuracy**: 100% (all objects to correct bins)
- **Average IK Convergence**: ~100-150 steps per position
- **Transport Clearance**: 40cm (10cm safety margin above bins)
