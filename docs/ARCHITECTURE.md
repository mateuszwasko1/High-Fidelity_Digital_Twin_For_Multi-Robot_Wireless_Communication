# Project architecture overview

This document explains how the codebase is structured and how data/control flow through the system so an AI (or a human reader) can reason about it quickly.

## Repository layout

- `pipeline.py` — main entrypoint that wires everything together (simulation, detection, planning, control)
- `src/`
  - `simulation_setup.py` — initializes PyBullet (plane + Franka Panda), camera, bins, and spawns objects
  - `object_finder.py` — finds the nearest object with depth clustering and returns a cropped RGB image + world (x,y,z)
  - `vlm_detector.py` — VLM interface + CLIP implementation to classify shape from the cropped image
  - `arm_movement/`
    - `controller.py` — high-level pick/move/drop behaviors; pluggable motion planner
    - `inverse_kinematics.py` — IK motion planner using PyBullet’s `calculateInverseKinematics`
    - `vlm_planner.py` — placeholder for future VLM-driven motion planning (not implemented)
- `environment.yml` — Conda environment spec
- `README.md` — setup and git workflow notes

## End-to-end flow (happy path)

1. `pipeline.py` starts the simulation in GUI mode using helpers from `simulation_setup.py`.
2. Four open-top bins are created at the edges of the workspace; five random objects are spawned on the table.
3. Each iteration:
   - The overhead camera captures RGB + depth; `object_finder.py` masks the robot/bins, segments the depth, and finds the closest object.
   - `object_finder.py` returns a cropped 224×224 RGB image of that object and its world coordinates (x,y,z).
   - `vlm_detector.py` (CLIP) classifies the crop as one of: sphere, triangle, rectangle (with confidence).
   - `arm_movement/controller.py` drives the arm via the IK planner to pick the object, lift, and drop it into the matching bin.
   - Failed picks are remembered (position blacklist) so detection avoids retrying very close spots.

## Key modules and responsibilities

- `simulation_setup.py`
  - Physics init: gravity, solver/substep tuning, debug visualizer settings
  - Camera: top-down (yaw=270°, pitch=-90°), FOV=60°, 640×480; converts PyBullet depth buffer to meters
  - World: plane + Franka Panda URDF; open-top bins (walls only) at Y=±0.55; spawns random objects (sphere/triangle/rectangle)
- `object_finder.py`
  - Builds a depth-based mask for objects (closer than floor by a margin)
  - Connected components to isolate candidates; for each, computes centroid, height, and world coordinates
  - Returns nearest object + a white-background crop (224×224) ready for VLM
- `vlm_detector.py`
  - Abstract detector interface
  - `CLIPDetector`: loads OpenAI CLIP (ViT-B/32), pre-encodes three descriptive prompts, and does image–text similarity to infer shape
- `arm_movement/controller.py`
  - High-level behaviors: move-to, open/close gripper with force feedback, pick routine, drop routine, ready pose
  - Uses `inverse_kinematics.py` to compute joint targets and executes them with position control

## Control/data flow diagram (Mermaid)

```mermaid
flowchart TD
    A[pipeline.py] --> B[initialize_physics / load_environment]
    A --> C[OverheadCamera]
    A --> D[create_sorting_containers]
    A --> E[spawn_debug_grid_objects]
    A --> F[create_detector(CLIP)]
    A --> G[ArmController(planner=IK)]

    subgraph Detection Loop
      C --> H[get_cropped_image(rgb, depth)]
      H --> I[find_closest_object]
      I --> J[cropped 224×224 + world (x,y,z)]
      J --> K[CLIPDetector.detect_object]
      K --> L{shape}
    end

    L -->|sphere/triangle/rectangle| M[select bin]
    J --> G
    G --> N[pick -> lift -> move -> drop]
```

## Notable design choices

- Speed and robustness: reduced solver iters/substeps; masked camera regions for bins; blacklist of failed positions
- CLIP label engineering: uses clear, shape-focused prompts rather than class names for better separation
- Planner pluggability: you can switch to a VLM planner in the future by implementing `vlm_planner.py`

## Environment expectations

- Conda env (`environment.yml`) should provide: Python 3.9, PyTorch (CPU OK), OpenCV, PyBullet, Pillow, numpy, tqdm; pip installs CLIP from GitHub and optional RL libs (gymnasium, SB3)
- Git must be on PATH for `git+https://github.com/openai/CLIP.git`

## Typical run

- Activate env: `conda activate bullet39`
- Start: `python pipeline.py`
- A PyBullet GUI window appears; the arm starts sorting after initialization
