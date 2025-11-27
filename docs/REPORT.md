# High‑Fidelity Digital Twin for Multi‑Robot Wireless Communication

A consolidated technical report of the simulation, perception, and control stack.

## Abstract
This project implements a tabletop sorting pipeline in PyBullet using a Franka Panda arm, an overhead camera, and a lightweight vision‑language classification step. The camera captures RGB‑D views; a depth‑based segmentation isolates candidate objects, the nearest object is cropped, and OpenAI CLIP (ViT‑B/32) classifies the crop into sphere, triangle, or rectangle. An inverse‑kinematics (IK) motion planner drives the robot to pick and place objects into color‑coded bins. The system emphasizes modularity and speed: reduced physics solver effort, masked camera regions, pre‑encoded CLIP text embeddings, and a simple policy to avoid retrying failed positions. Results demonstrate a functional end‑to‑end loop suitable for experimentation and extensions (e.g., VLM‑driven planning, color detection, and RL). We also provide environment guidance and documentation to support reproducible runs and report generation.

## 1. Introduction
Robotic picking and sorting on a tabletop requires robust perception under top‑down viewpoints and reliable motion generation in clutter. Simulation accelerates iteration by enabling fast, repeatable experiments. This work assembles a compact, modular pipeline: (i) PyBullet for physics and rendering, (ii) a depth‑aware object detector from a single overhead camera, (iii) CLIP for shape classification with minimal training, and (iv) IK control for pick‑and‑place. We target rapid prototyping, clear extensibility, and reasonable performance on CPU‑only machines.

Contributions:
- A complete, runnable sorting pipeline (camera → detection → classification → pick/place)
- Depth‑based detection tailored for top‑down scenes and quick CLIP‑based shape inference
- A pluggable motion‑planning interface (IK today; VLM‑based planner scaffolded)
- Documentation and prompts to streamline report generation with external AI tools

## 2. System Architecture
Project layout and responsibilities:
- `pipeline.py`: Entry point that composes setup, perception, VLM classification, and control
- `src/simulation_setup.py`: Physics init, camera model, bin construction, object spawning
- `src/object_finder.py`: Depth‑masking + connected components to find the nearest object; returns a 224×224 crop and world (x,y,z)
- `src/vlm_detector.py`: VLM interface; `CLIPDetector` (ViT‑B/32) classifies shape from the crop
- `src/arm_movement/controller.py`: High‑level arm routines (move, pick, release, ready pose) using a motion planner
- `src/arm_movement/inverse_kinematics.py`: IK‑based planner using PyBullet’s solver
- `src/arm_movement/vlm_planner.py`: Placeholder for future VLM‑driven motion generation

A standalone Mermaid diagram is provided in `docs/architecture_diagram.mmd`.

## 3. Methods
### 3.1 Simulation Setup
- PyBullet with GUI, gravity = (0,0,-9.81), real‑time disabled
- Performance tuning: fewer solver iterations/substeps; debug overlay off
- Franka Panda URDF on a plane; open‑top bins (walls only) at Y = ±0.55
- Five random objects spawned with minimum spacing to avoid overlap

### 3.2 Camera & Depth Conversion
- Overhead camera: height 1.5 m, yaw 270°, pitch −90°, FOV 60°, 640×480
- Convert PyBullet depth buffer to meters via perspective formula with near/far planes
- Mask robot and bin regions (via segmentation + conservative image margins)

### 3.3 Object Detection via Depth Clustering
- Build binary mask of valid pixels closer than floor by a margin (objects ≈ 5–20 cm tall)
- Morphological close/open to reduce noise; connected components for object candidates
- For each component: compute centroid and average depth; convert to world (x,y,z); skip locations near prior failures
- Return nearest object and a 224×224 RGB crop (white background elsewhere)

### 3.4 Shape Classification with CLIP
- Model: ViT‑B/32 for speed/accuracy balance; pre‑encode three descriptive prompts:
  - “a perfect circle or sphere”
  - “a triangle with three corners and three sides”
  - “a rectangle with four corners and four sides”
- Compare crop embedding to text embeddings; take argmax similarity; output shape + confidence

### 3.5 Planning & Control (IK)
- Compute IK with Panda limits and an adaptive rest pose; execute joint targets with POSITION_CONTROL
- Gripper routine: open (≈8 cm), descend to grasp, close with force/contact check, lift, transit, release, return to ready

### 3.6 Sorting Policy
- Shape → bin mapping: sphere→sphere_bin, triangle→triangle_bin, rectangle→rectangle_bin, unknown→mixed_bin
- Bins arranged front/back at left/right edges to simplify trajectories

## 4. Results & Evaluation Plan
This repository focuses on the integrated pipeline rather than formal benchmarking. A suggested evaluation protocol:
- Grasp success rate: (# successful picks)/(# attempts)
- Classification accuracy: agreement between CLIP prediction and ground‑truth shape
- Cycle time: mean and variance per pick→place
- Robustness: performance with closely spaced objects and partial occlusions
Optional logs: TensorBoard scalars for timings and success rates; screenshots of camera/depth masks per run.

## 5. Limitations
- Depth‑based detection assumes objects are raised above the table; flat/transparent items may fail
- Coarse bin masking at image edges may hide valid pixels
- Pinhole camera approximation without explicit calibration → cm‑level pose errors possible
- No collision‑aware planning; gripper success detection is heuristic
- No persistent multi‑object tracking; limited failure recovery
- CLIP returns shapes but not colors (currently reported as “unknown”)
- Heavy deps (PyTorch, TorchVision); CLIP installed from Git—requires Git on PATH

## 6. Future Work
- Perception: color extraction (HSV), size estimation, confidence calibration, multimodal VLM (e.g., Gemini) for richer labels
- Planning: implement the VLM planner, add collision avoidance and smooth trajectories, closed‑loop alignment during grasp
- RL: wrap as a `gymnasium` env; train grasp/bin policies with `stable‑baselines3`
- Realism: camera calibration, noise models, improved gripper dynamics
- Performance: headless mode, offscreen rendering, batched perception
- Tooling: unit tests, CI, structured logging, centralized config

## 7. Environment & Reproducibility
- Conda env specified in `environment.yml` (Python 3.9; PyBullet, OpenCV, PyTorch, etc.)
- Pip section installs CLIP from GitHub; ensure Git is installed and on PATH
- Typical run: `conda activate bullet39` then `python pipeline.py` (PyBullet GUI appears)

## 8. Conclusion
We presented a compact, modular simulation pipeline that detects, classifies, and sorts tabletop objects in PyBullet. The design favors speed and clarity, serving as a strong baseline for experimentation. Several extensions—VLM‑driven motion planning, collision‑aware trajectories, color detection, and RL policies—are straightforward next steps and are outlined for future work.
