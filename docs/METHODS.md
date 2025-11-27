# Methods

This document details the full methodology used by the project: simulation setup, sensing, perception, planning/control, and sorting policy. It closely matches the actual code so an external AI or reader can reconstruct the behavior without reading every line.

## 1. Simulation setup

- Engine: PyBullet (GUI by default for visual feedback)
- Robot: Franka Panda (fixed base)
- Ground: plane.urdf
- Gravity: (0, 0, -9.81)
- Performance tuning:
  - `numSolverIterations = 50` (down from ~150)
  - `numSubSteps = 1` (down from ~4)
  - Debug overlay disabled to reduce render overhead
- Real-time simulation disabled; manual stepping with a ~240 Hz loop when needed

### 1.1 Camera
- Overhead camera, top-down:
  - Height: 1.5 m above workspace
  - Yaw: 270° (robot at bottom of image)
  - Pitch: -90° (looking straight down)
  - FOV: 60°
  - Resolution: 640×480
- Returns: RGB, depth (non-linear buffer), segmentation mask
- Depth conversion: depth buffer → metric depth using the standard perspective conversion:
  - `z = (near * far) / (far - (far - near) * depth_buffer)`
  - Uses near=0.01, far=3.0

### 1.2 Workspace fixtures
- Four open-top bins (walls only, no bottom collision) at Y = ±0.55 (left, right); front-to-back arrangement:
  - Left side (Y=-0.55): sphere_bin (front), triangle_bin (back)
  - Right side (Y=+0.55): rectangle_bin (front), mixed_bin (back)
- Five random objects spawned on the table within a safe reachable area and with minimum spacing to avoid overlaps.

## 2. Perception pipeline

### 2.1 RGB+Depth acquisition
- Capture from overhead camera
- Mask robot and bin regions using segmentation IDs and a conservative image-edge mask (left/right margins where bins live) to reduce false detections.
- Produce:
  - RGB image with masked pixels set to white
  - Linear depth image (meters) with masked pixels as NaN

### 2.2 Object detection (depth clustering)
- Build a boolean mask of object pixels:
  - Valid (non-NaN) depth
  - Closer than the floor by a margin (objects stand 5–20 cm above the table)
- Morphological cleanup (close → open) to remove noise and merge small holes
- Connected components over the binary mask
- For each component:
  - Compute area threshold (skip tiny blobs)
  - Extract all depths, drop NaNs
  - Estimate object height from (max_depth - min_depth)
  - Compute centroid (cx,cy) and average depth; convert to world coordinates (x,y,z)
  - Skip if too close (< 8 cm) to a previously failed grasp position
- Choose the nearest object in the XY plane
- Extract a white-background crop around the object, padded and resized to 224×224 (optimal for CLIP speed/quality)

### 2.3 Shape classification (VLM / CLIP)
- Model: OpenAI CLIP ViT-B/32 (fast, accurate enough)
- Pre-encode three descriptive prompts:
  - "a perfect circle or sphere"
  - "a triangle with three corners and three sides"
  - "a rectangle with four corners and four sides"
- For a given crop:
  - Preprocess → image embedding → cosine similarity with pre-encoded text embeddings
  - Take argmax similarity; map description to shape {sphere, triangle, rectangle}
  - Confidence = softmax(similarity)
- Color classification is not performed by CLIP; currently reported as "unknown".

## 3. Planning and control

### 3.1 Motion planner (IK)
- Inverse kinematics via `p.calculateInverseKinematics` with Franka joint limits and an adaptive rest pose based on target direction.
- Output is joint targets for arm joints (gripper excluded), executed via POSITION_CONTROL with appropriate torque/velocity limits.

### 3.2 Gripper control
- Open: target 0.04 per finger (≈8 cm total opening), step the sim ~60 ticks
- Close: target 0.0; detect grasp success by motor force threshold and/or contact check on gripper links; step the sim to stabilize

### 3.3 Pick-and-place routine
1. Move to ready pose; open gripper
2. Move above the object by an approach height (default 0.15 m)
3. Descend to object z; stabilize a short time
4. Close gripper; verify grip (force/contact)
5. Lift back to approach height
6. Move to drop location (based on classified shape)
7. Open gripper to release; short settle; return to ready pose

## 4. Sorting policy
- Shape → bin mapping:
  - sphere → sphere_bin
  - triangle → triangle_bin
  - rectangle → rectangle_bin
  - unknown → mixed_bin
- Bins arranged front-to-back left/right so trajectories avoid crossing over the robot unnecessarily.

## 5. Performance considerations
- Reduced solver iterations and substeps to speed sim without noticeable instability
- Pre-encoded CLIP text embeddings avoid repeated tokenization/encoding
- Reduced camera resolution and efficient resize (INTER_LINEAR) for the object crop
- Avoid repeated attempts at previously failed positions; cap attempts per loop

## 6. Assumptions and calibration
- The depth-to-world conversion is approximate (ideal pinhole + known FOV/height); acceptable for sorting demo
- No explicit collision checking in IK (relies on reachable geometry and simple targets)
- Gripper force threshold tuned heuristically; grip success is additionally assumed if moderate forces are observed

## 7. Reproducibility
- Environment managed via Conda (`environment.yml`) plus pip section (CLIP via GitHub)
- Requires Git on PATH to install CLIP
- CPU-only operation is supported; GPU (CUDA) not required for the demo
