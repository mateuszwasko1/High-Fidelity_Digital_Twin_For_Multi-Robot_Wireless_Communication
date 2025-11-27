# Limitations

This document lists current technical and architectural limitations so they can be acknowledged in reports and guide future work.

## Perception
- Depth-based segmentation assumes objects are elevated relative to the table by 5–20 cm; very flat or transparent objects may be missed.
- Bin masking uses coarse image-edge regions; can hide valid pixels near edges → potential false negatives.
- Camera model is approximate (pinhole + FOV); no intrinsic/extrinsic calibration against the simulated camera matrices → centimeter-level position errors possible.
- CLIP shape classification is robust but not infallible; top-down textures or occlusions can reduce confidence.
- Object color is not computed (currently reported as "unknown").

## Planning and control
- IK planner has no collision checking; certain poses could intersect walls/robot if commanded incorrectly.
- Gripper success detection relies on motor force/contacts; no tactile sensing; occasionally optimistic in edge cases.
- No path/time optimization beyond basic IK target tracking; trajectories may be suboptimal.

## System behavior
- No persistent multi-object tracking; each iteration redetects objects from scratch.
- Failure recovery is limited to skipping nearby positions and returning to ready pose.
- The loop stops after a fixed number of consecutive detection failures or a max attempts cap.

## Environment & portability
- CLIP is installed from GitHub (git+https); requires Git installed and accessible on PATH.
- Heavy dependencies (PyTorch, TorchVision) are version/OS specific; wrong Python version can break installs.
- GUI rendering is required for visual feedback; headless mode isn’t provided as a first-class CLI option yet.

## Evaluation & testing
- No formal unit tests (e.g., for `_image_to_world`, mask creation, or IK convergence checks).
- No benchmark metrics baked in (e.g., grasp success rate, classification accuracy, wall clock timing).

## Documentation
- Minimal in-code docstrings; a richer developer doc would help (state machines, tuning tips, calibration steps).
