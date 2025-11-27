# Future Work

Areas to extend and mature the system, grouped by theme.

## Perception & VLM
- Integrate color extraction (HSV dominant color) to augment shape classification.
- Add size estimation and semantic attributes (e.g., "small red sphere") for richer sorting criteria.
- Replace CLIP with multimodal Gemini or other VLM for combined shape+color detection and natural language queries.
- Implement a cache for repeated frames (low motion phases) to reduce redundant classification.

## Motion planning & control
- Implement `VLMMotionPlanner` for goal inference from language instructions ("place the triangle in the mixed bin").
- Add collision-aware planning (sampling-based or optimization-based) to avoid walls and bin edges.
- Introduce smooth trajectory generation (time parameterization, jerk minimization).
- Closed-loop grasp adjustment using real-time re-detection (move until center alignment).

## Reinforcement learning integration
- Use `stable-baselines3` + `gymnasium` wrapper to train policies for grasp robustness or bin ordering.
- Curriculum learning: start with fewer objects, increase complexity (occlusions, varied shapes).
- Reward shaping: successful pick, correct bin placement, collision penalties.

## Calibration & realism
- Calibrate camera intrinsics/extrinsics against PyBullet matrices; apply distortion if simulating real hardware.
- Add noise models (depth noise, RGB blur) to test robustness.
- Simulate gripper force limits and slip dynamics for more realistic failure cases.

## Performance optimization
- Batch multiple perception steps (process next object while arm is moving).
- Use a smaller network variant or quantization for faster classification.
- Headless mode + offscreen rendering for CI and training.

## Metrics & evaluation
- Log: grasp success rate, classification accuracy, mean pick cycle time, number of failed attempts.
- Visualization dashboard (TensorBoard) for run summaries and timelines.
- Benchmark scenarios: cluttered table, partial occlusions, closely spaced similar shapes.

## Tooling & maintainability
- Add automated tests (unit + integration) and GitHub Actions CI matrix (Python versions, OS).
- Modular config file (`config.yaml`) for camera parameters, bin positions, tuning constants.
- Structured logging instead of print statements (log levels, timestamps, JSON output option).

## User interaction
- CLI flags: `--headless`, `--num-objects`, `--disable-vlm`, `--planner=ik|vlm`, `--record-run`.
- Live keyboard commands: pause, re-detect, cycle camera overlays.
- Save dataset of object crops + metadata for offline model improvements.
