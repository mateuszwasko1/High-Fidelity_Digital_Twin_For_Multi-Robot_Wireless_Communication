"""Performance benchmarking for the PyBullet sorting demo.

This script runs repeated simulations under configurable planner presets and collects:

1. Average cycle time (ms)
2. Trajectory deviation (m)
3. Success rate (%)

Each configuration executes multiple trials (default: 10). Metrics are
summarised per configuration and printed to stdout in the same tabular layout
as the project report. The cloud presets optionally add network latency jitter,
perception drift, and misclassification noise so the impact of delayed sensing
is visible in the aggregated metrics. By default the script evaluates the three
baseline planners shown in Table 1: ``Local (IK)``, ``Cloud (Simulated 100 ms
Delay)`` and ``Cloud (Simulated 200 ms Delay)``. Optionally, results can be
written to JSON via ``--output``.

Usage
-----
```bash
conda run -n bullet39 python experiments/performance_benchmark.py \
    --trials 10 \
    --configs local cloud_100 cloud_200 \
    --output benchmark_results.json
```

For real cloud inference, set the environment variable ``CLOUD_VLM_ENDPOINT``
to point at an HTTP endpoint that accepts base64-encoded PNG images and returns
JSON with ``name``, ``confidence`` and ``shape`` fields. If authentication is
required, set ``CLOUD_VLM_API_KEY`` as well. The payload format used here is::

    {
        "image_base64": "...",
        "metadata": {"source": "performance_benchmark"}
    }

Adjust ``CloudVLMDetector`` if your service expects a different schema.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import statistics
import time
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from PIL import Image

from tqdm.auto import tqdm

# Allow imports from src/ when running as a standalone script.
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from simulation_setup import (
    OverheadCamera,
    create_sorting_containers,
    initialize_physics,
    initialize_robot_gripper,
    load_environment,
    spawn_debug_grid_objects,
)
from object_finder import find_closest_object
from vlm_detector import VLMDetector, create_detector
from arm_movement import ArmController


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------


class CloudVLMDetector(VLMDetector):
    """Client for a remote VLM inference endpoint."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None, timeout: float = 30.0):
        super().__init__()
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def load_model(self):
        # Nothing to preload for HTTP calls, but maintain symmetry.
        self.loaded = True

    def detect_object(self, image: np.ndarray) -> Dict:
        import requests

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "image_base64": encoded,
            "metadata": {"source": "performance_benchmark"},
        }

        response = requests.post(
            self.endpoint,
            data=json.dumps(payload),
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Normalise keys with sensible defaults.
        return {
            "name": data.get("name", "remote_object"),
            "confidence": float(data.get("confidence", 0.0)),
            "shape": data.get("shape", "unknown"),
            "color": data.get("color", "unknown"),
        }


class LatencyInjectedDetector(VLMDetector):
    """Wrap another detector and simulate latency, jitter, and perception errors."""

    def __init__(
        self,
        base: VLMDetector,
        latency_ms: float,
        jitter_ms: float = 0.0,
        misclassification_rate: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.latency_mean_ms = max(0.0, latency_ms)
        self.latency_jitter_ms = max(0.0, jitter_ms)
        self.misclassification_rate = max(0.0, min(1.0, misclassification_rate))

    def load_model(self):
        if not getattr(self.base, "loaded", False):
            self.base.load_model()
        self.loaded = True

    def detect_object(self, image: np.ndarray) -> Dict:
        result = self.base.detect_object(image)

        simulated_latency_ms = max(
            0.0,
            random.gauss(self.latency_mean_ms, self.latency_jitter_ms),
        )
        if simulated_latency_ms > 0.0:
            time.sleep(simulated_latency_ms / 1000.0)

        misclassified = False
        failure_reason = ""
        if self.misclassification_rate > 0.0 and random.random() < self.misclassification_rate:
            misclassified = True
            failure_reason = "misclassification"
            plausible_shapes = ["sphere", "triangle", "rectangle", "unknown"]
            current_shape = (result.get("shape") or "unknown").lower()
            alternatives = [shape for shape in plausible_shapes if shape != current_shape]
            if alternatives:
                result["shape"] = random.choice(alternatives)
            else:
                result["shape"] = "unknown"
            result["confidence"] = float(max(0.05, result.get("confidence", 0.1) * 0.5))

        result["simulated_latency_ms"] = simulated_latency_ms
        result["simulated_misclassification"] = misclassified
        if failure_reason:
            result["simulated_failure_reason"] = failure_reason

        return result


@dataclass(frozen=True)
class BenchmarkConfigSpec:
    key: str
    label: str
    detector_mode: str
    planner_type: str = "ik"
    simulated_latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    drift_rate_m_per_s: float = 0.0
    misclassification_rate: float = 0.0


BENCHMARK_CONFIGS: Dict[str, BenchmarkConfigSpec] = {
    "local": BenchmarkConfigSpec(
        key="local",
        label="Local (IK)",
        detector_mode="local",
    ),
    "local_ik": BenchmarkConfigSpec(
        key="local_ik",
        label="Local (IK)",
        detector_mode="local",
    ),
    "cloud": BenchmarkConfigSpec(
        key="cloud",
        label="Cloud (HTTP)",
        detector_mode="cloud",
    ),
    "cloud_100": BenchmarkConfigSpec(
        key="cloud_100",
        label="Cloud (Simulated 100 ms Delay)",
        detector_mode="local",
        simulated_latency_ms=100.0,
        latency_jitter_ms=35.0,
        drift_rate_m_per_s=0.18,
        misclassification_rate=0.10,
    ),
    "cloud_100ms": BenchmarkConfigSpec(
        key="cloud_100ms",
        label="Cloud (Simulated 100 ms Delay)",
        detector_mode="local",
        simulated_latency_ms=100.0,
        latency_jitter_ms=35.0,
        drift_rate_m_per_s=0.18,
        misclassification_rate=0.10,
    ),
    "cloud_200": BenchmarkConfigSpec(
        key="cloud_200",
        label="Cloud (Simulated 200 ms Delay)",
        detector_mode="local",
        simulated_latency_ms=200.0,
        latency_jitter_ms=50.0,
        drift_rate_m_per_s=0.25,
        misclassification_rate=0.18,
    ),
    "cloud_200ms": BenchmarkConfigSpec(
        key="cloud_200ms",
        label="Cloud (Simulated 200 ms Delay)",
        detector_mode="local",
        simulated_latency_ms=200.0,
        latency_jitter_ms=50.0,
        drift_rate_m_per_s=0.25,
        misclassification_rate=0.18,
    ),
    "cloud_300": BenchmarkConfigSpec(
        key="cloud_300",
        label="Cloud (Simulated 300 ms Delay)",
        detector_mode="local",
        simulated_latency_ms=300.0,
        latency_jitter_ms=70.0,
        drift_rate_m_per_s=0.32,
        misclassification_rate=0.25,
    ),
    "cloud_600": BenchmarkConfigSpec(
        key="cloud_600",
        label="Cloud (Simulated 600 ms Delay)",
        detector_mode="local",
        simulated_latency_ms=600.0,
        latency_jitter_ms=120.0,
        drift_rate_m_per_s=0.45,
        misclassification_rate=0.35,
    ),
}


def build_detector(config: BenchmarkConfigSpec) -> VLMDetector:
    detector_mode = config.detector_mode.lower()
    if detector_mode == "local":
        detector = create_detector("clip")
    elif detector_mode == "cloud":
        endpoint = os.getenv("CLOUD_VLM_ENDPOINT")
        if not endpoint:
            raise RuntimeError(
                "CLOUD_VLM_ENDPOINT must be set for the 'cloud' detector configuration."
            )
        api_key = os.getenv("CLOUD_VLM_API_KEY")
        detector = CloudVLMDetector(endpoint=endpoint, api_key=api_key)
    else:
        raise ValueError(f"Unknown detector mode '{detector_mode}'.")

    detector.load_model()

    if (
        config.simulated_latency_ms > 0.0
        or config.latency_jitter_ms > 0.0
        or config.misclassification_rate > 0.0
    ):
        latency_wrapper = LatencyInjectedDetector(
            detector,
            latency_ms=config.simulated_latency_ms,
            jitter_ms=config.latency_jitter_ms,
            misclassification_rate=config.misclassification_rate,
        )
        latency_wrapper.load_model()
        return latency_wrapper

    return detector


# ---------------------------------------------------------------------------
# Metrics data structures
# ---------------------------------------------------------------------------


@dataclass
class CycleMetrics:
    cycle_time_ms: float
    trajectory_deviation_m: float
    success: bool
    detector_latency_ms: float
    failure_reason: str = ""


@dataclass
class TrialMetrics:
    cycles: List[CycleMetrics] = field(default_factory=list)

    @property
    def attempts(self) -> int:
        return len(self.cycles)

    @property
    def successes(self) -> int:
        return sum(1 for c in self.cycles if c.success)

    def summary(self) -> Dict[str, float]:
        if not self.cycles:
            return {
                "avg_cycle_time_ms": 0.0,
                "avg_deviation_m": 0.0,
                "success_rate": 0.0,
                "avg_detector_latency_ms": 0.0,
                "failure_counts": {},
            }
        avg_cycle = statistics.mean(c.cycle_time_ms for c in self.cycles)
        avg_dev = statistics.mean(c.trajectory_deviation_m for c in self.cycles)
        avg_latency = statistics.mean(c.detector_latency_ms for c in self.cycles)
        success_rate = (self.successes / self.attempts) * 100 if self.attempts else 0.0
        failure_counts = Counter(c.failure_reason for c in self.cycles if c.failure_reason)
        return {
            "avg_cycle_time_ms": avg_cycle,
            "avg_deviation_m": avg_dev,
            "success_rate": success_rate,
            "avg_detector_latency_ms": avg_latency,
            "failure_counts": dict(failure_counts),
        }


@dataclass
class ConfigResult:
    name: str
    trials: List[TrialMetrics] = field(default_factory=list)

    def aggregate(self) -> Dict[str, float]:
        if not self.trials:
            return {
                "avg_cycle_time_ms": 0.0,
                "std_cycle_time_ms": 0.0,
                "avg_deviation_m": 0.0,
                "std_deviation_m": 0.0,
                "success_rate": 0.0,
                "std_success_rate": 0.0,
                "avg_detector_latency_ms": 0.0,
                "std_detector_latency_ms": 0.0,
                "failure_counts": {},
            }

        cycle_values = [c.cycle_time_ms for t in self.trials for c in t.cycles]
        deviation_values = [c.trajectory_deviation_m for t in self.trials for c in t.cycles]
        successes = sum(t.successes for t in self.trials)
        attempts = sum(t.attempts for t in self.trials)
        latency_values = [c.detector_latency_ms for t in self.trials for c in t.cycles]
        failure_counts = Counter(
            c.failure_reason for t in self.trials for c in t.cycles if c.failure_reason
        )

        def _mean_std(values: List[float]) -> Tuple[float, float]:
            if not values:
                return 0.0, 0.0
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            return mean_val, std_val

        cycle_mean, cycle_std = _mean_std(cycle_values)
        deviation_mean, deviation_std = _mean_std(deviation_values)
        latency_mean, latency_std = _mean_std(latency_values)
        success_rate = (successes / attempts) * 100 if attempts else 0.0
        success_std = 0.0
        if attempts > 1:
            p_hat = successes / attempts
            success_std = math.sqrt(max(p_hat * (1 - p_hat), 0.0) / attempts) * 100

        return {
            "avg_cycle_time_ms": cycle_mean,
            "std_cycle_time_ms": cycle_std,
            "avg_deviation_m": deviation_mean,
            "std_deviation_m": deviation_std,
            "success_rate": success_rate,
            "std_success_rate": success_std,
            "avg_detector_latency_ms": latency_mean,
            "std_detector_latency_ms": latency_std,
            "failure_counts": dict(failure_counts),
        }

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "aggregate": self.aggregate(),
            "trials": [t.summary() for t in self.trials],
        }


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


BIN_REGIONS: Sequence[Tuple[float, float, float, float]] = (
    (0.05, 0.25, -0.65, -0.45),
    (0.35, 0.55, -0.65, -0.45),
    (0.05, 0.25, 0.45, 0.65),
    (0.35, 0.55, 0.45, 0.65),
)

DROP_LOCATIONS: Dict[str, Tuple[float, float, float]] = {
    "sphere_bin": (0.15, -0.55, 0.35),
    "triangle_bin": (0.45, -0.55, 0.35),
    "rectangle_bin": (0.15, 0.55, 0.35),
    "mixed_bin": (0.45, 0.55, 0.35),
}

SHAPE_TO_BIN: Dict[str, str] = {
    "sphere": "sphere_bin",
    "triangle": "triangle_bin",
    "rectangle": "rectangle_bin",
}


def move_and_measure(controller: ArmController, target_pos: Tuple[float, float, float], target_orientation=None) -> Tuple[bool, float]:
    ok = controller.move_to_position(target_pos, target_orientation)
    current_pos, _ = controller.motion_planner.get_current_ee_pose()
    deviation = float(np.linalg.norm(np.array(target_pos) - np.array(current_pos)))
    return ok, deviation


def perform_single_pick(
    detector: VLMDetector,
    camera: OverheadCamera,
    robot_id: int,
    controller: ArmController,
    exclude_ids: Iterable[int],
    failed_positions: List[Tuple[float, float, float]],
    config: BenchmarkConfigSpec,
) -> Optional[CycleMetrics]:
    detection_start = time.perf_counter()

    closest_object = find_closest_object(
        camera,
        robot_id,
        camera_height=camera.camera_height,
        exclude_ids=list(exclude_ids),
        exclude_regions=BIN_REGIONS,
        exclude_positions=failed_positions,
        debug=False,
    )

    if not closest_object or closest_object.image is None:
        return None

    # Reject objects behind the robot base.
    if closest_object.location[0] < 0.0:
        failed_positions.append(closest_object.location)
        return None

    classification = detector.detect_object(closest_object.image)
    shape = (classification.get("shape") or "unknown").lower()
    detector_latency_ms = float(classification.get("simulated_latency_ms", 0.0))
    failure_reason = classification.get("simulated_failure_reason", "")

    if failure_reason:
        failed_positions.append(tuple(float(x) for x in closest_object.location))
        cycle_time_ms = (time.perf_counter() - detection_start) * 1000.0
        return CycleMetrics(
            cycle_time_ms=cycle_time_ms,
            trajectory_deviation_m=0.0,
            success=False,
            detector_latency_ms=detector_latency_ms,
            failure_reason=failure_reason,
        )

    cycle_deviation = 0.0
    success = True
    drift_applied = False

    controller.move_to_ready_position()
    controller.open_gripper()

    object_pos = tuple(float(x) for x in closest_object.location)
    approach_height = 0.10
    approach_pos = (object_pos[0], object_pos[1], object_pos[2] + approach_height)

    if config.drift_rate_m_per_s > 0.0 and detector_latency_ms > 0.0:
        drift_magnitude = config.drift_rate_m_per_s * (detector_latency_ms / 1000.0)
        if drift_magnitude > 0.0:
            drift_applied = True
            angle = random.uniform(0.0, 2.0 * math.pi)
            object_pos = (
                object_pos[0] + drift_magnitude * math.cos(angle),
                object_pos[1] + drift_magnitude * math.sin(angle),
                object_pos[2],
            )
            approach_pos = (object_pos[0], object_pos[1], object_pos[2] + approach_height)

    ok, deviation = move_and_measure(controller, approach_pos)
    cycle_deviation += deviation
    if not ok:
        success = False
        failure_reason = failure_reason or "ik_failure"

    if success:
        grasp_pos = (object_pos[0], object_pos[1], object_pos[2])
        ok, deviation = move_and_measure(controller, grasp_pos)
        cycle_deviation += deviation
        if not ok:
            success = False
            failure_reason = failure_reason or "ik_failure"

    if success:
        for _ in range(24):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        grasp_ok = controller.close_gripper()
        grip_verified = controller.verify_grip()
        success = grasp_ok and grip_verified
        if not success:
            failure_reason = failure_reason or ("stale_pick" if drift_applied else "grip_failure")

    if success:
        lift_pos = (object_pos[0], object_pos[1], object_pos[2] + approach_height)
        ok, deviation = move_and_measure(controller, lift_pos)
        cycle_deviation += deviation
        if not ok:
            success = False
            failure_reason = failure_reason or "ik_failure"

    if success:
        target_bin = SHAPE_TO_BIN.get(shape)
        if target_bin is None:
            success = False
            failure_reason = failure_reason or "unknown_shape"
        else:
            drop_pos = DROP_LOCATIONS[target_bin]
            transit_pos = (drop_pos[0], drop_pos[1], max(drop_pos[2], lift_pos[2] + 0.05))

            ok, deviation = move_and_measure(controller, transit_pos)
            cycle_deviation += deviation
            if not ok:
                success = False
                failure_reason = failure_reason or "ik_failure"

            if success:
                ok, deviation = move_and_measure(controller, drop_pos)
                cycle_deviation += deviation
                if not ok:
                    success = False
                    failure_reason = failure_reason or "ik_failure"

    if success and classification.get("simulated_misclassification"):
        success = False
        failure_reason = failure_reason or "misclassification"

    if success:
        controller.open_gripper()
        for _ in range(72):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    else:
        controller.open_gripper()
        failed_positions.append(object_pos)

    controller.move_to_ready_position()

    cycle_time_ms = (time.perf_counter() - detection_start) * 1000.0
    if not success and not failure_reason:
        failure_reason = "execution_failure"
    return CycleMetrics(
        cycle_time_ms=cycle_time_ms,
        trajectory_deviation_m=cycle_deviation,
        success=success,
        detector_latency_ms=detector_latency_ms,
        failure_reason=failure_reason,
    )


def run_trial(config: BenchmarkConfigSpec, trial_index: int, seed: Optional[int] = None) -> TrialMetrics:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    physics_client = initialize_physics(gui_mode=False)
    _, robot_id = load_environment()
    initialize_robot_gripper(robot_id, [9, 10])
    camera = OverheadCamera()

    containers = create_sorting_containers()
    objects = spawn_debug_grid_objects()

    # Allow physics to settle.
    for _ in range(120):
        p.stepSimulation()

    detector = build_detector(config)
    controller = ArmController(
        robot_id,
        end_effector_link_index=11,
        gripper_joints=[9, 10],
        planner_type=config.planner_type,
        verbose=False,
    )

    container_ids: List[int] = []
    for item in containers.values():
        if isinstance(item["id"], list):
            container_ids.extend(item["id"])
        else:
            container_ids.append(item["id"])

    trial_metrics = TrialMetrics()
    failed_positions: List[Tuple[float, float, float]] = []
    max_attempts = 20
    target_successes = max(1, len(objects))
    success_count = 0

    attempts = 0
    while attempts < max_attempts:
        cycle_metrics = perform_single_pick(
            detector,
            camera,
            robot_id,
            controller,
            container_ids,
            failed_positions,
            config,
        )

        if cycle_metrics is None:
            attempts += 1
            continue

        trial_metrics.cycles.append(cycle_metrics)
        if cycle_metrics.success:
            success_count += 1
        attempts += 1

        if all(pos[2] < 0.05 for pos in failed_positions[-3:]):
            break

        if success_count >= target_successes:
            break

    p.disconnect(physics_client)
    return trial_metrics


def run_benchmark(configs: Sequence[BenchmarkConfigSpec], trials: int, seed: Optional[int] = None) -> List[ConfigResult]:
    results: List[ConfigResult] = []
    for config in tqdm(configs, desc="Planner configs", unit="config"):
        config_result = ConfigResult(name=config.label)
        trial_iter = tqdm(range(trials), desc=config.label, unit="trial", leave=False)
        for trial_idx in trial_iter:
            trial_seed = None if seed is None else seed + trial_idx
            trial_metrics = run_trial(config, trial_idx, seed=trial_seed)
            config_result.trials.append(trial_metrics)
            summary = trial_metrics.summary()
            trial_iter.set_postfix(
                attempts=trial_metrics.attempts,
                successes=trial_metrics.successes,
                cycle_ms=f"{summary['avg_cycle_time_ms']:.1f}",
                dev_m=f"{summary['avg_deviation_m']:.3f}",
                latency_ms=f"{summary['avg_detector_latency_ms']:.0f}",
            )
        results.append(config_result)
    return results


def print_summary(results: Sequence[ConfigResult]) -> None:
    headers = [
        "Planner Type",
        "Avg. Cycle Time (ms)",
        "Trajectory Deviation (m)",
        "Success Rate (%)",
    ]
    rows = []
    aggregates: List[Tuple[ConfigResult, Dict[str, object]]] = []
    for result in results:
        summary = result.aggregate()
        aggregates.append((result, summary))
        rows.append(
            [
                result.name,
                f"{summary['avg_cycle_time_ms']:.1f} ± {summary['std_cycle_time_ms']:.1f}",
                f"{summary['avg_deviation_m']:.3f} ± {summary['std_deviation_m']:.3f}",
                f"{summary['success_rate']:.1f} ± {summary['std_success_rate']:.1f}",
            ]
        )

    widths = [
        max(len(headers[idx]), max((len(row[idx]) for row in rows), default=0))
        for idx in range(len(headers))
    ]

    header_line = "  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    separator = "-" * len(header_line)

    print("Table 1: Baseline Performance Comparison")
    print(separator)
    print(header_line)
    print(separator)

    for row in rows:
        formatted = []
        for idx, cell in enumerate(row):
            if idx == 0:
                formatted.append(cell.ljust(widths[idx]))
            else:
                formatted.append(cell.rjust(widths[idx]))
        print("  ".join(formatted))
    print(separator)

    for result, summary in aggregates:
        print(
            f"{result.name} detector latency -> "
            f"{summary['avg_detector_latency_ms']:.1f} ± {summary['std_detector_latency_ms']:.1f} ms"
        )
        failure_counts = summary.get("failure_counts", {})
        if not failure_counts:
            continue
        total_failures = sum(failure_counts.values())
        breakdown = ", ".join(
            f"{reason}: {count} ({(count/total_failures)*100:.1f}%)"
            for reason, count in sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)
        )
        print(f"{result.name} failures -> {breakdown}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run performance benchmarks for the robot sorting simulation.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["local", "cloud_100", "cloud_200"],
        help="Planner presets to evaluate (default: local cloud_100 cloud_200).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per configuration (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed to make object placement reproducible.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write metrics as JSON.",
    )
    return parser.parse_args()


def resolve_config_keys(config_keys: Sequence[str]) -> List[BenchmarkConfigSpec]:
    if not config_keys:
        config_keys = ["local", "cloud_100", "cloud_200"]

    resolved: List[BenchmarkConfigSpec] = []
    available = ", ".join(sorted(BENCHMARK_CONFIGS))
    for key in config_keys:
        spec = BENCHMARK_CONFIGS.get(key.lower())
        if not spec:
            raise ValueError(f"Unknown configuration '{key}'. Available options: {available}")
        resolved.append(spec)
    return resolved


def main() -> None:
    args = parse_args()
    try:
        config_specs = resolve_config_keys(args.configs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results = run_benchmark(config_specs, args.trials, seed=args.seed)
    print_summary(results)

    if args.output:
        payload = {
            "configs": [result.to_dict() for result in results],
            "trials_per_config": args.trials,
            "detector_configs": [spec.key for spec in config_specs],
        }
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
