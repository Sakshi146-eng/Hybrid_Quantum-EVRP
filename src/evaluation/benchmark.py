"""
Benchmark result recording and comparative evaluation runner module.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.classical.solution import VRPSolution
from src.vrp.instance import VRPInstance


@dataclass
class BenchmarkResult:
    """
    Standardized benchmark result record for comparing algorithms (GA, ACO, SA, TS, RL-QAOA)
    on the same problem instances.
    """
    instance_name: str
    algorithm: str
    num_customers: int
    num_vehicles: int
    vehicle_capacity: float
    total_distance: float
    feasible: bool
    runtime_seconds: float
    routes: List[List[int]] = field(default_factory=list)
    vehicle_loads: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts result record to plain JSON-serializable dictionary."""
        return {
            "instance_name": self.instance_name,
            "algorithm": self.algorithm,
            "num_customers": self.num_customers,
            "num_vehicles": self.num_vehicles,
            "vehicle_capacity": float(self.vehicle_capacity) if self.vehicle_capacity is not None else float("inf"),
            "total_distance": float(round(self.total_distance, 4)),
            "feasible": bool(self.feasible),
            "runtime_seconds": float(round(self.runtime_seconds, 6)),
            "routes": [list(r) for r in self.routes],
            "vehicle_loads": [float(round(l, 4)) for l in self.vehicle_loads],
        }

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Saves or appends benchmark result to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = []
        if path.is_file() and path.stat().st_size > 0:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except json.JSONDecodeError:
                existing = []

        existing.append(self.to_dict())

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)


def record_benchmark_result(
    solution: VRPSolution,
    instance: VRPInstance,
    instance_name: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> BenchmarkResult:
    """
    Creates a BenchmarkResult record from a VRPSolution and VRPInstance.

    Args:
        solution (VRPSolution): Output solution from solver.
        instance (VRPInstance): Problem instance.
        instance_name (Optional[str]): Explicit name of instance (defaults to instance.raw_data name if available).
        save_path (Optional[Union[str, Path]]): Optional JSON file path to append/save result.

    Returns:
        BenchmarkResult: Formatted result record object.
    """
    if instance_name is None:
        if hasattr(instance, "raw_data") and isinstance(instance.raw_data, dict):
            instance_name = instance.raw_data.get("name", "unknown_instance")
        else:
            instance_name = "unknown_instance"

    result = BenchmarkResult(
        instance_name=instance_name,
        algorithm=solution.algorithm_name,
        num_customers=instance.num_customers,
        num_vehicles=instance.num_vehicles,
        vehicle_capacity=instance.vehicle_capacity if instance.vehicle_capacity is not None else float("inf"),
        total_distance=solution.total_distance,
        feasible=solution.feasible,
        runtime_seconds=solution.runtime_seconds,
        routes=solution.routes,
        vehicle_loads=solution.vehicle_loads,
    )

    if save_path is not None:
        result.save_json(save_path)

    return result


class BenchmarkRunner:
    """
    Orchestrates execution of benchmark experiments.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run_comparative_experiment(self) -> Dict[str, Any]:
        """
        Runs comparative benchmark across algorithms (e.g., GA, ACO, SA, TS, RL-QAOA).
        """
        raise NotImplementedError(
            "TODO: Full comparative benchmark runner to be implemented in future phases."
        )
