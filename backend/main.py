"""
FastAPI Backend Server for Quantum-Assisted EVRP.
Exposes CEC-2020 E-CVRP benchmark discovery, problem instance details, and Python GA solver execution.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.e_cvrp_loader import load_e_cvrp_instance, parse_e_cvrp_file
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.evaluation.benchmark import record_benchmark_result

# Resolve project root and data directory
ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT_DIR / "data" / "raw" / "e_cvrp"
RESULTS_DIR = ROOT_DIR / "results"

app = FastAPI(
    title="Quantum-Assisted EVRP API Server",
    description="FastAPI REST API bridging CEC-2020 E-CVRP Benchmark dataset & Python GA baseline to React dashboard UI.",
    version="1.0.0",
)

# Enable CORS for local Vite frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GARequest(BaseModel):
    """Request schema for running Genetic Algorithm on a benchmark instance."""
    instance_name: str = Field(..., example="E-n29-k4-s7.evrp")
    seed: Optional[int] = Field(42, example=42)
    population_size: Optional[int] = Field(100, example=100)
    generations: Optional[int] = Field(200, example=200)
    crossover_rate: Optional[float] = Field(0.85, example=0.85)
    mutation_rate: Optional[float] = Field(0.25, example=0.25)


def _resolve_instance_path(instance_name: str) -> Path:
    """Helper to locate .evrp instance file inside BENCHMARK_DIR."""
    if not instance_name.endswith(".evrp"):
        file_name = f"{instance_name}.evrp"
    else:
        file_name = instance_name

    target_path = BENCHMARK_DIR / file_name

    if not target_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark instance '{instance_name}' not found at path {target_path}. Please check data/raw/e_cvrp/ directory."
        )

    return target_path


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/benchmark/instances")
def list_benchmark_instances() -> Dict[str, List[Dict[str, Any]]]:
    """
    Discovers and lists all available .evrp benchmark files in data/raw/e_cvrp/.
    """
    if not BENCHMARK_DIR.is_dir():
        return {"instances": []}

    evrp_files = sorted(BENCHMARK_DIR.glob("*.evrp"))
    instances_list = []

    for file_path in evrp_files:
        try:
            parsed = parse_e_cvrp_file(file_path)
            instances_list.append({
                "name": parsed["name"] if parsed["name"].endswith(".evrp") else f"{parsed['name']}.evrp",
                "customers": parsed["dimension"] - 1 - parsed["num_stations"],
                "vehicles": parsed["num_vehicles"],
                "vehicle_capacity": parsed["vehicle_capacity"],
                "optimal_value": parsed["optimal_value"],
                "stations": parsed["num_stations"],
            })
        except Exception:
            # Fallback basic file info if parse fails
            instances_list.append({
                "name": file_path.name,
                "customers": 0,
                "vehicles": 0,
                "vehicle_capacity": 0.0,
            })

    return {"instances": instances_list}


@app.get("/api/benchmark/instances/{instance_name}")
def get_benchmark_instance_details(instance_name: str) -> Dict[str, Any]:
    """
    Returns full details for a specific benchmark instance (depot, customers, capacity, EV metadata).
    """
    file_path = _resolve_instance_path(instance_name)
    instance = load_e_cvrp_instance(file_path)

    raw_meta = getattr(instance, "raw_data", {})
    depot_coord = instance.coordinates[0]

    customers_data = []
    for i in range(1, instance.num_nodes):
        coord = instance.coordinates[i]
        demand = float(instance.demands[i])
        customers_data.append({
            "id": i,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "demand": demand,
        })

    # EV charging stations metadata (for informational display only)
    charging_stations = []
    station_nids = raw_meta.get("station_nodes", [])
    node_coords = raw_meta.get("node_coords", {})
    for st_id in station_nids:
        if st_id in node_coords:
            c = node_coords[st_id]
            charging_stations.append({"id": st_id, "x": float(c[0]), "y": float(c[1])})

    tot_demand = float(instance.demands.sum())

    return {
        "name": raw_meta.get("name", file_path.name),
        "num_customers": instance.num_customers,
        "num_vehicles": instance.num_vehicles,
        "vehicle_capacity": instance.vehicle_capacity,
        "total_demand": tot_demand,
        "optimal_value": raw_meta.get("optimal_value"),
        "energy_capacity": raw_meta.get("energy_capacity"),
        "energy_consumption": raw_meta.get("energy_consumption"),
        "num_stations": raw_meta.get("num_stations", len(charging_stations)),
        "depot": {
            "id": 0,
            "x": float(depot_coord[0]),
            "y": float(depot_coord[1]),
        },
        "customers": customers_data,
        "charging_stations": charging_stations,
    }


@app.post("/api/benchmark/ga")
def run_benchmark_ga(req: GARequest) -> Dict[str, Any]:
    """
    Loads specified .evrp benchmark instance, runs Python GeneticAlgorithmSolver,
    records results, and returns complete solution data for frontend visualization.
    """
    file_path = _resolve_instance_path(req.instance_name)
    instance = load_e_cvrp_instance(file_path)

    solver = GeneticAlgorithmSolver(
        population_size=req.population_size or 100,
        generations=req.generations or 200,
        crossover_rate=req.crossover_rate or 0.85,
        mutation_rate=req.mutation_rate or 0.25,
        seed=req.seed if req.seed is not None else 42,
    )

    solution = solver.solve(instance)

    # Record benchmark result to JSON
    out_file = RESULTS_DIR / "ga_benchmark_results.json"
    record_benchmark_result(
        solution=solution,
        instance=instance,
        instance_name=req.instance_name,
        save_path=out_file
    )

    depot_coord = instance.coordinates[0]
    customers_data = []
    for i in range(1, instance.num_nodes):
        coord = instance.coordinates[i]
        demand = float(instance.demands[i])
        customers_data.append({
            "id": i,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "demand": demand,
        })

    # Format routes list
    formatted_routes = []
    for idx, r in enumerate(solution.routes, start=1):
        load = solution.vehicle_loads[idx - 1] if idx - 1 < len(solution.vehicle_loads) else 0.0
        formatted_routes.append({
            "vehicle": idx,
            "route": [int(node) for node in r],
            "load": float(round(load, 2)),
        })

    raw_meta = getattr(instance, "raw_data", {})
    station_nids = raw_meta.get("station_nodes", [])
    node_coords = raw_meta.get("node_coords", {})
    charging_stations = [{"id": s, "x": float(node_coords[s][0]), "y": float(node_coords[s][1])} for s in station_nids if s in node_coords]

    return {
        "instance_name": req.instance_name,
        "algorithm": solution.algorithm_name,
        "num_customers": instance.num_customers,
        "num_vehicles": instance.num_vehicles,
        "vehicle_capacity": instance.vehicle_capacity,
        "total_demand": float(instance.demands.sum()),
        "total_distance": float(round(solution.total_distance, 4)),
        "runtime_seconds": float(round(solution.runtime_seconds, 6)),
        "feasible": bool(solution.feasible),
        "depot": {
            "id": 0,
            "x": float(depot_coord[0]),
            "y": float(depot_coord[1]),
        },
        "customers": customers_data,
        "charging_stations": charging_stations,
        "routes": formatted_routes,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
