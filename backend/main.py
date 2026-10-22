"""
FastAPI Backend Server for Quantum-Assisted EVRP.
Exposes CEC-2020 E-CVRP benchmark discovery, problem instance details, and Python GA solver execution.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.e_cvrp_loader import load_e_cvrp_instance, parse_e_cvrp_file, load_evrp_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.evaluation.benchmark import record_benchmark_result
from src.quantum.qubo import extract_sub_instance
from src.quantum.qaoa_solver import QAOAVRPSolver
from src.quantum.comparison import run_quantum_classical_comparison
from src.evrp.benchmark import (
    run_evrp_comparison,
    run_evrp_multi_seed_comparison,
    run_evrp_ablation_study,
)
from src.rl.agent import EVRLAgent

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


class QAOARequest(BaseModel):
    """Request schema for running Simulated QAOA on a benchmark sub-instance."""
    instance_name: str = Field(..., example="E-n29-k4-s7.evrp")
    customer_count: Optional[int] = Field(3, ge=2, le=4, example=3)
    customer_indices: Optional[List[int]] = Field(None, example=[1, 2, 3])
    p_depth: Optional[int] = Field(1, ge=1, le=5, example=1)
    shots: Optional[int] = Field(1024, ge=100, le=8192, example=1024)
    max_iter: Optional[int] = Field(50, ge=10, le=200, example=50)
    seed: Optional[int] = Field(42, example=42)


class CompareRequest(BaseModel):
    """Request schema for comparing Exact, GA, GA+LocalSearch, QAOA, and Hybrid GA-QAOA."""
    instance_name: str = Field(..., example="E-n29-k4-s7.evrp")
    customer_count: Optional[int] = Field(3, ge=2, le=8, example=3)
    customer_indices: Optional[List[int]] = Field(None, example=[1, 2, 3])
    p_depth: Optional[int] = Field(1, ge=1, le=5, example=1)
    shots: Optional[int] = Field(1024, ge=100, le=8192, example=1024)
    max_iter: Optional[int] = Field(50, ge=10, le=200, example=50)
    ga_population_size: Optional[int] = Field(50, ge=10, le=300, example=50)
    ga_generations: Optional[int] = Field(100, ge=10, le=500, example=100)
    hybrid_iterations: Optional[int] = Field(3, ge=1, le=10, example=3)
    seed: Optional[int] = Field(42, example=42)


class EVRPCompareRequest(BaseModel):
    """Request schema for full EVRP comparative evaluation."""
    instance_name: str = Field(..., example="E-n29-k4-s7.evrp")
    customer_count: Optional[int] = Field(4, ge=2, le=50, example=4)
    customer_indices: Optional[List[int]] = Field(None, example=[1, 2, 3, 4])
    algorithms: Optional[List[str]] = Field(None, example=["ga", "ga_ls", "rl", "rl_ls", "conventional_qaoa", "rl_qaoa", "rl_qaoa_ls", "full_hybrid"])
    battery_capacity: Optional[float] = Field(80.0, example=80.0)
    base_energy_rate: Optional[float] = Field(0.2, example=0.2)
    load_factor: Optional[float] = Field(0.2, example=0.2)
    charging_power: Optional[float] = Field(50.0, example=50.0)
    charging_price: Optional[float] = Field(0.25, example=0.25)
    seed: Optional[int] = Field(42, example=42)


class EVRPMultiSeedRequest(BaseModel):
    """Request schema for evaluating multiple random seeds and computing mean +/- std."""
    instance_name: str = Field("E-n29-k4-s7.evrp", example="E-n29-k4-s7.evrp")
    customer_count: Optional[int] = Field(4, ge=2, le=25, example=4)
    algorithms: Optional[List[str]] = Field(None, example=["ga", "ga_ls", "rl", "rl_ls", "conventional_qaoa", "rl_qaoa", "full_hybrid"])
    seeds: Optional[List[int]] = Field([42, 101, 2024], example=[42, 101, 2024])
    battery_capacity: Optional[float] = Field(80.0, example=80.0)
    base_energy_rate: Optional[float] = Field(0.2, example=0.2)
    load_factor: Optional[float] = Field(0.2, example=0.2)
    charging_power: Optional[float] = Field(50.0, example=50.0)
    charging_price: Optional[float] = Field(0.25, example=0.25)


class EVRPScalabilityRequest(BaseModel):
    """Request schema for customer-count scalability analysis."""
    instance_name: str = Field("E-n29-k4-s7.evrp", example="E-n29-k4-s7.evrp")
    customer_counts: Optional[List[int]] = Field([4, 8, 12], example=[4, 8, 12])
    algorithms: Optional[List[str]] = Field(["ga", "rl", "conventional_qaoa", "rl_qaoa", "full_hybrid"], example=["ga", "rl", "conventional_qaoa", "rl_qaoa", "full_hybrid"])
    battery_capacity: Optional[float] = Field(80.0, example=80.0)
    base_energy_rate: Optional[float] = Field(0.2, example=0.2)
    load_factor: Optional[float] = Field(0.2, example=0.2)
    charging_power: Optional[float] = Field(50.0, example=50.0)
    charging_price: Optional[float] = Field(0.25, example=0.25)
    seed: Optional[int] = Field(42, example=42)


class RLTrainRequest(BaseModel):
    """Request schema for training EVRP Reinforcement Learning agent in background."""
    instance_name: str = Field("E-n29-k4-s7.evrp", example="E-n29-k4-s7.evrp")
    customer_count: Optional[int] = Field(4, ge=2, le=10, example=4)
    episodes: Optional[int] = Field(30, ge=5, le=200, example=30)
    lr: Optional[float] = Field(0.001, example=0.001)
    gamma: Optional[float] = Field(0.99, example=0.99)
    seed: Optional[int] = Field(42, example=42)


# Background state tracking for RL Training
rl_training_state: Dict[str, Any] = {
    "status": "idle",
    "progress": 0.0,
    "current_episode": 0,
    "total_episodes": 0,
    "instance_name": None,
    "last_reward": 0.0,
    "history": [],
    "error_message": None,
    "checkpoint_exists": (RESULTS_DIR / "ev_rl_checkpoint.pt").is_file(),
}


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


@app.post("/api/benchmark/qaoa")
def run_benchmark_qaoa(req: QAOARequest) -> Dict[str, Any]:
    """
    Runs Simulated QAOA on a small sub-instance extracted from the benchmark dataset.
    Validates problem size (<= 4 customers -> 16 qubits) to avoid unbounded simulator memory.
    """
    file_path = _resolve_instance_path(req.instance_name)
    parent_instance = load_e_cvrp_instance(file_path)

    # Validate problem size
    num_cust = req.customer_count or 3
    if req.customer_indices is not None:
        num_cust = len(req.customer_indices)

    if num_cust < 2:
        raise HTTPException(
            status_code=400,
            detail="QAOA routing requires at least 2 customer nodes."
        )
    if num_cust > 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Simulator limit exceeded: Requested {num_cust} customers, which requires "
                f"{num_cust**2} qubits. The classical statevector simulator is limited to <= 4 "
                f"customers (16 qubits) to ensure safe, interactive response times."
            )
        )

    try:
        sub_instance = extract_sub_instance(
            parent_instance=parent_instance,
            customer_indices=req.customer_indices,
            customer_count=req.customer_count,
        )

        solver = QAOAVRPSolver(
            p_depth=req.p_depth or 1,
            shots=req.shots or 1024,
            max_iter=req.max_iter or 50,
            seed=req.seed if req.seed is not None else 42,
        )
        solution = solver.solve(sub_instance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QAOA execution failed: {str(e)}")

    depot_coord = sub_instance.coordinates[0]
    customers_data = []
    for i in range(1, sub_instance.num_nodes):
        coord = sub_instance.coordinates[i]
        customers_data.append({
            "id": i,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "demand": float(sub_instance.demands[i]),
        })

    formatted_routes = []
    for idx, r in enumerate(solution.routes, start=1):
        formatted_routes.append({
            "vehicle": idx,
            "route": [int(node) for node in r],
            "load": 0.0,
        })

    return {
        "instance_name": req.instance_name,
        "sub_instance_name": sub_instance.raw_data.get("name"),
        "algorithm": solution.algorithm_name,
        "num_customers": sub_instance.num_customers,
        "total_distance": float(round(solution.total_distance, 4)),
        "runtime_seconds": float(round(solution.runtime_seconds, 6)),
        "feasible": bool(solution.feasible),
        "depot": {"id": 0, "x": float(depot_coord[0]), "y": float(depot_coord[1])},
        "customers": customers_data,
        "routes": formatted_routes,
        "metadata": solution.metadata,
    }


@app.post("/api/benchmark/compare")
def run_benchmark_compare(req: CompareRequest) -> Dict[str, Any]:
    """
    Runs Exact Baseline, Classical GA, and Simulated QAOA on the exact same benchmark sub-instance.
    Computes comparative metrics, timing breakdowns, and optimality gaps.
    """
    file_path = _resolve_instance_path(req.instance_name)
    parent_instance = load_e_cvrp_instance(file_path)

    # Validate problem size
    num_cust = req.customer_count or 3
    if req.customer_indices is not None:
        num_cust = len(req.customer_indices)

    if num_cust < 2:
        raise HTTPException(
            status_code=400,
            detail="Comparison requires at least 2 customer nodes."
        )
    if num_cust > 8:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Evaluation limit exceeded: Requested {num_cust} customers. "
                f"Supported comparison range is 2 to 8 customers to maintain exact baseline calibration "
                f"and interactive quantum simulation."
            )
        )

    out_file = RESULTS_DIR / "quantum_classical_comparison.json"
    try:
        record = run_quantum_classical_comparison(
            instance=parent_instance,
            customer_indices=req.customer_indices,
            customer_count=req.customer_count,
            qaoa_params={
                "p_depth": req.p_depth or 1,
                "shots": req.shots or 1024,
                "max_iter": req.max_iter or 50,
            },
            ga_params={
                "population_size": req.ga_population_size or 50,
                "generations": req.ga_generations or 100,
            },
            hybrid_params={
                "max_hybrid_iterations": req.hybrid_iterations or 3,
            },
            seed=req.seed if req.seed is not None else 42,
            save_path=out_file,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparative evaluation failed: {str(e)}")

    return record


def _run_rl_training_background(req: RLTrainRequest):
    """Background worker for RL policy training."""
    global rl_training_state
    try:
        rl_training_state["status"] = "training"
        rl_training_state["instance_name"] = req.instance_name
        rl_training_state["progress"] = 0.0
        rl_training_state["current_episode"] = 0
        rl_training_state["total_episodes"] = req.episodes or 30
        rl_training_state["history"] = []
        rl_training_state["error_message"] = None

        file_path = _resolve_instance_path(req.instance_name)
        ev_inst = load_evrp_instance(
            file_path=file_path,
            customer_count=req.customer_count,
            seed=req.seed or 42,
        )

        agent = EVRLAgent(
            lr=req.lr or 0.001,
            gamma=req.gamma or 0.99,
            seed=req.seed or 42,
        )

        total_episodes = req.episodes or 30
        history = []
        for ep in range(1, total_episodes + 1):
            ep_metric = agent.train_step(ev_inst)
            history.append(ep_metric)
            rl_training_state["current_episode"] = ep
            rl_training_state["progress"] = round(ep / total_episodes, 2)
            rl_training_state["last_reward"] = round(ep_metric["reward"], 2)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = RESULTS_DIR / "ev_rl_checkpoint.pt"
        agent.save(ckpt_path)

        rl_training_state["status"] = "completed"
        rl_training_state["progress"] = 1.0
        rl_training_state["history"] = history
        rl_training_state["error_message"] = None
        rl_training_state["checkpoint_saved"] = True
        rl_training_state["checkpoint_exists"] = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        rl_training_state["status"] = "error"
        rl_training_state["error_message"] = f"{type(e).__name__}: {str(e)}"


@app.post("/api/evrp/rl/train")
def train_rl_model(req: RLTrainRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Triggers asynchronous RL policy training on an EVRP instance."""
    global rl_training_state
    if rl_training_state["status"] == "training":
        raise HTTPException(
            status_code=409,
            detail="RL training is already in progress. Please check /api/evrp/rl/status."
        )

    background_tasks.add_task(_run_rl_training_background, req)
    return {
        "message": f"Started RL training for {req.episodes} episodes on {req.instance_name}",
        "status": "training"
    }


@app.get("/api/evrp/rl/status")
def get_rl_training_status() -> Dict[str, Any]:
    """Returns current status, progress, and reward trajectory of background RL training."""
    global rl_training_state
    ckpt_path = RESULTS_DIR / "ev_rl_checkpoint.pt"
    rl_training_state["checkpoint_exists"] = ckpt_path.is_file()
    return rl_training_state


@app.post("/api/evrp/compare")
def compare_evrp_algorithms(req: EVRPCompareRequest) -> Dict[str, Any]:
    """
    Executes fair multi-algorithm comparison on a unified EVRP instance:
      - EV-GA (Classical baseline)
      - EV-GA + Local Search
      - RL (Reinforcement Learning)
      - RL + Local Search
      - RL + QAOA (Quantum-assisted local optimization)
      - Full Hybrid (RL + QAOA + EV Local Search)
    Returns complete metrics, route coordinates, and leg-by-leg battery traces.
    """
    file_path = _resolve_instance_path(req.instance_name)

    # Scale to matching CEC-2020 benchmark if customer_count exceeds small instances
    if req.customer_count is not None:
        if req.customer_count > 60 and ("E-n29" in req.instance_name or "E-n60" in req.instance_name or "E-n30" in req.instance_name or "E-n35" in req.instance_name):
            file_path = _resolve_instance_path("M-n110-k10-s9.evrp")
        elif req.customer_count > 28 and ("E-n29" in req.instance_name or "E-n30" in req.instance_name or "E-n35" in req.instance_name or "E-n37" in req.instance_name):
            file_path = _resolve_instance_path("E-n60-k5-s9.evrp")

    ev_inst = load_evrp_instance(
        file_path=file_path,
        customer_count=req.customer_count,
        customer_indices=req.customer_indices,
        battery_capacity=req.battery_capacity,
        energy_consumption_rate=req.base_energy_rate,
        load_factor=req.load_factor if req.load_factor is not None else 0.2,
        charging_power=req.charging_power if req.charging_power is not None else 50.0,
        charging_price=req.charging_price if req.charging_price is not None else 0.25,
        seed=req.seed if req.seed is not None else 42,
    )

    ckpt_path = RESULTS_DIR / "ev_rl_checkpoint.pt"
    rl_agent = None
    if ckpt_path.is_file():
        try:
            rl_agent = EVRLAgent(seed=req.seed or 42)
            rl_agent.load(ckpt_path)
        except Exception:
            rl_agent = None

    comparison_record = run_evrp_comparison(
        instance=ev_inst,
        algorithms=req.algorithms,
        rl_agent=rl_agent,
        seed=req.seed if req.seed is not None else 42,
    )

    depot_coord = ev_inst.coordinates[0]
    customers_data = []
    for c_idx in ev_inst.customer_indices:
        customers_data.append({
            "id": int(c_idx),
            "x": float(ev_inst.coordinates[c_idx][0]),
            "y": float(ev_inst.coordinates[c_idx][1]),
            "demand": float(ev_inst.demands[c_idx]),
        })

    stations_data = []
    for s_idx in ev_inst.charging_stations:
        stations_data.append({
            "id": int(s_idx),
            "x": float(ev_inst.coordinates[s_idx][0]),
            "y": float(ev_inst.coordinates[s_idx][1]),
        })

    comparison_record["depot"] = {"id": 0, "x": float(depot_coord[0]), "y": float(depot_coord[1])}
    comparison_record["customers"] = customers_data
    comparison_record["charging_stations"] = stations_data

    # Save to disk
    out_file = RESULTS_DIR / "evrp_multi_algorithm_comparison.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(comparison_record, f, indent=2)
    except Exception:
        pass

    return comparison_record


@app.get("/api/evrp/ablation")
def get_evrp_ablation_results() -> Dict[str, Any]:
    """Returns persistent or freshly computed ablation study results."""
    target = RESULTS_DIR / "evrp_ablation_study_results.json"
    if not target.is_file():
        target = ROOT_DIR / "evrp_ablation_study_results.json"

    if not target.is_file():
        # Compute fresh ablation study on default instance
        file_path = _resolve_instance_path("E-n29-k4-s7.evrp")
        ev_inst = load_evrp_instance(file_path=file_path, customer_count=4, seed=42)
        fresh_ablation = run_evrp_ablation_study(ev_inst, seed=42)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(fresh_ablation, f, indent=2)
        return fresh_ablation

    with open(target, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return raw_data


@app.post("/api/evrp/ablation/run")
def run_live_ablation_study(req: RLTrainRequest) -> Dict[str, Any]:
    """Executes live 6-component ablation study isolating hybrid mechanisms."""
    file_path = _resolve_instance_path(req.instance_name)
    ev_inst = load_evrp_instance(file_path=file_path, customer_count=req.customer_count, seed=req.seed or 42)
    ablation_result = run_evrp_ablation_study(ev_inst, seed=req.seed or 42)
    target = RESULTS_DIR / "evrp_ablation_study_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(ablation_result, f, indent=2)
    return ablation_result


@app.post("/api/evrp/multi-seed")
def evaluate_multi_seed(req: EVRPMultiSeedRequest) -> Dict[str, Any]:
    """Runs rigorous multi-seed comparison across seeds and aggregates Mean +/- Std."""
    file_path = _resolve_instance_path(req.instance_name)
    ev_inst = load_evrp_instance(
        file_path=file_path,
        customer_count=req.customer_count,
        battery_capacity=req.battery_capacity or 80.0,
        energy_consumption_rate=req.base_energy_rate or 0.2,
        load_factor=req.load_factor if req.load_factor is not None else 0.2,
        charging_power=req.charging_power if req.charging_power is not None else 50.0,
        charging_price=req.charging_price if req.charging_price is not None else 0.25,
        seed=req.seeds[0] if req.seeds else 42,
    )

    ckpt_path = RESULTS_DIR / "ev_rl_checkpoint.pt"
    rl_agent = None
    if ckpt_path.is_file():
        try:
            rl_agent = EVRLAgent(seed=42)
            rl_agent.load(ckpt_path)
        except Exception:
            rl_agent = None

    return run_evrp_multi_seed_comparison(
        instance=ev_inst,
        algorithms=req.algorithms,
        seeds=req.seeds,
        rl_agent=rl_agent,
    )


@app.get("/api/evrp/scalability/validated")
def get_validated_scalability() -> Dict[str, Any]:
    """
    Returns the persistent validated scalability benchmark results directly from
    results/validated_research_experiments.json across N = 4, 8, 12, 20, 50, and 100 customers.
    """
    target = RESULTS_DIR / "validated_research_experiments.json"
    if not target.is_file():
        target = ROOT_DIR / "results" / "validated_research_experiments.json"
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Validated scalability results file not found.")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": "success",
        "timestamp": data.get("timestamp"),
        "customer_counts": [4, 8, 12, 20, 50, 100],
        "scalability_progression": data.get("phase_5_scalability_progression", {}),
    }


@app.post("/api/evrp/scalability")
def evaluate_scalability(req: EVRPScalabilityRequest) -> Dict[str, Any]:
    """Evaluates algorithm scaling across increasing customer counts."""
    file_path = _resolve_instance_path(req.instance_name)
    counts = req.customer_counts or [4, 8, 12]
    scaling_results = {}

    for c_count in counts:
        try:
            ev_inst = load_evrp_instance(
                file_path=file_path,
                customer_count=c_count,
                battery_capacity=req.battery_capacity or 80.0,
                energy_consumption_rate=req.base_energy_rate or 0.2,
                load_factor=req.load_factor if req.load_factor is not None else 0.2,
                charging_power=req.charging_power if req.charging_power is not None else 50.0,
                charging_price=req.charging_price if req.charging_price is not None else 0.25,
                seed=req.seed or 42,
            )
            res = run_evrp_comparison(
                instance=ev_inst,
                algorithms=req.algorithms,
                seed=req.seed or 42,
            )
            scaling_results[str(c_count)] = {
                "customer_count": c_count,
                "results": {k: {"cost": v["total_cost"], "dist": v["total_distance"], "runtime": v["runtime_seconds"], "feasible": v["feasible"]} for k, v in res["results"].items()},
            }
        except Exception as e:
            scaling_results[str(c_count)] = {"error": str(e)}

    return {
        "instance_name": req.instance_name,
        "scalability_curve": scaling_results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
