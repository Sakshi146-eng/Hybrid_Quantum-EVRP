# Quantum-Assisted Electric Vehicle Routing Using Hybrid Optimization

## Overview

This research project explores solving the **Electric Vehicle Routing Problem (EVRP)** using a hybrid quantum-classical optimization approach. Specifically, we follow a staged research methodology starting from reproducing a base **RL-QAOA** framework for the standard **Vehicle Routing Problem (VRP)** and extending it to incorporate electric vehicle (EV) specific constraints.

### Base Paper Reference
> **Aman and Sakib (2026)** - *Reinforcement Learning based Quantum Approximate Optimization Algorithm for Vehicle Routing Problem*

---

## Benchmark Dataset: CEC-2020 E-CVRP Benchmark

We use the **CEC-2020 E-CVRP Benchmark Test Suite** published by Mavrovouniotis et al. (*IEEE Congress on Evolutionary Computation, 2020*) as our primary benchmark dataset.

- **Source Repository**: [Mavrovouniotis/e-cvrp_benchmark_instances](https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances)
- **Why Selected**: It provides standardized, realistic problem instances of varying scale (from 22 to 1000+ nodes) derived from classic CVRP benchmarks (Augerat, Christofides, Uchoa) with added EV charging infrastructure parameters.
- **Phase 1 Classical Usage**: In this phase, we extract only the classical CVRP information (depot, customer coordinates, customer demands, vehicle fleet count, and payload capacity) to run classical solvers (Genetic Algorithm) as our baseline.
- **Future EVRP Usage**: EV-specific parameters (battery capacity, energy consumption rate, charging station locations) are preserved in metadata and will be incorporated when extending solvers to full E-CVRP constraints in later phases.

> **Important Note**: The current classical GA baseline solves the CVRP formulation extracted from these benchmark instances. It does **not** claim to solve full E-CVRP battery/charging decisions yet.

---

## Staged Methodology Pipeline

```
E-CVRP Benchmark Test Suite
        ↓ (extract CVRP fields)
VRP / CVRP Classical Baselines (GA, ACO, SA, TS)
        ↓
QUBO Formulation (VRP)
        ↓
QAOA Optimization (VRP)
        ↓
RL-QAOA Parameter Optimization
        ↓ (incorporate EV fields: battery & charging)
EVRP Classical & Quantum-Assisted Solvers
```

---

## Project Structure

```text
quantum-evrp/
├── README.md                 # Main project overview and methodology
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusion rules
├── pyproject.toml            # Package setup configuration
├── configs/                  # Configuration files (YAML)
│   └── default_config.yaml
├── data/                     # Datasets & benchmark instructions
│   ├── raw/                  # Pristine .evrp benchmark files
│   ├── processed/            # Parsed graph cache
│   └── README.md             # Dataset acquisition & loading guide
├── examples/                 # Demonstration scripts
│   ├── demo_vrp_foundation.py
│   ├── demo_ga_solver.py
│   └── demo_ga_benchmark.py  # Run GA on real CEC-2020 E-CVRP benchmark
├── frontend/                 # React + Vite Dashboard
│   ├── src/
│   │   ├── components/       # Route visualizer & UI components
│   │   ├── utils/            # VRP client-side logic
   │   │   ├── App.jsx           # Main dashboard UI
│   │   └── index.css         # Styling system
│   └── package.json
├── src/                      # Core Python source code package
│   ├── data/                 # Dataset loader (e_cvrp_loader.py)
│   ├── vrp/                  # VRP distance matrices, route & validator
│   ├── classical/            # BaseVRPSolver & GeneticAlgorithmSolver
│   ├── qubo/                 # QUBO formulation & matrix generation
│   ├── qaoa/                 # QAOA circuit construction & execution
│   ├── rl/                   # Reinforcement Learning environments & agents
│   ├── evrp/                 # EV-specific constraints & feasibility checks
│   ├── evaluation/           # Metrics, BenchmarkResult & recorder
│   └── utils/                # Utility modules (config, logger, seed)
├── notebooks/                # Exploratory notebooks (01_vrp_foundation.ipynb)
├── tests/                    # Unit and integration test suite (36 tests)
└── results/                  # Generated benchmark results (ga_benchmark_results.json)
```

---

## How to Run the Project

### 1. Running Python Unit Tests

To execute all 36 unit tests covering VRP foundation, GA baseline, benchmark dataset loader, and result recording:

```bash
python tests/run_tests.py
```

---

### 2. Running the GA Benchmark Demonstration

To run the Genetic Algorithm on a real CEC-2020 benchmark instance (`E-n29-k4-s7.evrp`):

```bash
python examples/demo_ga_benchmark.py
```

To run on a specific `.evrp` file path:

```bash
python examples/demo_ga_benchmark.py data/raw/e_cvrp/F-n49-k4-s4.evrp
```

---

### 3. Launching the Web Dashboard (Frontend)

To run the interactive React + Vite web dashboard locally:

```bash
cd frontend
npm install
npm run dev
```

Open your browser at `http://localhost:5173/` to view the UI.

---

## Current Implementation Status

- **Classical VRP Foundation**: ✅ Complete (`VRPInstance`, `VRPValidator`, distance matrix)
- **Classical Solver Framework**: ✅ Complete (`BaseVRPSolver`, `VRPSolution`, `GeneticAlgorithmSolver`)
- **CEC-2020 E-CVRP Benchmark Loader**: ✅ Complete (`load_e_cvrp_instance`, `parse_e_cvrp_file`)
- **Benchmark Experiment & Result Recorder**: ✅ Complete (`BenchmarkResult`, `record_benchmark_result`)
- **QUBO Formulation**: 🔲 Not implemented (Coming Soon)
- **QAOA Solver**: 🔲 Not implemented (Coming Soon)
- **Reinforcement Learning**: 🔲 Not implemented (Coming Soon)
- **EVRP Constraints & Energy Model**: 🔲 Not implemented (Coming Soon - Phase 2)

---

## License & Citation
*Research project repository. All theoretical extensions build upon Aman and Sakib (2026).*
