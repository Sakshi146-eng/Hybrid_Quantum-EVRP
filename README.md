# Quantum-Assisted Electric Vehicle Routing Using Hybrid Optimization

## Overview

This research project explores solving the **Electric Vehicle Routing Problem (EVRP)** using a hybrid quantum-classical optimization approach. Specifically, we follow a staged research methodology starting from reproducing a base **RL-QAOA** framework for the standard **Vehicle Routing Problem (VRP)** and extending it to incorporate electric vehicle (EV) specific constraints.

### Base Paper Reference
> **Aman and Sakib (2026)** - *Reinforcement Learning based Quantum Approximate Optimization Algorithm for Vehicle Routing Problem*

---

## Staged Methodology Pipeline

```
VRP (Classical Baseline)
  ↓
QUBO Formulation (Coming Soon)
  ↓
QAOA Optimization (Coming Soon)
  ↓
RL Optimizes QAOA Parameters (Gamma, Beta) (Coming Soon)
  ↓
Extend VRP to EVRP (Phase 2) (Coming Soon)
  ↓
Evaluation & Benchmarking (Phase 3) (Coming Soon)
```

---

## Project Structure

```
quantum-evrp/
├── README.md                 # Main project overview and methodology
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusion rules
├── pyproject.toml            # Package setup configuration
├── configs/                  # Configuration files (YAML)
│   └── default_config.yaml
├── data/                     # Datasets
├── examples/                 # Demonstration scripts
│   └── demo_vrp_foundation.py
├── frontend/                 # React + Vite Dashboard
│   ├── src/
│   │   ├── components/       # Route visualizer & UI components
│   │   ├── utils/            # VRP client-side logic
│   │   ├── App.jsx           # Main dashboard UI
│   │   └── index.css         # Styling system
│   └── package.json
├── src/                      # Core Python source code package
│   ├── data/                 # Dataset loading and parsing
│   ├── vrp/                  # VRP distance matrices, route & validator
│   ├── qubo/                 # QUBO formulation & matrix generation
│   ├── qaoa/                 # QAOA circuit construction & execution
│   ├── rl/                   # Reinforcement Learning environments & agents
│   ├── evrp/                 # EV-specific constraints & feasibility checks
│   ├── evaluation/           # Metrics, benchmarking & plotting
│   └── utils/                # Utility modules (config, logger, seed)
├── notebooks/                # Exploratory notebooks (01_vrp_foundation.ipynb)
├── tests/                    # Unit and integration test suite
└── results/                  # Generated benchmark results & plots
```

---

## How to Run the Project

### 1. Launching the Web Dashboard (Frontend)

To run the interactive React + Vite web dashboard locally:

```bash
cd frontend
npm install
npm run dev
```

Open your browser at `http://localhost:5173/` to view:
- **Header & Subtitle**: Hybrid Quantum-Classical Optimization branding
- **Project Overview**: VRP, EVRP, QUBO, QAOA, RL concepts
- **VRP Demo**: Interactive 2D route graph visualizer, customer table, distance matrix, and payload validation
- **Methodology Pipeline**: Visual pipeline with "Coming Soon" badges
- **Status Table**: Current module implementation progress

---

### 2. Running Python Unit Tests

To execute the test suite verifying distance matrix symmetry, zero diagonal, route costs, depot start/end checks, and vehicle capacity bounds:

```bash
python tests/run_tests.py
```

---

### 3. Running the Python Demonstration Script

To run the terminal-based classical VRP demonstration:

```bash
python examples/demo_vrp_foundation.py
```

---

## Current Status

- **Classical VRP Foundation**: ✅ Complete
- **QUBO Formulation**: 🔲 Not implemented (Coming Soon)
- **QAOA Solver**: 🔲 Not implemented (Coming Soon)
- **Reinforcement Learning**: 🔲 Not implemented (Coming Soon)
- **EVRP Extension**: 🔲 Not implemented (Coming Soon)
- **Evaluation**: 🔲 Not implemented (Coming Soon)

---

## License & Citation
*Research project repository. All theoretical extensions build upon Aman and Sakib (2026).*
