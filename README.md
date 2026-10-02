# Quantum-Assisted Electric Vehicle Routing Using Hybrid Optimization

## Overview

This research project explores solving the **Electric Vehicle Routing Problem (EVRP)** using a hybrid quantum-classical optimization approach. Specifically, we follow a staged research methodology starting from reproducing a base **RL-QAOA** framework for the standard **Vehicle Routing Problem (VRP)** and extending it to incorporate electric vehicle (EV) specific constraints.

### Base Paper Reference
> **Aman and Sakib (2026)** - *Reinforcement Learning based Quantum Approximate Optimization Algorithm for Vehicle Routing Problem*

---

## Staged Methodology Pipeline

```
VRP
  ↓
QUBO Formulation
  ↓
QAOA Optimization
  ↓
RL Optimizes QAOA Parameters (Gamma, Beta)
  ↓
Reproduce / Validate Base RL-QAOA VRP Approach (Phase 1)
  ↓
Extend VRP to EVRP (Phase 2)
  ↓
Add EV-Specific Constraints (Battery, Energy, Charging)
  ↓
Extended EVRP QUBO
  ↓
QAOA + RL Pipeline
  ↓
Feasible EV Routing Solutions
  ↓
Evaluation & Benchmarking (Phase 3)
  ↓
Limitation Analysis & Refinements (Phase 4)
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
│   ├── raw/                  # Original, immutable datasets
│   └── processed/            # Cleaned / formatted datasets
├── src/                      # Core source code package
│   ├── data/                 # Dataset loading and parsing
│   ├── vrp/                  # VRP distance matrices and route structures
│   ├── qubo/                 # QUBO formulation & matrix generation
│   ├── qaoa/                 # QAOA circuit construction & execution
│   ├── rl/                   # Reinforcement Learning environments & agents
│   ├── evrp/                 # EV-specific constraints & feasibility checks
│   ├── evaluation/           # Metrics, benchmarking & plotting
│   └── utils/                # Utility modules (config, logger, seed)
├── notebooks/                # Exploratory notebooks (no core logic)
├── tests/                    # Unit and integration test suite
├── results/                  # Generated benchmark results & plots
└── docs/                     # Documentation & mathematical formulations
```

---

## Getting Started

### Prerequisites
- Python 3.9+

### Setup Environment
```bash
# Clone or navigate to directory
cd quantum-evrp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode with dependencies
pip install -e .
```

### Running Tests
```bash
pytest tests/
```

---

## License & Citation
*Research project repository. All theoretical extensions build upon Aman and Sakib (2026).*
