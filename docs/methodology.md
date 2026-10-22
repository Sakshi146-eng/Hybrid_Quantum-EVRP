# Theoretical Methodology and Mathematical Formulations

> **Note**: This document will contain the exact mathematical equations, QUBO variable encodings, and RL formulation once fully verified against the base paper.

## 1. Base Paper Details
- **Title**: Reinforcement Learning based Quantum Approximate Optimization Algorithm for Vehicle Routing Problem
- **Authors**: Aman and Sakib (2026)
- **Target Problem**: Vehicle Routing Problem (VRP)

---

## 2. VRP QUBO Formulation (Phase 1)
*TODO: Document exact mathematical formulation upon verification.*

- **Decision Variable Encoding**: `x_{i,j,t}` or `x_{i,t}` - *[TO BE DETERMINED from base paper]*
- **Objective Terms**: Minimize distance/cost.
- **Constraints & Penalty Terms**:
  - Visit each customer exactly once
  - Subtour elimination constraints
  - Vehicle capacity constraints (if applicable)

---

## 3. QAOA Parameter Optimization via RL
*TODO: Document state space, action space, reward function, and RL agent architecture.*

- **State**: Vector representing QAOA state (e.g., expectation value, variance, iteration step). *[TO BE DETERMINED]*
- **Action**: Parameter updates `(Δγ, Δβ)` or direct parameter values `(γ, β)`. *[TO BE DETERMINED]*
- **Reward**: Reduction in expected cost / energy value or solution quality metric. *[TO BE DETERMINED]*

---

## 4. EVRP Extension (Phase 2)
*TODO: Document EVRP extension once Phase 1 is validated.*

- **Battery State Encoding**: Energy consumption along arcs.
- **Charging Station Nodes**: Re-charging logic and dwell time constraints.
- **Feasibility Verification Rules**:
  - Distance check: `Distance(u, v) <= MaxRange`
  - Battery check: `RemainingBattery(v) >= 0`
