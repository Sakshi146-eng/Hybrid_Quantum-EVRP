"""
Quantum and Hybrid Optimization Module for VRP / EVRP.
Includes QUBO matrix formulation, QAOA quantum circuit solver, exact baseline,
Hybrid GA-QAOA, and Full RL + QAOA + EV-aware Hybrid solvers.
"""
from src.quantum.qubo import VRPQuboBuilder, extract_sub_instance
from src.quantum.ev_qubo import EVRouteQuboBuilder
from src.quantum.qaoa_solver import QAOAVRPSolver
from src.quantum.exact_solver import ExactBruteForceSolver
from src.quantum.hybrid_solver import HybridGAQAOASolver
from src.quantum.adaptive_rl_qaoa import (
    ConventionalQAOASolver,
    RLGuidedQAOASolver,
    RLGuidedQAOALocalSearchSolver,
    MasterFullHybridEVRPSolver,
)
from src.quantum.rl_qaoa_hybrid import RLQAOASolver, FullHybridEVRPSolver

__all__ = [
    "VRPQuboBuilder",
    "EVRouteQuboBuilder",
    "extract_sub_instance",
    "QAOAVRPSolver",
    "ExactBruteForceSolver",
    "HybridGAQAOASolver",
    "ConventionalQAOASolver",
    "RLGuidedQAOASolver",
    "RLGuidedQAOALocalSearchSolver",
    "MasterFullHybridEVRPSolver",
    "RLQAOASolver",
    "FullHybridEVRPSolver",
]
