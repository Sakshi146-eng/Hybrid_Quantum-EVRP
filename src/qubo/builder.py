"""
QUBO Matrix Builder.
Formulates VRP / EVRP instances into Quadratic Unconstrained Binary Optimization (QUBO) matrices.

NOTE: Exact decision variable encoding, penalty terms, and penalty weights
MUST be verified against the base paper (Aman and Sakib, 2026) before full implementation.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
from src.vrp.instance import VRPInstance


class QUBOBuilder:
    """
    Constructs the QUBO matrix (Q) for a given routing instance.
    """

    def __init__(self, vrp_instance: VRPInstance, config: Dict[str, Any]):
        """
        Args:
            vrp_instance (VRPInstance): Problem instance.
            config (Dict[str, Any]): Configuration dictionary containing penalty weights.
        """
        self.instance = vrp_instance
        self.config = config
        self.num_vars = 0
        self.Q_matrix = None

    def build_vrp_qubo() -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Formulates the standard VRP as a QUBO matrix.

        Returns:
            Tuple[np.ndarray, Dict[str, int]]:
                - Q_matrix: Upper triangular or symmetric N x N QUBO matrix.
                - var_map: Mapping from binary variable names to matrix indices.

        TODO: TO BE DETERMINED FROM BASE PAPER (Aman and Sakib, 2026):
        - Variable encoding: x_{i,j,t} or x_{i,t}
        - Exact penalty formulation for customer visit constraint
        - Subtour elimination strategy
        - Capacity penalty formulation
        """
        raise NotImplementedError(
            "TODO: QUBO formulation must be verified from base paper (Aman & Sakib, 2026) before implementation."
        )

    def decode_solution(self, bitstring: str) -> List[List[int]]:
        """
        Decodes a binary solution bitstring into structured vehicle routes.

        Args:
            bitstring (str): Binary string result from QAOA measurement (e.g., '101001').

        Returns:
            List[List[int]]: List of ordered node indices for each vehicle route.

        TODO: TO BE DETERMINED FROM BASE PAPER.
        """
        raise NotImplementedError(
            "TODO: Solution decoding logic must be verified from base paper before implementation."
        )
