"""
QUBO Matrix Builder (Legacy Bridge).
Wraps the modular src.quantum.qubo.VRPQuboBuilder implementation.
"""
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from src.vrp.instance import VRPInstance
from src.quantum.qubo import VRPQuboBuilder


class QUBOBuilder:
    """
    Constructs the QUBO matrix (Q) for a given routing instance.
    """

    def __init__(self, vrp_instance: VRPInstance, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            vrp_instance (VRPInstance): Problem instance.
            config (Optional[Dict[str, Any]]): Configuration dictionary containing penalty weights.
        """
        self.instance = vrp_instance
        self.config = config or {}
        penalty_weight = self.config.get("penalty_weight", None)
        penalty_multiplier = self.config.get("penalty_multiplier", 2.0)
        self.builder = VRPQuboBuilder(
            instance=vrp_instance,
            penalty_multiplier=penalty_multiplier,
            penalty_weight=penalty_weight,
        )
        self.num_vars = self.builder.num_vars
        self.Q_matrix = None
        self.offset = 0.0

    def build_vrp_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Formulates the routing problem as a QUBO matrix.

        Returns:
            Tuple[np.ndarray, float]: Q_matrix and constant offset.
        """
        self.Q_matrix, self.offset = self.builder.build_qubo()
        return self.Q_matrix, self.offset

    def decode_solution(self, bitstring: str) -> List[List[int]]:
        """
        Decodes a binary solution bitstring into structured vehicle routes.

        Args:
            bitstring (str): Binary string result from QAOA measurement.

        Returns:
            List[List[int]]: Vehicle routes.
        """
        route, is_valid, _ = self.builder.decode_bitstring(bitstring)
        return [route]
