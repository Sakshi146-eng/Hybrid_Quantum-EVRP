"""
Distance matrix calculations for VRP and EVRP instances.
"""
from typing import Optional
import numpy as np


def compute_distance_matrix(coordinates: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Computes pairwise distance matrix for node coordinates.

    Args:
        coordinates (np.ndarray): Array of shape (N, 2) or (N, D) representing node coordinates.
        metric (str): Distance metric to use ("euclidean" currently supported).

    Returns:
        np.ndarray: (N, N) symmetric pairwise distance matrix.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2:
        raise ValueError(f"Coordinates must be a 2D array of shape (N, D), got shape {coords.shape}")

    metric_clean = metric.lower().strip()
    if metric_clean == "euclidean":
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    else:
        raise ValueError(f"Unsupported distance metric: '{metric}'")

    return dist_matrix


def validate_distance_matrix(matrix: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Validates properties of a distance matrix:
    - 2D square matrix
    - Non-negative elements
    - Zero diagonal elements
    - Symmetry (for Euclidean distance)

    Args:
        matrix (np.ndarray): Distance matrix to check.
        rtol (float): Relative tolerance for floating point comparisons.
        atol (float): Absolute tolerance for floating point comparisons.

    Returns:
        bool: True if matrix passes all distance matrix properties.
    """
    mat = np.asarray(matrix)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    if np.any(mat < 0):
        return False
    if not np.allclose(np.diag(mat), 0.0, rtol=rtol, atol=atol):
        return False
    if not np.allclose(mat, mat.T, rtol=rtol, atol=atol):
        return False
    return True
