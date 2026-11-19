"""
Data loading and preprocessing package.
"""
from src.data.loader import DataLoader
from src.data.e_cvrp_loader import load_e_cvrp_instance, parse_e_cvrp_file

__all__ = ["DataLoader", "load_e_cvrp_instance", "parse_e_cvrp_file"]
