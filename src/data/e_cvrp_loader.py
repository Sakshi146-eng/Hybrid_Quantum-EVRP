"""
Loader for CEC-2020 E-CVRP Benchmark Test Suite instances.
Parses .evrp instance files and converts classical CVRP fields (depot, customer coordinates,
demands, vehicle capacity, and vehicle fleet count) into standard VRPInstance objects.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

from src.vrp.distance import compute_distance_matrix
from src.vrp.instance import VRPInstance


def parse_e_cvrp_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parses a CEC-2020 E-CVRP benchmark instance file (.evrp format).

    Args:
        file_path (Union[str, Path]): Absolute or relative path to the .evrp file.

    Returns:
        Dict[str, Any]: Dictionary containing parsed header metadata and raw section data.

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If essential headers or sections are missing.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"E-CVRP benchmark file not found at: {path}")

    headers: Dict[str, Any] = {}
    node_coords: Dict[int, List[float]] = {}
    demands: Dict[int, float] = {}
    station_nodes: List[int] = []
    depot_nodes: List[int] = []

    current_section = "HEADER"

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            # Section header checks
            if line_str.startswith("NODE_COORD_SECTION"):
                current_section = "NODE_COORD"
                continue
            elif line_str.startswith("DEMAND_SECTION"):
                current_section = "DEMAND"
                continue
            elif line_str.startswith("STATIONS_COORD_SECTION") or line_str.startswith("STATION_COORD_SECTION") or line_str.startswith("STATIONS_SECTION"):
                current_section = "STATIONS_COORD"
                continue
            elif line_str.startswith("DEPOT_SECTION"):
                current_section = "DEPOT"
                continue
            elif line_str.startswith("EOF"):
                break

            if current_section == "HEADER":
                if ":" in line_str:
                    key, val = line_str.split(":", 1)
                    key_clean = key.strip().upper()
                    val_clean = val.strip()

                    if key_clean in ("VEHICLES", "DIMENSION", "STATIONS"):
                        try:
                            headers[key_clean] = int(val_clean)
                        except ValueError:
                            headers[key_clean] = val_clean
                    elif key_clean in ("CAPACITY", "ENERGY_CAPACITY", "ENERGY_CONSUMPTION"):
                        try:
                            headers[key_clean] = float(val_clean)
                        except ValueError:
                            headers[key_clean] = val_clean
                    elif key_clean == "OPTIMAL_VALUE":
                        try:
                            # Might be integer or float e.g. 383 or float/string "740 (upper bound)"
                            val_first = val_clean.split()[0]
                            headers[key_clean] = float(val_first) if "." in val_first else int(val_first)
                        except ValueError:
                            headers[key_clean] = val_clean
                    else:
                        headers[key_clean] = val_clean

            elif current_section == "NODE_COORD":
                parts = line_str.split()
                if len(parts) >= 3:
                    try:
                        nid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        node_coords[nid] = [x, y]
                    except ValueError:
                        pass

            elif current_section == "DEMAND":
                parts = line_str.split()
                if len(parts) >= 2:
                    try:
                        nid = int(parts[0])
                        dem = float(parts[1])
                        demands[nid] = dem
                    except ValueError:
                        pass

            elif current_section == "STATIONS_COORD":
                parts = line_str.split()
                if parts:
                    try:
                        nid = int(parts[0])
                        station_nodes.append(nid)
                    except ValueError:
                        pass

            elif current_section == "DEPOT":
                parts = line_str.split()
                if parts:
                    try:
                        nid = int(parts[0])
                        if nid == -1:
                            current_section = "HEADER"
                        else:
                            depot_nodes.append(nid)
                    except ValueError:
                        pass

    if "DIMENSION" not in headers:
        headers["DIMENSION"] = len(node_coords)
    if "VEHICLES" not in headers:
        headers["VEHICLES"] = 1
    if "CAPACITY" not in headers:
        headers["CAPACITY"] = float("inf")

    return {
        "headers": headers,
        "name": headers.get("NAME", path.stem),
        "comment": headers.get("COMMENT", ""),
        "type": headers.get("TYPE", "EVRP"),
        "optimal_value": headers.get("OPTIMAL_VALUE", None),
        "num_vehicles": int(headers.get("VEHICLES", 1)),
        "dimension": int(headers.get("DIMENSION", len(node_coords))),
        "num_stations": int(headers.get("STATIONS", len(station_nodes))),
        "vehicle_capacity": float(headers.get("CAPACITY", 0.0)),
        "energy_capacity": headers.get("ENERGY_CAPACITY", None),
        "energy_consumption": headers.get("ENERGY_CONSUMPTION", None),
        "edge_weight_type": headers.get("EDGE_WEIGHT_TYPE", "EUC_2D"),
        "node_coords": node_coords,
        "demands": demands,
        "station_nodes": station_nodes,
        "depot_nodes": depot_nodes,
        "file_path": str(path),
    }


def load_e_cvrp_instance(
    file_path: Union[str, Path], include_stations: bool = False
) -> VRPInstance:
    """
    Loads a CEC-2020 E-CVRP benchmark file and converts classical VRP/CVRP data into VRPInstance.

    Args:
        file_path (Union[str, Path]): Path to the .evrp benchmark instance file.
        include_stations (bool): If False (default), filters out charging stations so the VRPInstance
                                 contains only depot and customer nodes.

    Returns:
        VRPInstance: Populated classical VRPInstance object with distance matrix, coordinates,
                     demands, vehicle capacity, fleet size, and depot index.
    """
    parsed = parse_e_cvrp_file(file_path)

    node_coords = parsed["node_coords"]
    raw_demands = parsed["demands"]

    if not parsed["depot_nodes"]:
        depot_node_id = 1
    else:
        depot_node_id = parsed["depot_nodes"][0]

    station_set = set(parsed["station_nodes"])

    # Determine ordered list of nodes for VRPInstance
    all_nids = sorted(node_coords.keys())

    if not include_stations:
        # Exclude station nodes for classical CVRP baseline
        selected_nids = [nid for nid in all_nids if nid == depot_node_id or nid not in station_set]
    else:
        selected_nids = all_nids

    # Depot MUST be at index 0 in VRPInstance
    if depot_node_id in selected_nids:
        selected_nids.remove(depot_node_id)
    ordered_nids = [depot_node_id] + selected_nids

    # Construct coordinates array (N, 2)
    coords_list = [node_coords[nid] for nid in ordered_nids]
    coordinates = np.array(coords_list, dtype=float)

    # Construct demands array (N,)
    demands_list = [float(raw_demands.get(nid, 0.0)) for nid in ordered_nids]
    # Ensure depot demand is explicitly 0
    demands_list[0] = 0.0
    demands = np.array(demands_list, dtype=float)

    # Distance matrix calculation (Euclidean 2D)
    distance_matrix = compute_distance_matrix(coordinates, metric="euclidean")

    num_vehicles = int(parsed["num_vehicles"])
    vehicle_capacity = float(parsed["vehicle_capacity"])

    instance = VRPInstance(
        distance_matrix=distance_matrix,
        coordinates=coordinates,
        num_vehicles=num_vehicles,
        depot_idx=0,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        distance_metric="euclidean",
    )

    # Preserve raw EV metadata for future EVRP phases
    raw_data = dict(parsed)
    raw_data["mapped_node_ids"] = ordered_nids
    raw_data["depot_node_id"] = depot_node_id
    instance.raw_data = raw_data

    return instance


def load_evrp_instance(
    file_path: Union[str, Path],
    customer_count: Optional[int] = None,
    customer_indices: Optional[List[int]] = None,
    battery_capacity: Optional[float] = None,
    energy_consumption_rate: Optional[float] = None,
    battery_reserve: Optional[float] = None,
    load_factor: float = 0.2,
    charging_power: float = 50.0,
    charging_price: float = 0.25,
    seed: Optional[int] = 42,
):
    """
    Loads a CEC-2020 E-CVRP benchmark file and converts it into a full EVRPInstance
    incorporating depot, customers, charging stations, and battery constraints.
    """
    from src.evrp.instance import EVRPInstance

    parsed = parse_e_cvrp_file(file_path)
    node_coords = parsed["node_coords"]
    raw_demands = parsed["demands"]

    depot_node_id = parsed["depot_nodes"][0] if parsed["depot_nodes"] else 1
    station_nodes = list(parsed["station_nodes"])
    station_set = set(station_nodes)

    all_nids = sorted(node_coords.keys())
    available_customers = [
        nid for nid in all_nids if nid != depot_node_id and nid not in station_set
    ]

    # Select customers
    if customer_indices is not None:
        selected_customers = [nid for nid in customer_indices if nid in available_customers]
    elif customer_count is not None and customer_count < len(available_customers):
        if seed is not None:
            rng = np.random.RandomState(seed)
            selected_customers = sorted(
                rng.choice(available_customers, size=customer_count, replace=False).tolist()
            )
        else:
            selected_customers = available_customers[:customer_count]
    else:
        selected_customers = available_customers

    # Select charging stations (up to 4 stations if available)
    selected_stations = station_nodes[:4] if len(station_nodes) > 4 else station_nodes

    # Construct unified ordered node list: [depot, ...customers, ...stations]
    ordered_nids = [depot_node_id] + selected_customers + selected_stations
    c_count = len(selected_customers)
    s_count = len(selected_stations)

    # Station indices in the unified instance
    station_indices = list(range(1 + c_count, 1 + c_count + s_count))
    customer_indices_mapped = list(range(1, 1 + c_count))

    coords_list = [node_coords[nid] for nid in ordered_nids]
    coordinates = np.array(coords_list, dtype=float)

    demands_list = [float(raw_demands.get(nid, 0.0)) for nid in ordered_nids]
    demands_list[0] = 0.0  # depot demand = 0
    for s_idx in station_indices:
        demands_list[s_idx] = 0.0  # station demand = 0
    demands = np.array(demands_list, dtype=float)

    distance_matrix = compute_distance_matrix(coordinates, metric="euclidean")

    # Battery capacity and consumption calibration
    raw_energy_cap = parsed.get("energy_capacity")
    raw_energy_cons = parsed.get("energy_consumption")

    rate = float(energy_consumption_rate) if energy_consumption_rate is not None else (
        float(raw_energy_cons) if (raw_energy_cons and float(raw_energy_cons) > 0) else 0.25
    )

    if battery_capacity is not None:
        b_cap = float(battery_capacity)
    elif raw_energy_cap is not None and float(raw_energy_cap) > 0:
        # Scale benchmark battery capacity proportionally to sub-instance distance scale
        raw_b = float(raw_energy_cap)
        b_cap = raw_b
    else:
        # Calibrate default battery capacity so single legs are feasible but long tours require charging
        max_d = float(np.max(distance_matrix))
        b_cap = max_d * rate * 3.5

    reserve = float(battery_reserve) if battery_reserve is not None else float(0.05 * b_cap)

    num_vehicles = int(parsed["num_vehicles"])
    vehicle_capacity = float(parsed["vehicle_capacity"])

    instance = EVRPInstance(
        distance_matrix=distance_matrix,
        coordinates=coordinates,
        battery_capacity=b_cap,
        energy_consumption_rate=rate,
        charging_stations=station_indices,
        initial_battery=b_cap,
        battery_reserve=reserve,
        load_factor=load_factor,
        charging_power=charging_power,
        charging_price_per_kwh=charging_price,
        num_vehicles=num_vehicles,
        depot_idx=0,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        customer_subset=customer_indices_mapped,
    )

    raw_data = dict(parsed)
    raw_data["mapped_node_ids"] = ordered_nids
    raw_data["depot_node_id"] = depot_node_id
    raw_data["selected_customers_parent_ids"] = selected_customers
    raw_data["selected_stations_parent_ids"] = selected_stations
    instance.raw_data = raw_data

    return instance
