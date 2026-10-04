# Benchmark Data Storage & Directory Structure

This directory stores raw benchmark datasets and processed VRP/EVRP problem data.

---

## Benchmark Source

We use the **CEC-2020 Electric Capacitated Vehicle Routing Problem (E-CVRP) Benchmark Test Suite** published by Mavrovouniotis et al. (IEEE CEC 2020):
- **Repository**: [https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances](https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances)
- **Reference**:
  > M. Mavrovouniotis, C. Menelaou, S. Timotheou, G. Ellinas, C. Panayiotou and M. Polycarpou, "A Benchmark Test Suite for the Electric Capacitated Vehicle Routing Problem," *2020 IEEE Congress on Evolutionary Computation (CEC)*, 2020, pp. 1-8.

---

## Expected Directory Structure

```text
data/
├── raw/
│   └── e_cvrp/
│       ├── E-n29-k4-s7.evrp
│       ├── E-n30-k3-s7.evrp
│       ├── E-n35-k3-s5.evrp
│       ├── F-n49-k4-s4.evrp
│       ├── M-n110-k10-s9.evrp
│       ├── X-n147-k7-s4.evrp
│       └── ...
├── processed/
└── README.md
```

> **Note**: Raw dataset contents in `data/raw/` (except `.gitkeep`) are excluded from git version control via `.gitignore`.

---

## How to Obtain the Dataset

To download/clone the raw CEC-2020 E-CVRP benchmark dataset into the expected local path:

```bash
git clone https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances data/raw/e_cvrp
```

Alternatively, download the `.evrp` files directly from GitHub and place them inside `data/raw/e_cvrp/`.

---

## How to Load an Instance in Python

To load a benchmark instance into our standard `VRPInstance` class:

```python
from src.data.e_cvrp_loader import load_e_cvrp_instance

# Load E-CVRP benchmark instance (extracts classical CVRP fields)
instance = load_e_cvrp_instance("data/raw/e_cvrp/E-n29-k4-s7.evrp")

print(f"Number of customers: {instance.num_customers}")
print(f"Number of vehicles:  {instance.num_vehicles}")
print(f"Vehicle capacity:    {instance.vehicle_capacity}")
print(f"Total demand:        {instance.demands.sum()}")
```

Accessing raw EV metadata for future EVRP phases:

```python
raw_meta = instance.raw_data
print(f"Battery Capacity:    {raw_meta['energy_capacity']}")
print(f"Energy Consumption:  {raw_meta['energy_consumption']}")
print(f"Recharging Stations: {raw_meta['station_node_ids']}")
```
