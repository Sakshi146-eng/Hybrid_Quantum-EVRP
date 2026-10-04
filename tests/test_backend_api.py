"""
Unit tests for FastAPI backend API endpoints.
"""
from backend.main import (
    health_check,
    list_benchmark_instances,
    get_benchmark_instance_details,
    run_benchmark_ga,
    GARequest,
)


def test_api_health():
    res = health_check()
    assert res == {"status": "ok"}


def test_api_benchmark_instances_list():
    res = list_benchmark_instances()
    assert "instances" in res
    assert isinstance(res["instances"], list)
    assert len(res["instances"]) >= 24
    names = [inst["name"] for inst in res["instances"]]
    assert "E-n29-k4-s7.evrp" in names


def test_api_benchmark_instance_details():
    data = get_benchmark_instance_details("E-n29-k4-s7.evrp")
    assert data["name"] == "E-n29-k4-s7.evrp"
    assert data["num_customers"] == 21
    assert data["num_vehicles"] == 4
    assert data["vehicle_capacity"] == 6000.0
    assert data["total_demand"] == 22500.0
    assert "depot" in data
    assert "customers" in data
    assert len(data["customers"]) == 21


def test_api_benchmark_ga_execution():
    req = GARequest(
        instance_name="E-n29-k4-s7.evrp",
        seed=42,
        population_size=30,
        generations=20
    )
    data = run_benchmark_ga(req)
    assert data["instance_name"] == "E-n29-k4-s7.evrp"
    assert data["algorithm"] == "Genetic Algorithm"
    assert data["num_customers"] == 21
    assert data["num_vehicles"] == 4
    assert data["vehicle_capacity"] == 6000.0
    assert data["total_distance"] > 0.0
    assert "routes" in data
    assert len(data["routes"]) > 0

    for r in data["routes"]:
        assert "vehicle" in r
        assert "route" in r
        assert "load" in r
        assert r["route"][0] == 0
        assert r["route"][-1] == 0
