/**
 * API client layer communicating with FastAPI backend.
 * Dynamically probes candidate ports (8000, 8080, 8001) for maximum flexibility.
 */

const CANDIDATE_PORTS = [8000, 8080, 8001];
let activeApiBase = 'http://localhost:8000/api';

/**
 * Checks backend health status across candidate ports and selects active URL.
 * @returns {Promise<boolean>}
 */
export async function checkBackendHealth() {
  for (const port of CANDIDATE_PORTS) {
    const url = `http://localhost:${port}/api`;
    try {
      const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'ok') {
          activeApiBase = url;
          return true;
        }
      }
    } catch (err) {
      // Continue checking next candidate port
    }
  }
  return false;
}

/**
 * Fetches available CEC-2020 .evrp benchmark instances from backend.
 * @returns {Promise<Array<{name: string, customers: number, vehicles: number, vehicle_capacity: number, optimal_value: any, stations: number}>>}
 */
export async function getBenchmarkInstances() {
  const res = await fetch(`${activeApiBase}/benchmark/instances`);
  if (!res.ok) {
    throw new Error(`Failed to fetch benchmark instances (HTTP ${res.status})`);
  }
  const data = await res.json();
  return data.instances || [];
}

/**
 * Fetches parsed details for a specific benchmark instance.
 * @param {string} instanceName - e.g. "E-n29-k4-s7.evrp"
 * @returns {Promise<Object>}
 */
export async function getBenchmarkInstance(instanceName) {
  const res = await fetch(`${activeApiBase}/benchmark/instances/${encodeURIComponent(instanceName)}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch instance details for '${instanceName}' (HTTP ${res.status})`);
  }
  return await res.json();
}

/**
 * Runs Python Genetic Algorithm on backend for the selected benchmark instance.
 * @param {string} instanceName 
 * @param {number} [seed=42] 
 * @param {Object} [config={}] - Optional parameters (population_size, generations)
 * @returns {Promise<Object>} Full VRPSolution response dictionary from backend
 */
export async function runBenchmarkGA(instanceName, seed = 42, config = {}) {
  const payload = {
    instance_name: instanceName,
    seed: seed,
    population_size: config.population_size || 100,
    generations: config.generations || 200,
    crossover_rate: config.crossover_rate || 0.85,
    mutation_rate: config.mutation_rate || 0.25,
  };

  const res = await fetch(`${activeApiBase}/benchmark/ga`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `GA execution failed (HTTP ${res.status})`);
  }

  return await res.json();
}
