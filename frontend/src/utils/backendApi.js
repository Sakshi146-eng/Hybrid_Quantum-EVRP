/**
 * API client layer communicating with FastAPI backend.
 * Dynamically probes candidate ports (8000, 8080, 8001) for maximum flexibility.
 */

const CANDIDATE_PORTS = [8001, 8000, 8080];
let activeApiBase = 'http://localhost:8001/api';

/**
 * Checks backend health status across candidate ports and selects active URL.
 * @returns {Promise<boolean>}
 */
export async function checkBackendHealth() {
  for (const port of CANDIDATE_PORTS) {
    const url = `http://localhost:${port}/api`;
    try {
      const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1200) });
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

/**
 * Runs Simulated QAOA on backend for a small benchmark sub-instance.
 * @param {string} instanceName 
 * @param {number} [customerCount=3]
 * @param {Object} [config={}]
 * @returns {Promise<Object>}
 */
export async function runBenchmarkQAOA(instanceName, customerCount = 3, config = {}) {
  const payload = {
    instance_name: instanceName,
    customer_count: customerCount,
    customer_indices: config.customer_indices || null,
    p_depth: config.p_depth || 1,
    shots: config.shots || 1024,
    max_iter: config.max_iter || 50,
    seed: config.seed !== undefined ? config.seed : 42,
  };

  const res = await fetch(`${activeApiBase}/benchmark/qaoa`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `QAOA execution failed (HTTP ${res.status})`);
  }

  return await res.json();
}

/**
 * Runs Exact Baseline, Classical GA, and Simulated QAOA comparison on backend.
 * @param {string} instanceName 
 * @param {number} [customerCount=3]
 * @param {Object} [config={}]
 * @returns {Promise<Object>}
 */
export async function runBenchmarkCompare(instanceName, customerCount = 3, config = {}) {
  const payload = {
    instance_name: instanceName,
    customer_count: customerCount,
    customer_indices: config.customer_indices || null,
    p_depth: config.p_depth || 1,
    shots: config.shots || 1024,
    max_iter: config.max_iter || 50,
    ga_population_size: config.ga_population_size || 50,
    ga_generations: config.ga_generations || 100,
    hybrid_iterations: config.hybrid_iterations || 3,
    seed: config.seed !== undefined ? config.seed : 42,
  };

  const res = await fetch(`${activeApiBase}/benchmark/compare`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Comparison failed (HTTP ${res.status})`);
  }

  return await res.json();
}

/**
 * Runs 6-algorithm EVRP comparative evaluation on backend.
 * @param {string} instanceName
 * @param {number} [customerCount=4]
 * @param {Object} [config={}]
 * @returns {Promise<Object>}
 */
export async function runEVRPCompare(instanceName, customerCount = 4, config = {}) {
  const payload = {
    instance_name: instanceName,
    customer_count: customerCount,
    customer_indices: config.customer_indices || null,
    algorithms: config.algorithms || ["ga", "ga_ls", "rl", "rl_ls", "rl_qaoa", "full_hybrid"],
    battery_capacity: config.battery_capacity || null,
    base_energy_rate: config.base_energy_rate || null,
    load_factor: config.load_factor !== undefined ? config.load_factor : 0.2,
    charging_power: config.charging_power || 50.0,
    charging_price: config.charging_price || 0.25,
    seed: config.seed !== undefined ? config.seed : 42,
  };

  const res = await fetch(`${activeApiBase}/evrp/compare`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `EVRP comparison failed (HTTP ${res.status})`);
  }

  return await res.json();
}

/**
 * Triggers background RL training on an EVRP instance.
 * @param {string} instanceName
 * @param {number} [customerCount=4]
 * @param {number} [episodes=30]
 * @param {Object} [config={}]
 * @returns {Promise<Object>}
 */
export async function trainRLAgent(instanceName, customerCount = 4, episodes = 30, config = {}) {
  const payload = {
    instance_name: instanceName,
    customer_count: customerCount,
    episodes: episodes,
    lr: config.lr || 0.001,
    gamma: config.gamma || 0.99,
    seed: config.seed !== undefined ? config.seed : 42,
  };

  const res = await fetch(`${activeApiBase}/evrp/rl/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `RL training failed (HTTP ${res.status})`);
  }

  return await res.json();
}

/**
 * Queries current RL training status and reward history.
 * @returns {Promise<Object>}
 */
export async function getRLTrainingStatus() {
  const res = await fetch(`${activeApiBase}/evrp/rl/status`);
  if (!res.ok) {
    throw new Error(`Failed to query RL status (HTTP ${res.status})`);
  }
  return await res.json();
}

/**
 * Fetches persistent ablation study results across multiple random seeds.
 * @returns {Promise<Object>}
 */
export async function getEVRPAblationResults() {
  const res = await fetch(`${activeApiBase}/evrp/ablation`);
  if (!res.ok) {
    throw new Error(`Failed to fetch ablation results (HTTP ${res.status})`);
  }
  return await res.json();
}

/**
 * Runs a live 6-component ablation study.
 */
export async function runLiveAblationStudy(instanceName, customerCount = 4, seed = 42) {
  const res = await fetch(`${activeApiBase}/evrp/ablation/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ instance_name: instanceName, customer_count: customerCount, seed }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Ablation study failed (HTTP ${res.status})`);
  }
  return await res.json();
}

/**
 * Executes multi-seed evaluation across multiple random seeds.
 */
export async function evaluateMultiSeed(instanceName, customerCount = 4, algorithms = null, seeds = [42, 101, 2024]) {
  const payload = {
    instance_name: instanceName,
    customer_count: customerCount,
    algorithms: algorithms,
    seeds: seeds,
  };
  const res = await fetch(`${activeApiBase}/evrp/multi-seed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Multi-seed comparison failed (HTTP ${res.status})`);
  }
  return await res.json();
}

/**
 * Evaluates customer-count scalability curve.
 */
export async function evaluateScalability(instanceName, customerCounts = [4, 8, 12], algorithms = null, config = {}) {
  const payload = {
    instance_name: instanceName,
    customer_counts: customerCounts,
    algorithms: algorithms,
    battery_capacity: config.battery_capacity !== undefined ? config.battery_capacity : 80.0,
    base_energy_rate: config.base_energy_rate !== undefined ? config.base_energy_rate : 0.2,
    load_factor: config.load_factor !== undefined ? config.load_factor : 0.2,
    charging_power: config.charging_power !== undefined ? config.charging_power : 50.0,
    charging_price: config.charging_price !== undefined ? config.charging_price : 0.25,
    seed: config.seed !== undefined ? config.seed : 42,
  };
  const res = await fetch(`${activeApiBase}/evrp/scalability`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Scalability evaluation failed (HTTP ${res.status})`);
  }
  return await res.json();
}

/**
 * Fetches the persistent validated scalability benchmark results across
 * 4, 8, 12, 20, 50, and 100 customers directly from the backend.
 */
export async function getValidatedScalabilityResults() {
  const res = await fetch(`${activeApiBase}/evrp/scalability/validated`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Failed to fetch validated scalability results (HTTP ${res.status})`);
  }
  return await res.json();
}
