/**
 * Decoupled Solver API Client / Runner Layer.
 * Provides standardized execution of VRP algorithms returning VRPSolution dictionaries.
 * Prepared for clean integration with future Python backend API endpoints.
 */

import { evaluateRoute, validateVRPInstanceConfig } from './vrpLogic';

/**
 * Runs a Genetic Algorithm on a given VRPInstance object using client-side JS implementation
 * matching Python's GeneticAlgorithmSolver logic.
 */
export function runGeneticAlgorithm(instance, config = {}) {
  // First check instance configuration feasibility
  const configVal = validateVRPInstanceConfig(instance);
  if (!configVal.isValid) {
    throw new Error(configVal.errors[0]);
  }

  const {
    populationSize = 50,
    generations = 100,
    crossoverRate = 0.8,
    mutationRate = 0.2,
    tournamentSize = 3,
    eliteSize = 2,
  } = config;

  const startTime = performance.now();

  const customerIndices = instance.nodes
    .filter((n) => n.id !== instance.depotIdx)
    .map((n) => n.id);

  if (customerIndices.length === 0) {
    const r = [[instance.depotIdx, instance.depotIdx]];
    return formatVRPSolution('Genetic Algorithm', r, instance, 0.0);
  }

  // 1. Generate Initial Population
  let population = [];
  for (let i = 0; i < populationSize; i++) {
    const perm = [...customerIndices];
    shuffleArray(perm);
    population.push(perm);
  }

  let bestChromosome = [...customerIndices];
  let bestFitness = -Infinity;
  let bestRoutes = null;
  let bestDistance = Infinity;
  const historyDistance = [];

  // Helper: Decode chromosome to vehicle routes (Greedy Capacity Split)
  const decodeChromosome = (chrom) => {
    const depot = instance.depotIdx;
    const cap = instance.vehicleCapacity;
    const demands = instance.demands;

    const routes = [];
    let currentRoute = [depot];
    let currentLoad = 0;

    for (const cust of chrom) {
      const d = demands[cust] || 0;
      if (cap && currentLoad + d > cap && currentRoute.length > 1) {
        currentRoute.push(depot);
        routes.push(currentRoute);
        currentRoute = [depot, cust];
        currentLoad = d;
      } else {
        currentRoute.push(cust);
        currentLoad += d;
      }
    }
    if (currentRoute.length > 1) {
      currentRoute.push(depot);
      routes.push(currentRoute);
    }
    return routes;
  };

  // Helper: Fitness Evaluation
  const evaluateFitness = (chrom) => {
    const rts = decodeChromosome(chrom);
    let totalDist = 0;
    let totalPenalty = 0;

    for (const r of rts) {
      const evalRes = evaluateRoute(instance, r);
      totalDist += evalRes.totalDistance;
      if (!evalRes.isValid) {
        totalPenalty += 500;
      }
    }

    if (instance.numVehicles && rts.length > instance.numVehicles) {
      totalPenalty += (rts.length - instance.numVehicles) * 1000;
    }

    const fitness = 1.0 / (totalDist + totalPenalty + 1e-6);
    return { fitness, totalDist, routes: rts };
  };

  // 2. Evolution Loop
  for (let gen = 0; gen < generations; gen++) {
    const evalResults = population.map((chrom) => evaluateFitness(chrom));
    const fitnesses = evalResults.map((r) => r.fitness);

    for (let i = 0; i < population.length; i++) {
      if (fitnesses[i] > bestFitness) {
        bestFitness = fitnesses[i];
        bestChromosome = [...population[i]];
        bestRoutes = evalResults[i].routes;
        bestDistance = evalResults[i].totalDist;
      }
    }
    historyDistance.push(bestDistance);

    // Elitism
    const indices = Array.from({ length: populationSize }, (_, idx) => idx);
    indices.sort((a, b) => fitnesses[b] - fitnesses[a]);

    const newPopulation = [];
    for (let e = 0; e < Math.min(eliteSize, populationSize); e++) {
      newPopulation.push([...population[indices[e]]]);
    }

    // Tournament selection & OX Crossover & Swap Mutation
    while (newPopulation.length < populationSize) {
      const p1 = tournamentSelect(population, fitnesses, tournamentSize);
      const p2 = tournamentSelect(population, fitnesses, tournamentSize);

      let offspring = Math.random() < crossoverRate ? orderCrossover(p1, p2) : [...p1];
      if (Math.random() < mutationRate) {
        offspring = swapMutation(offspring);
      }
      newPopulation.push(offspring);
    }
    population = newPopulation;
  }

  const endTime = performance.now();
  const runtimeSeconds = (endTime - startTime) / 1000.0;

  return formatVRPSolution('Genetic Algorithm', bestRoutes, instance, runtimeSeconds, {
    generations,
    populationSize,
    crossoverRate,
    mutationRate,
    bestDistanceHistory: historyDistance,
  });
}

/**
 * Format solution into standardized VRPSolution dictionary shape
 */
function formatVRPSolution(algorithmName, routes, instance, runtimeSeconds, metadata = {}) {
  let totalDistance = 0;
  const vehicleLoads = [];
  let isFeasible = true;
  const visitedCustomers = new Set();

  for (const r of routes) {
    const evalRes = evaluateRoute(instance, r);
    totalDistance += evalRes.totalDistance;
    vehicleLoads.push(evalRes.totalDemand);
    if (!evalRes.isValid) {
      isFeasible = false;
    }
    for (const node of r) {
      if (node !== instance.depotIdx) {
        visitedCustomers.add(node);
      }
    }
  }

  // Check all customers visited exactly once
  if (visitedCustomers.size !== instance.numCustomers) {
    isFeasible = false;
  }

  if (instance.numVehicles && routes.length > instance.numVehicles) {
    isFeasible = false;
  }

  return {
    algorithm_name: algorithmName,
    routes: routes,
    total_distance: parseFloat(totalDistance.toFixed(2)),
    feasible: isFeasible,
    runtime_seconds: parseFloat(runtimeSeconds.toFixed(6)),
    vehicle_loads: vehicleLoads,
    metadata,
  };
}

// Genetic Helpers
function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function tournamentSelect(pop, fitnesses, k) {
  let bestIdx = Math.floor(Math.random() * pop.length);
  for (let i = 1; i < k; i++) {
    const randIdx = Math.floor(Math.random() * pop.length);
    if (fitnesses[randIdx] > fitnesses[bestIdx]) {
      bestIdx = randIdx;
    }
  }
  return pop[bestIdx];
}

function orderCrossover(p1, p2) {
  const len = p1.length;
  if (len <= 2) return [...p1];
  const cut1 = Math.floor(Math.random() * (len - 1));
  const cut2 = Math.floor(Math.random() * (len - cut1)) + cut1;

  const offspring = Array(len).fill(null);
  for (let i = cut1; i <= cut2; i++) {
    offspring[i] = p1[i];
  }

  const copiedSet = new Set(offspring.filter((x) => x !== null));
  const fillVals = p2.filter((x) => !copiedSet.has(x));

  let fIdx = 0;
  for (let i = 0; i < len; i++) {
    if (offspring[i] === null) {
      offspring[i] = fillVals[fIdx++];
    }
  }
  return offspring;
}

function swapMutation(chrom) {
  const res = [...chrom];
  if (res.length >= 2) {
    const i = Math.floor(Math.random() * res.length);
    let j = Math.floor(Math.random() * res.length);
    while (i === j) j = Math.floor(Math.random() * res.length);
    [res[i], res[j]] = [res[j], res[i]];
  }
  return res;
}

/**
 * Main Solver Dispatcher preparing for future API connection
 */
export async function runClassicalSolver(algorithmKey, instance, config = {}) {
  if (algorithmKey === 'ga') {
    return runGeneticAlgorithm(instance, config);
  } else if (algorithmKey === 'aco' || algorithmKey === 'sa' || algorithmKey === 'tabu_search') {
    throw new Error(`Algorithm '${algorithmKey.toUpperCase()}' is currently under development (Coming Soon).`);
  } else {
    throw new Error(`Unknown algorithm key '${algorithmKey}'.`);
  }
}
