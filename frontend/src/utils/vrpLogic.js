/**
 * Frontend JavaScript implementation of Classical VRP toy instances & distance calculations.
 * Supports both fixed deterministic instances and random synthetic instance generation.
 */

export function getDeterministicToyInstance(numCustomers = 3, vehicleCapacity = 10.0) {
  let coords = [];

  if (numCustomers === 3) {
    coords = [
      { id: 0, type: 'Depot', x: 0.0, y: 0.0, demand: 0.0 },
      { id: 1, type: 'Cust 1', x: 0.0, y: 3.0, demand: 2.0 },
      { id: 2, type: 'Cust 2', x: 4.0, y: 0.0, demand: 3.0 },
      { id: 3, type: 'Cust 3', x: 4.0, y: 3.0, demand: 4.0 },
    ];
  } else if (numCustomers === 4) {
    coords = [
      { id: 0, type: 'Depot', x: 0.0, y: 0.0, demand: 0.0 },
      { id: 1, type: 'Cust 1', x: 0.0, y: 3.0, demand: 2.0 },
      { id: 2, type: 'Cust 2', x: 4.0, y: 0.0, demand: 3.0 },
      { id: 3, type: 'Cust 3', x: 4.0, y: 3.0, demand: 4.0 },
      { id: 4, type: 'Cust 4', x: 0.0, y: 4.0, demand: 2.0 },
    ];
  } else {
    // 5 customers
    coords = [
      { id: 0, type: 'Depot', x: 0.0, y: 0.0, demand: 0.0 },
      { id: 1, type: 'Cust 1', x: 0.0, y: 3.0, demand: 2.0 },
      { id: 2, type: 'Cust 2', x: 4.0, y: 0.0, demand: 3.0 },
      { id: 3, type: 'Cust 3', x: 4.0, y: 3.0, demand: 4.0 },
      { id: 4, type: 'Cust 4', x: 0.0, y: 4.0, demand: 2.0 },
      { id: 5, type: 'Cust 5', x: 3.0, y: 0.0, demand: 1.0 },
    ];
  }

  return buildInstanceFromNodes(coords, numCustomers, vehicleCapacity);
}

export function generateSyntheticInstance(numCustomers = 3, vehicleCapacity = 10.0) {
  const coords = [
    { id: 0, type: 'Depot', x: 0.0, y: 0.0, demand: 0.0 }
  ];

  for (let i = 1; i <= numCustomers; i++) {
    // Random coordinates between 0.5 and 5.0
    const rx = parseFloat((Math.random() * 5.0).toFixed(1));
    const ry = parseFloat((Math.random() * 5.0).toFixed(1));
    const rdemand = parseFloat((Math.floor(Math.random() * 4) + 1).toFixed(1));

    coords.push({
      id: i,
      type: `Cust ${i}`,
      x: rx,
      y: ry,
      demand: rdemand,
    });
  }

  return buildInstanceFromNodes(coords, numCustomers, vehicleCapacity);
}

function buildInstanceFromNodes(nodes, numCustomers, vehicleCapacity) {
  const numNodes = nodes.length;
  const demands = nodes.map(n => n.demand);
  const distanceMatrix = Array.from({ length: numNodes }, () => Array(numNodes).fill(0));

  for (let i = 0; i < numNodes; i++) {
    for (let j = 0; j < numNodes; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      distanceMatrix[i][j] = Math.sqrt(dx * dx + dy * dy);
    }
  }

  // Default candidate route visiting all customers sequentially
  const candidateRoute = [0];
  for (let i = 1; i < numNodes; i++) {
    candidateRoute.push(i);
  }
  candidateRoute.push(0);

  return {
    numNodes,
    numCustomers,
    depotIdx: 0,
    vehicleCapacity: parseFloat(vehicleCapacity) || 10.0,
    nodes,
    demands,
    distanceMatrix,
    candidateRoute,
  };
}

export function evaluateRoute(instance, route) {
  const { distanceMatrix, demands, vehicleCapacity, depotIdx } = instance;
  const errors = [];

  if (!route || route.length < 2) {
    return {
      isValid: false,
      totalDistance: 0,
      totalDemand: 0,
      capacityValid: false,
      errors: ['Route is empty or contains less than 2 nodes.'],
    };
  }

  let dist = 0;
  let totalDemand = 0;

  // Check valid integer node indices
  for (let i = 0; i < route.length; i++) {
    const node = route[i];
    if (typeof node !== 'number' || isNaN(node) || node < 0 || node >= instance.numNodes) {
      errors.push(`Node index ${node} out of valid range [0, ${instance.numNodes - 1}].`);
    }
  }

  if (errors.length > 0) {
    return {
      isValid: false,
      totalDistance: 0,
      totalDemand: 0,
      capacityValid: false,
      errors,
    };
  }

  // Check depot endpoints
  if (route[0] !== depotIdx || route[route.length - 1] !== depotIdx) {
    errors.push(`Route must start and end at depot (index ${depotIdx}), got start=${route[0]}, end=${route[route.length - 1]}.`);
  }

  // Calculate distance and demand
  for (let i = 0; i < route.length; i++) {
    const node = route[i];
    if (node !== depotIdx) {
      totalDemand += demands[node];
    }
    if (i < route.length - 1) {
      const nextNode = route[i + 1];
      dist += distanceMatrix[node][nextNode];
    }
  }

  let capacityValid = true;
  if (totalDemand > vehicleCapacity) {
    errors.push(`Total visited demand (${totalDemand.toFixed(1)}) exceeds vehicle capacity (${vehicleCapacity.toFixed(1)}).`);
    capacityValid = false;
  }

  return {
    isValid: errors.length === 0,
    totalDistance: parseFloat(dist.toFixed(2)),
    totalDemand: parseFloat(totalDemand.toFixed(1)),
    capacityValid,
    errors,
  };
}
