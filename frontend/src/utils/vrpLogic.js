/**
 * Frontend JavaScript implementation of Classical VRP instances & distance calculations.
 * Supports user-configured instances, dynamic customer tables, synthetic random generation,
 * and comprehensive feasibility validation.
 */

export function getDeterministicToyInstance(numCustomers = 5, vehicleCapacity = 10.0, numVehicles = 2) {
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
    // 5 to 20 customers default grid construction
    coords = [
      { id: 0, type: 'Depot', x: 0.0, y: 0.0, demand: 0.0 },
      { id: 1, type: 'Cust 1', x: 0.0, y: 3.0, demand: 2.0 },
      { id: 2, type: 'Cust 2', x: 4.0, y: 0.0, demand: 3.0 },
      { id: 3, type: 'Cust 3', x: 4.0, y: 3.0, demand: 4.0 },
      { id: 4, type: 'Cust 4', x: 0.0, y: 4.0, demand: 2.0 },
      { id: 5, type: 'Cust 5', x: 3.0, y: 0.0, demand: 1.0 },
    ];

    // For N > 5, generate remaining nodes in grid layout
    for (let i = 6; i <= numCustomers; i++) {
      const gx = parseFloat(((i % 5) * 1.5 + 1.0).toFixed(1));
      const gy = parseFloat((Math.floor(i / 5) * 1.5 + 1.0).toFixed(1));
      const gd = parseFloat(((i % 4) + 1.0).toFixed(1));
      coords.push({
        id: i,
        type: `Cust ${i}`,
        x: gx,
        y: gy,
        demand: gd,
      });
    }
  }

  return buildInstanceFromNodes(coords, numCustomers, vehicleCapacity, numVehicles);
}

export function generateRandomCustomers(numCustomers = 5, vehicleCapacity = 10.0, numVehicles = 2, depotX = 0.0, depotY = 0.0) {
  const coords = [
    { id: 0, type: 'Depot', x: parseFloat(depotX) || 0.0, y: parseFloat(depotY) || 0.0, demand: 0.0 }
  ];

  for (let i = 1; i <= numCustomers; i++) {
    const rx = parseFloat((Math.random() * 6.0 + 0.5).toFixed(1));
    const ry = parseFloat((Math.random() * 6.0 + 0.5).toFixed(1));
    const rdemand = parseFloat((Math.floor(Math.random() * 4) + 1).toFixed(1));

    coords.push({
      id: i,
      type: `Cust ${i}`,
      x: rx,
      y: ry,
      demand: rdemand,
    });
  }

  return buildInstanceFromNodes(coords, numCustomers, vehicleCapacity, numVehicles);
}

export function buildInstanceFromNodes(nodes, numCustomers, vehicleCapacity, numVehicles = 2) {
  const numNodes = nodes.length;
  const demands = nodes.map((n) => parseFloat(n.demand) || 0.0);
  const distanceMatrix = Array.from({ length: numNodes }, () => Array(numNodes).fill(0));

  for (let i = 0; i < numNodes; i++) {
    for (let j = 0; j < numNodes; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      distanceMatrix[i][j] = Math.sqrt(dx * dx + dy * dy);
    }
  }

  return {
    numNodes,
    numCustomers: nodes.length - 1,
    numVehicles: parseInt(numVehicles, 10) || 1,
    depotIdx: 0,
    vehicleCapacity: parseFloat(vehicleCapacity) || 10.0,
    nodes,
    demands,
    distanceMatrix,
  };
}

/**
 * Comprehensive VRP Instance Feasibility & Syntax Validator
 */
export function validateVRPInstanceConfig(instance) {
  if (!instance || !instance.nodes || !instance.demands) {
    return { isValid: false, errors: ['Invalid instance data structure.'] };
  }

  const { numCustomers, vehicleCapacity, numVehicles, demands, nodes } = instance;
  const errors = [];

  if (numCustomers < 1) {
    errors.push('Number of customers must be at least 1.');
  }

  if (numVehicles < 1) {
    errors.push('Number of vehicles must be at least 1.');
  }

  if (vehicleCapacity <= 0 || isNaN(vehicleCapacity)) {
    errors.push('Vehicle capacity must be greater than 0.');
  }

  // Check valid numeric values for coordinates and demands
  for (const n of nodes) {
    if (isNaN(n.x) || isNaN(n.y) || !isFinite(n.x) || !isFinite(n.y)) {
      errors.push(`Invalid coordinate values for Node ${n.id}.`);
    }
    if (isNaN(n.demand) || n.demand < 0 || !isFinite(n.demand)) {
      errors.push(`Invalid demand value for Node ${n.id}.`);
    }
  }

  const customerDemands = demands.slice(1);
  const maxDemand = customerDemands.length > 0 ? Math.max(...customerDemands) : 0;
  const totalDemand = customerDemands.reduce((a, b) => a + b, 0);
  const maxFleetCapacity = numVehicles * vehicleCapacity;

  if (vehicleCapacity < maxDemand) {
    errors.push(
      `Invalid configuration: vehicle capacity (${vehicleCapacity}) is smaller than the largest customer demand (${maxDemand}).`
    );
  } else if (maxFleetCapacity < totalDemand) {
    errors.push(
      `Instance infeasible: total demand (${totalDemand}) exceeds total fleet capacity (${maxFleetCapacity}).`
    );
  }

  return {
    isValid: errors.length === 0,
    errors,
    maxDemand,
    totalDemand,
    numVehicles,
    maxFleetCapacity,
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
