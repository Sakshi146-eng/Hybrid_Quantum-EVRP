import React, { useState, useEffect } from 'react';
import {
  Truck,
  BatteryCharging,
  Binary,
  Cpu,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
  Shuffle,
  Activity,
  Sliders,
  Database,
  Grid,
  Zap,
  Server,
  RefreshCw,
} from 'lucide-react';

import {
  getDeterministicToyInstance,
  generateRandomCustomers,
  buildInstanceFromNodes,
  validateVRPInstanceConfig,
} from './utils/vrpLogic';
import { runClassicalSolver } from './utils/solverApi';
import {
  getBenchmarkInstances,
  getBenchmarkInstance,
  runBenchmarkGA,
  runBenchmarkQAOA,
  runBenchmarkCompare,
  checkBackendHealth,
} from './utils/backendApi';
import RouteVisualizer from './components/RouteVisualizer';
import EVRPDashboard from './components/EVRPDashboard';
import ErrorBoundary from './components/ErrorBoundary';

export default function App() {
  // Mode Selector State: 'demo' (Interactive Demo) vs 'benchmark' (CEC-2020 Benchmark) vs 'quantum' (Quantum-Classical Comparison)
  const [dataSourceMode, setDataSourceMode] = useState('demo');

  // --- DEMO MODE STATES ---
  const [numCustomers, setNumCustomers] = useState(5);
  const [numVehicles, setNumVehicles] = useState(2);
  const [capacity, setCapacity] = useState(15.0);
  const [depotX, setDepotX] = useState(0.0);
  const [depotY, setDepotY] = useState(0.0);
  const [customerNodes, setCustomerNodes] = useState([]);

  // --- BENCHMARK MODE STATES ---
  const [benchmarkInstances, setBenchmarkInstances] = useState([]);
  const [selectedInstanceName, setSelectedInstanceName] = useState('');
  const [benchmarkDetails, setBenchmarkDetails] = useState(null);
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [backendError, setBackendError] = useState(null);
  const [isFetchingBenchmark, setIsFetchingBenchmark] = useState(false);

  // Common Algorithm & Hyperparameter State
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('ga');
  const [gaPopSize, setGaPopSize] = useState(100);
  const [gaGenerations, setGaGenerations] = useState(200);
  const [gaCrossoverRate, setGaCrossoverRate] = useState(0.85);
  const [gaMutationRate, setGaMutationRate] = useState(0.25);
  const [gaSeed, setGaSeed] = useState(42);

  // --- QUANTUM VS CLASSICAL COMPARISON STATES ---
  const [compareCustomerCount, setCompareCustomerCount] = useState(4);
  const [qaoaDepth, setQaoaDepth] = useState(2);
  const [qaoaShots, setQaoaShots] = useState(1024);
  const [qaoaMaxIter, setQaoaMaxIter] = useState(35);
  const [compareSeed, setCompareSeed] = useState(42);
  const [compareGaPopSize, setCompareGaPopSize] = useState(50);
  const [compareGaGenerations, setCompareGaGenerations] = useState(100);
  const [hybridIterations, setHybridIterations] = useState(3);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [activeCompareView, setActiveCompareView] = useState('hybrid');
  const [isComparing, setIsComparing] = useState(false);

  // Solution & Toast State
  const [instance, setInstance] = useState(null);
  const [solution, setSolution] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  const showToast = (msg, type = 'info') => {
    setToastMessage({ text: msg, type });
    setTimeout(() => {
      setToastMessage(null);
    }, 4000);
  };

  // --- DEMO MODE INITIALIZATION ---
  useEffect(() => {
    if (dataSourceMode === 'demo') {
      const baseInst = getDeterministicToyInstance(numCustomers, capacity, numVehicles);
      const custs = baseInst.nodes.filter((n) => n.id !== 0);
      setCustomerNodes(custs);
      setSolution(null);
    }
  }, [numCustomers, dataSourceMode]);

  useEffect(() => {
    if (dataSourceMode === 'demo') {
      const fullNodes = [
        { id: 0, type: 'Depot', x: parseFloat(depotX) || 0.0, y: parseFloat(depotY) || 0.0, demand: 0.0 },
        ...customerNodes.map((c, i) => ({
          id: i + 1,
          type: `Cust ${i + 1}`,
          x: parseFloat(c.x) || 0.0,
          y: parseFloat(c.y) || 0.0,
          demand: parseFloat(c.demand) || 0.0,
        })),
      ];
      const inst = buildInstanceFromNodes(fullNodes, customerNodes.length, capacity, numVehicles);
      setInstance(inst);
    }
  }, [customerNodes, depotX, depotY, capacity, numVehicles, dataSourceMode]);

  // --- BENCHMARK MODE DISCOVERY ---
  const loadBenchmarkDiscovery = async () => {
    setIsFetchingBenchmark(true);
    setBackendError(null);
    try {
      const isHealthy = await checkBackendHealth();
      setBackendHealthy(isHealthy);

      if (!isHealthy) {
        setBackendError('FastAPI backend server is offline. Please start uvicorn backend.main:app --reload on http://localhost:8000.');
        setIsFetchingBenchmark(false);
        return;
      }

      const instList = await getBenchmarkInstances();
      setBenchmarkInstances(instList);

      if (instList.length > 0) {
        const defaultName = instList.some((i) => i.name === 'E-n29-k4-s7.evrp')
          ? 'E-n29-k4-s7.evrp'
          : instList[0].name;
        setSelectedInstanceName(defaultName);
        await fetchAndSetBenchmarkDetails(defaultName);
      }
    } catch (err) {
      setBackendError(err.message);
    } finally {
      setIsFetchingBenchmark(false);
    }
  };

  const fetchAndSetBenchmarkDetails = async (instanceName) => {
    try {
      setIsFetchingBenchmark(true);
      const details = await getBenchmarkInstance(instanceName);
      setBenchmarkDetails(details);
      setSolution(null);

      // Build VRPInstance object format for RouteVisualizer
      const nodes = [
        { id: 0, type: 'Depot', x: details.depot.x, y: details.depot.y, demand: 0 },
        ...details.customers.map((c) => ({
          id: c.id,
          type: `Cust ${c.id}`,
          x: c.x,
          y: c.y,
          demand: c.demand,
        })),
      ];

      setInstance({
        nodes,
        numCustomers: details.num_customers,
        numVehicles: details.num_vehicles,
        vehicleCapacity: details.vehicle_capacity,
        totalDemand: details.total_demand,
        depotIdx: 0,
        charging_stations: details.charging_stations || [],
      });
    } catch (err) {
      setBackendError(err.message);
    } finally {
      setIsFetchingBenchmark(false);
    }
  };

  useEffect(() => {
    if (dataSourceMode === 'benchmark' || dataSourceMode === 'quantum') {
      loadBenchmarkDiscovery();
    }
  }, [dataSourceMode]);

  const handleBenchmarkSelect = async (e) => {
    const name = e.target.value;
    setSelectedInstanceName(name);
    await fetchAndSetBenchmarkDetails(name);
  };

  const handleRunComparison = async () => {
    setIsComparing(true);
    setBackendError(null);

    try {
      if (!selectedInstanceName) {
        throw new Error('Please select a valid benchmark instance.');
      }

      const compData = await runBenchmarkCompare(
        selectedInstanceName,
        parseInt(compareCustomerCount, 10) || 4,
        {
          p_depth: parseInt(qaoaDepth, 10) || 2,
          shots: parseInt(qaoaShots, 10) || 1024,
          max_iter: parseInt(qaoaMaxIter, 10) || 35,
          ga_population_size: parseInt(compareGaPopSize, 10) || 50,
          ga_generations: parseInt(compareGaGenerations, 10) || 100,
          hybrid_iterations: parseInt(hybridIterations, 10) || 3,
          seed: parseInt(compareSeed, 10) || 42,
        }
      );

      setComparisonResult(compData);
      showToast(
        `Comparison completed! Exact: ${compData.exact_baseline.total_distance.toFixed(2)} | GA: ${compData.classical_ga.total_distance.toFixed(2)} | Hybrid: ${compData.hybrid_ga_qaoa.total_distance.toFixed(2)}`,
        'success'
      );
    } catch (err) {
      setBackendError(err.message);
      showToast(err.message, 'error');
    } finally {
      setIsComparing(false);
    }
  };

  // Sub-instance format for RouteVisualizer during comparison view
  const subInstanceForVisualizer = comparisonResult ? {
    nodes: [
      { id: 0, type: 'Depot', x: comparisonResult.visualization_nodes.depot.x, y: comparisonResult.visualization_nodes.depot.y, demand: 0 },
      ...comparisonResult.visualization_nodes.customers.map((c) => ({
        id: c.id,
        type: `Cust ${c.id}`,
        x: c.x,
        y: c.y,
        demand: c.demand,
      })),
    ],
    charging_stations: [],
  } : null;

  const routesForVisualizer = comparisonResult ? (
    activeCompareView === 'exact'
      ? comparisonResult.exact_baseline?.routes || []
      : activeCompareView === 'ga'
      ? comparisonResult.classical_ga?.routes || []
      : activeCompareView === 'ga_ls'
      ? comparisonResult.ga_local_search?.routes || []
      : activeCompareView === 'qaoa'
      ? comparisonResult.qaoa_simulator?.routes || []
      : comparisonResult.hybrid_ga_qaoa?.routes || []
  ) : [];

  // Demo customer table cell editing
  const handleCustomerChange = (index, field, value) => {
    const updated = [...customerNodes];
    updated[index] = { ...updated[index], [field]: value };
    setCustomerNodes(updated);
    setSolution(null);
  };

  const handleGenerateRandomCustomers = () => {
    const newInst = generateRandomCustomers(numCustomers, capacity, numVehicles, depotX, depotY);
    const custs = newInst.nodes.filter((n) => n.id !== 0);
    setCustomerNodes(custs);
    setSolution(null);
    showToast(`Generated random customer coordinates for ${numCustomers} customers.`, 'success');
  };

  const configVal = instance && dataSourceMode === 'demo'
    ? validateVRPInstanceConfig(instance)
    : { isValid: true, errors: [] };

  // --- SOLVER EXECUTION ROUTER ---
  const handleRunAlgorithm = async () => {
    setIsRunning(true);
    setBackendError(null);

    try {
      if (dataSourceMode === 'benchmark') {
        if (!selectedInstanceName) {
          throw new Error('Please select a valid benchmark instance.');
        }

        const solData = await runBenchmarkGA(selectedInstanceName, parseInt(gaSeed, 10) || 42, {
          population_size: parseInt(gaPopSize, 10) || 100,
          generations: parseInt(gaGenerations, 10) || 200,
          crossover_rate: parseFloat(gaCrossoverRate) || 0.85,
          mutation_rate: parseFloat(gaMutationRate) || 0.25,
        });

        // Convert backend route response format to solution state
        const sol = {
          algorithm_name: solData.algorithm,
          total_distance: solData.total_distance,
          runtime_seconds: solData.runtime_seconds,
          feasible: solData.feasible,
          vehicle_loads: solData.routes.map((r) => r.load),
          routes: solData.routes.map((r) => r.route),
        };

        setSolution(sol);
        if (sol.feasible) {
          showToast(`GA solved benchmark ${solData.instance_name}! Total Distance: ${solData.total_distance} units. Feasible ✓`, 'success');
        } else {
          showToast(`GA completed ${solData.instance_name} with distance ${solData.total_distance} units. Feasible: NO`, 'warning');
        }
      } else {
        // Interactive Demo mode client-side execution
        if (!configVal.isValid) {
          showToast(configVal.errors[0], 'error');
          setIsRunning(false);
          return;
        }

        const sol = await runClassicalSolver(selectedAlgorithm, instance, {
          populationSize: parseInt(gaPopSize, 10) || 50,
          generations: parseInt(gaGenerations, 10) || 100,
          crossoverRate: parseFloat(gaCrossoverRate) || 0.8,
          mutationRate: parseFloat(gaMutationRate) || 0.2,
        });

        setSolution(sol);
        if (sol.feasible) {
          showToast(`${sol.algorithm_name} solved instance! Distance: ${sol.total_distance} units. Feasible Solution ✓`, 'success');
        } else {
          showToast(`${sol.algorithm_name} produced infeasible route.`, 'warning');
        }
      }
    } catch (err) {
      setBackendError(err.message);
      showToast(err.message, 'error');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="app-container">
      {/* Background Glowing Orbs */}
      <div className="bg-glow">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
      </div>

      {/* Floating Notification Toast */}
      {toastMessage && (
        <div
          style={{
            position: 'fixed',
            bottom: '2rem',
            right: '2rem',
            zIndex: 9999,
            padding: '1rem 1.5rem',
            borderRadius: '0.75rem',
            background:
              toastMessage.type === 'success'
                ? '#064e3b'
                : toastMessage.type === 'warning'
                ? '#78350f'
                : '#1e293b',
            color: '#ffffff',
            border: '1px solid rgba(255,255,255,0.2)',
            boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            fontFamily: 'var(--font-heading)',
            fontSize: '0.95rem',
          }}
        >
          <Sparkles size={18} color="#34d399" />
          <span>{toastMessage.text}</span>
        </div>
      )}

      {/* 1. PROJECT HEADER */}
      <header className="header">
        <div className="header-badge">
          <Sparkles size={16} /> CEC-2020 E-CVRP Benchmark & FastAPI Python GA Integrated
        </div>
        <h1 className="header-title">Quantum-Assisted Electric Vehicle Routing</h1>
        <h2 className="header-subtitle">Using Hybrid Optimization</h2>
        <p style={{ maxWidth: '780px', margin: '1rem auto 0', color: 'var(--text-muted)', fontSize: '0.975rem' }}>
          Real Python execution bridge connecting CEC-2020 E-CVRP benchmark test suite, classical Genetic Algorithm, and FastAPI backend to interactive visualization.
        </p>
      </header>

      {/* 2. MODE SELECTOR TABS */}
      <section className="section" style={{ marginBottom: '1.5rem' }}>
        <div
          style={{
            display: 'flex',
            justify: 'center',
            gap: '1rem',
            background: '#0f172a',
            padding: '0.6rem 1rem',
            borderRadius: '1rem',
            border: '1px solid rgba(255,255,255,0.08)',
            maxWidth: '820px',
            margin: '0 auto',
          }}
        >
          <button
            style={{
              flex: 1,
              padding: '0.65rem 1.25rem',
              borderRadius: '0.75rem',
              border: 'none',
              fontWeight: '700',
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              background: dataSourceMode === 'demo' ? 'var(--primary-cyan)' : 'transparent',
              color: dataSourceMode === 'demo' ? '#0f172a' : 'var(--text-muted)',
              transition: 'all 0.2s ease',
            }}
            onClick={() => setDataSourceMode('demo')}
          >
            <Sliders size={18} /> Interactive Demo
          </button>

          <button
            style={{
              flex: 1,
              padding: '0.65rem 1.25rem',
              borderRadius: '0.75rem',
              border: 'none',
              fontWeight: '700',
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              background: dataSourceMode === 'benchmark' ? 'var(--primary-cyan)' : 'transparent',
              color: dataSourceMode === 'benchmark' ? '#0f172a' : 'var(--text-muted)',
              transition: 'all 0.2s ease',
            }}
            onClick={() => setDataSourceMode('benchmark')}
          >
            <Database size={18} /> CEC-2020 Benchmark (GA)
          </button>

          <button
            style={{
              flex: 1,
              padding: '0.65rem 1.25rem',
              borderRadius: '0.75rem',
              border: 'none',
              fontWeight: '700',
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              background: dataSourceMode === 'quantum' ? 'var(--primary-cyan)' : 'transparent',
              color: dataSourceMode === 'quantum' ? '#0f172a' : 'var(--text-muted)',
              transition: 'all 0.2s ease',
            }}
            onClick={() => setDataSourceMode('quantum')}
          >
            <Cpu size={18} /> Quantum vs Classical (QAOA vs GA)
          </button>

          <button
            style={{
              flex: 1,
              padding: '0.65rem 1.25rem',
              borderRadius: '0.75rem',
              border: 'none',
              fontWeight: '700',
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              background: dataSourceMode === 'evrp' ? 'var(--primary-cyan)' : 'transparent',
              color: dataSourceMode === 'evrp' ? '#0f172a' : 'var(--text-muted)',
              transition: 'all 0.2s ease',
            }}
            onClick={() => setDataSourceMode('evrp')}
          >
            <Zap size={18} /> Quantum-Assisted EVRP (RL + QAOA)
          </button>
        </div>
      </section>

      {/* 3. VRP SIMULATOR, QUANTUM COMPARISON, OR EVRP DASHBOARD */}
      {dataSourceMode === 'evrp' ? (
        <ErrorBoundary>
          <EVRPDashboard />
        </ErrorBoundary>
      ) : dataSourceMode === 'quantum' ? (
        <section className="section">
          <h3 className="section-title">
            <Cpu className="section-title-icon" size={24} />
            Quantum-Classical Hybrid Optimization & Benchmark Comparison
          </h3>

          {/* Backend Error Banner */}
          {backendError && (
            <div
              style={{
                marginBottom: '1.5rem',
                padding: '1rem 1.25rem',
                background: 'rgba(244,63,94,0.12)',
                border: '1px solid #f43f5e',
                borderRadius: '0.75rem',
                color: '#fb7185',
                fontSize: '0.9rem',
                lineHeight: '1.5',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.75rem',
              }}
            >
              <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong style={{ display: 'block', marginBottom: '0.2rem' }}>Backend Connection Error</strong>
                {backendError}
                <div style={{ marginTop: '0.5rem' }}>
                  <button
                    className="btn btn-secondary"
                    style={{ padding: '0.3rem 0.75rem', fontSize: '0.8rem' }}
                    onClick={loadBenchmarkDiscovery}
                  >
                    <RefreshCw size={14} /> Retry Connection
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="demo-container">
            {/* Left Control Panel */}
            <div className="control-panel">
              <h4
                style={{
                  fontFamily: 'var(--font-heading)',
                  fontSize: '1.1rem',
                  marginBottom: '1.25rem',
                  color: '#f8fafc',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}
              >
                <Server size={18} color="var(--primary-cyan)" /> Experiment Configuration
              </h4>

              {/* Benchmark Dropdown Selector */}
              <div className="form-group">
                <label className="form-label">Select CEC-2020 Benchmark Instance</label>
                <select
                  className="form-select"
                  value={selectedInstanceName}
                  onChange={handleBenchmarkSelect}
                  disabled={isFetchingBenchmark || benchmarkInstances.length === 0}
                >
                  {benchmarkInstances.map((inst) => (
                    <option key={inst.name} value={inst.name}>
                      {inst.name} ({inst.customers} cust, {inst.vehicles} veh, Q={inst.vehicle_capacity})
                    </option>
                  ))}
                </select>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginTop: '0.35rem' }}>
                  Discovered from <code style={{ color: 'var(--primary-cyan)' }}>data/raw/e_cvrp/</code>
                </span>
              </div>

              {/* Sub-instance Customer Subset Selector */}
              <div className="form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label className="form-label">Sub-instance Customer Count</label>
                  <strong style={{ color: 'var(--primary-cyan)', fontSize: '0.9rem' }}>
                    {compareCustomerCount} Customers ({compareCustomerCount * compareCustomerCount} Qubits)
                  </strong>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.35rem' }}>
                  {[2, 3, 4].map((count) => (
                    <button
                      key={count}
                      type="button"
                      style={{
                        flex: 1,
                        padding: '0.5rem 0.75rem',
                        borderRadius: '0.5rem',
                        border: compareCustomerCount === count ? '1px solid var(--primary-cyan)' : '1px solid rgba(255,255,255,0.1)',
                        background: compareCustomerCount === count ? 'rgba(6, 182, 212, 0.15)' : '#0f172a',
                        color: compareCustomerCount === count ? 'var(--primary-cyan)' : 'var(--text-muted)',
                        fontWeight: '700',
                        fontSize: '0.85rem',
                        cursor: 'pointer',
                      }}
                      onClick={() => setCompareCustomerCount(count)}
                    >
                      {count} Cust ({count * count}Q)
                    </button>
                  ))}
                </div>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginTop: '0.4rem', lineHeight: '1.4' }}>
                  Position-based QUBO encoding requires N = C² qubits. Constrained to ≤ 4 customers (16 qubits) for stable classical simulation.
                </span>
              </div>

              {/* QAOA Hyperparameters */}
              <div
                style={{
                  background: '#0f172a',
                  padding: '1rem',
                  borderRadius: '0.75rem',
                  border: '1px solid rgba(255,255,255,0.06)',
                  marginBottom: '1rem',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '0.75rem',
                    fontSize: '0.85rem',
                    color: 'var(--primary-cyan)',
                    fontWeight: '600',
                  }}
                >
                  <Cpu size={16} /> QAOA Simulator Hyperparameters
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <div>
                    <label className="form-label" style={{ fontSize: '0.75rem' }}>Depth (p-layers)</label>
                    <input
                      type="number"
                      min="1"
                      max="5"
                      className="form-input"
                      style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                      value={qaoaDepth}
                      onChange={(e) => setQaoaDepth(e.target.value)}
                    />
                  </div>

                  <div>
                    <label className="form-label" style={{ fontSize: '0.75rem' }}>Aer Shots</label>
                    <input
                      type="number"
                      step="256"
                      min="100"
                      max="8192"
                      className="form-input"
                      style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                      value={qaoaShots}
                      onChange={(e) => setQaoaShots(e.target.value)}
                    />
                  </div>

                  <div>
                    <label className="form-label" style={{ fontSize: '0.75rem' }}>COBYLA Max Iter</label>
                    <input
                      type="number"
                      min="10"
                      max="200"
                      className="form-input"
                      style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                      value={qaoaMaxIter}
                      onChange={(e) => setQaoaMaxIter(e.target.value)}
                    />
                  </div>

                  <div>
                    <label className="form-label" style={{ fontSize: '0.75rem' }}>Random Seed</label>
                    <input
                      type="number"
                      className="form-input"
                      style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                      value={compareSeed}
                      onChange={(e) => setCompareSeed(e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* GA Hyperparameters */}
              <div
                style={{
                  background: '#0f172a',
                  padding: '1rem',
                  borderRadius: '0.75rem',
                  border: '1px solid rgba(255,255,255,0.06)',
                  marginBottom: '1.25rem',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '0.75rem',
                    fontSize: '0.85rem',
                    color: '#a78bfa',
                    fontWeight: '600',
                  }}
                >
                  <Sliders size={16} /> Classical GA Baseline Settings
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <div>
                    <label className="form-label" style={{ fontSize: '0.75rem' }}>GA Pop Size</label>
                    <input
                      type="number"
                      className="form-input"
                      style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                      value={compareGaPopSize}
                      onChange={(e) => setCompareGaPopSize(e.target.value)}
                    />
                  </div>

                  <div>
                    <label className="form-label" style={{ fontSize: '0.75rem' }}>GA Generations</label>
                    <input
                      type="number"
                      className="form-input"
                      style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                      value={compareGaGenerations}
                      onChange={(e) => setCompareGaGenerations(e.target.value)}
                    />
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <button
                className="btn btn-primary"
                style={{
                  width: '100%',
                  opacity: isComparing ? 0.6 : 1.0,
                  cursor: isComparing ? 'not-allowed' : 'pointer',
                }}
                onClick={handleRunComparison}
                disabled={isComparing}
              >
                <Play size={18} /> {isComparing ? 'Running Quantum & Classical Solvers...' : 'Run Quantum vs Classical Comparison'}
              </button>
            </div>

            {/* Right Display Panel */}
            <div className="display-panel">
              {/* Scientific Transparency & Limitations Notice */}
              <div
                style={{
                  background: 'rgba(99, 102, 241, 0.08)',
                  border: '1px solid rgba(99, 102, 241, 0.25)',
                  borderRadius: '0.75rem',
                  padding: '1rem 1.25rem',
                  marginBottom: '1.25rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#a5b4fc', fontWeight: '700', fontSize: '0.875rem', marginBottom: '0.35rem' }}>
                  <Zap size={16} /> Scientific Transparency & Simulation Model
                </div>
                <div style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                  <strong style={{ color: '#f1f5f9' }}>Simulated QAOA:</strong> Executed using Qiskit Aer statevector simulation on classical CPU. No physical quantum processor was used; no quantum supremacy or speedup is claimed.
                </div>
                <div style={{ fontSize: '0.825rem', color: '#fbbf24', marginTop: '0.35rem', lineHeight: '1.5' }}>
                  <strong>Research Scope:</strong> Evaluates single-vehicle routing on identical customer subsets. EV battery capacity, energy depletion, and charging-station visits are excluded in this Phase 1 comparison.
                </div>
              </div>

              {comparisonResult ? (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '0.85rem', marginBottom: '1.25rem' }}>
                    {/* Card 1: Exact Brute Force */}
                    <div
                      style={{
                        background: '#0f172a',
                        padding: '1rem',
                        borderRadius: '0.75rem',
                        border: '1px solid rgba(16, 185, 129, 0.35)',
                        position: 'relative',
                      }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#34d399', fontWeight: '700', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <CheckCircle2 size={14} /> EXACT BASELINE
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Ground Truth Reference</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#34d399', fontFamily: 'var(--font-heading)', margin: '0.35rem 0' }}>
                        {comparisonResult.exact_baseline.total_distance.toFixed(4)} <span style={{ fontSize: '0.75rem' }}>units</span>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                        <div>Runtime: <strong style={{ color: '#f8fafc' }}>{comparisonResult.exact_baseline.runtime_seconds.toFixed(6)} s</strong></div>
                        <div>Feasibility: <strong style={{ color: '#34d399' }}>VALID ✓</strong></div>
                        <div>Optimality Gap: <strong style={{ color: '#34d399' }}>0.00%</strong></div>
                        <div>Route: <span className="route-code" style={{ fontSize: '0.7rem' }}>{comparisonResult.exact_baseline.routes[0]?.join(' → ')}</span></div>
                      </div>
                    </div>

                    {/* Card 2: Classical GA */}
                    <div
                      style={{
                        background: '#0f172a',
                        padding: '1rem',
                        borderRadius: '0.75rem',
                        border: '1px solid rgba(168, 85, 247, 0.35)',
                        position: 'relative',
                      }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#a855f7', fontWeight: '700', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <Sliders size={14} /> CLASSICAL GA
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Evolutionary Baseline</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#c084fc', fontFamily: 'var(--font-heading)', margin: '0.35rem 0' }}>
                        {comparisonResult.classical_ga.total_distance.toFixed(4)} <span style={{ fontSize: '0.75rem' }}>units</span>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                        <div>Runtime: <strong style={{ color: '#f8fafc' }}>{comparisonResult.classical_ga.runtime_seconds.toFixed(6)} s</strong></div>
                        <div>Feasibility: <strong style={{ color: comparisonResult.classical_ga.feasible ? '#34d399' : '#fb7185' }}>{comparisonResult.classical_ga.feasible ? 'VALID ✓' : 'INFEASIBLE ✗'}</strong></div>
                        <div>Optimality Gap: <strong style={{ color: comparisonResult.classical_ga.optimality_gap_percent === 0 ? '#34d399' : '#fbbf24' }}>{comparisonResult.classical_ga.optimality_gap_percent !== null ? `${comparisonResult.classical_ga.optimality_gap_percent.toFixed(2)}%` : 'N/A'}</strong></div>
                        <div>Route: <span className="route-code" style={{ fontSize: '0.7rem' }}>{comparisonResult.classical_ga.routes[0]?.join(' → ')}</span></div>
                      </div>
                    </div>

                    {/* Card 3: GA + Local Search */}
                    <div
                      style={{
                        background: '#0f172a',
                        padding: '1rem',
                        borderRadius: '0.75rem',
                        border: '1px solid rgba(59, 130, 246, 0.35)',
                        position: 'relative',
                      }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#60a5fa', fontWeight: '700', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <Activity size={14} /> GA + LOCAL SEARCH
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>GA + 2-Opt / Relocate</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#93c5fd', fontFamily: 'var(--font-heading)', margin: '0.35rem 0' }}>
                        {comparisonResult.ga_local_search ? comparisonResult.ga_local_search.total_distance.toFixed(4) : 'N/A'} <span style={{ fontSize: '0.75rem' }}>units</span>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                        <div>Runtime: <strong style={{ color: '#f8fafc' }}>{comparisonResult.ga_local_search?.runtime_seconds.toFixed(6)} s</strong></div>
                        <div>Feasibility: <strong style={{ color: comparisonResult.ga_local_search?.feasible ? '#34d399' : '#fb7185' }}>{comparisonResult.ga_local_search?.feasible ? 'VALID ✓' : 'INFEASIBLE ✗'}</strong></div>
                        <div>Optimality Gap: <strong style={{ color: comparisonResult.ga_local_search?.optimality_gap_percent === 0 ? '#34d399' : '#fbbf24' }}>{comparisonResult.ga_local_search?.optimality_gap_percent !== null ? `${comparisonResult.ga_local_search.optimality_gap_percent.toFixed(2)}%` : 'N/A'}</strong></div>
                        <div>vs GA Gain: <strong style={{ color: comparisonResult.ga_local_search?.improvement_over_ga_percent > 0 ? '#34d399' : '#94a3b8' }}>{comparisonResult.ga_local_search?.improvement_over_ga_percent?.toFixed(2)}%</strong></div>
                      </div>
                    </div>

                    {/* Card 4: QAOA Simulator */}
                    <div
                      style={{
                        background: '#0f172a',
                        padding: '1rem',
                        borderRadius: '0.75rem',
                        border: '1px solid rgba(6, 182, 212, 0.35)',
                        position: 'relative',
                      }}
                    >
                      <div style={{ fontSize: '0.75rem', color: 'var(--primary-cyan)', fontWeight: '700', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <Cpu size={14} /> QAOA SIMULATOR
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Direct QUBO Simulation</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--primary-cyan)', fontFamily: 'var(--font-heading)', margin: '0.35rem 0' }}>
                        {comparisonResult.qaoa_simulator?.total_distance !== null ? `${comparisonResult.qaoa_simulator.total_distance.toFixed(4)} units` : 'N/A (>16Q)'}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                        <div>Runtime: <strong style={{ color: '#f8fafc' }}>{comparisonResult.qaoa_simulator?.runtime_seconds.toFixed(6)} s</strong></div>
                        <div>Feasibility: <strong style={{ color: comparisonResult.qaoa_simulator?.feasible ? '#34d399' : '#fb7185' }}>{comparisonResult.qaoa_simulator?.feasible ? 'VALID ✓' : (comparisonResult.qaoa_simulator?.repair_applied ? 'REPAIRED (Infeasible Raw) ✗' : 'N/A')}</strong></div>
                        <div>Raw Valid Rate: <strong style={{ color: '#38bdf8' }}>{comparisonResult.qaoa_simulator?.raw_valid_sample_rate_percent ?? comparisonResult.qaoa_simulator?.valid_sample_rate_percent ?? 0}%</strong></div>
                        <div>Optimality Gap: <strong style={{ color: comparisonResult.qaoa_simulator?.optimality_gap_percent === 0 ? '#34d399' : '#fbbf24' }}>{comparisonResult.qaoa_simulator?.optimality_gap_percent !== null ? `${comparisonResult.qaoa_simulator.optimality_gap_percent.toFixed(2)}%` : 'N/A'}</strong></div>
                      </div>
                    </div>

                    {/* Card 5: Hybrid GA-QAOA */}
                    <div
                      style={{
                        background: '#0f172a',
                        padding: '1rem',
                        borderRadius: '0.75rem',
                        border: '1px solid rgba(245, 158, 11, 0.45)',
                        position: 'relative',
                        boxShadow: '0 0 15px rgba(245, 158, 11, 0.08)',
                      }}
                    >
                      <div style={{ fontSize: '0.75rem', color: '#fbbf24', fontWeight: '700', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <Sparkles size={14} /> HYBRID GA–QAOA
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Staged Hybrid Optimization</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#fbbf24', fontFamily: 'var(--font-heading)', margin: '0.35rem 0' }}>
                        {comparisonResult.hybrid_ga_qaoa ? comparisonResult.hybrid_ga_qaoa.total_distance.toFixed(4) : 'N/A'} <span style={{ fontSize: '0.75rem' }}>units</span>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                        <div>Runtime: <strong style={{ color: '#f8fafc' }}>{comparisonResult.hybrid_ga_qaoa?.runtime_seconds.toFixed(6)} s</strong></div>
                        <div>Feasibility: <strong style={{ color: comparisonResult.hybrid_ga_qaoa?.feasible ? '#34d399' : '#fb7185' }}>{comparisonResult.hybrid_ga_qaoa?.feasible ? 'VALID ✓' : 'INFEASIBLE ✗'}</strong></div>
                        <div>Optimality Gap: <strong style={{ color: comparisonResult.hybrid_ga_qaoa?.optimality_gap_percent === 0 ? '#34d399' : '#fbbf24' }}>{comparisonResult.hybrid_ga_qaoa?.optimality_gap_percent !== null ? `${comparisonResult.hybrid_ga_qaoa.optimality_gap_percent.toFixed(2)}%` : 'N/A'}</strong></div>
                        <div>vs GA Gain: <strong style={{ color: comparisonResult.hybrid_ga_qaoa?.improvement_over_ga_percent > 0 ? '#34d399' : '#38bdf8' }}>{comparisonResult.hybrid_ga_qaoa?.improvement_over_ga_percent?.toFixed(2)}%</strong></div>
                      </div>
                    </div>
                  </div>

                  {/* Comprehensive 5-Algorithm Comparison Table */}
                  <div style={{ background: '#0f172a', padding: '1rem', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.06)', marginBottom: '1.25rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <h5 style={{ fontFamily: 'var(--font-heading)', fontSize: '0.95rem', color: '#f8fafc', margin: 0, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <Grid size={16} color="var(--primary-cyan)" /> Side-by-Side Algorithm Benchmark Matrix
                      </h5>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Instance: <code style={{ color: '#a5b4fc' }}>{comparisonResult.sub_instance_name}</code>
                      </span>
                    </div>

                    <div className="table-scroll-container">
                      <table className="matrix-table" style={{ fontSize: '0.8rem' }}>
                        <thead>
                          <tr>
                            <th>Algorithm</th>
                            <th>Best Route Distance</th>
                            <th>Feasibility</th>
                            <th>Optimality Gap</th>
                            <th>Improvement vs GA</th>
                            <th>QAOA Raw Valid Rate</th>
                            <th>Total Runtime</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td style={{ fontWeight: '700', color: '#34d399' }}>Exact Baseline (Brute-Force)</td>
                            <td style={{ fontWeight: '700' }}>{comparisonResult.exact_baseline.total_distance.toFixed(4)}</td>
                            <td><span className="badge badge-complete">VALID ✓</span></td>
                            <td>0.00%</td>
                            <td>—</td>
                            <td>100% (Enumerated)</td>
                            <td>{comparisonResult.exact_baseline.runtime_seconds.toFixed(6)} s</td>
                          </tr>
                          <tr>
                            <td style={{ fontWeight: '700', color: '#c084fc' }}>Classical GA</td>
                            <td style={{ fontWeight: '700' }}>{comparisonResult.classical_ga.total_distance.toFixed(4)}</td>
                            <td>{comparisonResult.classical_ga.feasible ? <span className="badge badge-complete">VALID ✓</span> : <span className="badge badge-invalid">INFEASIBLE ✗</span>}</td>
                            <td>{comparisonResult.classical_ga.optimality_gap_percent !== null ? `${comparisonResult.classical_ga.optimality_gap_percent.toFixed(2)}%` : 'N/A'}</td>
                            <td>0.00% (Baseline)</td>
                            <td>—</td>
                            <td>{comparisonResult.classical_ga.runtime_seconds.toFixed(6)} s</td>
                          </tr>
                          {comparisonResult.ga_local_search && (
                            <tr>
                              <td style={{ fontWeight: '700', color: '#93c5fd' }}>GA + Local Search (2-Opt)</td>
                              <td style={{ fontWeight: '700' }}>{comparisonResult.ga_local_search.total_distance.toFixed(4)}</td>
                              <td>{comparisonResult.ga_local_search.feasible ? <span className="badge badge-complete">VALID ✓</span> : <span className="badge badge-invalid">INFEASIBLE ✗</span>}</td>
                              <td>{comparisonResult.ga_local_search.optimality_gap_percent !== null ? `${comparisonResult.ga_local_search.optimality_gap_percent.toFixed(2)}%` : 'N/A'}</td>
                              <td style={{ color: comparisonResult.ga_local_search.improvement_over_ga_percent > 0 ? '#34d399' : 'inherit', fontWeight: '700' }}>
                                {comparisonResult.ga_local_search.improvement_over_ga_percent > 0 ? `+${comparisonResult.ga_local_search.improvement_over_ga_percent.toFixed(2)}%` : `${comparisonResult.ga_local_search.improvement_over_ga_percent.toFixed(2)}%`}
                              </td>
                              <td>—</td>
                              <td>{comparisonResult.ga_local_search.runtime_seconds.toFixed(6)} s</td>
                            </tr>
                          )}
                          {comparisonResult.qaoa_simulator && (
                            <tr>
                              <td style={{ fontWeight: '700', color: 'var(--primary-cyan)' }}>QAOA Simulator (Direct QUBO)</td>
                              <td style={{ fontWeight: '700' }}>{comparisonResult.qaoa_simulator.total_distance !== null ? comparisonResult.qaoa_simulator.total_distance.toFixed(4) : 'N/A (>16Q)'}</td>
                              <td>{comparisonResult.qaoa_simulator.feasible ? <span className="badge badge-complete">VALID ✓</span> : (comparisonResult.qaoa_simulator.repair_applied ? <span style={{ color: '#fbbf24', fontSize: '0.75rem' }}>Repaired (Raw Inv)</span> : <span className="badge badge-invalid">N/A</span>)}</td>
                              <td>{comparisonResult.qaoa_simulator.optimality_gap_percent !== null ? `${comparisonResult.qaoa_simulator.optimality_gap_percent.toFixed(2)}%` : 'N/A'}</td>
                              <td>—</td>
                              <td style={{ color: '#38bdf8', fontWeight: '700' }}>{comparisonResult.qaoa_simulator.raw_valid_sample_rate_percent ?? comparisonResult.qaoa_simulator.valid_sample_rate_percent ?? 0}%</td>
                              <td>{comparisonResult.qaoa_simulator.runtime_seconds.toFixed(6)} s</td>
                            </tr>
                          )}
                          {comparisonResult.hybrid_ga_qaoa && (
                            <tr style={{ background: 'rgba(245, 158, 11, 0.08)' }}>
                              <td style={{ fontWeight: '700', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                                <Sparkles size={14} /> Hybrid GA–QAOA
                              </td>
                              <td style={{ fontWeight: '700', color: '#fbbf24' }}>{comparisonResult.hybrid_ga_qaoa.total_distance.toFixed(4)}</td>
                              <td>{comparisonResult.hybrid_ga_qaoa.feasible ? <span className="badge badge-complete">VALID ✓</span> : <span className="badge badge-invalid">INFEASIBLE ✗</span>}</td>
                              <td style={{ fontWeight: '700', color: comparisonResult.hybrid_ga_qaoa.optimality_gap_percent === 0 ? '#34d399' : '#fbbf24' }}>
                                {comparisonResult.hybrid_ga_qaoa.optimality_gap_percent !== null ? `${comparisonResult.hybrid_ga_qaoa.optimality_gap_percent.toFixed(2)}%` : 'N/A'}
                              </td>
                              <td style={{ color: comparisonResult.hybrid_ga_qaoa.improvement_over_ga_percent > 0 ? '#34d399' : '#38bdf8', fontWeight: '700' }}>
                                {comparisonResult.hybrid_ga_qaoa.improvement_over_ga_percent > 0 ? `+${comparisonResult.hybrid_ga_qaoa.improvement_over_ga_percent.toFixed(2)}%` : `${comparisonResult.hybrid_ga_qaoa.improvement_over_ga_percent.toFixed(2)}%`}
                              </td>
                              <td>{comparisonResult.hybrid_ga_qaoa.qaoa_improvements_count} QAOA Subproblem Wins</td>
                              <td>{comparisonResult.hybrid_ga_qaoa.runtime_seconds.toFixed(6)} s</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Route Visualizer with 5-Way Toggle */}
                  <div className="visualizer-card" style={{ marginBottom: '1.25rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                      <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1rem', color: '#f8fafc', margin: 0 }}>
                        Sub-instance Route Visualizer ({comparisonResult.sub_instance_name})
                      </h4>
                      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                        <button
                          type="button"
                          style={{
                            padding: '0.35rem 0.65rem',
                            borderRadius: '0.4rem',
                            border: activeCompareView === 'exact' ? '1px solid #10b981' : '1px solid rgba(255,255,255,0.1)',
                            background: activeCompareView === 'exact' ? 'rgba(16, 185, 129, 0.2)' : 'transparent',
                            color: activeCompareView === 'exact' ? '#34d399' : 'var(--text-muted)',
                            fontSize: '0.75rem',
                            fontWeight: '600',
                            cursor: 'pointer',
                          }}
                          onClick={() => setActiveCompareView('exact')}
                        >
                          Exact Route
                        </button>
                        <button
                          type="button"
                          style={{
                            padding: '0.35rem 0.65rem',
                            borderRadius: '0.4rem',
                            border: activeCompareView === 'ga' ? '1px solid #a855f7' : '1px solid rgba(255,255,255,0.1)',
                            background: activeCompareView === 'ga' ? 'rgba(168, 85, 247, 0.2)' : 'transparent',
                            color: activeCompareView === 'ga' ? '#c084fc' : 'var(--text-muted)',
                            fontSize: '0.75rem',
                            fontWeight: '600',
                            cursor: 'pointer',
                          }}
                          onClick={() => setActiveCompareView('ga')}
                        >
                          Classical GA
                        </button>
                        <button
                          type="button"
                          style={{
                            padding: '0.35rem 0.65rem',
                            borderRadius: '0.4rem',
                            border: activeCompareView === 'ga_ls' ? '1px solid #3b82f6' : '1px solid rgba(255,255,255,0.1)',
                            background: activeCompareView === 'ga_ls' ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                            color: activeCompareView === 'ga_ls' ? '#93c5fd' : 'var(--text-muted)',
                            fontSize: '0.75rem',
                            fontWeight: '600',
                            cursor: 'pointer',
                          }}
                          onClick={() => setActiveCompareView('ga_ls')}
                        >
                          GA + Local Search
                        </button>
                        <button
                          type="button"
                          style={{
                            padding: '0.35rem 0.65rem',
                            borderRadius: '0.4rem',
                            border: activeCompareView === 'qaoa' ? '1px solid var(--primary-cyan)' : '1px solid rgba(255,255,255,0.1)',
                            background: activeCompareView === 'qaoa' ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                            color: activeCompareView === 'qaoa' ? 'var(--primary-cyan)' : 'var(--text-muted)',
                            fontSize: '0.75rem',
                            fontWeight: '600',
                            cursor: 'pointer',
                          }}
                          onClick={() => setActiveCompareView('qaoa')}
                        >
                          QAOA Simulator
                        </button>
                        <button
                          type="button"
                          style={{
                            padding: '0.35rem 0.65rem',
                            borderRadius: '0.4rem',
                            border: activeCompareView === 'hybrid' ? '1px solid #f59e0b' : '1px solid rgba(255,255,255,0.1)',
                            background: activeCompareView === 'hybrid' ? 'rgba(245, 158, 11, 0.2)' : 'transparent',
                            color: activeCompareView === 'hybrid' ? '#fbbf24' : 'var(--text-muted)',
                            fontSize: '0.75rem',
                            fontWeight: '700',
                            cursor: 'pointer',
                          }}
                          onClick={() => setActiveCompareView('hybrid')}
                        >
                          ★ Hybrid GA–QAOA
                        </button>
                      </div>
                    </div>

                    <RouteVisualizer
                      instance={subInstanceForVisualizer}
                      routes={routesForVisualizer}
                    />
                  </div>

                  {/* Top Quantum Measurement Bitstrings Table */}
                  {comparisonResult.qaoa_simulator?.top_candidates && comparisonResult.qaoa_simulator.top_candidates.length > 0 && (
                    <div style={{ background: '#0f172a', padding: '1rem', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <h5 style={{ fontFamily: 'var(--font-heading)', fontSize: '0.9rem', color: '#f8fafc', marginBottom: '0.6rem' }}>
                        Top Sampled Quantum Measurement Bitstrings (Qiskit Aer Simulator)
                      </h5>
                      <div className="table-scroll-container" style={{ maxHeight: '200px' }}>
                        <table className="matrix-table" style={{ fontSize: '0.75rem' }}>
                          <thead>
                            <tr>
                              <th>Bitstring ({comparisonResult.qubits_required}Q)</th>
                              <th>Counts</th>
                              <th>Prob</th>
                              <th>Energy</th>
                              <th>Decoded Route</th>
                              <th>Validity</th>
                            </tr>
                          </thead>
                          <tbody>
                            {comparisonResult.qaoa_simulator.top_candidates.map((c, idx) => (
                              <tr key={idx}>
                                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary-cyan)' }}>{c.bitstring}</td>
                                <td>{c.count}</td>
                                <td>{(c.probability * 100).toFixed(1)}%</td>
                                <td>{c.energy.toFixed(2)}</td>
                                <td style={{ fontFamily: 'var(--font-mono)' }}>{c.route?.join(' → ')}</td>
                                <td>
                                  {c.is_valid ? (
                                    <span style={{ color: '#34d399', fontWeight: '700' }}>VALID ✓ (Raw Sample)</span>
                                  ) : (
                                    <span style={{ color: '#fb7185' }}>INVALID ✗</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div style={{ padding: '3rem 1.5rem', textAlign: 'center', color: 'var(--text-muted)', background: '#0f172a', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <Cpu size={32} color="var(--primary-cyan)" style={{ marginBottom: '0.75rem', opacity: 0.8 }} />
                  <div style={{ fontWeight: '600', color: '#f8fafc', marginBottom: '0.35rem' }}>
                    No Comparison Executed Yet
                  </div>
                  <div style={{ fontSize: '0.85rem' }}>
                    Configure the sub-instance size (2–4 customers) and click <strong>'Run Quantum vs Classical Comparison'</strong> to evaluate Exact, GA, and QAOA simultaneously.
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      ) : (
        /* 3. VRP SIMULATOR CONTROLS & INSTANCE BUILDER */
        <section className="section">
          <h3 className="section-title">
            <Cpu className="section-title-icon" size={24} />
            {dataSourceMode === 'benchmark' ? 'CEC-2020 Benchmark GA Solver' : 'VRP Simulator & Instance Configuration'}
          </h3>

        {/* Backend Error Banner */}
        {backendError && (
          <div
            style={{
              marginBottom: '1.5rem',
              padding: '1rem 1.25rem',
              background: 'rgba(244,63,94,0.12)',
              border: '1px solid #f43f5e',
              borderRadius: '0.75rem',
              color: '#fb7185',
              fontSize: '0.9rem',
              lineHeight: '1.5',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.75rem',
            }}
          >
            <AlertTriangle size={20} style={{ shrink: 0, marginTop: '2px' }} />
            <div>
              <strong style={{ display: 'block', marginBottom: '0.2rem' }}>Backend Connection Error</strong>
              {backendError}
              <div style={{ marginTop: '0.5rem' }}>
                <button
                  className="btn btn-secondary"
                  style={{ padding: '0.3rem 0.75rem', fontSize: '0.8rem' }}
                  onClick={loadBenchmarkDiscovery}
                >
                  <RefreshCw size={14} /> Retry Connection
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="demo-container">
          {/* Controls Panel */}
          <div className="control-panel">
            <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', marginBottom: '1.25rem', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {dataSourceMode === 'benchmark' ? <Server size={18} color="var(--primary-cyan)" /> : <Sliders size={18} color="var(--primary-cyan)" />}
              {dataSourceMode === 'benchmark' ? 'Benchmark Settings' : 'Instance Configuration'}
            </h4>

            {/* Benchmark Dropdown Selector */}
            {dataSourceMode === 'benchmark' ? (
              <div className="form-group">
                <label className="form-label">Select .evrp Benchmark Instance</label>
                <select
                  className="form-select"
                  value={selectedInstanceName}
                  onChange={handleBenchmarkSelect}
                  disabled={isFetchingBenchmark || benchmarkInstances.length === 0}
                >
                  {benchmarkInstances.map((inst) => (
                    <option key={inst.name} value={inst.name}>
                      {inst.name} ({inst.customers} cust, {inst.vehicles} veh, Q={inst.vehicle_capacity})
                    </option>
                  ))}
                </select>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginTop: '0.35rem' }}>
                  Discovered from <code style={{ color: 'var(--primary-cyan)' }}>data/raw/e_cvrp/</code> directory
                </span>
              </div>
            ) : (
              <>
                {/* Algorithm Selector */}
                <div className="form-group">
                  <label className="form-label">Select Classical Algorithm</label>
                  <select
                    className="form-select"
                    value={selectedAlgorithm}
                    onChange={(e) => setSelectedAlgorithm(e.target.value)}
                  >
                    <option value="ga">Genetic Algorithm (GA) — Active ✅</option>
                    <option value="aco" disabled>Ant Colony Optimization (ACO) — Coming Soon 🔲</option>
                    <option value="sa" disabled>Simulated Annealing (SA) — Coming Soon 🔲</option>
                    <option value="tabu_search" disabled>Tabu Search (TS) — Coming Soon 🔲</option>
                  </select>
                </div>

                {/* Number of Customers Slider (5 - 20) */}
                <div className="form-group">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <label className="form-label">Number of Customers</label>
                    <strong style={{ color: 'var(--primary-cyan)', fontSize: '0.9rem' }}>{numCustomers}</strong>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="20"
                    step="1"
                    style={{ width: '100%', accentColor: 'var(--primary-cyan)' }}
                    value={numCustomers}
                    onChange={(e) => setNumCustomers(parseInt(e.target.value, 10))}
                  />
                </div>

                {/* Number of Vehicles */}
                <div className="form-group">
                  <label className="form-label">Number of Vehicles</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    className="form-input"
                    value={numVehicles}
                    onChange={(e) => setNumVehicles(parseInt(e.target.value, 10) || 1)}
                  />
                </div>

                {/* Vehicle Capacity */}
                <div className="form-group">
                  <label className="form-label">Vehicle Capacity (units)</label>
                  <input
                    type="number"
                    step="1"
                    min="1"
                    className="form-input"
                    value={capacity}
                    onChange={(e) => setCapacity(e.target.value)}
                  />
                </div>

                {/* Depot Coordinates */}
                <div className="form-group">
                  <label className="form-label">Depot Coordinates (X, Y)</label>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                    <input
                      type="number"
                      step="0.5"
                      className="form-input"
                      placeholder="X"
                      value={depotX}
                      onChange={(e) => setDepotX(e.target.value)}
                    />
                    <input
                      type="number"
                      step="0.5"
                      className="form-input"
                      placeholder="Y"
                      value={depotY}
                      onChange={(e) => setDepotY(e.target.value)}
                    />
                  </div>
                </div>

                <button
                  className="btn btn-secondary"
                  style={{ width: '100%', marginBottom: '1rem' }}
                  onClick={handleGenerateRandomCustomers}
                >
                  <Shuffle size={18} /> Generate Customers (Random 2D)
                </button>
              </>
            )}

            {/* GA Hyperparameters */}
            <div style={{ background: '#0f172a', padding: '1rem', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.06)', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', fontSize: '0.85rem', color: 'var(--primary-cyan)', fontWeight: '600' }}>
                <Sliders size={16} /> Python GA Hyperparameters
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                <div>
                  <label className="form-label" style={{ fontSize: '0.75rem' }}>Pop Size</label>
                  <input
                    type="number"
                    className="form-input"
                    style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                    value={gaPopSize}
                    onChange={(e) => setGaPopSize(e.target.value)}
                  />
                </div>

                <div>
                  <label className="form-label" style={{ fontSize: '0.75rem' }}>Generations</label>
                  <input
                    type="number"
                    className="form-input"
                    style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                    value={gaGenerations}
                    onChange={(e) => setGaGenerations(e.target.value)}
                  />
                </div>

                <div>
                  <label className="form-label" style={{ fontSize: '0.75rem' }}>Crossover Rate</label>
                  <input
                    type="number"
                    step="0.05"
                    className="form-input"
                    style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                    value={gaCrossoverRate}
                    onChange={(e) => setGaCrossoverRate(e.target.value)}
                  />
                </div>

                <div>
                  <label className="form-label" style={{ fontSize: '0.75rem' }}>Mutation Rate</label>
                  <input
                    type="number"
                    step="0.05"
                    className="form-input"
                    style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                    value={gaMutationRate}
                    onChange={(e) => setGaMutationRate(e.target.value)}
                  />
                </div>

                {dataSourceMode === 'benchmark' && (
                  <div style={{ gridColumn: 'span 2' }}>
                    <label className="form-label" style={{ fontSize: '0.75rem' }}>Random Seed</label>
                    <input
                      type="number"
                      className="form-input"
                      style={{ padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                      value={gaSeed}
                      onChange={(e) => setGaSeed(e.target.value)}
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Infeasibility Warning Banner (Demo mode) */}
            {dataSourceMode === 'demo' && !configVal.isValid && (
              <div style={{ marginBottom: '1.25rem', padding: '0.85rem 1rem', background: 'rgba(244,63,94,0.12)', border: '1px solid #f43f5e', borderRadius: '0.6rem', color: '#fb7185', fontSize: '0.85rem', lineHeight: '1.4' }}>
                <div style={{ fontWeight: '700', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <AlertTriangle size={16} /> Configuration Infeasible
                </div>
                {configVal.errors[0]}
              </div>
            )}

            {/* Action Button */}
            <button
              className="btn btn-primary"
              style={{
                width: '100%',
                opacity: (dataSourceMode === 'demo' && !configVal.isValid) || isRunning ? 0.5 : 1.0,
                cursor: (dataSourceMode === 'demo' && !configVal.isValid) || isRunning ? 'not-allowed' : 'pointer',
              }}
              onClick={handleRunAlgorithm}
              disabled={(dataSourceMode === 'demo' && !configVal.isValid) || isRunning}
            >
              <Play size={18} /> {isRunning ? 'Running Genetic Algorithm...' : 'Run Genetic Algorithm'}
            </button>
          </div>

          {/* Right Panel: Instance Specifications & Details */}
          <div className="display-panel">
            {/* Instance Specification Summary Card */}
            {dataSourceMode === 'benchmark' ? (
              benchmarkDetails ? (
                <div className="card" style={{ background: '#0f172a' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
                    <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.05rem', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Database size={18} className="card-icon" /> Benchmark Instance: {benchmarkDetails.name}
                    </h4>
                    <span className="badge badge-complete"><CheckCircle2 size={12} /> CEC-2020 Loaded</span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.85rem', fontSize: '0.875rem', marginBottom: '1rem' }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Customers</span>
                      <strong>{benchmarkDetails.num_customers}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Vehicles Fleet</span>
                      <strong>{benchmarkDetails.num_vehicles}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Vehicle Capacity</span>
                      <strong>{benchmarkDetails.vehicle_capacity} units</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Total Demand</span>
                      <strong style={{ color: 'var(--primary-cyan)' }}>{benchmarkDetails.total_demand} units</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Reference Value</span>
                      <strong style={{ color: '#a5b4fc' }}>{benchmarkDetails.optimal_value ?? 'N/A'}</strong>
                    </div>
                  </div>

                  {/* Informational EV Parameters Box */}
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1rem', borderRadius: '0.5rem', border: '1px stroke rgba(255,255,255,0.06)' }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: '700', color: '#34d399', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <Zap size={14} /> Informational EV Metadata (Phase 2 EVRP Extensions)
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.6rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      <div>Battery Capacity: <strong style={{ color: '#f8fafc' }}>{benchmarkDetails.energy_capacity ?? 'N/A'}</strong></div>
                      <div>Energy Consumption: <strong style={{ color: '#f8fafc' }}>{benchmarkDetails.energy_consumption ?? 'N/A'}</strong></div>
                      <div>Charging Stations: <strong style={{ color: '#f8fafc' }}>{benchmarkDetails.num_stations ?? 0}</strong></div>
                    </div>
                    <p style={{ fontSize: '0.725rem', color: 'var(--text-muted)', marginTop: '0.4rem', fontStyle: 'italic', marginBottom: 0 }}>
                      Note: Current GA baseline evaluates classical CVRP routing. Battery & charging station constraints will be enforced in Phase 2.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="card" style={{ background: '#0f172a', padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  Loading benchmark instance details...
                </div>
              )
            ) : (
              configVal && (
                <div className="card" style={{ background: '#0f172a' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
                    <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.05rem', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Database size={18} className="card-icon" /> VRP Instance Specifications
                    </h4>
                    {configVal.isValid ? (
                      <span className="badge badge-complete"><CheckCircle2 size={12} /> Configuration Valid</span>
                    ) : (
                      <span className="badge badge-invalid"><AlertTriangle size={12} /> Instance Infeasible</span>
                    )}
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.85rem', fontSize: '0.875rem' }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Customers</span>
                      <strong>{numCustomers}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Vehicles</span>
                      <strong>{numVehicles}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Vehicle Capacity</span>
                      <strong>{capacity} units</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Total Demand</span>
                      <strong style={{ color: 'var(--primary-cyan)' }}>{configVal.totalDemand} units</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Max Customer Demand</span>
                      <strong style={{ color: configVal.vehicleCapacity < configVal.maxDemand ? '#fb7185' : '#34d399' }}>
                        {configVal.maxDemand} units
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>Total Fleet Capacity</span>
                      <strong style={{ color: configVal.maxFleetCapacity < configVal.totalDemand ? '#fb7185' : '#34d399' }}>
                        {configVal.maxFleetCapacity} units
                      </strong>
                    </div>
                  </div>
                </div>
              )
            )}

            {/* Customer Table (Interactive Demo only) */}
            {dataSourceMode === 'demo' && (
              <div className="card">
                <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.05rem', marginBottom: '0.85rem', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Grid size={18} color="var(--primary-cyan)" /> Customer Node & Demand Table (User Editable)
                </h4>

                <div className="table-scroll-container" style={{ maxHeight: '220px' }}>
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th style={{ minWidth: '110px' }}>Customer ID</th>
                        <th style={{ minWidth: '130px' }}>X Coordinate</th>
                        <th style={{ minWidth: '130px' }}>Y Coordinate</th>
                        <th style={{ minWidth: '140px' }}>Demand (units)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {customerNodes.map((cust, idx) => (
                        <tr key={cust.id}>
                          <th style={{ color: 'var(--primary-cyan)' }}>Cust {cust.id}</th>
                          <td>
                            <input
                              type="number"
                              step="0.1"
                              className="form-input"
                              style={{ padding: '0.2rem 0.4rem', fontSize: '0.85rem', textAlign: 'center' }}
                              value={cust.x}
                              onChange={(e) => handleCustomerChange(idx, 'x', e.target.value)}
                            />
                          </td>
                          <td>
                            <input
                              type="number"
                              step="0.1"
                              className="form-input"
                              style={{ padding: '0.2rem 0.4rem', fontSize: '0.85rem', textAlign: 'center' }}
                              value={cust.y}
                              onChange={(e) => handleCustomerChange(idx, 'y', e.target.value)}
                            />
                          </td>
                          <td>
                            <input
                              type="number"
                              step="0.5"
                              min="0"
                              className="form-input"
                              style={{ padding: '0.2rem 0.4rem', fontSize: '0.85rem', textAlign: 'center', fontWeight: '700', color: '#34d399' }}
                              value={cust.demand}
                              onChange={(e) => handleCustomerChange(idx, 'demand', e.target.value)}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Animated Route Visualizer & VRPSolution Card */}
            <div className="visualizer-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', color: '#f8fafc' }}>
                  EV Routing Simulator & Visualizer
                </h4>
                {solution ? (
                  solution.feasible ? (
                    <span className="badge badge-valid"><ShieldCheck size={14} /> FEASIBLE SOLUTION</span>
                  ) : (
                    <span className="badge badge-invalid"><AlertTriangle size={14} /> INFEASIBLE ROUTE</span>
                  )
                ) : (
                  <span className="badge badge-pending"><Clock size={14} /> NO SOLUTION GENERATED YET</span>
                )}
              </div>

              {/* Dynamic SVG Visualizer with Animated EVs */}
              <RouteVisualizer
                instance={instance}
                routes={solution ? solution.routes : []}
              />

              {/* VRPSolution Results Display */}
              {solution ? (
                <div style={{ marginTop: '1.25rem' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                    <div style={{ background: '#0f172a', padding: '0.85rem 1rem', borderRadius: '0.6rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Algorithm</div>
                      <div style={{ fontSize: '0.95rem', fontWeight: '700', color: '#f8fafc' }}>
                        {solution.algorithm_name}
                      </div>
                    </div>

                    <div style={{ background: '#0f172a', padding: '0.85rem 1rem', borderRadius: '0.6rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Total Distance</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: '700', color: 'var(--primary-cyan)', fontFamily: 'var(--font-heading)' }}>
                        {solution.total_distance.toFixed(4)} units
                      </div>
                    </div>

                    <div style={{ background: '#0f172a', padding: '0.85rem 1rem', borderRadius: '0.6rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Execution Runtime</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#a5b4fc', fontFamily: 'var(--font-mono)' }}>
                        {solution.runtime_seconds.toFixed(4)} s
                      </div>
                    </div>

                    <div style={{ background: '#0f172a', padding: '0.85rem 1rem', borderRadius: '0.6rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Feasible</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: '700', color: solution.feasible ? '#34d399' : '#fb7185' }}>
                        {solution.feasible ? 'YES' : 'NO'}
                      </div>
                    </div>
                  </div>

                  {/* Vehicle Routes & Payload Breakdown */}
                  <h5 style={{ fontFamily: 'var(--font-heading)', fontSize: '0.95rem', marginBottom: '0.75rem', color: '#e2e8f0' }}>
                    Vehicle Route Breakdown ({solution.routes.length} vehicle(s))
                  </h5>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {solution.routes.map((r, rIdx) => (
                      <div
                        key={rIdx}
                        style={{
                          background: '#0f172a',
                          padding: '0.75rem 1rem',
                          borderRadius: '0.5rem',
                          border: '1px solid rgba(255,255,255,0.06)',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          flexWrap: 'wrap',
                          gap: '0.5rem',
                        }}
                      >
                        <div>
                          <strong style={{ color: 'var(--primary-cyan)', fontSize: '0.85rem', marginRight: '0.5rem' }}>
                            Vehicle {rIdx + 1}:
                          </strong>
                          <span className="route-code" style={{ fontSize: '0.85rem' }}>
                            {r.join(' → ')}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                          Load: <strong>{solution.vehicle_loads[rIdx] || 0} / {instance?.vehicleCapacity} units</strong>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                  No solution generated yet. Click <strong>'Run Genetic Algorithm'</strong> to execute Python GA on the benchmark instance.
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    )}

      {/* 4. METHODOLOGY PIPELINE */}
      <section className="section">
        <h3 className="section-title">
          <BrainCircuit className="section-title-icon" size={24} /> Overall Research Pipeline
        </h3>
        <div className="pipeline-container">
          <div className="pipeline-step active">
            <div className="pipeline-step-title">Classical VRP & GA</div>
            <span className="badge badge-complete"><CheckCircle2 size={12} /> Active Baseline</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step active">
            <div className="pipeline-step-title">QUBO Formulation</div>
            <span className="badge badge-complete"><CheckCircle2 size={12} /> Active Encoding</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step active">
            <div className="pipeline-step-title">QAOA Optimization</div>
            <span className="badge badge-complete"><CheckCircle2 size={12} /> Active Simulator</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step active">
            <div className="pipeline-step-title">RL Controller</div>
            <span className="badge badge-complete"><CheckCircle2 size={12} /> Active Policy</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step active">
            <div className="pipeline-step-title">EVRP Extension</div>
            <span className="badge badge-complete"><CheckCircle2 size={12} /> Active Model</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step active">
            <div className="pipeline-step-title">Evaluation</div>
            <span className="badge badge-complete"><CheckCircle2 size={12} /> Active Comparison</span>
          </div>
        </div>
      </section>

      {/* 5. DEVELOPMENT STATUS SECTION */}
      <section className="section">
        <h3 className="section-title">
          <Activity className="section-title-icon" size={24} /> Module Implementation Status
        </h3>
        <div className="table-scroll-container">
          <table className="status-table">
            <thead>
              <tr>
                <th style={{ minWidth: '200px' }}>Module / Algorithm</th>
                <th style={{ minWidth: '320px' }}>Description</th>
                <th style={{ minWidth: '150px' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Classical VRP Foundation</strong></td>
                <td>VRPInstance, Euclidean distance matrix, route distance calculation, and validator</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>CEC-2020 E-CVRP Benchmark Loader</strong></td>
                <td>Dataset parser (e_cvrp_loader.py) and FastAPI REST API backend bridge</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>Genetic Algorithm (GA)</strong></td>
                <td>Permutation chromosome, OX crossover, swap mutation, elitism, greedy capacity decoding</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>Exact Brute-Force Baseline</strong></td>
                <td>Exhaustive permutation solver for ground-truth optimality gap calibration</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>QUBO Formulation</strong></td>
                <td>Lucas position-based binary variable mapping (N = C²) & Ising Hamiltonian translation</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>QAOA Quantum Solver</strong></td>
                <td>Parameterized quantum circuit, COBYLA classical optimizer & AerSimulator sampling</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>Reinforcement Learning (RL Controller)</strong></td>
                <td>Gym-compatible Actor-Critic policy with dynamic action masking over physically reachable nodes</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>EVRP Battery & Charging Feasibility</strong></td>
                <td>Leg-by-leg state-of-charge tracking, non-linear energy dissipation, CS station visits, and charging cost</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>Hybrid Optimization Suite (RL + QAOA + EV-LS)</strong></td>
                <td>Unified multi-algorithm framework comparing Classical GA, Local Search, RL Policy, Quantum QAOA, and Full Hybrid</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>Ant Colony Optimization (ACO)</strong></td>
                <td>Pheromone matrix update and probabilistic ant route construction</td>
                <td><span className="badge badge-pending">⏳ Planned</span></td>
              </tr>
              <tr>
                <td><strong>Simulated Annealing (SA)</strong></td>
                <td>Metropolis acceptance probability and temperature cooling schedule</td>
                <td><span className="badge badge-pending">⏳ Planned</span></td>
              </tr>
              <tr>
                <td><strong>Tabu Search (TS)</strong></td>
                <td>Tabu list memory tenure and neighborhood move search</td>
                <td><span className="badge badge-pending">⏳ Planned</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
