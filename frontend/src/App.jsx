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
  Plus,
  Trash2,
} from 'lucide-react';

import {
  getDeterministicToyInstance,
  generateRandomCustomers,
  buildInstanceFromNodes,
  validateVRPInstanceConfig,
} from './utils/vrpLogic';
import { runClassicalSolver } from './utils/solverApi';
import RouteVisualizer from './components/RouteVisualizer';

export default function App() {
  // Instance Controls State
  const [numCustomers, setNumCustomers] = useState(5); // Default 5 (5-20 supported)
  const [numVehicles, setNumVehicles] = useState(2);
  const [capacity, setCapacity] = useState(15.0);

  // Depot Coordinates
  const [depotX, setDepotX] = useState(0.0);
  const [depotY, setDepotY] = useState(0.0);

  // Customer Nodes Array State
  const [customerNodes, setCustomerNodes] = useState([]);

  // Selected Algorithm State
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('ga');

  // GA Hyperparameters
  const [gaPopSize, setGaPopSize] = useState(50);
  const [gaGenerations, setGaGenerations] = useState(100);
  const [gaCrossoverRate, setGaCrossoverRate] = useState(0.8);
  const [gaMutationRate, setGaMutationRate] = useState(0.2);

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

  // Initialize or update customer nodes when numCustomers changes
  useEffect(() => {
    const baseInst = getDeterministicToyInstance(numCustomers, capacity, numVehicles);
    // Extract customer nodes (excluding depot)
    const custs = baseInst.nodes.filter((n) => n.id !== 0);
    setCustomerNodes(custs);
  }, [numCustomers]);

  // Re-build VRPInstance whenever nodes, depot, capacity, or numVehicles change
  useEffect(() => {
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
  }, [customerNodes, depotX, depotY, capacity, numVehicles]);

  // Handle Customer Table Cell Editing
  const handleCustomerChange = (index, field, value) => {
    const updated = [...customerNodes];
    updated[index] = {
      ...updated[index],
      [field]: value,
    };
    setCustomerNodes(updated);
    setSolution(null);
  };

  // Generate Random Customer Coordinates while preserving editable table
  const handleGenerateRandomCustomers = () => {
    const newInst = generateRandomCustomers(numCustomers, capacity, numVehicles, depotX, depotY);
    const custs = newInst.nodes.filter((n) => n.id !== 0);
    setCustomerNodes(custs);
    setSolution(null);
    showToast(`Generated random customer coordinates for ${numCustomers} customers.`, 'success');
  };

  // Instance Configuration Feasibility Validation
  const configVal = instance ? validateVRPInstanceConfig(instance) : { isValid: true, errors: [] };

  // Run Selected Classical Solver
  const handleRunAlgorithm = async () => {
    if (!instance) return;

    if (!configVal.isValid) {
      showToast(configVal.errors[0], 'error');
      return;
    }

    setIsRunning(true);
    try {
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
    } catch (err) {
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
          <Sparkles size={16} /> VRP Simulator & Classical GA Baseline Operational
        </div>
        <h1 className="header-title">Quantum-Assisted Electric Vehicle Routing</h1>
        <h2 className="header-subtitle">Using Hybrid Optimization</h2>
        <p style={{ maxWidth: '780px', margin: '1rem auto 0', color: 'var(--text-muted)', fontSize: '0.975rem' }}>
          The project studies a hybrid quantum-classical optimization framework for Vehicle Routing and extends it toward the Electric Vehicle Routing Problem.
        </p>
      </header>

      {/* 2. CLASSICAL VRP SIMULATOR CONTROLS & INSTANCE BUILDER */}
      <section className="section">
        <h3 className="section-title">
          <Cpu className="section-title-icon" size={24} /> VRP Simulator & Instance Configuration
        </h3>

        <div className="demo-container">
          {/* Controls Panel */}
          <div className="control-panel">
            <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', marginBottom: '1.25rem', color: '#f8fafc' }}>
              Instance Configuration
            </h4>

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

            {/* GA Hyperparameters */}
            {selectedAlgorithm === 'ga' && (
              <div style={{ background: '#0f172a', padding: '1rem', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.06)', marginBottom: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', fontSize: '0.85rem', color: 'var(--primary-cyan)', fontWeight: '600' }}>
                  <Sliders size={16} /> GA Hyperparameters
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
                </div>
              </div>
            )}

            {/* Infeasibility Warning Banner */}
            {!configVal.isValid && (
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
              style={{ width: '100%', opacity: !configVal.isValid || isRunning ? 0.5 : 1.0, cursor: !configVal.isValid || isRunning ? 'not-allowed' : 'pointer' }}
              onClick={handleRunAlgorithm}
              disabled={!configVal.isValid || isRunning}
            >
              <Play size={18} /> {isRunning ? 'Running Optimization...' : 'Run Algorithm'}
            </button>
          </div>

          {/* Right Panel: Instance Specifications & Customer Table */}
          <div className="display-panel">
            {/* Instance Specification Summary Card */}
            {configVal && (
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
            )}

            {/* Interactive User-Editable Customer Table */}
            <div className="card">
              <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.05rem', marginBottom: '0.85rem', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Grid size={18} color="var(--primary-cyan)" /> Customer Node & Demand Table (User Editable)
              </h4>

              <div className="table-scroll-container" style={{ maxHeight: '250px' }}>
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
                        {solution.total_distance} units
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
                          justify: 'space-between',
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
                          Load: <strong>{solution.vehicle_loads[rIdx] || 0} / {instance.vehicleCapacity} units</strong>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                  No solution generated yet. Select an algorithm and click <strong>'Run Algorithm'</strong>.
                </div>
              )}
            </div>

            {/* Pairwise Distance Matrix Table */}
            {instance && (
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
                  <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.05rem', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Grid size={18} color="var(--primary-cyan)" /> Pairwise Euclidean Distance Matrix (N x N)
                  </h4>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Values represent pairwise Euclidean distances between nodes</span>
                </div>

                <div className="table-scroll-container" style={{ maxHeight: '380px' }}>
                  <table className="matrix-table" style={{ minWidth: `${Math.max(600, (instance.nodes?.length || 0) * 60)}px` }}>
                    <thead>
                      <tr>
                        <th style={{ minWidth: '80px' }}>Node</th>
                        {instance.nodes.map((n) => (
                          <th key={n.id} style={{ minWidth: '60px' }}>{n.type}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {instance.distanceMatrix.map((row, i) => (
                        <tr key={i}>
                          <th>{instance.nodes[i].type}</th>
                          {row.map((val, j) => (
                            <td key={j} className={i === j ? 'diagonal' : ''}>
                              {val.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* 3. METHODOLOGY PIPELINE */}
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

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">QUBO Formulation</div>
            <span className="badge badge-pending"><Clock size={12} /> Under Development</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">QAOA Optimization</div>
            <span className="badge badge-pending"><Clock size={12} /> Under Development</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">RL Controller</div>
            <span className="badge badge-pending"><Clock size={12} /> Under Development</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">EVRP Extension</div>
            <span className="badge badge-pending"><Clock size={12} /> Under Development</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">Evaluation</div>
            <span className="badge badge-pending"><Clock size={12} /> Under Development</span>
          </div>
        </div>
      </section>

      {/* 4. DEVELOPMENT STATUS SECTION */}
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
                <td><strong>Classical Solver Architecture</strong></td>
                <td>BaseVRPSolver and VRPSolution standardized input/output interfaces</td>
                <td><span className="badge badge-complete">✅ Complete</span></td>
              </tr>
              <tr>
                <td><strong>Genetic Algorithm (GA)</strong></td>
                <td>Permutation chromosome, OX crossover, swap mutation, elitism, greedy capacity decoding</td>
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
              <tr>
                <td><strong>QUBO Formulation</strong></td>
                <td>Penalty matrices mapping VRP constraints to quadratic terms</td>
                <td><span className="badge badge-pending">🔄 In Development</span></td>
              </tr>
              <tr>
                <td><strong>QAOA Quantum Solver</strong></td>
                <td>Parameterized quantum circuit execution on quantum simulators</td>
                <td><span className="badge badge-pending">🔄 In Development</span></td>
              </tr>
              <tr>
                <td><strong>Reinforcement Learning</strong></td>
                <td>Policy network optimizing QAOA parameters during quantum sampling</td>
                <td><span className="badge badge-pending">🔄 In Development</span></td>
              </tr>
              <tr>
                <td><strong>EVRP Extension</strong></td>
                <td>Battery capacity, energy dissipation, and charging station visits</td>
                <td><span className="badge badge-pending">🔄 In Development</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
