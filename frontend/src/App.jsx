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
  Info,
  Shuffle,
  Edit3,
} from 'lucide-react';

import {
  getDeterministicToyInstance,
  generateSyntheticInstance,
  evaluateRoute,
} from './utils/vrpLogic';
import RouteVisualizer from './components/RouteVisualizer';

export default function App() {
  const [numCustomers, setNumCustomers] = useState(3);
  const [capacity, setCapacity] = useState(10.0);
  const [mode, setMode] = useState('deterministic'); // 'deterministic' or 'synthetic'

  const [instance, setInstance] = useState(null);
  const [routeString, setRouteString] = useState('[0, 1, 2, 3, 0]');
  const [evaluation, setEvaluation] = useState(null);
  const [toastMessage, setToastMessage] = useState(null);

  // Helper to show temporary notification toasts
  const showToast = (msg, type = 'info') => {
    setToastMessage({ text: msg, type });
    setTimeout(() => {
      setToastMessage(null);
    }, 4000);
  };

  // Generate VRP Instance (Triggered on 'Generate VRP' or parameters change)
  const handleGenerateVRP = () => {
    let inst;
    if (mode === 'synthetic') {
      inst = generateSyntheticInstance(numCustomers, capacity);
      showToast(`Generated random synthetic VRP instance with ${numCustomers} customers.`, 'success');
    } else {
      inst = getDeterministicToyInstance(numCustomers, capacity);
      showToast(`Loaded deterministic ${numCustomers}-customer VRP instance.`, 'info');
    }

    // Set default route string matching the instance nodes
    const defaultRoute = inst.candidateRoute;
    const defaultRouteStr = `[${defaultRoute.join(', ')}]`;

    setInstance(inst);
    setRouteString(defaultRouteStr);

    // Evaluate default route
    const evalRes = evaluateRoute(inst, defaultRoute);
    setEvaluation(evalRes);
  };

  // Auto-generate on initial render or mode/customer count change
  useEffect(() => {
    handleGenerateVRP();
  }, [numCustomers, mode]);

  // Handle Route Evaluation when 'Run Demo / Validate Route' is clicked or capacity changes
  const handleValidateRoute = () => {
    if (!instance) return;

    // Parse route string e.g. "[0, 1, 3, 2, 0]" or "0, 1, 3, 2, 0"
    let parsedRoute = [];
    try {
      const cleanStr = routeString.replace(/\[|\]/g, '').trim();
      if (cleanStr.length > 0) {
        parsedRoute = cleanStr.split(',').map((s) => parseInt(s.trim(), 10));
      }
    } catch (err) {
      showToast('Invalid route format. Use format like [0, 1, 2, 0]', 'error');
      return;
    }

    // Update instance capacity if user modified the capacity input
    const currentInst = {
      ...instance,
      vehicleCapacity: parseFloat(capacity) || 10.0,
    };
    setInstance(currentInst);

    const evalRes = evaluateRoute(currentInst, parsedRoute);
    setEvaluation(evalRes);

    if (evalRes.isValid) {
      showToast(`Route validated! Total distance: ${evalRes.totalDistance} units. Feasible ✓`, 'success');
    } else {
      showToast(`Route validation failed: ${evalRes.errors[0]}`, 'warning');
    }
  };

  // Helper to set preset routes
  const handleSelectPresetRoute = (presetType) => {
    if (!instance) return;
    const n = instance.numNodes;

    let routeArr = [];
    if (presetType === 'sequential') {
      // 0 -> 1 -> 2 -> ... -> 0
      routeArr = Array.from({ length: n }, (_, i) => i);
      routeArr.push(0);
    } else if (presetType === 'reverse') {
      // 0 -> n-1 -> n-2 -> ... -> 1 -> 0
      routeArr = [0];
      for (let i = n - 1; i >= 1; i--) routeArr.push(i);
      routeArr.push(0);
    } else if (presetType === 'subtour') {
      // 0 -> 1 -> 2 -> 0 (Leaves out some nodes)
      routeArr = [0, 1, Math.min(2, n - 1), 0];
    } else if (presetType === 'invalid_start') {
      // 1 -> 2 -> 3 -> 0 (No depot start)
      routeArr = [1, Math.min(2, n - 1), 0];
    }

    const str = `[${routeArr.join(', ')}]`;
    setRouteString(str);

    // Auto evaluate preset
    const currentInst = { ...instance, vehicleCapacity: parseFloat(capacity) || 10.0 };
    const evalRes = evaluateRoute(currentInst, routeArr);
    setEvaluation(evalRes);
  };

  return (
    <div className="app-container">
      {/* Background Orbs */}
      <div className="bg-glow">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
      </div>

      {/* Floating Toast Notification */}
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

      {/* 1. HEADER */}
      <header className="header">
        <div className="header-badge">
          <Sparkles size={16} /> Phase 1: Classical VRP Baseline Operational
        </div>
        <h1 className="header-title">Quantum-Assisted Electric Vehicle Routing</h1>
        <h2 className="header-subtitle">Hybrid Quantum-Classical Optimization</h2>
      </header>

      {/* 2. PROJECT OVERVIEW SECTION */}
      <section className="section">
        <h3 className="section-title">
          <Layers className="section-title-icon" size={24} /> Project Overview
        </h3>
        <div className="grid-cards">
          <div className="card">
            <div className="card-header">
              <span className="card-title">VRP</span>
              <div className="card-icon"><Truck size={20} /></div>
            </div>
            <div className="card-body">
              <strong>Vehicle Routing Problem:</strong> Classical combinatorial baseline finding optimal paths for fleet delivery to customers with minimal distance.
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">EVRP</span>
              <div className="card-icon"><BatteryCharging size={20} /></div>
            </div>
            <div className="card-body">
              <strong>Electric VRP:</strong> Extends classical VRP with electric vehicle battery constraints, energy consumption rates, and charging station stops.
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">QUBO</span>
              <div className="card-icon"><Binary size={20} /></div>
            </div>
            <div className="card-body">
              <strong>QUBO Formulation:</strong> Quadratic Unconstrained Binary Optimization mapping route constraints into quadratic penalty matrices.
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">QAOA</span>
              <div className="card-icon"><Cpu size={20} /></div>
            </div>
            <div className="card-body">
              <strong>QAOA Solver:</strong> Quantum Approximate Optimization Algorithm parameterized by \(\gamma, \beta\) variational layers executed on QPUs.
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Reinforcement Learning</span>
              <div className="card-icon"><BrainCircuit size={20} /></div>
            </div>
            <div className="card-body">
              <strong>RL Controller:</strong> Dynamic policy agent optimizing QAOA parameters during runtime to accelerate quantum convergence.
            </div>
          </div>
        </div>
      </section>

      {/* 3. VRP DEMO SECTION */}
      <section className="section">
        <h3 className="section-title">
          <Truck className="section-title-icon" size={24} /> Classical VRP Foundation Demo
        </h3>

        <div className="demo-container">
          {/* Controls Panel */}
          <div className="control-panel">
            <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', marginBottom: '1.25rem', color: '#f8fafc' }}>
              Instance Configuration
            </h4>

            {/* Mode Selection */}
            <div className="form-group">
              <label className="form-label">Instance Data Source</label>
              <select
                className="form-select"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
              >
                <option value="deterministic">Deterministic (Fixed Integer Coords)</option>
                <option value="synthetic">Random Synthetic Generator</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Number of Customers</label>
              <select
                className="form-select"
                value={numCustomers}
                onChange={(e) => setNumCustomers(parseInt(e.target.value))}
              >
                <option value={3}>3 Customers (4 Nodes Total)</option>
                <option value={4}>4 Customers (5 Nodes Total)</option>
                <option value={5}>5 Customers (6 Nodes Total)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Vehicle Capacity (Payload Limit)</label>
              <input
                type="number"
                step="1"
                min="1"
                className="form-input"
                value={capacity}
                onChange={(e) => setCapacity(e.target.value)}
              />
            </div>

            <button
              className="btn btn-primary"
              style={{ width: '100%', marginTop: '0.5rem' }}
              onClick={handleGenerateVRP}
            >
              {mode === 'synthetic' ? <Shuffle size={18} /> : <RotateCcw size={18} />} Generate VRP Instance
            </button>

            <hr style={{ borderColor: 'rgba(255,255,255,0.08)', margin: '1.5rem 0' }} />

            {/* Route Candidate Controls */}
            <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.05rem', marginBottom: '1rem', color: '#f8fafc' }}>
              Candidate Route Selection
            </h4>

            <div className="form-group">
              <label className="form-label">Route Node Sequence</label>
              <input
                type="text"
                className="form-input"
                style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', color: 'var(--primary-cyan)' }}
                value={routeString}
                onChange={(e) => setRouteString(e.target.value)}
                placeholder="[0, 1, 2, 3, 0]"
              />
            </div>

            {/* Route Presets */}
            <div className="form-group">
              <label className="form-label">Load Preset Candidate Routes</label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                <button className="btn btn-secondary" style={{ padding: '0.5rem', fontSize: '0.8rem' }} onClick={() => handleSelectPresetRoute('sequential')}>
                  Sequential
                </button>
                <button className="btn btn-secondary" style={{ padding: '0.5rem', fontSize: '0.8rem' }} onClick={() => handleSelectPresetRoute('reverse')}>
                  Reverse
                </button>
                <button className="btn btn-secondary" style={{ padding: '0.5rem', fontSize: '0.8rem' }} onClick={() => handleSelectPresetRoute('subtour')}>
                  Sub-route
                </button>
                <button className="btn btn-secondary" style={{ padding: '0.5rem', fontSize: '0.8rem', color: '#fb7185' }} onClick={() => handleSelectPresetRoute('invalid_start')}>
                  Invalid Start
                </button>
              </div>
            </div>

            <button
              className="btn btn-secondary"
              style={{ width: '100%', marginTop: '0.75rem', background: 'rgba(6, 182, 212, 0.15)', borderColor: '#06b6d4', color: '#38bdf8' }}
              onClick={handleValidateRoute}
            >
              <Play size={18} /> Run Demo / Validate Route
            </button>
          </div>

          {/* Display & Results Panel */}
          {instance && evaluation && (
            <div className="display-panel">
              {/* Visualizer & Route Results */}
              <div className="visualizer-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', color: '#f8fafc' }}>
                    Interactive 2D Route Visualizer
                  </h4>
                  {evaluation.isValid ? (
                    <span className="badge badge-valid">
                      <ShieldCheck size={14} /> VALID ROUTE
                    </span>
                  ) : (
                    <span className="badge badge-invalid">
                      <AlertTriangle size={14} /> INVALID ROUTE
                    </span>
                  )}
                </div>

                {/* SVG Visualizer */}
                <RouteVisualizer
                  instance={instance}
                  route={(() => {
                    try {
                      return routeString.replace(/\[|\]/g, '').split(',').map((s) => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
                    } catch {
                      return instance.candidateRoute;
                    }
                  })()}
                />

                <div style={{ marginTop: '1.25rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
                  <div style={{ background: '#0f172a', padding: '0.85rem 1rem', borderRadius: '0.6rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Evaluated Route</div>
                    <div className="route-code" style={{ marginTop: '0.25rem', fontSize: '0.85rem' }}>
                      {routeString}
                    </div>
                  </div>

                  <div style={{ background: '#0f172a', padding: '0.85rem 1rem', borderRadius: '0.6rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Total Route Distance</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: '700', color: 'var(--primary-cyan)', fontFamily: 'var(--font-heading)' }}>
                      {evaluation.totalDistance} units
                    </div>
                  </div>

                  <div style={{ background: '#0f172a', padding: '0.85rem 1rem', borderRadius: '0.6rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Demand / Capacity</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: '700', color: evaluation.capacityValid ? '#34d399' : '#fb7185', fontFamily: 'var(--font-heading)' }}>
                      {evaluation.totalDemand} / {instance.vehicleCapacity}
                    </div>
                  </div>
                </div>

                {evaluation.errors.length > 0 && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.3)', borderRadius: '0.5rem', color: '#fb7185', fontSize: '0.85rem' }}>
                    <strong>Validation Errors:</strong> {evaluation.errors.join('; ')}
                  </div>
                )}
              </div>

              {/* Distance Matrix Table */}
              <div className="card">
                <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.05rem', marginBottom: '0.85rem', color: '#f8fafc' }}>
                  Pairwise Euclidean Distance Matrix (N x N)
                </h4>
                <div style={{ overflowX: 'auto' }}>
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th>Node</th>
                        {instance.nodes.map((n) => (
                          <th key={n.id}>{n.type}</th>
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
            </div>
          )}
        </div>
      </section>

      {/* 4. METHODOLOGY PIPELINE */}
      <section className="section">
        <h3 className="section-title">
          <BrainCircuit className="section-title-icon" size={24} /> Methodology Pipeline
        </h3>
        <div className="pipeline-container">
          <div className="pipeline-step active">
            <div className="pipeline-step-title">Classical VRP</div>
            <span className="badge badge-complete"><CheckCircle2 size={12} /> Active</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">QUBO</div>
            <span className="badge badge-pending"><Clock size={12} /> Coming Soon</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">QAOA</div>
            <span className="badge badge-pending"><Clock size={12} /> Coming Soon</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">RL Tuning</div>
            <span className="badge badge-pending"><Clock size={12} /> Coming Soon</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">EVRP Extension</div>
            <span className="badge badge-pending"><Clock size={12} /> Coming Soon</span>
          </div>
          <div className="pipeline-arrow"><ArrowRight size={20} /></div>

          <div className="pipeline-step pending">
            <div className="pipeline-step-title">Evaluation</div>
            <span className="badge badge-pending"><Clock size={12} /> Coming Soon</span>
          </div>
        </div>
      </section>

      {/* 5. STATUS SECTION */}
      <section className="section">
        <h3 className="section-title">
          <CheckCircle2 className="section-title-icon" size={24} /> Project Module Implementation Status
        </h3>
        <table className="status-table">
          <thead>
            <tr>
              <th>Module / Phase</th>
              <th>Description</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Classical VRP Foundation</strong></td>
              <td>Data representation, toy instances, Euclidean distance matrix, route distance calculation, and basic validation</td>
              <td><span className="badge badge-complete">✅ Complete</span></td>
            </tr>
            <tr>
              <td><strong>QUBO Formulation</strong></td>
              <td>Penalty weight matrices mapping VRP constraints to quadratic terms</td>
              <td><span className="badge badge-pending">🔲 Not implemented</span></td>
            </tr>
            <tr>
              <td><strong>QAOA Quantum Solver</strong></td>
              <td>Parameterized quantum circuit execution on quantum simulators</td>
              <td><span className="badge badge-pending">🔲 Not implemented</span></td>
            </tr>
            <tr>
              <td><strong>Reinforcement Learning</strong></td>
              <td>Policy network optimizing QAOA parameters during quantum sampling</td>
              <td><span className="badge badge-pending">🔲 Not implemented</span></td>
            </tr>
            <tr>
              <td><strong>EVRP Extension</strong></td>
              <td>Battery capacity, energy dissipation, and charging station visits</td>
              <td><span className="badge badge-pending">🔲 Not implemented</span></td>
            </tr>
            <tr>
              <td><strong>Evaluation & Benchmarks</strong></td>
              <td>Solution quality, quantum speedup, and convergence rate comparison</td>
              <td><span className="badge badge-pending">🔲 Not implemented</span></td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
