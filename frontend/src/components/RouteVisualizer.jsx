import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FastForward, Navigation } from 'lucide-react';

const ROUTE_STYLES = [
  { stroke: '#06b6d4', glow: 'rgba(6, 182, 212, 0.4)', bg: '#06b6d4', name: 'Vehicle 1' },
  { stroke: '#10b981', glow: 'rgba(16, 185, 129, 0.4)', bg: '#10b981', name: 'Vehicle 2' },
  { stroke: '#a855f7', glow: 'rgba(168, 85, 247, 0.4)', bg: '#a855f7', name: 'Vehicle 3' },
  { stroke: '#f43f5e', glow: 'rgba(244, 63, 94, 0.4)', bg: '#f43f5e', name: 'Vehicle 4' },
  { stroke: '#f59e0b', glow: 'rgba(245, 158, 11, 0.4)', bg: '#f59e0b', name: 'Vehicle 5' },
];

export default function RouteVisualizer({ instance, routes }) {
  if (!instance || !instance.nodes) return null;

  const { nodes } = instance;

  // Animation Control States
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0.0); // 0.0 to 1.0
  const [speed, setSpeed] = useState(1.0); // 0.5x, 1x, 2x
  const [vehicleFilter, setVehicleFilter] = useState('all');

  const animRef = useRef(null);

  // Dynamic SVG Scaling for 5-20 Nodes
  const padding = 65;
  const width = 600;
  const height = 320;

  const maxX = Math.max(...nodes.map((n) => n.x), 6.0);
  const maxY = Math.max(...nodes.map((n) => n.y), 6.0);

  const scaleX = (x) => padding + (x / maxX) * (width - 2 * padding);
  const scaleY = (y) => height - padding - (y / maxY) * (height - 2 * padding);

  // Normalize routes
  let routeList = [];
  if (Array.isArray(routes)) {
    if (routes.length > 0 && Array.isArray(routes[0])) {
      routeList = routes;
    } else if (routes.length > 0 && typeof routes[0] === 'number') {
      routeList = [routes];
    }
  }

  // Filter routes based on vehicle selection dropdown
  const activeRoutes = vehicleFilter === 'all'
    ? routeList
    : routeList.filter((_, idx) => idx === parseInt(vehicleFilter, 10));

  // Animation Loop Effect
  useEffect(() => {
    let lastTime = performance.now();

    const animate = (now) => {
      const dt = (now - lastTime) / 1000.0;
      lastTime = now;

      if (isPlaying) {
        setProgress((prev) => {
          const next = prev + dt * 0.12 * speed;
          if (next >= 1.0) {
            setIsPlaying(false);
            return 1.0;
          }
          return next;
        });
        animRef.current = requestAnimationFrame(animate);
      }
    };

    if (isPlaying) {
      animRef.current = requestAnimationFrame(animate);
    } else {
      cancelAnimationFrame(animRef.current);
    }

    return () => cancelAnimationFrame(animRef.current);
  }, [isPlaying, speed]);

  const handleReplay = () => {
    setProgress(0.0);
    setIsPlaying(true);
  };

  // Helper to interpolate position along route sequence for progress value [0 - 1]
  const calculateVehiclePosition = (routeNodes) => {
    if (!routeNodes || routeNodes.length < 2) return null;

    const numSegments = routeNodes.length - 1;
    const globalT = Math.min(Math.max(progress, 0.0), 1.0);
    const totalSegPos = globalT * numSegments;
    const segIdx = Math.min(Math.floor(totalSegPos), numSegments - 1);
    const segProgress = totalSegPos - segIdx;

    const fromNode = nodes[routeNodes[segIdx]];
    const toNode = nodes[routeNodes[segIdx + 1]];

    if (!fromNode || !toNode) return null;

    const x1 = scaleX(fromNode.x);
    const y1 = scaleY(fromNode.y);
    const x2 = scaleX(toNode.x);
    const y2 = scaleY(toNode.y);

    const currX = x1 + (x2 - x1) * segProgress;
    const currY = y1 + (y2 - y1) * segProgress;

    // Angle of motion
    const angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);

    const isAtDepot = toNode.id === instance.depotIdx;
    const statusText = isAtDepot
      ? `Returning to Depot (0)`
      : `Visiting C${toNode.id} (${toNode.demand} units)`;

    return {
      x: currX,
      y: currY,
      angle,
      fromNode,
      toNode,
      segIdx,
      statusText,
    };
  };

  // Generate SVG path string with curve displacement
  const createRoutePathD = (routeNodes, vIdx) => {
    if (!routeNodes || routeNodes.length < 2) return '';
    const curveOffset = (vIdx - (routeList.length - 1) / 2) * 14;

    const coords = routeNodes
      .map((idx) => {
        const node = nodes[idx];
        if (!node) return null;
        return { x: scaleX(node.x), y: scaleY(node.y) };
      })
      .filter(Boolean);

    if (coords.length < 2) return '';

    let d = `M ${coords[0].x} ${coords[0].y}`;
    for (let i = 0; i < coords.length - 1; i++) {
      const start = coords[i];
      const end = coords[i + 1];

      if (curveOffset !== 0) {
        const midX = (start.x + end.x) / 2;
        const midY = (start.y + end.y) / 2;
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const nx = -dy / len;
        const ny = dx / len;

        const ctrlX = midX + nx * curveOffset;
        const ctrlY = midY + ny * curveOffset;

        d += ` Q ${ctrlX} ${ctrlY}, ${end.x} ${end.y}`;
      } else {
        d += ` L ${end.x} ${end.y}`;
      }
    }
    return d;
  };

  return (
    <div className="visualizer-canvas-wrapper" style={{ position: 'relative', width: '100%' }}>
      {/* Simulation Controls Toolbar */}
      <div
        style={{
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          marginBottom: '0.75rem',
          background: '#0f172a',
          padding: '0.75rem 1rem',
          borderRadius: '0.75rem',
          border: '1px solid rgba(255,255,255,0.06)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {isPlaying ? (
            <button className="btn btn-secondary" style={{ padding: '0.45rem 0.85rem', fontSize: '0.85rem' }} onClick={() => setIsPlaying(false)}>
              <Pause size={15} /> Pause
            </button>
          ) : (
            <button className="btn btn-primary" style={{ padding: '0.45rem 0.85rem', fontSize: '0.85rem' }} onClick={() => setIsPlaying(true)}>
              <Play size={15} /> Play EV Simulation
            </button>
          )}

          <button className="btn btn-secondary" style={{ padding: '0.45rem 0.85rem', fontSize: '0.85rem' }} onClick={handleReplay}>
            <RotateCcw size={15} /> Replay
          </button>
        </div>

        {/* Speed Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          <FastForward size={15} /> Speed:
          {[0.5, 1.0, 2.0].map((s) => (
            <button
              key={s}
              style={{
                padding: '0.25rem 0.55rem',
                borderRadius: '0.4rem',
                border: '1px solid rgba(255,255,255,0.1)',
                background: speed === s ? 'var(--primary-indigo)' : 'rgba(255,255,255,0.05)',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.75rem',
                fontWeight: '600',
              }}
              onClick={() => setSpeed(s)}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Vehicle Filter Selector */}
        {routeList.length > 1 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            <Navigation size={15} /> Animate:
            <select
              className="form-select"
              style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem', width: 'auto' }}
              value={vehicleFilter}
              onChange={(e) => setVehicleFilter(e.target.value)}
            >
              <option value="all">All Vehicles</option>
              {routeList.map((_, idx) => (
                <option key={idx} value={idx}>Vehicle {idx + 1}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* SVG Routing Map Canvas */}
      <svg
        width="100%"
        height="320"
        viewBox={`0 0 ${width} ${height}`}
        style={{ background: '#0b1120', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        <defs>
          <filter id="glowEffect" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Grid Background Lines */}
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((val) => (
          <g key={val}>
            <line
              x1={scaleX(val)}
              y1={padding - 25}
              x2={scaleX(val)}
              y2={height - padding + 25}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="3 3"
            />
            <line
              x1={padding - 25}
              y1={scaleY(val)}
              x2={width - padding + 25}
              y2={scaleY(val)}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="3 3"
            />
          </g>
        ))}

        {/* Render Vehicle Routes */}
        {activeRoutes.map((r, idx) => {
          const origIdx = vehicleFilter === 'all' ? idx : parseInt(vehicleFilter, 10);
          const pathD = createRoutePathD(r, origIdx);
          if (!pathD) return null;
          const style = ROUTE_STYLES[origIdx % ROUTE_STYLES.length];

          return (
            <path
              key={idx}
              d={pathD}
              fill="none"
              stroke={style.stroke}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              filter="url(#glowEffect)"
            />
          );
        })}

        {/* Animated EV Markers */}
        {routeList.map((r, rIdx) => {
          if (vehicleFilter !== 'all' && parseInt(vehicleFilter, 10) !== rIdx) return null;
          const pos = calculateVehiclePosition(r);
          if (!pos) return null;
          const style = ROUTE_STYLES[rIdx % ROUTE_STYLES.length];

          return (
            <g key={rIdx} transform={`translate(${pos.x}, ${pos.y}) rotate(${pos.angle})`}>
              {/* Pulse Ring around active EV */}
              <circle r="14" fill={style.glow} />

              {/* Animated Vehicle Car Marker */}
              <rect x="-10" y="-6" width="20" height="12" rx="3" fill={style.stroke} stroke="#fff" strokeWidth="1.5" />
              <circle cx="-5" cy="-6" r="2" fill="#fff" />
              <circle cx="5" cy="-6" r="2" fill="#fff" />
              <circle cx="-5" cy="6" r="2" fill="#fff" />
              <circle cx="5" cy="6" r="2" fill="#fff" />
            </g>
          );
        })}

        {/* Render Depot & Customer Nodes */}
        {nodes.map((node) => {
          const isDepot = node.id === instance.depotIdx;
          const cx = scaleX(node.x);
          const cy = scaleY(node.y);

          return (
            <g key={node.id} transform={`translate(${cx}, ${cy})`}>
              {/* Outer Glow Ring */}
              <circle
                r={isDepot ? 15 : 11}
                fill={isDepot ? 'rgba(245, 158, 11, 0.25)' : 'rgba(6, 182, 212, 0.2)'}
                stroke={isDepot ? '#f59e0b' : '#06b6d4'}
                strokeWidth="2"
              />

              {/* Core Circle */}
              <circle r={isDepot ? 6 : 4.5} fill={isDepot ? '#fbbf24' : '#38bdf8'} />

              {/* Node Labels */}
              <text
                x="0"
                y={isDepot ? -22 : 24}
                textAnchor="middle"
                fill={isDepot ? '#fbbf24' : '#e2e8f0'}
                fontSize="11"
                fontWeight="700"
                fontFamily="Outfit, sans-serif"
              >
                {isDepot ? `Depot (0)` : `C${node.id} (${node.demand}u)`}
              </text>
              <text
                x="0"
                y={isDepot ? -9 : 35}
                textAnchor="middle"
                fill="rgba(255,255,255,0.45)"
                fontSize="9"
                fontFamily="Fira Code, monospace"
              >
                ({node.x.toFixed(1)}, {node.y.toFixed(1)})
              </text>
            </g>
          );
        })}
      </svg>

      {/* Progress & Live EV Status Display */}
      {routeList.length > 0 && (
        <div style={{ marginTop: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {/* Progress Bar */}
          <div style={{ background: '#0f172a', padding: '0.75rem 1rem', borderRadius: '0.6rem', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              <span>Simulation Progress</span>
              <strong>{Math.round(progress * 100)}%</strong>
            </div>
            <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '999px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${progress * 100}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #06b6d4, #6366f1)',
                  transition: 'width 0.1s linear',
                }}
              ></div>
            </div>
          </div>

          {/* Active Stop Indicators */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
            {routeList.map((r, rIdx) => {
              if (vehicleFilter !== 'all' && parseInt(vehicleFilter, 10) !== rIdx) return null;
              const pos = calculateVehiclePosition(r);
              if (!pos) return null;
              const style = ROUTE_STYLES[rIdx % ROUTE_STYLES.length];

              return (
                <div
                  key={rIdx}
                  style={{
                    flex: '1',
                    minWidth: '220px',
                    background: '#0f172a',
                    padding: '0.6rem 0.85rem',
                    borderRadius: '0.5rem',
                    border: `1px solid ${style.stroke}`,
                    fontSize: '0.825rem',
                  }}
                >
                  <strong style={{ color: style.stroke, marginRight: '0.4rem' }}>Vehicle {rIdx + 1}:</strong>
                  <span style={{ color: '#e2e8f0' }}>{pos.statusText}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
