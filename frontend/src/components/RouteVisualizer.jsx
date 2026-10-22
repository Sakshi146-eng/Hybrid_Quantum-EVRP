import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FastForward, Navigation, Zap } from 'lucide-react';

// Visually distinct color styles per vehicle route
const ROUTE_STYLES = [
  { stroke: '#06b6d4', glow: 'rgba(6, 182, 212, 0.4)', bg: '#06b6d4', name: 'Vehicle 1' },
  { stroke: '#10b981', glow: 'rgba(16, 185, 129, 0.4)', bg: '#10b981', name: 'Vehicle 2' },
  { stroke: '#a855f7', glow: 'rgba(168, 85, 247, 0.4)', bg: '#a855f7', name: 'Vehicle 3' },
  { stroke: '#f43f5e', glow: 'rgba(244, 63, 94, 0.4)', bg: '#f43f5e', name: 'Vehicle 4' },
  { stroke: '#f59e0b', glow: 'rgba(245, 158, 11, 0.4)', bg: '#f59e0b', name: 'Vehicle 5' },
  { stroke: '#ec4899', glow: 'rgba(236, 72, 153, 0.4)', bg: '#ec4899', name: 'Vehicle 6' },
  { stroke: '#3b82f6', glow: 'rgba(59, 130, 246, 0.4)', bg: '#3b82f6', name: 'Vehicle 7' },
  { stroke: '#84cc16', glow: 'rgba(132, 204, 22, 0.4)', bg: '#84cc16', name: 'Vehicle 8' },
];

export default function RouteVisualizer({ instance, routes }) {
  if (!instance || (!instance.nodes && !instance.nodeLookup)) return null;

  const rawNodes = instance.nodes || [];
  const chargingStations = instance.charging_stations || [];

  // Map nodes by their unique id for safe O(1) lookup
  const nodeLookup = React.useMemo(() => {
    const lookup = {};
    if (instance.nodeLookup && typeof instance.nodeLookup === 'object') {
      Object.assign(lookup, instance.nodeLookup);
    }
    if (Array.isArray(rawNodes)) {
      rawNodes.forEach((n, idx) => {
        if (n) {
          if (n.id !== undefined) lookup[n.id] = n;
          lookup[idx] = lookup[idx] || n;
        }
      });
    }
    if (Array.isArray(chargingStations)) {
      chargingStations.forEach((s) => {
        if (s && s.id !== undefined) {
          lookup[s.id] = lookup[s.id] || {
            id: s.id,
            x: s.x,
            y: s.y,
            type: `Station ${s.id}`,
            demand: 0,
            isStation: true,
          };
        }
      });
    }
    return lookup;
  }, [instance, rawNodes, chargingStations]);

  const getNode = (id) => nodeLookup[id] || (rawNodes && rawNodes[id]) || null;

  // Animation Control States
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0.0); // 0.0 to 1.0
  const [speed, setSpeed] = useState(1.0); // 0.5x, 1x, 2x
  const [vehicleFilter, setVehicleFilter] = useState('all');

  const animRef = useRef(null);

  // Spacious 1000 x 650 SVG Coordinate Viewport
  const width = 1000;
  const height = 650;
  const padding = 70;

  // Compute bounding box across all known nodes
  const allValidNodes = Object.values(nodeLookup).filter(
    (n) => n && typeof n.x === 'number' && typeof n.y === 'number'
  );

  const allX = allValidNodes.map((n) => n.x);
  const allY = allValidNodes.map((n) => n.y);

  const minX = allX.length > 0 ? Math.min(...allX) : 0;
  const maxX = allX.length > 0 ? Math.max(...allX) : 100;
  const minY = allY.length > 0 ? Math.min(...allY) : 0;
  const maxY = allY.length > 0 ? Math.max(...allY) : 100;

  const rangeX = maxX - minX > 0 ? maxX - minX : 1.0;
  const rangeY = maxY - minY > 0 ? maxY - minY : 1.0;

  const scaleX = (x) => padding + ((x - minX) / rangeX) * (width - 2 * padding);
  const scaleY = (y) => height - padding - ((y - minY) / rangeY) * (height - 2 * padding);

  // Normalize routes
  let routeList = [];
  if (Array.isArray(routes)) {
    if (routes.length > 0 && Array.isArray(routes[0])) {
      routeList = routes;
    } else if (routes.length > 0 && typeof routes[0] === 'number') {
      routeList = [routes];
    }
  }

  // Active routes based on vehicle filter dropdown
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
          const next = prev + dt * 0.1 * speed;
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

  // Interpolate vehicle position along route sequence for progress value [0 - 1]
  const calculateVehiclePosition = (routeNodes) => {
    if (!routeNodes || routeNodes.length < 2) return null;

    const numSegments = routeNodes.length - 1;
    const globalT = Math.min(Math.max(progress, 0.0), 1.0);
    const totalSegPos = globalT * numSegments;
    const segIdx = Math.min(Math.floor(totalSegPos), numSegments - 1);
    const segProgress = totalSegPos - segIdx;

    const fromNode = getNode(routeNodes[segIdx]);
    const toNode = getNode(routeNodes[segIdx + 1]);

    if (!fromNode || !toNode || fromNode.x === undefined || toNode.x === undefined) return null;

    const x1 = scaleX(fromNode.x);
    const y1 = scaleY(fromNode.y);
    const x2 = scaleX(toNode.x);
    const y2 = scaleY(toNode.y);

    const currX = x1 + (x2 - x1) * segProgress;
    const currY = y1 + (y2 - y1) * segProgress;

    const angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);

    const isAtDepot = toNode.id === (instance.depotIdx ?? 0);
    const isStation = chargingStations.some((s) => s.id === toNode.id);
    const statusText = isAtDepot
      ? `Returning to Depot (0)`
      : isStation
      ? `Charging Station S${toNode.id}`
      : `Customer C${toNode.id}`;

    return {
      x: currX,
      y: currY,
      angle: isNaN(angle) ? 0 : angle,
      fromNode,
      toNode,
      segIdx,
      statusText,
    };
  };

  // Generate SVG path string with curve displacement
  const createRoutePathD = (routeNodes, vIdx) => {
    if (!routeNodes || routeNodes.length < 2) return '';
    const curveOffset = (vIdx - (routeList.length - 1) / 2) * 12;

    const coords = routeNodes
      .map((id) => {
        const node = getNode(id);
        if (!node || node.x === undefined || node.y === undefined) return null;
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
    <div style={{ width: '100%' }}>
      {/* Simulation Controls Toolbar */}
      <div
        style={{
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          marginBottom: '1rem',
          background: '#0f172a',
          padding: '0.85rem 1.25rem',
          borderRadius: '0.75rem',
          border: '1px solid rgba(255,255,255,0.08)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {isPlaying ? (
            <button className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }} onClick={() => setIsPlaying(false)}>
              <Pause size={16} /> Pause
            </button>
          ) : (
            <button className="btn btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }} onClick={() => setIsPlaying(true)}>
              <Play size={16} /> Play EV Simulation
            </button>
          )}

          <button className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }} onClick={handleReplay}>
            <RotateCcw size={16} /> Replay
          </button>
        </div>

        {/* Speed Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
          <FastForward size={16} /> Speed:
          {[0.5, 1.0, 2.0].map((s) => (
            <button
              key={s}
              style={{
                padding: '0.3rem 0.65rem',
                borderRadius: '0.4rem',
                border: '1px solid rgba(255,255,255,0.12)',
                background: speed === s ? 'var(--primary-indigo)' : 'rgba(255,255,255,0.05)',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.8rem',
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            <Navigation size={16} /> Animate:
            <select
              className="form-select"
              style={{ padding: '0.3rem 0.6rem', fontSize: '0.85rem', width: 'auto' }}
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

      {/* 620px Route Canvas Wrapper */}
      <div className="visualizer-canvas-wrapper">
        <svg
          width="100%"
          height="100%"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ background: '#0b1120', display: 'block' }}
        >
          <defs>
            <filter id="glowEffect" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Grid Background */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio, i) => {
            const gx = padding + ratio * (width - 2 * padding);
            const gy = padding + ratio * (height - 2 * padding);
            return (
              <g key={i}>
                <line x1={gx} y1={padding - 20} x2={gx} y2={height - padding + 20} stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" />
                <line x1={padding - 20} y1={gy} x2={width - padding + 20} y2={gy} stroke="rgba(255,255,255,0.03)" strokeDasharray="4 4" />
              </g>
            );
          })}

          {/* Render Vehicle Route Lines */}
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
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                filter="url(#glowEffect)"
              />
            );
          })}

          {/* Render Informational Charging Stations if present */}
          {chargingStations.map((st) => {
            if (!st || typeof st.x !== 'number' || typeof st.y !== 'number') return null;
            const cx = scaleX(st.x);
            const cy = scaleY(st.y);
            return (
              <g key={`st-${st.id}`} transform={`translate(${cx}, ${cy})`}>
                <rect x="-11" y="-11" width="22" height="22" rx="4" fill="rgba(16, 185, 129, 0.25)" stroke="#10b981" strokeWidth="2" />
                <circle r="3.5" fill="#34d399" />
                <text x="0" y="24" textAnchor="middle" fill="#34d399" fontSize="10" fontWeight="700">
                  S{st.id}
                </text>
              </g>
            );
          })}

          {/* Destination Node Pulse Rings during Animation */}
          {routeList.map((r, rIdx) => {
            if (vehicleFilter !== 'all' && parseInt(vehicleFilter, 10) !== rIdx) return null;
            const pos = calculateVehiclePosition(r);
            if (!pos || !pos.toNode || typeof pos.toNode.x !== 'number' || typeof pos.toNode.y !== 'number') return null;
            const tx = scaleX(pos.toNode.x);
            const ty = scaleY(pos.toNode.y);

            return (
              <circle
                key={`dest-${rIdx}`}
                cx={tx}
                cy={ty}
                r="16"
                fill="none"
                stroke="#38bdf8"
                strokeWidth="2"
                strokeDasharray="4 2"
              />
            );
          })}

          {/* Animated EV Markers */}
          {routeList.map((r, rIdx) => {
            if (vehicleFilter !== 'all' && parseInt(vehicleFilter, 10) !== rIdx) return null;
            const pos = calculateVehiclePosition(r);
            if (!pos || typeof pos.x !== 'number' || typeof pos.y !== 'number' || isNaN(pos.x) || isNaN(pos.y)) return null;
            const style = ROUTE_STYLES[rIdx % ROUTE_STYLES.length];

            return (
              <g key={`ev-${rIdx}`} transform={`translate(${pos.x}, ${pos.y}) rotate(${pos.angle})`}>
                <circle r="14" fill={style.glow} />
                <rect x="-10" y="-6" width="20" height="12" rx="3" fill={style.stroke} stroke="#fff" strokeWidth="1.5" />
                <circle cx="-5" cy="-6" r="2" fill="#fff" />
                <circle cx="5" cy="-6" r="2" fill="#fff" />
                <circle cx="-5" cy="6" r="2" fill="#fff" />
                <circle cx="5" cy="6" r="2" fill="#fff" />
              </g>
            );
          })}

          {/* Render Depot & Customer Nodes */}
          {allValidNodes.map((node) => {
            // Avoid duplicate rendering of charging stations already rendered above
            if (chargingStations.some((s) => s.id === node.id) || node.isStation) return null;

            const isDepot = node.id === (instance.depotIdx ?? 0);
            const cx = scaleX(node.x);
            const cy = scaleY(node.y);

            const labelOffsetY = cy > height / 2 ? 26 : -18;
            const coordOffsetY = cy > height / 2 ? 38 : -30;

            return (
              <g key={`node-${node.id}`} transform={`translate(${cx}, ${cy})`}>
                <circle
                  r={isDepot ? 16 : 11}
                  fill={isDepot ? 'rgba(245, 158, 11, 0.3)' : 'rgba(6, 182, 212, 0.22)'}
                  stroke={isDepot ? '#f59e0b' : '#06b6d4'}
                  strokeWidth="2.0"
                />
                <circle r={isDepot ? 7 : 5} fill={isDepot ? '#fbbf24' : '#38bdf8'} />

                <text
                  x="0"
                  y={labelOffsetY}
                  textAnchor="middle"
                  fill={isDepot ? '#fbbf24' : '#f8fafc'}
                  fontSize="11"
                  fontWeight="700"
                  fontFamily="Outfit, sans-serif"
                >
                  {isDepot ? `Depot (0)` : `C${node.id} (${node.demand || 0}u)`}
                </text>

                <text
                  x="0"
                  y={coordOffsetY}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.45)"
                  fontSize="8.5"
                  fontFamily="Fira Code, monospace"
                >
                  ({node.x.toFixed(1)}, {node.y.toFixed(1)})
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Dynamic Vehicle Route Legend */}
      {routeList.length > 0 && (
        <div
          style={{
            marginTop: '1rem',
            background: '#0f172a',
            padding: '0.85rem 1.25rem',
            borderRadius: '0.75rem',
            border: '1px solid rgba(255,255,255,0.08)',
          }}
        >
          <div style={{ fontSize: '0.875rem', fontWeight: '700', color: '#f8fafc', marginBottom: '0.5rem', fontFamily: 'var(--font-heading)' }}>
            Vehicle Route Legend & Navigation Status
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {routeList.map((r, rIdx) => {
              const style = ROUTE_STYLES[rIdx % ROUTE_STYLES.length];
              const pos = calculateVehiclePosition(r);

              return (
                <div key={rIdx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', fontSize: '0.85rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span
                      style={{
                        width: '14px',
                        height: '14px',
                        borderRadius: '4px',
                        backgroundColor: style.stroke,
                        display: 'inline-block',
                      }}
                    ></span>
                    <strong style={{ color: style.stroke }}>Vehicle {rIdx + 1}:</strong>
                    <span className="route-code" style={{ fontSize: '0.825rem' }}>
                      {r.join(' → ')}
                    </span>
                  </div>

                  {pos && (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', background: 'rgba(255,255,255,0.04)', padding: '0.2rem 0.6rem', borderRadius: '0.4rem' }}>
                      Vehicle {rIdx + 1} → <strong style={{ color: '#38bdf8' }}>{pos.statusText}</strong>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Simulation Progress Bar */}
          <div style={{ marginTop: '0.85rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              <span>EV Simulation Progress</span>
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
        </div>
      )}
    </div>
  );
}
