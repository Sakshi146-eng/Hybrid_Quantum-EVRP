import React from 'react';

// Visually distinct color palettes for vehicle routes
const ROUTE_STYLES = [
  { stroke: '#06b6d4', glow: 'rgba(6, 182, 212, 0.4)', bg: '#06b6d4', name: 'Vehicle 1 (Cyan)' },
  { stroke: '#10b981', glow: 'rgba(16, 185, 129, 0.4)', bg: '#10b981', name: 'Vehicle 2 (Emerald)' },
  { stroke: '#a855f7', glow: 'rgba(168, 85, 247, 0.4)', bg: '#a855f7', name: 'Vehicle 3 (Purple)' },
  { stroke: '#f43f5e', glow: 'rgba(244, 63, 94, 0.4)', bg: '#f43f5e', name: 'Vehicle 4 (Rose)' },
  { stroke: '#f59e0b', glow: 'rgba(245, 158, 11, 0.4)', bg: '#f59e0b', name: 'Vehicle 5 (Amber)' },
];

export default function RouteVisualizer({ instance, routes }) {
  if (!instance || !instance.nodes) return null;

  const { nodes } = instance;
  const padding = 60;
  const width = 540;
  const height = 290;

  // Calculate coordinate bounds for scaling
  const maxX = Math.max(...nodes.map((n) => n.x), 5.0);
  const maxY = Math.max(...nodes.map((n) => n.y), 5.0);

  const scaleX = (x) => padding + (x / maxX) * (width - 2 * padding);
  const scaleY = (y) => height - padding - (y / maxY) * (height - 2 * padding);

  // Normalize input routes
  let routeList = [];
  if (Array.isArray(routes)) {
    if (routes.length > 0 && Array.isArray(routes[0])) {
      routeList = routes;
    } else if (routes.length > 0 && typeof routes[0] === 'number') {
      routeList = [routes];
    }
  }

  // Generate SVG path string with curve displacement to avoid overlapping line ambiguity
  const createRoutePathD = (routeNodes, vehicleIdx) => {
    if (!routeNodes || routeNodes.length < 2) return '';

    // Offset curve per vehicle route index so overlapping horizontal/vertical segments remain distinct
    const curveOffset = (vehicleIdx - (routeList.length - 1) / 2) * 12;

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
        // Control point offset perpendicular to line segment
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
      <svg
        width="100%"
        height="300"
        viewBox={`0 0 ${width} ${height}`}
        style={{ background: '#0b1120', borderRadius: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        <defs>
          <filter id="glowEffect" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Grid background lines */}
        {[0, 1, 2, 3, 4, 5].map((val) => (
          <g key={val}>
            <line
              x1={scaleX(val)}
              y1={padding - 20}
              x2={scaleX(val)}
              y2={height - padding + 20}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="3 3"
            />
            <line
              x1={padding - 20}
              y1={scaleY(val)}
              x2={width - padding + 20}
              y2={scaleY(val)}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="3 3"
            />
          </g>
        ))}

        {/* Render Vehicle Routes */}
        {routeList.map((r, rIdx) => {
          const pathD = createRoutePathD(r, rIdx);
          if (!pathD) return null;
          const style = ROUTE_STYLES[rIdx % ROUTE_STYLES.length];

          return (
            <g key={rIdx}>
              <path
                d={pathD}
                fill="none"
                stroke={style.stroke}
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
                filter="url(#glowEffect)"
              />
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

              {/* Core Node Circle */}
              <circle r={isDepot ? 6 : 4.5} fill={isDepot ? '#fbbf24' : '#38bdf8'} />

              {/* Node Label & Coordinates */}
              <text
                x="0"
                y={isDepot ? -22 : 24}
                textAnchor="middle"
                fill={isDepot ? '#fbbf24' : '#e2e8f0'}
                fontSize="11"
                fontWeight="700"
                fontFamily="Outfit, sans-serif"
              >
                {isDepot ? `Depot (0)` : `C${node.id} (${node.demand} units)`}
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

      {/* Dynamic Vehicle Route Color Legend */}
      {routeList.length > 0 && (
        <div
          style={{
            marginTop: '0.75rem',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '1rem',
            alignItems: 'center',
            fontSize: '0.8rem',
            background: '#0f172a',
            padding: '0.6rem 1rem',
            borderRadius: '0.5rem',
            border: '1px solid rgba(255,255,255,0.06)',
          }}
        >
          <strong style={{ color: 'var(--text-muted)' }}>Visual Route Legend:</strong>
          {routeList.map((r, rIdx) => {
            const style = ROUTE_STYLES[rIdx % ROUTE_STYLES.length];
            return (
              <div key={rIdx} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '3px',
                    backgroundColor: style.stroke,
                    display: 'inline-block',
                  }}
                ></span>
                <span style={{ color: '#e2e8f0', fontFamily: 'var(--font-mono)' }}>
                  Vehicle {rIdx + 1}: {r.join(' → ')}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
