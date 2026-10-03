import React from 'react';

// Color palettes for multi-vehicle routes
const ROUTE_COLORS = [
  { stroke: '#06b6d4', glow: 'rgba(6, 182, 212, 0.4)', name: 'Cyan' },
  { stroke: '#10b981', glow: 'rgba(16, 185, 129, 0.4)', name: 'Emerald' },
  { stroke: '#a855f7', glow: 'rgba(168, 85, 247, 0.4)', name: 'Purple' },
  { stroke: '#f43f5e', glow: 'rgba(244, 63, 94, 0.4)', name: 'Rose' },
  { stroke: '#f59e0b', glow: 'rgba(245, 158, 11, 0.4)', name: 'Amber' },
];

export default function RouteVisualizer({ instance, routes }) {
  if (!instance || !instance.nodes) return null;

  const { nodes } = instance;
  const padding = 55;
  const width = 520;
  const height = 280;

  // Calculate coordinate bounds for scaling
  const maxX = Math.max(...nodes.map((n) => n.x), 5.0);
  const maxY = Math.max(...nodes.map((n) => n.y), 5.0);

  const scaleX = (x) => padding + (x / maxX) * (width - 2 * padding);
  const scaleY = (y) => height - padding - (y / maxY) * (height - 2 * padding);

  // Parse input routes (support single route [0,1,0] or array of routes [[0,1,0], [0,2,0]])
  let routeList = [];
  if (Array.isArray(routes)) {
    if (routes.length > 0 && Array.isArray(routes[0])) {
      routeList = routes;
    } else if (routes.length > 0 && typeof routes[0] === 'number') {
      routeList = [routes];
    }
  }

  return (
    <div className="visualizer-canvas-wrapper" style={{ position: 'relative', width: '100%', height: '300px' }}>
      <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} style={{ background: '#0b1120', borderRadius: '0.75rem' }}>
        <defs>
          <filter id="glowEffect" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Coordinate Grid Background Lines */}
        {[0, 1, 2, 3, 4, 5].map((val) => (
          <g key={val}>
            <line
              x1={scaleX(val)}
              y1={padding - 15}
              x2={scaleX(val)}
              y2={height - padding + 15}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="3 3"
            />
            <line
              x1={padding - 15}
              y1={scaleY(val)}
              x2={width - padding + 15}
              y2={scaleY(val)}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="3 3"
            />
          </g>
        ))}

        {/* Render Each Vehicle Route */}
        {routeList.map((r, rIdx) => {
          if (!r || r.length < 2) return null;
          const colorObj = ROUTE_COLORS[rIdx % ROUTE_COLORS.length];

          const points = r
            .map((idx) => {
              const node = nodes[idx];
              if (!node) return null;
              return `${scaleX(node.x)},${scaleY(node.y)}`;
            })
            .filter(Boolean);

          if (points.length < 2) return null;
          const pathD = `M ${points.join(' L ')}`;

          return (
            <g key={rIdx}>
              {/* Route Line */}
              <path
                d={pathD}
                fill="none"
                stroke={colorObj.stroke}
                strokeWidth="3.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                filter="url(#glowEffect)"
              />
            </g>
          );
        })}

        {/* Plot Nodes (Depot & Customers) */}
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

              {/* Inner Node Core */}
              <circle r={isDepot ? 6 : 4.5} fill={isDepot ? '#fbbf24' : '#38bdf8'} />

              {/* Node Title & Coordinates */}
              <text
                x="0"
                y={isDepot ? -22 : 24}
                textAnchor="middle"
                fill={isDepot ? '#fbbf24' : '#e2e8f0'}
                fontSize="11"
                fontWeight="700"
                fontFamily="Outfit, sans-serif"
              >
                {isDepot ? `Depot (0)` : `C${node.id} (${node.demand}t)`}
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
    </div>
  );
}
