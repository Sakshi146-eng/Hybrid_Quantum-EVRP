import React from 'react';

export default function RouteVisualizer({ instance, route }) {
  const { nodes, numNodes } = instance;
  const padding = 50;
  const width = 500;
  const height = 260;

  // Calculate coordinate bounds for scaling
  const maxX = Math.max(...nodes.map((n) => n.x), 5.0);
  const maxY = Math.max(...nodes.map((n) => n.y), 5.0);

  const scaleX = (x) => padding + (x / maxX) * (width - 2 * padding);
  const scaleY = (y) => height - padding - (y / maxY) * (height - 2 * padding);

  // Generate SVG path string for the candidate route
  let routePoints = [];
  if (route && route.length > 1) {
    routePoints = route
      .map((idx) => {
        const node = nodes[idx];
        if (!node) return null;
        return `${scaleX(node.x)},${scaleY(node.y)}`;
      })
      .filter(Boolean);
  }
  const pathD = routePoints.length > 0 ? `M ${routePoints.join(' L ')}` : '';

  return (
    <div className="visualizer-canvas-wrapper" style={{ position: 'relative', width: '100%', height: '280px' }}>
      <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} style={{ background: '#0b1120' }}>
        <defs>
          <linearGradient id="routeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" />
            <stop offset="50%" stopColor="#6366f1" />
            <stop offset="100%" stopColor="#8b5cf6" />
          </linearGradient>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Grid lines */}
        {[0, 1, 2, 3, 4, 5].map((val) => (
          <g key={val}>
            <line
              x1={scaleX(val)}
              y1={padding - 10}
              x2={scaleX(val)}
              y2={height - padding + 10}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="3 3"
            />
            <line
              x1={padding - 10}
              y1={scaleY(val)}
              x2={width - padding + 10}
              y2={scaleY(val)}
              stroke="rgba(255,255,255,0.04)"
              strokeDasharray="3 3"
            />
          </g>
        ))}

        {/* Route Path */}
        {pathD && (
          <path
            d={pathD}
            fill="none"
            stroke="url(#routeGradient)"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter="url(#glow)"
          />
        )}

        {/* Plot Nodes */}
        {nodes.map((node) => {
          const isDepot = node.id === 0;
          const cx = scaleX(node.x);
          const cy = scaleY(node.y);

          return (
            <g key={node.id} transform={`translate(${cx}, ${cy})`}>
              {/* Outer Glow Circle */}
              <circle
                r={isDepot ? 14 : 10}
                fill={isDepot ? 'rgba(245, 158, 11, 0.25)' : 'rgba(6, 182, 212, 0.2)'}
                stroke={isDepot ? '#f59e0b' : '#06b6d4'}
                strokeWidth="2"
              />

              {/* Node Core */}
              <circle r={isDepot ? 6 : 4} fill={isDepot ? '#fbbf24' : '#38bdf8'} />

              {/* Node Label */}
              <text
                x="0"
                y={isDepot ? -20 : 22}
                textAnchor="middle"
                fill={isDepot ? '#fbbf24' : '#e2e8f0'}
                fontSize="11"
                fontWeight="700"
                fontFamily="Outfit, sans-serif"
              >
                {isDepot ? 'Depot (0)' : `C${node.id} (${node.demand}t)`}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
