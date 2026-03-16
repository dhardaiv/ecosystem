import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip as RechartsTip,
} from 'recharts';

const SPARKLINE_MAX = 30;

function Sparkline({ data, color }) {
  if (!data || data.length < 2) return <div className="sparkline-wrap" />;
  const slice = data.slice(-SPARKLINE_MAX);
  const max = Math.max(...slice, 0.001);
  const w = 80;
  const h = 24;
  const pts = slice.map((v, i) => {
    const x = (i / (slice.length - 1)) * w;
    const y = h - (v / max) * (h - 2);
    return `${x},${y}`;
  });
  return (
    <div className="sparkline-wrap">
      <svg width={w} height={h} style={{ display: 'block' }}>
        <polyline
          points={pts.join(' ')}
          fill="none"
          stroke={color}
          strokeWidth={1}
          opacity={0.7}
        />
      </svg>
    </div>
  );
}

function MetricCell({ label, value, sparkData, sparkColor, warn }) {
  const displayVal = typeof value === 'number'
    ? Number.isInteger(value) ? value.toString() : value.toFixed(3)
    : '—';

  return (
    <div className="metric-cell">
      <div className="metric-cell-label">{label}</div>
      <div className={`metric-cell-value mono ${warn || ''}`}>{displayVal}</div>
      <Sparkline data={sparkData} color={sparkColor} />
    </div>
  );
}

export default function MetricsPanel({ metricsHistory, warnings }) {
  const latest = metricsHistory[metricsHistory.length - 1] || {};

  const mseHistory   = useMemo(() => metricsHistory.map(m => m.loss_mse  ?? 0), [metricsHistory]);
  const bceHistory   = useMemo(() => metricsHistory.map(m => m.loss_bce  ?? 0), [metricsHistory]);
  const ceHistory    = useMemo(() => metricsHistory.map(m => m.loss_ce   ?? 0), [metricsHistory]);
  const auxHistory   = useMemo(() => metricsHistory.map(m => m.loss_aux  ?? 0), [metricsHistory]);
  const ghostHistory = useMemo(() => metricsHistory.map(m => m.ghost_count ?? 0), [metricsHistory]);

  const ghostCount = latest.ghost_count ?? 0;
  const ghostWarn = ghostCount > 3 ? 'red' : ghostCount > 0 ? 'amber' : '';

  const chartData = useMemo(
    () => metricsHistory.map((m, i) => ({
      step: i,
      mse:  m.loss_mse  ?? null,
      bce:  m.loss_bce  ?? null,
      ce:   m.loss_ce   ?? null,
      aux:  m.loss_aux  ?? null,
    })),
    [metricsHistory]
  );

  const latestWarning = warnings && warnings.length > 0
    ? warnings[warnings.length - 1]
    : null;

  return (
    <div className="panel panel-metrics">
      <div className="panel-header">
        <span className="panel-title">Model Diagnostics</span>
      </div>
      <div className="metrics-inner">
        <div className="metric-grid">
          <MetricCell
            label="Position MSE"
            value={latest.loss_mse}
            sparkData={mseHistory}
            sparkColor="#a8c0cc"
          />
          <MetricCell
            label="Alive BCE"
            value={latest.loss_bce}
            sparkData={bceHistory}
            sparkColor="#00E5CC"
          />
          <MetricCell
            label="Food CE"
            value={latest.loss_ce}
            sparkData={ceHistory}
            sparkColor="#2d9e48"
          />
          <MetricCell
            label="Ghost Agents"
            value={ghostCount}
            sparkData={ghostHistory}
            sparkColor="#FF8C00"
            warn={ghostWarn}
          />
        </div>

        {latestWarning && (
          <div className="warning-log">{latestWarning}</div>
        )}

        <div className="loss-chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
              <CartesianGrid stroke="#1e262e" strokeDasharray="2 4" />
              <XAxis
                dataKey="step"
                tick={{ fill: '#6b7c8a', fontSize: 9, fontFamily: 'Space Mono' }}
                tickLine={false}
                axisLine={{ stroke: '#242b33' }}
              />
              <YAxis
                scale="log"
                domain={['auto', 'auto']}
                tick={{ fill: '#6b7c8a', fontSize: 9, fontFamily: 'Space Mono' }}
                tickLine={false}
                axisLine={false}
                width={36}
                tickFormatter={v => v < 0.01 ? v.toExponential(0) : v.toFixed(2)}
                allowDataKey={false}
              />
              <RechartsTip
                contentStyle={{
                  background: '#111418',
                  border: '1px solid #242b33',
                  fontFamily: 'Space Mono',
                  fontSize: 10,
                  color: '#a8c0cc',
                }}
                labelStyle={{ color: '#6b7c8a' }}
              />
              <Line type="monotone" dataKey="mse" stroke="#a8c0cc" strokeWidth={1.2} dot={false} name="MSE" isAnimationActive={false} connectNulls />
              <Line type="monotone" dataKey="bce" stroke="#00E5CC" strokeWidth={1.2} dot={false} name="BCE" isAnimationActive={false} connectNulls />
              <Line type="monotone" dataKey="ce"  stroke="#2d9e48" strokeWidth={1.2} dot={false} name="CE"  isAnimationActive={false} connectNulls />
              <Line type="monotone" dataKey="aux" stroke="#FF8C00" strokeWidth={1.2} dot={false} name="AUX" isAnimationActive={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
