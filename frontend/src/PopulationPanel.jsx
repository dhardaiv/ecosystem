import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ReferenceLine, ResponsiveContainer, Tooltip as RechartsTip,
  Area, ComposedChart,
} from 'recharts';

const PREY_COLOR = '#00E5CC';
const PRED_COLOR = '#FF8C00';

function CustomLabel({ viewBox, step }) {
  const { x, y } = viewBox;
  return (
    <text x={x + 4} y={y - 4} fill="#E84040" fontSize={9} fontFamily="Space Mono">
      ✕{step}
    </text>
  );
}

function EndLabel({ points, color, label }) {
  if (!points || points.length === 0) return null;
  const last = points[points.length - 1];
  if (!last || last.x == null || last.y == null) return null;
  return (
    <text
      x={last.x + 6}
      y={last.y + 4}
      fill={color}
      fontSize={9}
      fontFamily="Space Mono"
    >
      {label}
    </text>
  );
}

export default function PopulationPanel({ metricsHistory, extinctions }) {
  const data = useMemo(
    () => metricsHistory.map((m, i) => ({
      step: i,
      prey: m.prey_count,
      pred: m.pred_count,
      gap_min: Math.min(m.prey_count, m.pred_count),
      gap_max: Math.max(m.prey_count, m.pred_count),
    })),
    [metricsHistory]
  );

  const extinctionSteps = useMemo(
    () => extinctions || [],
    [extinctions]
  );

  return (
    <div className="panel panel-pop">
      <div className="panel-header">
        <span className="panel-title">Population</span>
      </div>
      <div className="pop-inner">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 48, left: 0, bottom: 4 }}>
            <CartesianGrid stroke="#1e262e" strokeDasharray="2 4" />
            <XAxis
              dataKey="step"
              tick={{ fill: '#6b7c8a', fontSize: 9, fontFamily: 'Space Mono' }}
              tickLine={false}
              axisLine={{ stroke: '#242b33' }}
            />
            <YAxis
              tick={{ fill: '#6b7c8a', fontSize: 9, fontFamily: 'Space Mono' }}
              tickLine={false}
              axisLine={false}
              width={30}
            />
            <RechartsTip
              contentStyle={{
                background: '#111418',
                border: '1px solid #242b33',
                fontFamily: 'Space Mono',
                fontSize: 11,
                color: '#a8c0cc',
              }}
              labelStyle={{ color: '#6b7c8a' }}
              itemStyle={{ color: '#a8c0cc' }}
            />

            {/* shaded band between pred and prey */}
            <Area
              type="monotone"
              dataKey="gap_max"
              stroke="none"
              fill="#00E5CC"
              fillOpacity={0.06}
              legendType="none"
              activeDot={false}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="gap_min"
              stroke="none"
              fill="#0a0c0e"
              fillOpacity={1}
              legendType="none"
              activeDot={false}
              isAnimationActive={false}
            />

            <Line
              type="monotone"
              dataKey="prey"
              stroke={PREY_COLOR}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              label={<EndLabel color={PREY_COLOR} label="PREY" />}
            />
            <Line
              type="monotone"
              dataKey="pred"
              stroke={PRED_COLOR}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              label={<EndLabel color={PRED_COLOR} label="PRED" />}
            />

            {extinctionSteps.map(({ step, type }) => (
              <ReferenceLine
                key={`ext-${step}-${type}`}
                x={step}
                stroke="#E84040"
                strokeDasharray="3 3"
                strokeWidth={1}
                label={<CustomLabel step={step} />}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
