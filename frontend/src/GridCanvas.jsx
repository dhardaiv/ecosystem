import { useRef, useEffect, useState, useCallback } from 'react';
import { useCanvasRenderer, hitTestAgent } from './useCanvasRenderer';

function AgentTooltip({ agent, x, y, cw, ch }) {
  if (!agent) return null;
  const tipW = 148;
  const tipH = 110;
  const left = x + tipW + 12 > cw ? x - tipW - 8 : x + 12;
  const top  = y + tipH + 12 > ch ? y - tipH - 8 : y + 12;

  return (
    <div
      className="agent-tooltip"
      style={{ left, top }}
    >
      <div className="agent-tooltip-row">
        <span className="agent-tooltip-label">ID</span>
        <span className="agent-tooltip-val mono">{agent.id}</span>
      </div>
      <div className="agent-tooltip-row">
        <span className="agent-tooltip-label">Type</span>
        <span
          className="agent-tooltip-val"
          style={{ color: agent.type === 'prey' ? 'var(--accent-prey)' : 'var(--accent-pred)' }}
        >
          {agent.type}
        </span>
      </div>
      <div className="agent-tooltip-row">
        <span className="agent-tooltip-label">Energy</span>
        <span className="agent-tooltip-val mono">{agent.energy?.toFixed(3)}</span>
      </div>
      <div className="agent-tooltip-row">
        <span className="agent-tooltip-label">Age</span>
        <span className="agent-tooltip-val mono">{agent.age?.toFixed(3)}</span>
      </div>
      <div className="agent-tooltip-row">
        <span className="agent-tooltip-label">p_alive</span>
        <span className="agent-tooltip-val mono">{agent.p_alive?.toFixed(3) ?? '—'}</span>
      </div>
    </div>
  );
}

export default function GridCanvas({ stateRef, gridFlash }) {
  const containerRef = useRef(null);
  const canvasRef    = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  const { onStepArrived } = useCanvasRenderer(canvasRef, stateRef);

  // Expose onStepArrived via stateRef so App can call it
  useEffect(() => {
    stateRef.current._onStepArrived = onStepArrived;
  }, [stateRef, onStepArrived]);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    const canvas    = canvasRef.current;
    if (!container || !canvas) return;

    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        canvas.width  = Math.floor(width);
        canvas.height = Math.floor(height);
      }
    });
    ro.observe(container);

    // set initial size
    canvas.width  = container.clientWidth;
    canvas.height = container.clientHeight;

    return () => ro.disconnect();
  }, []);

  const handleMouseMove = useCallback(e => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const agents = stateRef.current.agents || [];
    const cw = canvas.width;
    const ch = canvas.height;
    const hit = hitTestAgent(agents, mx, my, cw, ch);
    setTooltip(hit ? { agent: hit, x: mx, y: my, cw, ch } : null);
  }, [stateRef]);

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  return (
    <div
      className="panel panel-grid"
      style={{ position: 'relative', overflow: 'hidden' }}
    >
      <div className="panel-header">
        <span className="panel-title">Ecosystem Grid</span>
      </div>
      <div
        ref={containerRef}
        style={{ width: '100%', height: 'calc(100% - 33px)', position: 'relative' }}
      >
        <canvas
          ref={canvasRef}
          style={{ display: 'block', width: '100%', height: '100%' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
        {tooltip && (
          <AgentTooltip
            agent={tooltip.agent}
            x={tooltip.x}
            y={tooltip.y}
            cw={tooltip.cw}
            ch={tooltip.ch}
          />
        )}
        {gridFlash && <div className="grid-border-flash" key={gridFlash} />}
      </div>
    </div>
  );
}
