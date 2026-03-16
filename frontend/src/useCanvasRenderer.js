import { useEffect, useRef, useCallback } from 'react';

// ── Colour helpers ───────────────────────────────────────────
const PREY_COLOR  = '#00E5CC';
const PRED_COLOR  = '#FF8C00';
const FOOD_MAX    = '#2d6e3a';

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return [r, g, b];
}

const FOOD_RGB = hexToRgb(FOOD_MAX);

function foodColor(food) {
  const t = Math.max(0, Math.min(1, food));
  const r = Math.round(FOOD_RGB[0] * t);
  const g = Math.round(FOOD_RGB[1] * t + 8 * t);
  const b = Math.round(FOOD_RGB[2] * t);
  return `rgb(${r},${g},${b})`;
}

// ── Coordinate mapping ───────────────────────────────────────
function toPixel(nx, ny, cw, ch) {
  return [nx * cw, ny * ch];
}

// ── Draw food patches ────────────────────────────────────────
// p.x and p.y are integer grid indices (0..gridW-1, 0..gridH-1)
function drawPatches(ctx, patches, cw, ch, gridW, gridH) {
  // First fill the whole canvas with the "no food" base colour
  ctx.fillStyle = '#0d1210';
  ctx.fillRect(0, 0, cw, ch);

  if (!patches || patches.length === 0) return;

  const cellW = cw / gridW;
  const cellH = ch / gridH;
  patches.forEach(p => {
    if (!p.food || p.food < 0.02) return;
    ctx.fillStyle = foodColor(p.food);
    // p.x / p.y are integer grid coords
    ctx.fillRect(p.x * cellW, p.y * cellH, cellW + 0.5, cellH + 0.5);
  });
}

// ── Draw motion trails ───────────────────────────────────────
function drawTrails(ctx, agentHistory, agents, cw, ch) {
  const aliveIds = new Set((agents || []).filter(a => a.alive).map(a => a.id));
  Object.entries(agentHistory).forEach(([idStr, history]) => {
    const id = Number(idStr);
    if (!aliveIds.has(id) || history.length < 2) return;
    const agent = agents.find(a => a.id === id);
    if (!agent) return;
    const color = agent.type === 'prey' ? PREY_COLOR : PRED_COLOR;

    for (let i = 1; i < history.length; i++) {
      const t = i / (history.length - 1);
      const alpha = 0.05 + t * 0.35;
      const prev = history[i - 1];
      const curr = history[i];
      const [x0, y0] = toPixel(prev.x, prev.y, cw, ch);
      const [x1, y1] = toPixel(curr.x, curr.y, cw, ch);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  });
}

// ── Draw a predator diamond ──────────────────────────────────
function drawDiamond(ctx, cx, cy, size) {
  ctx.beginPath();
  ctx.moveTo(cx, cy - size);
  ctx.lineTo(cx + size, cy);
  ctx.lineTo(cx, cy + size);
  ctx.lineTo(cx - size, cy);
  ctx.closePath();
}

// ── Draw agents ──────────────────────────────────────────────
function drawAgents(ctx, agents, pulsePhase, cw, ch) {
  if (!agents) return;
  agents.forEach(agent => {
    if (!agent.alive) return;
    const [cx, cy] = toPixel(agent.x, agent.y, cw, ch);
    const energy = agent.energy ?? 0.5;
    const baseR = 3 + (energy * 3);
    const pulse = Math.sin(pulsePhase + agent.id * 1.3) * 0.5;
    const r = baseR + pulse;

    const isGhost = agent.p_alive !== undefined && agent.p_alive >= 0.3 && agent.p_alive < 0.5;
    const alpha = isGhost ? 0.3 : 1.0;
    ctx.globalAlpha = alpha;

    if (agent.type === 'prey') {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = PREY_COLOR;
      ctx.fill();
      if (isGhost) {
        ctx.beginPath();
        ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
        ctx.strokeStyle = PREY_COLOR;
        ctx.globalAlpha = 0.6;
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    } else {
      drawDiamond(ctx, cx, cy, r);
      ctx.fillStyle = PRED_COLOR;
      ctx.fill();
      if (isGhost) {
        drawDiamond(ctx, cx, cy, r + 2);
        ctx.strokeStyle = PRED_COLOR;
        ctx.globalAlpha = 0.6;
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.globalAlpha = 1;
  });
}

// ── Draw death flashes ───────────────────────────────────────
function drawDeathFlashes(ctx, deathFlashes, cw, ch) {
  deathFlashes.forEach(flash => {
    const [cx, cy] = toPixel(flash.x, flash.y, cw, ch);
    const alpha = flash.frame / 3;
    ctx.beginPath();
    ctx.arc(cx, cy, 7 + (3 - flash.frame), 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

// ── Hover hit-test ───────────────────────────────────────────
export function hitTestAgent(agents, mx, my, cw, ch) {
  if (!agents) return null;
  for (let i = agents.length - 1; i >= 0; i--) {
    const a = agents[i];
    if (!a.alive) continue;
    const [cx, cy] = toPixel(a.x, a.y, cw, ch);
    const dx = cx - mx;
    const dy = cy - my;
    if (Math.sqrt(dx * dx + dy * dy) <= 8) return a;
  }
  return null;
}

// ── Main hook ────────────────────────────────────────────────
export function useCanvasRenderer(canvasRef, stateRef) {
  const rafRef     = useRef(null);
  const flashesRef = useRef([]);    // [{x, y, frame}]
  const prevAgentsRef = useRef({});  // id -> {x, y} last rendered position
  const lerpAgentsRef = useRef({});  // id -> {fromX, fromY, toX, toY, startMs}

  const LERP_DURATION = 200;

  // called externally when a new step arrives — register new positions for lerp
  const onStepArrived = useCallback((newAgents, prevAgents) => {
    const now = performance.now();
    const prevMap = {};
    (prevAgents || []).forEach(a => { prevMap[a.id] = a; });

    // detect deaths → register flashes
    (prevAgents || []).forEach(prev => {
      if (prev.alive) {
        const curr = (newAgents || []).find(a => a.id === prev.id);
        if (!curr || !curr.alive) {
          flashesRef.current.push({ x: prev.x, y: prev.y, frame: 3 });
        }
      }
    });

    // register lerp targets
    (newAgents || []).forEach(a => {
      const prev = prevMap[a.id] || prevAgentsRef.current[a.id];
      lerpAgentsRef.current[a.id] = {
        fromX: prev ? prev.x : a.x,
        fromY: prev ? prev.y : a.y,
        toX: a.x,
        toY: a.y,
        startMs: now,
      };
    });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function frame() {
      const state = stateRef.current;
      const cw = canvas.width;
      const ch = canvas.height;
      const now = performance.now();

      // clear
      ctx.clearRect(0, 0, cw, ch);
      ctx.fillStyle = '#0a0c0e';
      ctx.fillRect(0, 0, cw, ch);

      const { patches, agents, gridWidth, gridHeight } = state;

      // 1. food patches
      drawPatches(ctx, patches, cw, ch, gridWidth || 20, gridHeight || 20);

      // 2. compute lerped positions for agents
      const lerpedAgents = (agents || []).map(a => {
        const l = lerpAgentsRef.current[a.id];
        if (!l) return a;
        const t = Math.min(1, (now - l.startMs) / LERP_DURATION);
        const smooth = t < 1 ? t * t * (3 - 2 * t) : 1;
        return { ...a, x: l.fromX + (l.toX - l.fromX) * smooth, y: l.fromY + (l.toY - l.fromY) * smooth };
      });

      // 3. motion trails
      drawTrails(ctx, state.agentHistory || {}, lerpedAgents, cw, ch);

      // 4. agents
      const pulsePhase = (Date.now() / 1000) * Math.PI * 2;
      drawAgents(ctx, lerpedAgents, pulsePhase, cw, ch);

      // 5. death flashes (tick down)
      drawDeathFlashes(ctx, flashesRef.current, cw, ch);
      flashesRef.current = flashesRef.current
        .map(f => ({ ...f, frame: f.frame - 1 }))
        .filter(f => f.frame > 0);

      // 6. store last rendered positions
      lerpedAgents.forEach(a => { prevAgentsRef.current[a.id] = { x: a.x, y: a.y }; });

      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [canvasRef, stateRef]);

  return { onStepArrived };
}
