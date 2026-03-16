import { useReducer, useRef, useEffect, useCallback, useState } from 'react';
import GridCanvas from './GridCanvas';
import PopulationPanel from './PopulationPanel';
import MetricsPanel from './MetricsPanel';
import ControlBar from './ControlBar';
import { startRollout, stepRollout, resetRollout } from './api';

// ── Reducer ──────────────────────────────────────────────────
const initialState = {
  rolloutId:       null,
  gridWidth:       20,
  gridHeight:      20,
  currentStep:     0,
  maxSteps:        100,
  isPlaying:       false,
  isPending:       false,
  isComplete:      false,
  speed:           1,
  seed:            42,
  noiseInjection:  0.0,
  agents:          [],
  patches:         [],
  agentHistory:    {},
  metricsHistory:  [],
  extinctions:     [],
  warnings:        [],
};

function buildAgentHistory(prev, agents, step) {
  const next = { ...prev };
  (agents || []).forEach(a => {
    if (!a.alive) return;
    const arr = next[a.id] ? [...next[a.id]] : [];
    arr.push({ x: a.x, y: a.y, step });
    if (arr.length > 5) arr.shift();
    next[a.id] = arr;
  });
  return next;
}

function detectExtinctions(prevMetrics, newMetrics, step) {
  const exts = [];
  if (!prevMetrics) return exts;
  if (prevMetrics.prey_count > 0 && newMetrics.prey_count === 0) exts.push({ step, type: 'prey' });
  if (prevMetrics.pred_count > 0 && newMetrics.pred_count === 0) exts.push({ step, type: 'pred' });
  return exts;
}

function reducer(state, action) {
  switch (action.type) {
    case 'START_ROLLOUT': {
      const { rolloutId, gridWidth, gridHeight, n_max_agents, initialState: s, maxSteps } = action.payload;
      const agents  = s.agents  || [];
      const patches = s.patches || [];
      return {
        ...state,
        rolloutId,
        gridWidth,
        gridHeight,
        maxSteps:      maxSteps || state.maxSteps,
        currentStep:   0,
        isComplete:    false,
        isPlaying:     false,
        isPending:     false,
        agents,
        patches,
        agentHistory:  buildAgentHistory({}, agents, 0),
        metricsHistory: [],
        extinctions:   [],
        warnings:      [],
      };
    }
    case 'STEP_COMPLETE': {
      const { step, state: s, metrics } = action.payload;
      const agents  = s.agents  || [];
      const patches = s.patches || [];
      const prevMetrics = state.metricsHistory[state.metricsHistory.length - 1];
      const newExts = detectExtinctions(prevMetrics, metrics, step);
      const newWarnings = [...state.warnings];
      if (metrics.ghost_count > 5) {
        newWarnings.push(`⚠ Death cascade risk detected at step ${step}`);
      }
      const isComplete = step >= state.maxSteps;
      return {
        ...state,
        currentStep:    step,
        isComplete,
        isPlaying:      isComplete ? false : state.isPlaying,
        isPending:      false,
        agents,
        patches,
        agentHistory:   buildAgentHistory(state.agentHistory, agents, step),
        metricsHistory: [...state.metricsHistory, metrics],
        extinctions:    [...state.extinctions, ...newExts],
        warnings:       newWarnings,
      };
    }
    case 'SET_PLAYING':  return { ...state, isPlaying: action.payload };
    case 'SET_SPEED':    return { ...state, speed: action.payload };
    case 'SET_SEED':     return { ...state, seed: action.payload };
    case 'SET_NOISE':    return { ...state, noiseInjection: action.payload };
    case 'SET_PENDING':  return { ...state, isPending: action.payload };
    case 'RESET':        return { ...initialState, seed: state.seed };
    default:             return state;
  }
}

// ── App ──────────────────────────────────────────────────────
export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [error, setError] = useState(null);
  const [gridFlash, setGridFlash] = useState(null);

  // keep a ref to state for canvas renderer (avoids stale closures in RAF)
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = { ...state };
  }, [state]);

  // auto-play timer ref
  const playTimerRef = useRef(null);

  // ── API calls ──────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    setError(null);
    dispatch({ type: 'SET_PENDING', payload: true });
    try {
      const res = await startRollout({
        episode_seed:     state.seed,
        n_steps:          state.maxSteps,
        noise_injection:  state.noiseInjection,
      });
      dispatch({
        type: 'START_ROLLOUT',
        payload: {
          rolloutId:    res.rollout_id,
          gridWidth:    res.grid_width,
          gridHeight:   res.grid_height,
          n_max_agents: res.n_max_agents,
          initialState: res.initial_state,
          maxSteps:     state.maxSteps,
        },
      });
    } catch (err) {
      setError(err.message);
      dispatch({ type: 'SET_PENDING', payload: false });
    }
  }, [state.seed, state.maxSteps, state.noiseInjection]);

  const handleStep = useCallback(async () => {
    if (!state.rolloutId || state.isPending || state.isComplete) return;
    setError(null);
    dispatch({ type: 'SET_PENDING', payload: true });

    const prevAgents = stateRef.current.agents;

    try {
      const res = await stepRollout(state.rolloutId);

      // trigger lerp
      if (stateRef.current._onStepArrived) {
        stateRef.current._onStepArrived(res.state.agents, prevAgents);
      }

      // check ghost cascade
      if (res.metrics.ghost_count > 5) {
        setGridFlash(Date.now());
      }

      dispatch({ type: 'STEP_COMPLETE', payload: res });
    } catch (err) {
      setError(err.message);
      dispatch({ type: 'SET_PENDING', payload: false });
    }
  }, [state.rolloutId, state.isPending, state.isComplete]);

  const handleReset = useCallback(async () => {
    dispatch({ type: 'SET_PLAYING', payload: false });
    setError(null);
    setGridFlash(null);
    try { await resetRollout(); } catch { /* best-effort */ }
    dispatch({ type: 'RESET' });
  }, []);

  // ── Auto-play ──────────────────────────────────────────────
  const handleStepRef = useRef(handleStep);
  useEffect(() => { handleStepRef.current = handleStep; }, [handleStep]);

  useEffect(() => {
    if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
      playTimerRef.current = null;
    }
    if (!state.isPlaying || state.isComplete) return;

    const interval = Math.round(500 / state.speed);
    playTimerRef.current = setInterval(() => {
      handleStepRef.current();
    }, interval);

    return () => clearInterval(playTimerRef.current);
  }, [state.isPlaying, state.speed, state.isComplete]);

  return (
    <>
      {error && (
        <div className="error-banner">
          <span>⚠ {error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <div className="app-shell" style={error ? { paddingTop: 36 } : {}}>
        <GridCanvas stateRef={stateRef} gridFlash={gridFlash} />

        <PopulationPanel
          metricsHistory={state.metricsHistory}
          extinctions={state.extinctions}
        />

        <MetricsPanel
          metricsHistory={state.metricsHistory}
          warnings={state.warnings}
        />

        <ControlBar
          state={state}
          dispatch={dispatch}
          onStart={handleStart}
          onStep={handleStep}
          onReset={handleReset}
        />
      </div>
    </>
  );
}
