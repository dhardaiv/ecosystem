import { useCallback } from 'react';

const SPEEDS = [0.5, 1, 2, 4];

export default function ControlBar({ state, dispatch, onStart, onStep, onReset }) {
  const {
    isPlaying, isPending, speed, seed, noiseInjection,
    currentStep, maxSteps, rolloutId, isComplete,
  } = state;

  const handleSeed = useCallback(e => {
    dispatch({ type: 'SET_SEED', payload: Number(e.target.value) || 0 });
  }, [dispatch]);

  const handleNoise = useCallback(e => {
    dispatch({ type: 'SET_NOISE', payload: parseFloat(e.target.value) });
  }, [dispatch]);

  const handleSpeed = useCallback(s => {
    dispatch({ type: 'SET_SPEED', payload: s });
  }, [dispatch]);

  const handlePlayPause = useCallback(() => {
    dispatch({ type: 'SET_PLAYING', payload: !isPlaying });
  }, [dispatch, isPlaying]);

  const started = !!rolloutId;

  return (
    <div className="control-bar">
      {/* Seed */}
      <div className="ctrl-group">
        <span className="ctrl-label">Seed</span>
        <input
          className="ctrl-input mono"
          type="number"
          value={seed}
          onChange={handleSeed}
          disabled={isPending || started}
          style={{ width: 56 }}
        />
      </div>

      <div className="divider" />

      {/* Noise */}
      <div className="ctrl-group">
        <span className="ctrl-label">Ghost noise</span>
        <input
          className="ctrl-slider"
          type="range"
          min={0}
          max={0.5}
          step={0.05}
          value={noiseInjection}
          onChange={handleNoise}
          disabled={isPending || started}
        />
        <span className="mono" style={{ fontSize: 11, color: 'var(--text-mono)', minWidth: 28 }}>
          {noiseInjection.toFixed(2)}
        </span>
      </div>

      <div className="divider" />

      {/* Start / Step / Play */}
      {!started ? (
        <button className="ctrl-btn" onClick={onStart} disabled={isPending}>
          Start
        </button>
      ) : (
        <>
          <button
            className="ctrl-btn"
            onClick={onStep}
            disabled={isPending || isPlaying || isComplete}
          >
            Step →
          </button>

          <button
            className={`ctrl-btn ${isPlaying ? 'active' : ''}`}
            onClick={handlePlayPause}
            disabled={isPending || isComplete}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
        </>
      )}

      <div className="divider" />

      {/* Speed */}
      <div className="ctrl-group">
        <span className="ctrl-label">Speed</span>
        <div className="speed-group">
          {SPEEDS.map(s => (
            <button
              key={s}
              className={`ctrl-btn ${speed === s ? 'active' : ''}`}
              onClick={() => handleSpeed(s)}
              disabled={isPending}
              style={{ padding: '5px 10px', fontSize: 11 }}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      <div className="divider" />

      {/* Reset */}
      <button className="ctrl-btn danger" onClick={onReset} disabled={isPending}>
        Reset
      </button>

      {/* Step counter — pushed to right */}
      <div style={{ marginLeft: 'auto' }} />
      <div className="step-counter">
        {isPending && <span className="pending-dot" />}
        <span className="mono">
          Step {String(currentStep).padStart(3, '0')} / {String(maxSteps).padStart(3, '0')}
        </span>
        {isComplete && <span className="badge-complete">Rollout complete</span>}
      </div>
    </div>
  );
}
