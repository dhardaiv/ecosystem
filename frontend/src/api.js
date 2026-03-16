const BASE = '';

async function post(path, body = {}) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => `HTTP ${res.status}`);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const text = await res.text().catch(() => `HTTP ${res.status}`);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export function startRollout({ episode_seed, n_steps, noise_injection }) {
  return post('/api/rollout/start', { episode_seed, n_steps, noise_injection });
}

export function stepRollout(rollout_id) {
  return post('/api/rollout/step', { rollout_id });
}

export function resetRollout() {
  return post('/api/rollout/reset');
}

export function getRolloutHistory(rollout_id) {
  return get(`/api/rollout/history?rollout_id=${rollout_id}`);
}
