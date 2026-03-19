from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
EAT = 4
KILL = 5
REPRODUCE = 6
DIE = 7


@dataclass
class Agent:
    species: int  # 0: prey, 1: predator
    x: int
    y: int
    e: float
    alive: bool = True


class PredatorPreyEnv:
    def __init__(self, H: int, W: int, N_prey: int, N_pred: int, feed_regen_rate: float):
        self.H = H
        self.W = W
        self.N_prey = N_prey
        self.N_pred = N_pred
        self.feed_regen_rate = feed_regen_rate
        self.rng = np.random.default_rng(42)
        self.feed = np.zeros((H, W), dtype=np.float32)
        self.agents: List[Agent] = []
        self.reset()

    def _empty_grid(self) -> np.ndarray:
        grid = -np.ones((self.H, self.W), dtype=np.int32)
        for i, a in enumerate(self.agents):
            if a.alive:
                grid[a.x, a.y] = i
        return grid

    def _random_empty_cell(self, occ: np.ndarray) -> Tuple[int, int]:
        empties = np.argwhere(occ < 0)
        idx = self.rng.integers(0, len(empties))
        x, y = empties[idx]
        return int(x), int(y)

    def reset(self) -> Dict[str, np.ndarray]:
        self.feed = self.rng.uniform(0.0, 1.0, size=(self.H, self.W)).astype(np.float32)
        self.agents = []
        occ = -np.ones((self.H, self.W), dtype=np.int32)
        for _ in range(self.N_prey):
            x, y = self._random_empty_cell(occ)
            occ[x, y] = 1
            self.agents.append(Agent(species=0, x=x, y=y, e=float(self.rng.uniform(0.4, 0.8))))
        for _ in range(self.N_pred):
            x, y = self._random_empty_cell(occ)
            occ[x, y] = 1
            self.agents.append(Agent(species=1, x=x, y=y, e=float(self.rng.uniform(0.4, 0.8))))
        return self._state_dict()

    def _state_dict(self) -> Dict[str, np.ndarray]:
        alive_idx = [i for i, a in enumerate(self.agents) if a.alive]
        n = len(alive_idx)
        e = np.zeros((n,), dtype=np.float32)
        pos = np.zeros((n, 2), dtype=np.int64)
        species = np.zeros((n, 3), dtype=np.float32)
        alive = np.ones((n,), dtype=np.float32)
        for k, idx in enumerate(alive_idx):
            a = self.agents[idx]
            e[k] = np.clip(a.e, 0.0, 1.0)
            pos[k] = np.array([a.x, a.y], dtype=np.int64)
            species[k, a.species] = 1.0
        return {"e": e, "pos": pos, "species": species, "patch": self.feed.copy(), "alive": alive}

    def _adjacent_cells(self, x: int, y: int) -> List[Tuple[int, int]]:
        nbrs = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.H and 0 <= ny < self.W:
                nbrs.append((nx, ny))
        return nbrs

    def _try_move(self, a: Agent, action: int, occ: np.ndarray):
        dx, dy = 0, 0
        if action == UP:
            dx = -1
        elif action == DOWN:
            dx = 1
        elif action == LEFT:
            dy = -1
        elif action == RIGHT:
            dy = 1
        nx, ny = a.x + dx, a.y + dy
        if 0 <= nx < self.H and 0 <= ny < self.W and occ[nx, ny] < 0:
            occ[a.x, a.y] = -1
            a.x, a.y = nx, ny
            occ[a.x, a.y] = 1

    def step(self, actions: np.ndarray):
        """
        actions: [N_alive] action ids in {0..7}.
        Returns: next_state, rewards, alive_mask
        """
        alive_agents = [a for a in self.agents if a.alive]
        if len(actions) != len(alive_agents):
            raise ValueError("actions length must equal number of alive agents.")

        pre_energy = np.array([a.e for a in alive_agents], dtype=np.float32)
        occ = self._empty_grid()

        # Process actions in random order to reduce ordering bias.
        order = self.rng.permutation(len(alive_agents))
        for oi in order:
            a = alive_agents[oi]
            if not a.alive:
                continue
            act = int(actions[oi])
            if act in (UP, DOWN, LEFT, RIGHT):
                a.e -= 0.01
                self._try_move(a, act, occ)
            elif act == EAT:
                if a.species == 0 and self.feed[a.x, a.y] > 1e-6:
                    a.e += 0.20
                    self.feed[a.x, a.y] = 0.0
                else:
                    a.e -= 0.005
            elif act == KILL:
                if a.species == 1:
                    killed = False
                    for nx, ny in self._adjacent_cells(a.x, a.y):
                        victim_idx = occ[nx, ny]
                        if victim_idx >= 0:
                            victim = self.agents[victim_idx]
                            if victim.alive and victim.species == 0:
                                victim.alive = False
                                victim.e = 0.0
                                occ[nx, ny] = -1
                                a.e += 0.15
                                killed = True
                                break
                    if not killed:
                        a.e -= 0.005
                else:
                    a.e -= 0.005
            elif act == REPRODUCE:
                if a.e > 0.7:
                    nbrs = self._adjacent_cells(a.x, a.y)
                    self.rng.shuffle(nbrs)
                    spawned = False
                    for nx, ny in nbrs:
                        if occ[nx, ny] < 0:
                            child_e = 0.30
                            self.agents.append(Agent(species=a.species, x=nx, y=ny, e=child_e, alive=True))
                            occ[nx, ny] = len(self.agents) - 1
                            a.e -= 0.30
                            spawned = True
                            break
                    if not spawned:
                        a.e -= 0.005
                else:
                    a.e -= 0.005
            elif act == DIE:
                a.e = 0.0
                a.alive = False
                occ[a.x, a.y] = -1
            else:
                a.e -= 0.005

            if a.e <= 0.0:
                a.e = 0.0
                a.alive = False
                if 0 <= a.x < self.H and 0 <= a.y < self.W:
                    occ[a.x, a.y] = -1
            a.e = float(np.clip(a.e, 0.0, 1.0))

        self.feed += self.feed_regen_rate * (1.0 - self.feed)
        self.feed = np.clip(self.feed, 0.0, 1.0).astype(np.float32)

        post_alive_agents = [a for a in self.agents if a.alive]
        post_energy = np.array([a.e for a in post_alive_agents], dtype=np.float32)

        # Reward aligned to currently alive set.
        min_len = min(len(pre_energy), len(post_energy))
        rewards = np.zeros((len(post_alive_agents),), dtype=np.float32)
        if min_len > 0:
            rewards[:min_len] = np.clip(post_energy[:min_len] - pre_energy[:min_len], -1.0, 1.0)

        alive_mask = np.ones((len(post_alive_agents),), dtype=np.float32)
        return self._state_dict(), rewards, alive_mask


if __name__ == "__main__":
    env = PredatorPreyEnv(H=16, W=16, N_prey=8, N_pred=4, feed_regen_rate=0.03)
    s = env.reset()
    assert s["patch"].shape == (16, 16)
    for _ in range(3):
        n_alive = len(s["e"])
        acts = np.random.randint(0, 8, size=(n_alive,))
        s, r, alive = env.step(acts)
        assert s["patch"].shape == (16, 16)
        assert np.all(np.isfinite(s["patch"]))
        assert r.ndim == 1
        assert alive.ndim == 1
    print("env.py self-test passed.")
