"""
Mesa EcosystemModel — the ground-truth simulation.

Snapshot format (numpy structured array per step):
  entity_id  int32
  species    int32    0=prey 1=predator 2=patch
  x          float32  normalised to [0,1]
  y          float32  normalised to [0,1]
  energy     float32  [0,1] (food level for patches)
  age        float32  normalised by max_age=500
  alive      float32  1.0 or 0.0
"""
import numpy as np
from typing import Optional, List
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from simulator.agents import PreyAgent, PredatorAgent, FoodPatch
from simulator.agents import SPECIES_PREY, SPECIES_PREDATOR, SPECIES_PATCH
from config import SimConfig

MAX_AGE_NORM = 500.0   # normalisation denominator for agent age


def _snapshot_from_model(model: "EcosystemModel") -> dict:
    """
    Collect a raw snapshot dict with two keys:
      'agents' — list of agent dicts (prey + predators)
      'patches' — list of patch dicts (all grid cells, row-major)
    """
    W, H = model.grid.width, model.grid.height
    agents_data = []
    for agent in model.schedule.agents:
        if isinstance(agent, (PreyAgent, PredatorAgent)):
            x_norm = agent.pos[0] / W
            y_norm = agent.pos[1] / H
            agents_data.append({
                "entity_id": agent.unique_id,
                "species":   agent.species,
                "x":         float(x_norm),
                "y":         float(y_norm),
                "energy":    float(np.clip(agent.energy, 0.0, 1.0)),
                "age":       float(agent.age / MAX_AGE_NORM),
                "alive":     1.0 if agent.alive else 0.0,
            })

    patches_data = []
    for ix in range(W):
        for iy in range(H):
            patch = model.patch_grid[(ix, iy)]
            patches_data.append({
                "entity_id": patch.unique_id,
                "species":   SPECIES_PATCH,
                "x":         float(ix / W),
                "y":         float(iy / H),
                "energy":    float(np.clip(patch.food, 0.0, 1.0)),
                "age":       0.0,
                "alive":     1.0,   # patches are always "alive"
            })

    return {"agents": agents_data, "patches": patches_data}


class EcosystemModel(Model):
    """
    Toroidal predator–prey–food ecosystem.

    Parameters
    ----------
    sim_cfg : SimConfig
    n_prey : int   override initial prey count (for IC variety)
    n_pred : int   override initial predator count
    food_density : float  override initial food coverage
    seed : int
    """

    def __init__(
        self,
        sim_cfg: Optional[SimConfig] = None,
        n_prey: Optional[int] = None,
        n_pred: Optional[int] = None,
        food_density: Optional[float] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.sim_cfg = sim_cfg or SimConfig()
        self.random.seed(seed)
        np.random.seed(seed)

        W = self.sim_cfg.grid_width
        H = self.sim_cfg.grid_height
        self.grid = MultiGrid(W, H, torus=True)
        self.schedule = RandomActivation(self)
        self._next_id = 0

        # ── Food patch layer (one patch per cell) ──────────────────────────
        self.patch_grid: dict[tuple, FoodPatch] = {}
        fd = food_density if food_density is not None else self.sim_cfg.initial_food_density
        for ix in range(W):
            for iy in range(H):
                food_val = float(self.random.random() < fd) * self.random.random()
                patch = FoodPatch(self._next_id, self, food=food_val)
                self._next_id += 1
                self.patch_grid[(ix, iy)] = patch
                self.grid.place_agent(patch, (ix, iy))
                self.schedule.add(patch)

        # ── Prey ──────────────────────────────────────────────────────────
        n_p = n_prey if n_prey is not None else self.sim_cfg.n_initial_prey
        for _ in range(n_p):
            self.spawn_prey(pos=None, energy=self.sim_cfg.prey_initial_energy)

        # ── Predators ─────────────────────────────────────────────────────
        n_d = n_pred if n_pred is not None else self.sim_cfg.n_initial_predators
        for _ in range(n_d):
            self.spawn_predator(pos=None, energy=self.sim_cfg.predator_initial_energy)

        # ── DataCollector ─────────────────────────────────────────────────
        self.datacollector = DataCollector(
            model_reporters={
                "snapshot": _snapshot_from_model,
                "n_prey":   lambda m: sum(1 for a in m.schedule.agents if isinstance(a, PreyAgent) and a.alive),
                "n_pred":   lambda m: sum(1 for a in m.schedule.agents if isinstance(a, PredatorAgent) and a.alive),
            }
        )

        # Collect step 0 before any agent acts
        self.datacollector.collect(self)

    # ── Spawn helpers ────────────────────────────────────────────────────────

    def _random_pos(self) -> tuple:
        return (
            self.random.randrange(self.grid.width),
            self.random.randrange(self.grid.height),
        )

    def spawn_prey(self, pos=None, energy: Optional[float] = None) -> PreyAgent:
        a = PreyAgent(self._next_id, self, energy=energy)
        self._next_id += 1
        if pos is None:
            pos = self._random_pos()
        self.grid.place_agent(a, pos)
        self.schedule.add(a)
        return a

    def spawn_predator(self, pos=None, energy: Optional[float] = None) -> PredatorAgent:
        a = PredatorAgent(self._next_id, self, energy=energy)
        self._next_id += 1
        if pos is None:
            pos = self._random_pos()
        self.grid.place_agent(a, pos)
        self.schedule.add(a)
        return a

    def kill_agent(self, agent):
        """Remove agent from grid and schedule."""
        if agent in self.schedule.agents:
            self.schedule.remove(agent)
        if agent.pos is not None:
            self.grid.remove_agent(agent)

    # ── Simulation step ─────────────────────────────────────────────────────

    def step(self):
        self.schedule.step()          # RandomActivation: random order, all agents
        self.datacollector.collect(self)

    # ── Episode runner ───────────────────────────────────────────────────────

    def run(self, n_steps: int) -> List[dict]:
        """Run n_steps and return list of snapshot dicts (length n_steps+1)."""
        for _ in range(n_steps):
            self.step()
        snapshots = self.datacollector.get_model_vars_dataframe()["snapshot"].tolist()
        return snapshots
