"""
Mesa agent definitions: PreyAgent, PredatorAgent, FoodPatch.

Energy convention: normalised to [0, 1] at all times. 
  - Prey/predators die when energy ≤ 0.
  - Reproduction fires when energy ≥ reproduce_threshold.
"""
import random
from typing import Optional
from mesa import Agent


# ── Species type constants ───────────────────────────────────────────────────
SPECIES_PREY = 0
SPECIES_PREDATOR = 1
SPECIES_PATCH = 2


class PreyAgent(Agent):
    """Herbivore that grazes food patches and flees predators."""

    def __init__(self, unique_id, model, energy: Optional[float] = None):
        super().__init__(unique_id, model)
        self.energy: float = energy if energy is not None else model.sim_cfg.prey_initial_energy
        self.age: int = 0
        self.alive: bool = True
        self.species: int = SPECIES_PREY

    # ── sub-steps (called inside step()) ────────────────────────────────────

    def _move(self):
        """Random walk on the toroidal grid."""
        cfg = self.model.sim_cfg
        possible = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_pos = self.random.choice(possible)
        self.model.grid.move_agent(self, new_pos)
        self.energy -= cfg.prey_move_energy_cost

    def _eat(self):
        """Consume food at current cell."""
        cfg = self.model.sim_cfg
        patch = self.model.patch_grid[self.pos]
        if patch.food > 0:
            eaten = min(patch.food, 1.0)          # eat all available food
            self.energy = min(1.0, self.energy + eaten * cfg.prey_eat_gain)
            patch.food = 0.0

    def _metabolise(self):
        self.energy -= self.model.sim_cfg.prey_metabolism

    def _reproduce(self):
        cfg = self.model.sim_cfg
        if self.energy >= cfg.prey_reproduce_threshold:
            self.energy -= cfg.prey_reproduce_cost
            offspring_energy = cfg.prey_initial_energy
            self.model.spawn_prey(self.pos, offspring_energy)

    def _check_death(self):
        if self.energy <= 0:
            self.alive = False
            self.model.kill_agent(self)

    def step(self):
        if not self.alive:
            return
        self.age += 1
        self._move()
        self._eat()
        self._metabolise()
        self._reproduce()
        self._check_death()


class PredatorAgent(Agent):
    """Carnivore that hunts nearby prey."""

    def __init__(self, unique_id, model, energy: Optional[float] = None):
        super().__init__(unique_id, model)
        self.energy: float = energy if energy is not None else model.sim_cfg.predator_initial_energy
        self.age: int = 0
        self.alive: bool = True
        self.species: int = SPECIES_PREDATOR

    def _hunt(self):
        """Move toward nearest prey in hunt_radius, eat if on same cell."""
        cfg = self.model.sim_cfg
        neighbours = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=True, radius=cfg.predator_hunt_radius
        )
        prey_list = [a for a in neighbours if isinstance(a, PreyAgent) and a.alive]
        if prey_list:
            target = self.random.choice(prey_list)
            # Move one step toward target
            dx = target.pos[0] - self.pos[0]
            dy = target.pos[1] - self.pos[1]
            # Normalise to one-step toroidal
            W, H = self.model.grid.width, self.model.grid.height
            dx = (dx + W // 2) % W - W // 2
            dy = (dy + H // 2) % H - H // 2
            step_x = self.pos[0] + (1 if dx > 0 else (-1 if dx < 0 else 0))
            step_y = self.pos[1] + (1 if dy > 0 else (-1 if dy < 0 else 0))
            new_pos = (step_x % W, step_y % H)
            self.model.grid.move_agent(self, new_pos)
            self.energy -= cfg.predator_move_energy_cost
            # Try to eat prey on new cell
            cell_contents = self.model.grid.get_cell_list_contents([self.pos])
            for agent in cell_contents:
                if isinstance(agent, PreyAgent) and agent.alive:
                    agent.alive = False
                    self.model.kill_agent(agent)
                    self.energy = min(1.0, self.energy + cfg.predator_eat_gain)
                    break
        else:
            # Random walk when no prey nearby
            possible = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
            new_pos = self.random.choice(possible)
            self.model.grid.move_agent(self, new_pos)
            self.energy -= cfg.predator_move_energy_cost

    def _metabolise(self):
        self.energy -= self.model.sim_cfg.predator_metabolism

    def _reproduce(self):
        cfg = self.model.sim_cfg
        if self.energy >= cfg.predator_reproduce_threshold:
            self.energy -= cfg.predator_reproduce_cost
            self.model.spawn_predator(self.pos, cfg.predator_initial_energy)

    def _check_death(self):
        if self.energy <= 0:
            self.alive = False
            self.model.kill_agent(self)

    def step(self):
        if not self.alive:
            return
        self.age += 1
        self._hunt()
        self._metabolise()
        self._reproduce()
        self._check_death()


class FoodPatch(Agent):
    """
    Static patch agent — one per grid cell.
    Carries a `food` level in [0, 1] that regrows each step.
    """

    def __init__(self, unique_id, model, food: float = 0.0):
        super().__init__(unique_id, model)
        self.food: float = food
        self.species: int = SPECIES_PATCH
        # FoodPatch is never 'dead'; alive=True always for masking purposes
        self.alive: bool = True
        self.age: int = 0                          # unused but keeps interface uniform

    def step(self):
        regrow = self.model.sim_cfg.food_regrow_rate
        self.food = min(1.0, self.food + regrow)
