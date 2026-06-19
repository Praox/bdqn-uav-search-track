from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class EnvConfig:
    grid_size: int = 20
    n_value1_targets: int = 3
    n_value2_targets: int = 1
    sensor_radius: int = 2
    track_radius: int = 1
    track_required: int = 3
    max_steps: int = 150
    seed: int | None = None

    step_penalty: float = -0.01
    new_cell_bonus: float = 0.01
    revisit_penalty: float = -0.005
    detect_bonus: float = 0.50
    track_progress_bonus: float = 0.05
    complete_bonus: float = 2.00
    invalid_track_penalty: float = -0.05
    completed_track_penalty: float = -0.03
    all_targets_bonus: float = 3.00


class SearchTrackEnv:
    """Minimal single-UAV search-and-track gridworld.

    This class intentionally mimics a tiny subset of the Gymnasium API:
    reset() -> obs, info
    step(action) -> obs, reward, terminated, truncated, info
    """

    SEARCH = 0
    TRACK = 1
    MOVES = {
        0: (0, 0),   # stay
        1: (-1, 0),  # up
        2: (1, 0),   # down
        3: (0, -1),  # left
        4: (0, 1),   # right
    }

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.action_dim = 10
        self.observation_shape = (5, self.cfg.grid_size, self.cfg.grid_size)
        self.reset()

    def reset(self) -> Tuple[np.ndarray, Dict]:
        g = self.cfg.grid_size
        self.t = 0
        self.drone_pos = np.array([self.rng.integers(g), self.rng.integers(g)], dtype=np.int64)
        self.visited = np.zeros((g, g), dtype=np.float32)
        self.visited[tuple(self.drone_pos)] = 1.0

        self.target_values = np.array([1] * self.cfg.n_value1_targets + [2] * self.cfg.n_value2_targets, dtype=np.float32)
        n_targets = len(self.target_values)
        forbidden = {tuple(self.drone_pos)}
        positions = []
        while len(positions) < n_targets:
            p = (int(self.rng.integers(g)), int(self.rng.integers(g)))
            if p not in forbidden:
                positions.append(p)
                forbidden.add(p)
        self.target_pos = np.array(positions, dtype=np.int64)
        self.detected = np.zeros(n_targets, dtype=bool)
        self.completed = np.zeros(n_targets, dtype=bool)
        self.track_progress = np.zeros(n_targets, dtype=np.int64)

        # Uniform prior over unknown cells. It is updated by search observations.
        self.belief = np.ones((g, g), dtype=np.float32)
        self.belief[tuple(self.drone_pos)] = 0.0
        self._normalize_belief()
        return self._obs(), self._info()

    def step(self, action: int):
        assert 0 <= action < self.action_dim, f"invalid action {action}"
        self.t += 1
        mode = action // 5
        move_id = action % 5
        dr, dc = self.MOVES[move_id]

        reward = self.cfg.step_penalty
        new_pos = self.drone_pos + np.array([dr, dc], dtype=np.int64)
        new_pos = np.clip(new_pos, 0, self.cfg.grid_size - 1)
        self.drone_pos = new_pos

        pos = tuple(self.drone_pos)
        if self.visited[pos] < 0.5:
            reward += self.cfg.new_cell_bonus
        else:
            reward += self.cfg.revisit_penalty
        self.visited[pos] = 1.0

        if mode == self.SEARCH:
            reward += self._search_update()
        else:
            reward += self._track_update()

        terminated = bool(np.all(self.completed))
        if terminated:
            reward += self.cfg.all_targets_bonus
        truncated = self.t >= self.cfg.max_steps
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _search_update(self) -> float:
        reward = 0.0
        visible = self._cells_in_radius(self.drone_pos, self.cfg.sensor_radius)
        for r, c in visible:
            self.belief[r, c] = 0.0

        for i, p in enumerate(self.target_pos):
            if not self.detected[i] and self._dist(self.drone_pos, p) <= self.cfg.sensor_radius:
                self.detected[i] = True
                reward += self.cfg.detect_bonus * float(self.target_values[i])
                # A detected target cell becomes high-confidence in the belief map.
                self.belief[tuple(p)] = max(self.belief[tuple(p)], 1.0)
        self._normalize_belief()
        return reward

    def _track_update(self) -> float:
        candidates = []
        for i, p in enumerate(self.target_pos):
            if self.detected[i] and self._dist(self.drone_pos, p) <= self.cfg.track_radius:
                candidates.append(i)

        if not candidates:
            return self.cfg.invalid_track_penalty

        # Prefer unfinished and high-value targets.
        candidates.sort(key=lambda i: (self.completed[i], -self.target_values[i], self._dist(self.drone_pos, self.target_pos[i])))
        i = candidates[0]
        if self.completed[i]:
            return self.cfg.completed_track_penalty

        self.track_progress[i] += 1
        reward = self.cfg.track_progress_bonus * float(self.target_values[i])
        if self.track_progress[i] >= self.cfg.track_required:
            self.completed[i] = True
            reward += self.cfg.complete_bonus * float(self.target_values[i])
        return reward

    def _obs(self) -> np.ndarray:
        g = self.cfg.grid_size
        drone = np.zeros((g, g), dtype=np.float32)
        drone[tuple(self.drone_pos)] = 1.0

        detected_value = np.zeros((g, g), dtype=np.float32)
        completed_map = np.zeros((g, g), dtype=np.float32)
        for i, p in enumerate(self.target_pos):
            if self.detected[i]:
                detected_value[tuple(p)] = self.target_values[i] / 2.0
            if self.completed[i]:
                completed_map[tuple(p)] = 1.0

        return np.stack([drone, self.belief, detected_value, completed_map, self.visited], axis=0).astype(np.float32)

    def _info(self) -> Dict:
        return {
            "t": self.t,
            "drone_pos": self.drone_pos.copy(),
            "detected": int(self.detected.sum()),
            "completed": int(self.completed.sum()),
            "target_values": self.target_values.copy(),
        }

    def _normalize_belief(self) -> None:
        s = float(self.belief.sum())
        if s > 1e-8:
            self.belief /= s

    def _cells_in_radius(self, center: np.ndarray, radius: int):
        g = self.cfg.grid_size
        cr, cc = int(center[0]), int(center[1])
        out = []
        for r in range(max(0, cr - radius), min(g, cr + radius + 1)):
            for c in range(max(0, cc - radius), min(g, cc + radius + 1)):
                if abs(r - cr) + abs(c - cc) <= radius:
                    out.append((r, c))
        return out

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> int:
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))
