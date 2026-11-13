"""Course of Action (COA) representation.

A COA assigns every blue unit a task:
  * target    - index of the objective to attack, or HOLD (-1) to stay in reserve
  * approach  - DIRECT (fastest route) or COVERED (threat-avoiding route)
  * delay     - H-hour offset in hours (synchronises arrivals)

This compact genome is what the optimizer searches over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOLD = -1
DIRECT, COVERED = 0, 1
APPROACH_NAMES = {DIRECT: "Direct", COVERED: "Covered"}


@dataclass(frozen=True)
class COA:
    target: tuple[int, ...]
    approach: tuple[int, ...]
    delay: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.target) == len(self.approach) == len(self.delay)):
            raise ValueError("COA genes must have equal length")

    @property
    def n_units(self) -> int:
        return len(self.target)

    def key(self) -> tuple:
        return (self.target, self.approach, self.delay)

    def committed(self) -> list[int]:
        return [i for i, t in enumerate(self.target) if t != HOLD]

    # ---- genetic operators ---------------------------------------------
    @classmethod
    def random(cls, n_units: int, n_obj: int, max_delay: int,
               rng: np.random.Generator, hold_prob: float = 0.1) -> COA:
        target = np.where(rng.random(n_units) < hold_prob, HOLD,
                          rng.integers(0, n_obj, n_units))
        return cls(tuple(int(x) for x in target),
                   tuple(int(x) for x in rng.integers(0, 2, n_units)),
                   tuple(int(x) for x in rng.integers(0, max_delay + 1, n_units)))

    def crossover(self, other: COA, rng: np.random.Generator) -> COA:
        """Uniform crossover at the unit level (a unit's task stays intact)."""
        mask = rng.random(self.n_units) < 0.5
        pick = lambda a, b: tuple(x if m else y for x, y, m in zip(a, b, mask))
        return COA(pick(self.target, other.target),
                   pick(self.approach, other.approach),
                   pick(self.delay, other.delay))

    def mutate(self, n_obj: int, max_delay: int, rng: np.random.Generator,
               rate: float = 0.2) -> COA:
        t, a, d = list(self.target), list(self.approach), list(self.delay)
        for i in range(self.n_units):
            if rng.random() < rate:
                t[i] = HOLD if rng.random() < 0.1 else int(rng.integers(0, n_obj))
            if rng.random() < rate:
                a[i] = 1 - a[i]
            if rng.random() < rate:
                d[i] = int(np.clip(d[i] + rng.choice([-1, 1]), 0, max_delay))
        return COA(tuple(t), tuple(a), tuple(d))
