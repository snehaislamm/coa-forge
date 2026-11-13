"""Scenario model: terrain, forces, objectives.

Everything here is fictional and abstract. Units carry notional combat-power
numbers, not real equipment data. The goal is to exercise planning logic,
not to model any real force.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

Cell = tuple[int, int]


@dataclass(frozen=True)
class Unit:
    """A manoeuvre unit with notional combat power."""

    uid: str
    side: str  # "blue" | "red"
    position: Cell
    strength: float  # notional combat power (0-100)
    firepower: float  # attrition coefficient multiplier
    speed: float  # cells per hour on cost-1 ground
    role: str = "manoeuvre"

    def __post_init__(self) -> None:
        if self.side not in {"blue", "red"}:
            raise ValueError(f"Unit {self.uid}: side must be 'blue' or 'red'")
        if self.strength <= 0 or self.speed <= 0:
            raise ValueError(f"Unit {self.uid}: strength and speed must be > 0")


@dataclass(frozen=True)
class Objective:
    oid: str
    name: str
    position: Cell
    value: float  # relative operational importance
    critical: bool = False


@dataclass
class Terrain:
    """Grid terrain with movement cost, cover and impassable water."""

    move_cost: np.ndarray  # >= 1, np.inf where impassable
    cover: np.ndarray  # 0..1, fraction of exposure removed
    elevation: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.move_cost.shape  # type: ignore[return-value]

    def passable(self, cell: Cell) -> bool:
        r, c = cell
        h, w = self.shape
        return 0 <= r < h and 0 <= c < w and np.isfinite(self.move_cost[r, c])

    @classmethod
    def generate(cls, height: int, width: int, seed: int,
                 water_level: float = 0.18, forest_density: float = 0.35) -> Terrain:
        """Procedural terrain from smoothed noise (reproducible by seed)."""
        rng = np.random.default_rng(seed)
        elev = gaussian_filter(rng.standard_normal((height, width)), sigma=width / 10)
        elev = (elev - elev.min()) / (np.ptp(elev) + 1e-9)

        forest = gaussian_filter(rng.standard_normal((height, width)), sigma=2.0)
        forest_mask = forest > np.quantile(forest, 1 - forest_density)

        gy, gx = np.gradient(elev * 10)
        slope = np.hypot(gx, gy)

        cost = 1.0 + 2.0 * slope + 0.6 * forest_mask
        cost[elev < water_level] = np.inf

        cover = np.clip(0.55 * forest_mask + 0.25 * np.clip(slope, 0, 1), 0, 0.85)
        return cls(move_cost=cost, cover=cover, elevation=elev)

    def clear_around(self, cells: Iterable[Cell], radius: int = 1) -> None:
        """Guarantee that key locations (units, objectives) are passable."""
        h, w = self.shape
        for r, c in cells:
            r0, r1 = max(0, r - radius), min(h, r + radius + 1)
            c0, c1 = max(0, c - radius), min(w, c + radius + 1)
            block = self.move_cost[r0:r1, c0:c1]
            block[~np.isfinite(block)] = 1.5


@dataclass
class Scenario:
    name: str
    description: str
    terrain: Terrain
    blue: list[Unit]
    red: list[Unit]
    red_reserve: list[Unit]
    objectives: list[Objective]
    params: dict = field(default_factory=dict)

    # ---- convenience -------------------------------------------------
    def objective_index(self, oid: str) -> int:
        return next(i for i, o in enumerate(self.objectives) if o.oid == oid)

    def defenders_at(self, obj: Objective, radius: int = 2) -> list[Unit]:
        return [u for u in self.red
                if abs(u.position[0] - obj.position[0]) <= radius
                and abs(u.position[1] - obj.position[1]) <= radius]

    def threat_map(self) -> np.ndarray:
        """Red influence map: sum of red firepower decaying with distance,
        reduced by local cover. Used for route risk and en-route attrition."""
        h, w = self.terrain.shape
        rr, cc = np.mgrid[0:h, 0:w]
        rng = float(self.params.get("red_weapon_range", 6.0))
        threat = np.zeros((h, w))
        for u in self.red:
            d = np.hypot(rr - u.position[0], cc - u.position[1])
            threat += u.firepower * u.strength / 100.0 * np.exp(-(d / rng) ** 2)
        return threat * (1.0 - self.terrain.cover)

    # ---- IO ------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict) -> Scenario:
        t = cfg["terrain"]
        terrain = Terrain.generate(t["height"], t["width"], t.get("seed", 0),
                                   t.get("water_level", 0.18),
                                   t.get("forest_density", 0.35))

        def mk(side: str, items: list[dict]) -> list[Unit]:
            return [Unit(uid=i["id"], side=side, position=tuple(i["pos"]),
                         strength=float(i["strength"]),
                         firepower=float(i.get("firepower", 1.0)),
                         speed=float(i.get("speed", 1.0)),
                         role=i.get("role", "manoeuvre")) for i in items]

        blue = mk("blue", cfg["blue"])
        red = mk("red", cfg["red"]["defenders"])
        reserve = mk("red", cfg["red"].get("reserve", []))
        objectives = [Objective(oid=o["id"], name=o.get("name", o["id"]),
                                position=tuple(o["pos"]), value=float(o["value"]),
                                critical=bool(o.get("critical", False)))
                      for o in cfg["objectives"]]
        terrain.clear_around([u.position for u in blue + red + reserve]
                             + [o.position for o in objectives])
        sc = cls(name=cfg["name"], description=cfg.get("description", ""),
                 terrain=terrain, blue=blue, red=red, red_reserve=reserve,
                 objectives=objectives, params=cfg.get("params", {}))
        sc.validate()
        return sc

    def validate(self) -> None:
        ids = [u.uid for u in self.blue + self.red + self.red_reserve]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate unit ids in scenario")
        if not self.objectives:
            raise ValueError("Scenario needs at least one objective")
        for thing in self.blue + self.red + self.red_reserve:
            if not self.terrain.passable(thing.position):
                raise ValueError(f"{thing.uid} placed on impassable terrain")
