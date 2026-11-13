"""Threat-aware A* route planning."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .scenario import Cell

_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
               (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


@dataclass(frozen=True)
class Route:
    cells: tuple[Cell, ...]
    distance_cost: float  # terrain-weighted length (cost-units)
    exposure: float  # integral of threat along the route

    def travel_time(self, speed: float) -> float:
        return self.distance_cost / speed


class RoutePlanner:
    """A* over an 8-connected grid.

    Edge cost = step * move_cost * (1 + threat_weight * threat).
    threat_weight = 0 gives the fastest route; larger values trade time
    for concealment ("covered approach").
    """

    def __init__(self, move_cost: np.ndarray, threat: np.ndarray):
        if move_cost.shape != threat.shape:
            raise ValueError("move_cost and threat must have the same shape")
        self.move_cost = move_cost
        self.threat = threat
        self._cache: dict[tuple[Cell, Cell, float], Route] = {}

    def plan(self, start: Cell, goal: Cell, threat_weight: float = 0.0) -> Route:
        key = (start, goal, threat_weight)
        if key not in self._cache:
            self._cache[key] = self._astar(start, goal, threat_weight)
        return self._cache[key]

    def _astar(self, start: Cell, goal: Cell, tw: float) -> Route:
        h, w = self.move_cost.shape
        weighted = self.move_cost * (1.0 + tw * self.threat)
        g = {start: 0.0}
        parent: dict[Cell, Cell] = {}
        heap = [(self._h(start, goal), 0.0, start)]
        closed: set[Cell] = set()
        while heap:
            _, gc, cur = heapq.heappop(heap)
            if cur == goal:
                return self._build(parent, start, goal)
            if cur in closed:
                continue
            closed.add(cur)
            r, c = cur
            for dr, dc, step in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                cost = weighted[nr, nc]
                if not np.isfinite(cost):
                    continue
                ng = gc + step * cost
                nxt = (nr, nc)
                if ng < g.get(nxt, math.inf):
                    g[nxt] = ng
                    parent[nxt] = cur
                    heapq.heappush(heap, (ng + self._h(nxt, goal), ng, nxt))
        raise ValueError(f"No route from {start} to {goal}")

    @staticmethod
    def _h(a: Cell, b: Cell) -> float:
        # Octile distance; admissible because move_cost >= 1.
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

    def _build(self, parent: dict[Cell, Cell], start: Cell, goal: Cell) -> Route:
        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        dist = exposure = 0.0
        for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
            step = math.hypot(r1 - r0, c1 - c0)
            dist += step * self.move_cost[r1, c1]
            exposure += step * self.move_cost[r1, c1] * self.threat[r1, c1]
        return Route(cells=tuple(cells), distance_cost=dist, exposure=exposure)
