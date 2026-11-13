"""Multi-objective COA search: NSGA-II with an optional ML surrogate pre-screen.

Objectives (all minimised):
    f0 = -mission success (0.5*E[value held] + 0.5*P[critical objective held])
    f1 =  E[blue combat power lost]
    f2 =  E[value-weighted time to secure]

The surrogate (see surrogate.py) lets us generate many more candidate COAs
per generation than we can afford to war-game, then send only the most
promising ones to the Monte Carlo simulator.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .coa import COA, HOLD
from .simulator import Outcome, WarGameSimulator
from .surrogate import Surrogate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- NSGA-II core
def non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Deb et al. (2002) fast non-dominated sort. Returns list of index arrays."""
    n = len(F)
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dominates = le & lt  # dominates[i, j]: i dominates j
    dom_count = dominates.sum(0)
    fronts, current = [], np.where(dom_count == 0)[0]
    while current.size:
        fronts.append(current)
        dom_count = dom_count - dominates[current].sum(0)
        dom_count[np.concatenate(fronts)] = -1
        current = np.where(dom_count == 0)[0]
    assert sum(len(f) for f in fronts) == n
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j])
        span = F[order[-1], j] - F[order[0], j]
        d[order[0]] = d[order[-1]] = np.inf
        if span > 0:
            d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return d


def rank_and_crowd(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rank, crowd = np.zeros(len(F), int), np.zeros(len(F))
    for r, front in enumerate(non_dominated_sort(F)):
        rank[front] = r
        crowd[front] = crowding_distance(F[front])
    return rank, crowd


def select_best(F: np.ndarray, k: int) -> np.ndarray:
    """Pick k indices by (rank asc, crowding desc)."""
    rank, crowd = rank_and_crowd(F)
    return np.lexsort((-crowd, rank))[:k]


def hypervolume_2d(value: np.ndarray, loss: np.ndarray, ref=(0.0, 1.0)) -> float:
    """Area dominated in (maximise value, minimise loss) space vs reference."""
    pts = sorted(zip(value, loss), key=lambda p: -p[0])
    hv, best_loss = 0.0, ref[1]
    for i, (v, lo) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else ref[0]
        best_loss = min(best_loss, lo)
        hv += max(v - nxt, 0) * max(ref[1] - best_loss, 0)
    return hv


# ---------------------------------------------------------------- driver
@dataclass
class OptimizerConfig:
    pop_size: int = 48
    generations: int = 30
    n_runs: int = 64  # Monte Carlo replications per evaluation
    max_delay: int = 4  # hours
    mutation_rate: float = 0.2
    use_surrogate: bool = True
    candidate_factor: int = 6  # candidates generated per evaluated child
    exploit_fraction: float = 0.5  # share of each batch chosen by the surrogate
    polish_starts: int = 3  # local-search starts after the GA (0 = off)
    polish_rounds: int = 4
    surrogate_warmup: int = 3  # generations before the surrogate is trusted
    seed: int = 42


@dataclass
class SearchResult:
    archive: dict[tuple, tuple[COA, Outcome]]
    pareto: list[tuple[COA, Outcome]]
    history: list[dict] = field(default_factory=list)
    surrogate_report: dict = field(default_factory=dict)


class COAOptimizer:
    def __init__(self, sim: WarGameSimulator, cfg: OptimizerConfig | None = None):
        self.sim = sim
        self.cfg = cfg or OptimizerConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.archive: dict[tuple, tuple[COA, Outcome]] = {}
        self.surrogate = Surrogate(sim, seed=self.cfg.seed)

    # -- helpers
    def _eval(self, coa: COA) -> Outcome:
        if coa.key() not in self.archive:
            # Common random numbers: same seed for every COA -> fairer comparison.
            self.archive[coa.key()] = (coa, self.sim.evaluate(coa, self.cfg.n_runs, seed=0))
        return self.archive[coa.key()][1]

    def _F(self, coas: list[COA]) -> np.ndarray:
        return np.array([self._eval(c).objectives() for c in coas])

    def _tournament(self, pop: list[COA], rank, crowd) -> COA:
        i, j = self.rng.integers(0, len(pop), 2)
        better = i if (rank[i], -crowd[i]) < (rank[j], -crowd[j]) else j
        return pop[better]

    def _offspring(self, pop: list[COA], F: np.ndarray, n: int) -> list[COA]:
        rank, crowd = rank_and_crowd(F)
        K, md = self.sim.n_obj, self.cfg.max_delay
        seen, out = set(self.archive), []
        tries = 0
        while len(out) < n and tries < n * 20:
            tries += 1
            child = self._tournament(pop, rank, crowd).crossover(
                self._tournament(pop, rank, crowd), self.rng)
            child = child.mutate(K, md, self.rng, self.cfg.mutation_rate)
            if child.key() not in seen:
                seen.add(child.key())
                out.append(child)
        return out

    # -- main loop
    def run(self) -> SearchResult:
        cfg, sim = self.cfg, self.sim
        pop: list[COA] = []
        while len(pop) < cfg.pop_size:
            c = COA.random(sim.n_blue, sim.n_obj, cfg.max_delay, self.rng)
            if c.key() not in {p.key() for p in pop}:
                pop.append(c)
        F = self._F(pop)
        history = [self._snapshot(0)]

        for gen in range(1, cfg.generations + 1):
            surrogate_on = cfg.use_surrogate and gen > cfg.surrogate_warmup
            if surrogate_on and (gen - cfg.surrogate_warmup - 1) % 5 == 0:
                self.surrogate.fit(*self._training_data())

            n_cand = cfg.pop_size * (cfg.candidate_factor if surrogate_on else 1)
            cands = self._offspring(pop, F, n_cand)
            if surrogate_on and len(cands) > cfg.pop_size:
                # Exploit/explore split: the surrogate picks part of the batch,
                # the rest is sampled at random so the search is not trapped
                # by the model's own biases.
                n_exploit = int(round(cfg.pop_size * cfg.exploit_fraction))
                pred = self.surrogate.predict(cands)
                chosen = list(select_best(pred, n_exploit))
                rest = np.setdiff1d(np.arange(len(cands)), chosen)
                chosen += list(self.rng.choice(rest, cfg.pop_size - n_exploit, replace=False))
                cands = [cands[i] for i in chosen]

            union = pop + cands
            UF = np.vstack([F, self._F(cands)]) if cands else F
            keep = select_best(UF, cfg.pop_size)
            pop, F = [union[i] for i in keep], UF[keep]
            history.append(self._snapshot(gen))
            log.info("gen %d  evals=%d  HV=%.3f", gen, len(self.archive), history[-1]["hypervolume"])

        if cfg.polish_starts:
            self.polish(cfg.polish_starts, cfg.polish_rounds)
            history.append(self._snapshot(cfg.generations + 1) | {"phase": "polish"})

        X, Y = self._training_data()
        report =self.surrogate.holdout_report(X, Y) if cfg.use_surrogate else {}
        return SearchResult(self.archive, self.pareto(), history, report)

    def neighbours(self, coa: COA) -> list[COA]:
        """All COAs that differ from `coa` by one unit's task."""
        out, K, md = [], self.sim.n_obj, self.cfg.max_delay
        for u in range(coa.n_units):
            t, a, d = list(coa.target), list(coa.approach), list(coa.delay)
            for k in [HOLD, *range(K)]:
                if k != t[u]:
                    out.append(COA(tuple(t[:u] + [k] + t[u + 1:]), coa.approach, coa.delay))
            out.append(COA(coa.target, tuple(a[:u] + [1 - a[u]] + a[u + 1:]), coa.delay))
            for dd in (d[u] - 1, d[u] + 1):
                if 0 <= dd <= md:
                    out.append(COA(coa.target, coa.approach, tuple(d[:u] + [dd] + d[u + 1:])))
        # Coordinated moves: re-task two units together onto the same objective.
        # Single moves can't find these because one unit alone loses the fight.
        for u in range(coa.n_units):
            for v in range(u + 1, coa.n_units):
                for k in range(K):
                    if coa.target[u] != k or coa.target[v] != k:
                        t = list(coa.target)
                        t[u] = t[v] = k
                        out.append(COA(tuple(t), coa.approach, coa.delay))
        return out

    def polish(self, n_starts: int, rounds: int) -> None:
        """Memetic step: steepest-ascent local search on mission success,
        started from the best Pareto COAs. GA finds the right region; local
        search squeezes out the last few points that random mutation misses."""
        starts = sorted(self.pareto(), key=lambda t: -t[1].mission)[:n_starts]
        for coa, out in starts:
            for _ in range(rounds):
                nbrs = self.neighbours(coa)
                F = self._F(nbrs)  # evaluated + archived (extends the Pareto front)
                i = int(np.argmin(F[:, 0]))
                if -F[i, 0] <= out.mission + 1e-3:
                    break
                coa, out = nbrs[i], self._eval(nbrs[i])

    def _training_data(self) -> tuple[list[COA], np.ndarray]:
        coas = [c for c, _ in self.archive.values()]
        return coas, np.array([o.objectives() for _, o in self.archive.values()])

    def pareto(self) -> list[tuple[COA, Outcome]]:
        items = list(self.archive.values())
        F = np.array([o.objectives() for _, o in items])
        return [items[i] for i in non_dominated_sort(F)[0]]

    def _snapshot(self, gen: int) -> dict:
        outs = [o for _, o in self.archive.values()]
        v = np.array([o.mission for o in outs])
        loss = np.array([o.loss for o in outs])
        return dict(generation=gen, evaluations=len(outs),
                    hypervolume=hypervolume_2d(v, loss),
                    best_mission=float(v.max()), min_loss_at_best=float(loss[v.argmax()]))
