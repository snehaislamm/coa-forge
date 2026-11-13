"""Stochastic war-game adjudicator.

Each COA is played out N times (Monte Carlo) against a reactive red force.
All N replications run at once as numpy arrays, so a COA evaluates in
milliseconds.

Model, per objective k and replication n:
  * Movement: arrival = route_time * LogNormal(0, sigma_move) + H-hour delay
  * Approach attrition: strength *= exp(-alpha * route_exposure * noise)
  * Close combat: Lanchester square law, time-stepped
        dB/dt = -beta_r * defender_adv * R
        dR/dt = -beta_b * B
    A side breaks (withdraws) when it falls below a fraction of its peak.
  * Red reserve: commits after a reaction delay. With probability p_detect it
    correctly finds the objective with the most value at risk; otherwise it
    guesses at random.
    It can counter-attack an objective that blue has already taken.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coa import COA, HOLD
from .pathing import Route, RoutePlanner
from .scenario import Scenario


@dataclass(frozen=True)
class Outcome:
    """Summary statistics of a Monte Carlo evaluation."""

    value: float  # E[fraction of objective value held at end-ex]
    p_critical: float  # P(all critical objectives held)
    loss: float  # E[fraction of blue combat power lost]
    time: float  # E[value-weighted hours to secure objectives] (unheld = t_max)
    value_std: float
    loss_std: float
    time_std: float
    n_runs: int
    held_prob: tuple[float, ...]  # P(objective k held at end-ex)

    @property
    def mission(self) -> float:
        """Mission success score: commander's intent says the critical
        objective is a must-have, so it counts as much as all value held."""
        return 0.5 * self.value + 0.5 * self.p_critical

    def objectives(self) -> np.ndarray:
        """Minimisation vector for the optimizer."""
        return np.array([-self.mission, self.loss, self.time])


DEFAULTS = dict(
    dt=0.1, t_max=12.0, sigma_move=0.15, sigma_eff=0.25,
    exposure_alpha=0.02, beta=0.06, defender_advantage=1.6,
    blue_break=0.4, red_break=0.35, reserve_reaction=2.0, p_detect=0.6,
    covered_threat_weight=4.0, red_weapon_range=6.0,
)


class WarGameSimulator:
    def __init__(self, scenario: Scenario):
        self.sc = scenario
        self.p = {**DEFAULTS, **scenario.params}
        self.threat = scenario.threat_map()
        self.planner = RoutePlanner(scenario.terrain.move_cost, self.threat)
        self.n_obj = len(scenario.objectives)
        self.n_blue = len(scenario.blue)
        self._precompute()

    # ------------------------------------------------------------------
    def _precompute(self) -> None:
        sc, p = self.sc, self.p
        tw = {0: 0.0, 1: p["covered_threat_weight"]}
        self.routes: dict[tuple[int, int, int], Route] = {}
        self.travel = np.zeros((self.n_blue, self.n_obj, 2))
        self.exposure = np.zeros((self.n_blue, self.n_obj, 2))
        for u, unit in enumerate(sc.blue):
            for k, obj in enumerate(sc.objectives):
                for m in (0, 1):
                    r = self.planner.plan(unit.position, obj.position, tw[m])
                    self.routes[(u, k, m)] = r
                    self.travel[u, k, m] = r.travel_time(unit.speed)
                    self.exposure[u, k, m] = r.exposure

        self.red0 = np.zeros(self.n_obj)
        self.red_fp = np.ones(self.n_obj)
        for k, obj in enumerate(sc.objectives):
            d = sc.defenders_at(obj)
            s = sum(u.strength for u in d)
            self.red0[k] = s
            if s > 0:
                self.red_fp[k] = sum(u.strength * u.firepower for u in d) / s

        self.reserve_strength = sum(u.strength for u in sc.red_reserve)
        self.reserve_fp = (sum(u.strength * u.firepower for u in sc.red_reserve)
                           / self.reserve_strength) if sc.red_reserve else 1.0
        self.reserve_time = np.zeros(self.n_obj)
        if sc.red_reserve:
            lead = max(sc.red_reserve, key=lambda u: u.strength)
            for k, obj in enumerate(sc.objectives):
                self.reserve_time[k] = self.planner.plan(
                    lead.position, obj.position, 0.0).travel_time(lead.speed)

        self.blue_strength = np.array([u.strength for u in sc.blue])
        self.blue_fp = np.array([u.firepower for u in sc.blue])
        self.values = np.array([o.value for o in sc.objectives])
        self.critical = np.array([o.critical for o in sc.objectives])

    def route(self, coa: COA, unit_idx: int) -> Route | None:
        t = coa.target[unit_idx]
        return None if t == HOLD else self.routes[(unit_idx, t, coa.approach[unit_idx])]

    # ------------------------------------------------------------------
    def evaluate(self, coa: COA, n_runs: int = 64, seed: int | None = 0) -> Outcome:
        p, K = self.p, self.n_obj
        rng = np.random.default_rng(seed)
        dt, t_max = p["dt"], p["t_max"]
        n_steps = int(round(t_max / dt))

        blue_add = np.zeros((n_steps + 1, K, n_runs))
        blue_fp_num = np.zeros(K)
        blue_fp_den = np.zeros(K)
        initial_total = self.blue_strength.sum()
        held_back = 0.0

        for u in range(self.n_blue):
            k = coa.target[u]
            if k == HOLD:
                held_back += self.blue_strength[u]
                continue
            m = coa.approach[u]
            arr = (self.travel[u, k, m] * rng.lognormal(0, p["sigma_move"], n_runs)
                   + coa.delay[u])
            att = np.exp(-p["exposure_alpha"] * self.exposure[u, k, m]
                         * rng.lognormal(0, p["sigma_eff"], n_runs))
            step = np.minimum(np.ceil(arr / dt).astype(int), n_steps)
            np.add.at(blue_add, (step, k, np.arange(n_runs)),
                      self.blue_strength[u] * att)
            blue_fp_num[k] += self.blue_strength[u] * self.blue_fp[u]
            blue_fp_den[k] += self.blue_strength[u]
        blue_fp = np.where(blue_fp_den > 0, blue_fp_num / np.maximum(blue_fp_den, 1e-9), 1.0)

        # Red reserve decision
        red_add = np.zeros((n_steps + 1, K, n_runs))
        red_fp_reserve_at = np.zeros((K, n_runs), dtype=bool)
        if self.reserve_strength > 0:
            # Red's "most dangerous" read: reinforce the objective with the most
            # value at risk (value x committed blue/red force ratio).
            ratio = np.minimum(blue_fp_num / np.maximum(self.red0 * self.red_fp, 1.0), 3.0)
            at_risk = self.values * ratio
            main_effort = int(np.argmax(at_risk)) if at_risk.any() else 0
            guess = rng.integers(0, K, n_runs)
            choice = np.where(rng.random(n_runs) < p["p_detect"], main_effort, guess)
            arr = (p["reserve_reaction"] + self.reserve_time[choice]
                   * rng.lognormal(0, p["sigma_move"], n_runs))
            step = np.minimum(np.ceil(arr / dt).astype(int), n_steps)
            np.add.at(red_add, (step, choice, np.arange(n_runs)), self.reserve_strength)
            red_fp_reserve_at[choice, np.arange(n_runs)] = True

        # Per-replication combat effectiveness noise
        eff_b = rng.lognormal(0, p["sigma_eff"], (K, n_runs)) * blue_fp[:, None]
        eff_r = (rng.lognormal(0, p["sigma_eff"], (K, n_runs))
                 * np.where(red_fp_reserve_at,
                            (self.red_fp[:, None] + self.reserve_fp) / 2,
                            self.red_fp[:, None]))

        B = np.zeros((K, n_runs))
        R = np.repeat(self.red0[:, None], n_runs, axis=1).astype(float)
        B_peak = np.zeros_like(B)
        R_peak = R.copy()
        withdrawn = np.zeros(n_runs)
        held = np.zeros((K, n_runs), dtype=bool)
        t_secured = np.full((K, n_runs), np.nan)
        beta, adv = p["beta"], p["defender_advantage"]

        for s in range(n_steps + 1):
            B += blue_add[s]
            R += red_add[s]
            B_peak = np.maximum(B_peak, B)
            R_peak = np.maximum(R_peak, R)

            fighting = (B > 0) & (R > 0)
            dB = beta * adv * eff_r * R * dt
            dR = beta * eff_b * B * dt
            B = np.where(fighting, np.maximum(B - dB, 0), B)
            R = np.where(fighting, np.maximum(R - dR, 0), R)

            red_breaks = fighting & (R < p["red_break"] * R_peak)
            R = np.where(red_breaks, 0.0, R)
            R_peak = np.where(red_breaks, 0.0, R_peak)

            blue_breaks = fighting & ~red_breaks & (B < p["blue_break"] * B_peak)
            withdrawn += np.where(blue_breaks, B, 0).sum(axis=0)
            B = np.where(blue_breaks, 0.0, B)
            B_peak = np.where(blue_breaks, 0.0, B_peak)

            now_held = (B > 0) & (R <= 0)
            t_secured = np.where(now_held & ~held, s * dt, t_secured)
            held = now_held

        survivors = B.sum(axis=0) + withdrawn + held_back
        loss = 1.0 - survivors / initial_total
        value = (self.values[:, None] * held).sum(axis=0) / self.values.sum()
        # Tempo: value-weighted time to secure; unheld objectives count as t_max.
        secured_at = np.where(held, t_secured, t_max)
        completion = (self.values[:, None] * secured_at).sum(axis=0) / self.values.sum()
        crit = held[self.critical].all(axis=0) if self.critical.any() else held.all(axis=0)

        return Outcome(
            value=float(value.mean()), p_critical=float(crit.mean()),
            loss=float(loss.mean()), time=float(completion.mean()),
            value_std=float(value.std()), loss_std=float(loss.std()),
            time_std=float(completion.std()), n_runs=n_runs,
            held_prob=tuple(float(x) for x in held.mean(axis=1)),
        )
