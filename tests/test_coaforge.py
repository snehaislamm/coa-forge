"""Unit tests (stdlib unittest; also runs under pytest)."""
import math
import unittest
from pathlib import Path

import numpy as np

from coaforge import COA, HOLD, Scenario, WarGameSimulator
from coaforge.brief import pick_candidates
from coaforge.optimizer import (
    COAOptimizer,
    OptimizerConfig,
    crowding_distance,
    hypervolume_2d,
    non_dominated_sort,
)
from coaforge.pathing import RoutePlanner

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "northern_bastion.yaml"


def tiny_scenario(**params) -> Scenario:
    return Scenario.from_dict({
        "name": "tiny", "terrain": {"height": 15, "width": 15, "seed": 1, "water_level": 0.0},
        "objectives": [{"id": "A", "pos": [2, 2], "value": 2, "critical": True},
                       {"id": "B", "pos": [2, 12], "value": 1}],
        "blue": [{"id": "b1", "pos": [13, 4], "strength": 100, "speed": 5},
                 {"id": "b2", "pos": [13, 10], "strength": 100, "speed": 5}],
        "red": {"defenders": [{"id": "r1", "pos": [2, 3], "strength": 40},
                              {"id": "r2", "pos": [2, 11], "strength": 40}]},
        "params": {"t_max": 12.0, **params},
    })


class TestScenario(unittest.TestCase):
    def test_loads_and_validates(self):
        sc = Scenario.from_yaml(SCENARIO)
        self.assertEqual(len(sc.objectives), 4)
        for u in sc.blue + sc.red + sc.red_reserve:
            self.assertTrue(sc.terrain.passable(u.position))

    def test_rejects_duplicate_ids(self):
        cfg = {"name": "x", "terrain": {"height": 5, "width": 5, "water_level": 0},
               "objectives": [{"id": "A", "pos": [0, 0], "value": 1}],
               "blue": [{"id": "u", "pos": [4, 4], "strength": 1},
                        {"id": "u", "pos": [4, 3], "strength": 1}],
               "red": {"defenders": []}}
        with self.assertRaises(ValueError):
            Scenario.from_dict(cfg)

    def test_threat_peaks_near_red(self):
        sc = tiny_scenario()
        t = sc.threat_map()
        self.assertGreater(t[2, 3] + t[2, 2], t[13, 7])


class TestPathing(unittest.TestCase):
    def test_straight_line_on_flat_ground(self):
        rp = RoutePlanner(np.ones((10, 10)), np.zeros((10, 10)))
        r = rp.plan((0, 0), (0, 9))
        self.assertAlmostEqual(r.distance_cost, 9.0)
        self.assertEqual(r.cells[0], (0, 0))
        self.assertEqual(r.cells[-1], (0, 9))

    def test_routes_around_water(self):
        cost = np.ones((7, 7))
        cost[0:6, 3] = np.inf  # wall with a gap at the bottom row
        r = RoutePlanner(cost, np.zeros_like(cost)).plan((0, 0), (0, 6))
        self.assertTrue(all(np.isfinite(cost[c]) for c in r.cells))
        self.assertIn((6, 3), r.cells)

    def test_covered_route_trades_time_for_exposure(self):
        threat = np.zeros((11, 11))
        threat[3:8, 3:8] = 5.0
        rp = RoutePlanner(np.ones((11, 11)), threat)
        direct, covered = rp.plan((5, 0), (5, 10), 0.0), rp.plan((5, 0), (5, 10), 10.0)
        self.assertLess(covered.exposure, direct.exposure)
        self.assertGreaterEqual(covered.distance_cost, direct.distance_cost)

    def test_unreachable_raises(self):
        cost = np.ones((5, 5))
        cost[:, 2] = np.inf
        with self.assertRaises(ValueError):
            RoutePlanner(cost, np.zeros_like(cost)).plan((0, 0), (0, 4))


class TestCOA(unittest.TestCase):
    def test_operators_preserve_shape_and_bounds(self):
        rng = np.random.default_rng(0)
        a, b = COA.random(6, 4, 3, rng), COA.random(6, 4, 3, rng)
        for _ in range(50):
            c = a.crossover(b, rng).mutate(4, 3, rng, rate=0.5)
            self.assertEqual(c.n_units, 6)
            self.assertTrue(all(t == HOLD or 0 <= t < 4 for t in c.target))
            self.assertTrue(all(0 <= d <= 3 for d in c.delay))

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            COA((0, 1), (0,), (0, 0))


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = WarGameSimulator(tiny_scenario())

    def test_all_hold_means_no_loss_no_value(self):
        o = self.sim.evaluate(COA((HOLD, HOLD), (0, 0), (0, 0)), 32)
        self.assertEqual(o.value, 0.0)
        self.assertAlmostEqual(o.loss, 0.0)

    def test_concentration_beats_dispersion_on_critical(self):
        massed = self.sim.evaluate(COA((0, 0), (0, 0), (0, 0)), 256)
        split = self.sim.evaluate(COA((0, 1), (0, 0), (0, 0)), 256)
        self.assertGreaterEqual(massed.p_critical, split.p_critical)

    def test_deterministic_with_seed(self):
        c = COA((0, 1), (1, 0), (0, 1))
        self.assertEqual(self.sim.evaluate(c, 64, seed=3), self.sim.evaluate(c, 64, seed=3))

    def test_metrics_in_range(self):
        o = self.sim.evaluate(COA((0, 1), (0, 1), (2, 0)), 128)
        for x in (o.value, o.p_critical, o.loss, *o.held_prob):
            self.assertTrue(0.0 <= x <= 1.0)
        self.assertLessEqual(o.time, self.sim.p["t_max"] + 1e-9)

    def test_overwhelming_defence_holds(self):
        sim = WarGameSimulator(tiny_scenario(defender_advantage=50.0))
        o = sim.evaluate(COA((0, 0), (0, 0), (0, 0)), 64)
        self.assertEqual(o.held_prob[0], 0.0)


class TestNSGA(unittest.TestCase):
    def test_non_dominated_sort(self):
        F = np.array([[1, 1], [2, 2], [1, 2], [0, 3], [3, 0]])
        fronts = non_dominated_sort(F)
        self.assertEqual(set(fronts[0]), {0, 3, 4})
        self.assertEqual(set(fronts[1]), {2})
        self.assertEqual(set(fronts[2]), {1})

    def test_crowding_extremes_infinite(self):
        d = crowding_distance(np.array([[0, 3], [1, 2], [2, 1], [3, 0]], float))
        self.assertTrue(math.isinf(d[0]) and math.isinf(d[3]))
        self.assertTrue(np.isfinite(d[1:3]).all())

    def test_hypervolume(self):
        self.assertAlmostEqual(hypervolume_2d(np.array([1.0]), np.array([0.0])), 1.0)
        self.assertAlmostEqual(hypervolume_2d(np.array([1.0, 0.5]), np.array([0.5, 0.0])), 0.75)

    def test_optimizer_improves_on_random(self):
        sim = WarGameSimulator(tiny_scenario())
        r = COAOptimizer(sim, OptimizerConfig(pop_size=16, generations=6, n_runs=32,
                                              surrogate_warmup=2, polish_starts=1)).run()
        self.assertGreaterEqual(r.history[-1]["hypervolume"], r.history[0]["hypervolume"])
        self.assertTrue(r.pareto)
        self.assertEqual(len(pick_candidates(r.pareto)), 3)


if __name__ == "__main__":
    unittest.main()
