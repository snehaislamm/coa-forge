"""ML surrogate: predicts war-game outcomes from engineered COA features.

The surrogate is a multi-output random forest. It is trained on every COA
the simulator has already adjudicated, and is used to pre-screen new
candidates. Feature importances double as a cheap sensitivity analysis:
which planning factors drive mission success and losses?
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .coa import COA, HOLD

if TYPE_CHECKING:
    from .simulator import WarGameSimulator

TARGET_NAMES = ["mission", "loss", "time"]


class Surrogate:
    def __init__(self, sim: WarGameSimulator, seed: int = 0):
        self.sim = sim
        self.seed = seed
        self.model: RandomForestRegressor | None = None
        objs = [o.oid for o in sim.sc.objectives]
        self.feature_names = (
            [f"{o}:strength" for o in objs] + [f"{o}:firepower" for o in objs]
            + [f"{o}:force_ratio" for o in objs] + [f"{o}:attacked" for o in objs]
            + [f"{o}:mean_arrival" for o in objs]
            + [f"{o}:arrival_spread" for o in objs] + [f"{o}:exposure" for o in objs]
            + ["held_in_reserve", "n_objectives_attacked"]
        )

    def features(self, coa: COA) -> np.ndarray:
        s, K = self.sim, self.sim.n_obj
        strength, fire, expo = np.zeros(K), np.zeros(K), np.zeros(K)
        arrivals: list[list[float]] = [[] for _ in range(K)]
        reserve = 0.0
        for u, k in enumerate(coa.target):
            if k == HOLD:
                reserve += s.blue_strength[u]
                continue
            m = coa.approach[u]
            strength[k] += s.blue_strength[u]
            fire[k] += s.blue_strength[u] * s.blue_fp[u]
            expo[k] += s.exposure[u, k, m] * s.blue_strength[u]
            arrivals[k].append(s.travel[u, k, m] + coa.delay[u])
        ratio = fire / np.maximum(s.red0 * s.red_fp, 1.0)
        mean_arr = np.array([np.mean(a) if a else s.p["t_max"] for a in arrivals])
        spread = np.array([np.ptp(a) if len(a) > 1 else 0.0 for a in arrivals])
        attacked = (strength > 0).astype(float)
        return np.concatenate([strength, fire, ratio, attacked, mean_arr, spread, expo,
                               [reserve, (strength > 0).sum()]])

    def fit(self, coas: list[COA], Y: np.ndarray) -> None:
        X = np.array([self.features(c) for c in coas])
        self.model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                           n_jobs=-1, random_state=self.seed)
        self.model.fit(X, Y)

    def predict(self, coas: list[COA]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Surrogate not fitted")
        return self.model.predict(np.array([self.features(c) for c in coas]))

    def holdout_report(self, coas: list[COA], Y: np.ndarray) -> dict:
        """Out-of-sample accuracy on a 25% hold-out + feature importances."""
        X = np.array([self.features(c) for c in coas])
        Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.25, random_state=self.seed)
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                  n_jobs=-1, random_state=self.seed).fit(Xtr, ytr)
        pred = m.predict(Xte)
        r2 = {name: float(r2_score(yte[:, i], pred[:, i]))
              for i, name in enumerate(TARGET_NAMES)}
        imp = sorted(zip(self.feature_names, m.feature_importances_),
                     key=lambda t: -t[1])
        return {"n_train": len(Xtr), "n_test": len(Xte), "r2": r2,
                "top_features": [(n, float(v)) for n, v in imp[:10]]}
