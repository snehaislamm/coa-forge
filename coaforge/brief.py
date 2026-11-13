"""Turn the Pareto front into a decision brief.

Picks three distinct COAs that a commander would actually want to compare,
re-adjudicates them with many more replications (and fresh random numbers,
so optimizer luck does not carry over), stress-tests them against a smarter
or dumber red reserve, and writes a COA comparison matrix.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .coa import APPROACH_NAMES, COA, HOLD
from .scenario import Scenario
from .simulator import Outcome, WarGameSimulator

WEIGHTS = {"Mission success": 0.5, "Force protection": 0.3, "Tempo": 0.2}


@dataclass
class Candidate:
    label: str
    intent: str
    coa: COA
    outcome: Outcome  # high-fidelity re-evaluation
    robustness: dict[float, Outcome]  # p_detect -> outcome
    score: float = 0.0


def pick_candidates(pareto: list[tuple[COA, Outcome]],
                    min_p_critical: float = 0.5) -> list[tuple[str, str, COA]]:
    # A COA that usually loses the critical objective is not a real option.
    viable = [p for p in pareto if p[1].p_critical >= min_p_critical]
    if len(viable) >= 3:
        pareto = viable
    v = np.array([o.mission for _, o in pareto])
    loss = np.array([o.loss for _, o in pareto])
    used: set[tuple] = set()

    def first_unused(order) -> COA:
        for i in order:
            if pareto[i][0].key() not in used:
                used.add(pareto[i][0].key())
                return pareto[i][0]
        raise ValueError("Pareto front too small")

    decisive = first_unused(np.lexsort((loss, -v)))
    ok = np.where(v >= 0.75 * v.max())[0]
    economy = first_unused(ok[np.argsort(loss[ok])])
    vn = (v - v.min()) / (np.ptp(v) + 1e-9)
    ln = (loss - loss.min()) / (np.ptp(loss) + 1e-9)
    balanced = first_unused(np.argsort(np.hypot(1 - vn, ln)))
    return [
        ("COA 1 - DECISIVE", "Maximise mission success; accept higher losses.", decisive),
        ("COA 2 - ECONOMY OF FORCE", "Keep >=75% of the best mission score at the lowest cost.", economy),
        ("COA 3 - BALANCED", "Knee of the value/loss trade-off curve.", balanced),
    ]


def evaluate_candidates(sc: Scenario, sim: WarGameSimulator,
                        pareto: list[tuple[COA, Outcome]],
                        n_runs: int = 2000, seed: int = 12345,
                        p_detect_levels=(0.2, 0.6, 0.95)) -> list[Candidate]:
    variants = {pd: WarGameSimulator(dataclasses.replace(sc, params={**sc.params, "p_detect": pd}))
                for pd in p_detect_levels}
    cands = []
    for label, intent, coa in pick_candidates(pareto):
        out = sim.evaluate(coa, n_runs, seed=seed)
        rob = {pd: s.evaluate(coa, n_runs // 2, seed=seed + 1) for pd, s in variants.items()}
        cands.append(Candidate(label, intent, coa, out, rob))
    _score(cands)
    return cands


def _score(cands: list[Candidate]) -> None:
    """Weighted decision matrix, each criterion min-max scaled across COAs."""
    crit = {
        "Mission success": np.array([c.outcome.mission for c in cands]),
        "Force protection": -np.array([c.outcome.loss for c in cands]),
        "Tempo": -np.array([c.outcome.time for c in cands]),
    }
    total = np.zeros(len(cands))
    for name, x in crit.items():
        total += WEIGHTS[name] * (x - x.min()) / (np.ptp(x) + 1e-9)
    for c, s in zip(cands, total):
        c.score = float(s)


def ci95(mean: float, std: float, n: int) -> str:
    return f"{mean:.2f} ± {1.96 * std / np.sqrt(n):.2f}"


def sync_matrix(sim: WarGameSimulator, coa: COA) -> list[dict]:
    rows = []
    for u, unit in enumerate(sim.sc.blue):
        k = coa.target[u]
        if k == HOLD:
            rows.append(dict(unit=unit.uid, task="Reserve (hold)", objective="-",
                             approach="-", h_hour="-", eta="-", exposure="-"))
            continue
        m = coa.approach[u]
        rows.append(dict(
            unit=unit.uid, task="Seize", objective=sim.sc.objectives[k].oid,
            approach=APPROACH_NAMES[m], h_hour=f"H+{coa.delay[u]}",
            eta=f"H+{sim.travel[u, k, m] + coa.delay[u]:.1f}",
            exposure=f"{sim.exposure[u, k, m]:.1f}"))
    return rows


def to_markdown(sc: Scenario, sim: WarGameSimulator, cands: list[Candidate],
                report: dict | None = None) -> str:
    best = max(cands, key=lambda c: c.score)
    n = cands[0].outcome.n_runs
    L = [f"# COA Decision Brief: {sc.name}", "",
         "_Fictional scenario. Generated automatically by COA-Forge._", "",
         "## Situation", sc.description.strip(), "",
         "## Recommendation",
         f"**{best.label}** (weighted score {best.score:.2f}). {best.intent}", "",
         "## COA comparison matrix",
         f"Each COA re-adjudicated over {n} Monte Carlo runs with fresh random numbers. "
         "Values are mean ± 95% CI.", "",
         "| | " + " | ".join(c.label for c in cands) + " |",
         "|---|" + "---|" * len(cands),
         "| **Mission success score** | " + " | ".join(f"**{c.outcome.mission:.2f}**" for c in cands) + " |",
         "| Objective value held | " + " | ".join(ci95(c.outcome.value, c.outcome.value_std, n) for c in cands) + " |",
         "| P(critical objective held) | " + " | ".join(f"{c.outcome.p_critical:.0%}" for c in cands) + " |",
         "| Blue combat power lost | " + " | ".join(ci95(c.outcome.loss, c.outcome.loss_std, n) for c in cands) + " |",
         "| Tempo (value-weighted hrs) | " + " | ".join(ci95(c.outcome.time, c.outcome.time_std, n) for c in cands) + " |",
         "| **Weighted score** | " + " | ".join(f"**{c.score:.2f}**" for c in cands) + " |",
         "", "Weights: " + ", ".join(f"{k} {v:.0%}" for k, v in WEIGHTS.items()), "",
         "## Objective-by-objective probability of holding at end-ex", "",
         "| Objective | Value | " + " | ".join(c.label for c in cands) + " |",
         "|---|---|" + "---|" * len(cands)]
    for k, o in enumerate(sc.objectives):
        L.append(f"| {o.oid} ({o.name}){' ★' if o.critical else ''} | {o.value:g} | "
                 + " | ".join(f"{c.outcome.held_prob[k]:.0%}" for c in cands) + " |")
    L += ["", "## War-game: sensitivity to red intelligence",
          "How each COA holds up if the red reserve reads blue's main effort "
          "poorly, as expected, or almost perfectly (P(detect)).", "",
          "| P(detect) | " + " | ".join(c.label for c in cands) + " |",
          "|---|" + "---|" * len(cands)]
    for pd in cands[0].robustness:
        L.append(f"| {pd:.2f} | " + " | ".join(
            f"mission {c.robustness[pd].mission:.2f} / loss {c.robustness[pd].loss:.2f}"
            for c in cands) + " |")
    for c in cands:
        L += ["", f"## {c.label}", f"_{c.intent}_", "",
              "| Unit | Task | Objective | Approach | Start | ETA | Route exposure |",
              "|---|---|---|---|---|---|---|"]
        for r in sync_matrix(sim, c.coa):
            L.append("| " + " | ".join(str(v) for v in r.values()) + " |")
    if report:
        L += ["", "## Surrogate model (ML pre-screen)",
              f"Random forest trained on {report['n_train']} war-gamed COAs; "
              f"hold-out R² on {report['n_test']}: "
              + ", ".join(f"{k} {v:.2f}" for k, v in report["r2"].items()) + ".", "",
              "Top drivers of outcome (feature importance):", ""]
        L += [f"{i + 1}. `{n}` ({v:.2f})" for i, (n, v) in enumerate(report["top_features"][:6])]
    return "\n".join(L) + "\n"
