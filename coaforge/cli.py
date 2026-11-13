"""Command-line entry point.

    python -m coaforge scenarios/northern_bastion.yaml --out results/
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from . import viz
from .brief import evaluate_candidates, to_markdown
from .optimizer import COAOptimizer, OptimizerConfig
from .scenario import Scenario
from .simulator import WarGameSimulator

log = logging.getLogger("coaforge")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Automated COA generation and war-gaming")
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--runs", type=int, default=64, help="Monte Carlo runs per evaluation")
    ap.add_argument("--final-runs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-surrogate", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    t0 = time.perf_counter()
    sc = Scenario.from_yaml(a.scenario)
    sim = WarGameSimulator(sc)
    cfg = OptimizerConfig(pop_size=a.pop, generations=a.generations, n_runs=a.runs,
                          use_surrogate=not a.no_surrogate, seed=a.seed)
    result = COAOptimizer(sim, cfg).run()
    method = "NSGA-II + surrogate" if cfg.use_surrogate else "NSGA-II"

    cands = evaluate_candidates(sc, sim, result.pareto, n_runs=a.final_runs)

    out, figs = a.out, a.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    viz.plot_coa_maps(sim, cands, figs / "coa_maps.png")
    viz.plot_pareto(result, cands, figs / "pareto.png")
    viz.plot_convergence({method: [result.history]}, figs / "convergence.png")
    viz.plot_objective_probs(sim, cands, figs / "objective_probs.png")
    if result.surrogate_report:
        viz.plot_importance(result.surrogate_report, figs / "feature_importance.png")

    (out / "brief.md").write_text(to_markdown(sc, sim, cands, result.surrogate_report),
                                  encoding="utf-8")
    payload = {
        "scenario": sc.name,
        "config": asdict(cfg),
        "evaluations": len(result.archive),
        "runtime_s": round(time.perf_counter() - t0, 1),
        "history": result.history,
        "surrogate": result.surrogate_report,
        "selected": [{"label": c.label, "coa": asdict(c.coa), "outcome": asdict(c.outcome),
                      "score": c.score} for c in cands],
        "pareto": [{"coa": asdict(c), "outcome": asdict(o)} for c, o in result.pareto],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(result.archive)} COAs war-gamed, {len(result.pareto)} Pareto-optimal, "
          f"{payload['runtime_s']}s. Brief: {out / 'brief.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
