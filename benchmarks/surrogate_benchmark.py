"""Does the ML surrogate actually help? Compare it with plain NSGA-II over
several seeds at equal simulation budgets.

    python benchmarks/surrogate_benchmark.py scenarios/northern_bastion.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coaforge import Scenario, WarGameSimulator, viz  # noqa: E402
from coaforge.optimizer import COAOptimizer, OptimizerConfig  # noqa: E402


def main(path: str, seeds=range(5), generations: int = 30,
         out: Path = Path("results")) -> str:
    sim = WarGameSimulator(Scenario.from_yaml(path))
    checkpoints = [10, 20, 30]
    histories: dict[str, list[list[dict]]] = {}
    best: dict[str, list[float]] = {}
    for name, use in [("NSGA-II", False), ("NSGA-II + surrogate", True)]:
        histories[name], best[name] = [], []
        for s in seeds:
            # Polish off: compare the global search methods on their own.
            r = COAOptimizer(sim, OptimizerConfig(generations=generations, seed=s,
                                                  use_surrogate=use, polish_starts=0)).run()
            histories[name].append(r.history)
            best[name].append(max(o.mission for _, o in r.pareto))

    (out / "figures").mkdir(parents=True, exist_ok=True)
    viz.plot_convergence(histories, out / "figures" / "benchmark.png")

    evals = {g: histories["NSGA-II"][0][g]["evaluations"] for g in checkpoints}
    lines = [f"Mean ± std over {len(list(seeds))} seeds. Hypervolume in (mission, loss) space; higher is better.", "",
             "| Method | " + " | ".join(f"HV @ {evals[g]} evals" for g in checkpoints)
             + " | Best mission score |",
             "|---|" + "---|" * (len(checkpoints) + 1)]
    for name, runs in histories.items():
        cells = []
        for g in checkpoints:
            v = [h[g]["hypervolume"] for h in runs]
            cells.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}")
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {np.mean(best[name]):.3f} ± {np.std(best[name]):.3f} |")
    table = "\n".join(lines)
    (out / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return table


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scenarios/northern_bastion.yaml")
