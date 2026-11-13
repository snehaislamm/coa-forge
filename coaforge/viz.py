"""Figures for the decision brief (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

BG, FG, GRID = "#0f1419", "#d7dde4", "#2a323c"
UNIT_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ba68c8", "#f06292", "#fff176",
               "#4dd0e1", "#aed581"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "text.color": FG, "axes.titleweight": "bold",
    "font.size": 10, "legend.facecolor": "#161c23", "legend.edgecolor": GRID,
})


def _terrain(ax, sim):
    t = sim.sc.terrain
    water = ~np.isfinite(t.move_cost)
    ax.imshow(t.elevation, cmap="gray", alpha=0.55, origin="upper")
    ax.imshow(np.where(t.cover > 0.4, 1, np.nan), cmap=ListedColormap(["#2e5d34"]),
              alpha=0.6, origin="upper")
    ax.imshow(np.where(water, 1, np.nan), cmap=ListedColormap(["#1f4e79"]), origin="upper")
    ax.contour(sim.threat, levels=6, colors="#ef5350", linewidths=0.6, alpha=0.7)
    for o in sim.sc.objectives:
        ax.scatter(o.position[1], o.position[0], marker="*", s=260 if o.critical else 170,
                   c="#ffd54f", edgecolors="black", zorder=5)
        ax.annotate(o.oid.replace("OBJ_", ""), (o.position[1], o.position[0]),
                    xytext=(5, -10), textcoords="offset points", fontsize=8,
                    color="#ffd54f", weight="bold")
    for u in sim.sc.red + sim.sc.red_reserve:
        ax.scatter(u.position[1], u.position[0], marker="D", s=45, c="#ef5350",
                   edgecolors="black", zorder=5)
    ax.set_xticks([]), ax.set_yticks([])


def plot_coa_maps(sim, cands, path: Path) -> None:
    fig, axes = plt.subplots(1, len(cands), figsize=(6 * len(cands), 5.2))
    for ax, c in zip(np.atleast_1d(axes), cands):
        _terrain(ax, sim)
        for u, unit in enumerate(sim.sc.blue):
            col = UNIT_COLORS[u % len(UNIT_COLORS)]
            ax.scatter(unit.position[1], unit.position[0], marker="s", s=50, c=col,
                       edgecolors="black", zorder=6)
            route = sim.route(c.coa, u)
            if route is None:
                ax.annotate("R", (unit.position[1], unit.position[0]), xytext=(4, 4),
                            textcoords="offset points", color=col, fontsize=8)
                continue
            rc = np.array(route.cells)
            ls = "--" if c.coa.approach[u] else "-"
            ax.plot(rc[:, 1], rc[:, 0], ls, color=col, lw=2, label=unit.uid)
        ax.set_title(f"{c.label}\nmission {c.outcome.mission:.2f} · loss {c.outcome.loss:.2f}",
                     fontsize=10)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.012, "Solid = direct approach · dashed = covered approach · "
             "red contours = threat · ★ objective · ◆ red unit · R = held in reserve",
             ha="center", fontsize=9, color="#9aa5b1")
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pareto(result, cands, path: Path) -> None:
    outs = [o for _, o in result.archive.values()]
    par = [o for _, o in result.pareto]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter([o.loss for o in outs], [o.mission for o in outs], s=8, c="#4a5562",
               label=f"Evaluated COAs ({len(outs)})")
    sc = ax.scatter([o.loss for o in par], [o.mission for o in par], s=26,
                    c=[o.time for o in par], cmap="viridis_r", edgecolors="none",
                    label=f"Pareto-optimal ({len(par)})")
    fig.colorbar(sc, ax=ax, label="Tempo (value-weighted hrs, lower = faster)")
    for c, col in zip(cands, ["#ff7043", "#29b6f6", "#ffee58"]):
        ax.scatter(c.outcome.loss, c.outcome.mission, s=160, facecolors="none",
                   edgecolors=col, linewidths=2)
        ax.annotate(c.label.split(" - ")[0], (c.outcome.loss, c.outcome.mission),
                    xytext=(8, -4), textcoords="offset points", color=col, weight="bold")
    ax.set_xlabel("Expected blue combat power lost  →  worse")
    ax.set_ylabel("Mission success score  →  better")
    ax.set_title("Trade-off space explored by NSGA-II")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_convergence(histories: dict[str, list[list[dict]]], path: Path) -> None:
    """histories: method name -> list of runs (one history per seed)."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (name, runs), col in zip(histories.items(), ["#9aa5b1", "#4fc3f7"]):
        runs = [[x for x in h if x.get("phase") != "polish"] for h in runs]
        ev = np.array([x["evaluations"] for x in runs[0]])
        hv = np.array([[x["hypervolume"] for x in h] for h in runs])
        mu = hv.mean(0)
        ax.plot(ev, mu, "-", lw=2, color=col,
                label=f"{name} (mean of {len(runs)} seed{'s' * (len(runs) > 1)})")
        if len(runs) > 1:
            ax.fill_between(ev, mu - hv.std(0), mu + hv.std(0), color=col, alpha=0.2)
    ax.set_xlabel("War-game evaluations (simulation budget)")
    ax.set_ylabel("Hypervolume (mission × loss)")
    ax.set_title("Search convergence")
    ax.grid(color=GRID, lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_objective_probs(sim, cands, path: Path) -> None:
    K = sim.n_obj
    x = np.arange(K)
    w = 0.8 / len(cands)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (c, col) in enumerate(zip(cands, ["#ff7043", "#29b6f6", "#ffee58"])):
        ax.bar(x + i * w - 0.4 + w / 2, c.outcome.held_prob, w, color=col,
               label=c.label.split(" - ")[0])
    ax.set_xticks(x, [f"{o.oid.replace('OBJ_', '')}\n(value {o.value:g})"
                      for o in sim.sc.objectives])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(held at end of exercise)")
    ax.set_title("Where each COA wins and loses")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(report: dict, path: Path) -> None:
    feats = report["top_features"][:8][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([f for f, _ in feats], [v for _, v in feats], color="#4fc3f7")
    ax.set_xlabel("Random-forest feature importance")
    ax.set_title("What drives the outcome? (surrogate sensitivity analysis)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
