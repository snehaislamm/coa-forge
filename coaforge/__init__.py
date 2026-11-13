"""COA-Forge: automated Course of Action generation and war-gaming."""
from . import viz  # noqa: F401
from .coa import COA, COVERED, DIRECT, HOLD
from .scenario import Objective, Scenario, Terrain, Unit
from .simulator import Outcome, WarGameSimulator

__all__ = ["COA", "COVERED", "DIRECT", "HOLD", "Objective", "Outcome",
           "Scenario", "Terrain", "Unit", "WarGameSimulator"]
__version__ = "0.1.0"
