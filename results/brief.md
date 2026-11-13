# COA Decision Brief: Exercise NORTHERN BASTION

_Fictional scenario. Generated automatically by COA-Forge._

## Situation
A blue task force must secure key terrain on a remote northern peninsula before an opposing force can reinforce it. Four objectives of unequal value are held by small defending detachments. A mobile red reserve sits in depth and will react once blue's main effort is detected.

## Recommendation
**COA 1 - DECISIVE** (weighted score 0.70). Maximise mission success; accept higher losses.

## COA comparison matrix
Each COA re-adjudicated over 2000 Monte Carlo runs with fresh random numbers. Values are mean ± 95% CI.

| | COA 1 - DECISIVE | COA 2 - ECONOMY OF FORCE | COA 3 - BALANCED |
|---|---|---|---|
| **Mission success score** | **0.79** | **0.69** | **0.73** |
| Objective value held | 0.63 ± 0.01 | 0.39 ± 0.00 | 0.48 ± 0.00 |
| P(critical objective held) | 95% | 98% | 98% |
| Blue combat power lost | 0.25 ± 0.00 | 0.17 ± 0.00 | 0.20 ± 0.00 |
| Tempo (value-weighted hrs) | 13.51 ± 0.04 | 14.11 ± 0.03 | 13.71 ± 0.03 |
| **Weighted score** | **0.70** | **0.30** | **0.53** |

Weights: Mission success 50%, Force protection 30%, Tempo 20%

## Objective-by-objective probability of holding at end-ex

| Objective | Value | COA 1 - DECISIVE | COA 2 - ECONOMY OF FORCE | COA 3 - BALANCED |
|---|---|---|---|---|
| OBJ_ANVIL (Airstrip) ★ | 4 | 95% | 98% | 98% |
| OBJ_HARBOR (Harbour) | 3 | 0% | 0% | 0% |
| OBJ_SPIRE (Relay site) | 2 | 78% | 0% | 0% |
| OBJ_FORD (River crossing) | 1 | 91% | 0% | 90% |

## War-game: sensitivity to red intelligence
How each COA holds up if the red reserve reads blue's main effort poorly, as expected, or almost perfectly (P(detect)).

| P(detect) | COA 1 - DECISIVE | COA 2 - ECONOMY OF FORCE | COA 3 - BALANCED |
|---|---|---|---|
| 0.20 | mission 0.78 / loss 0.24 | mission 0.69 / loss 0.14 | mission 0.73 / loss 0.18 |
| 0.60 | mission 0.78 / loss 0.26 | mission 0.68 / loss 0.17 | mission 0.73 / loss 0.20 |
| 0.95 | mission 0.77 / loss 0.27 | mission 0.68 / loss 0.19 | mission 0.73 / loss 0.21 |

## COA 1 - DECISIVE
_Maximise mission success; accept higher losses._

| Unit | Task | Objective | Approach | Start | ETA | Route exposure |
|---|---|---|---|---|---|---|
| B1-MECH | Seize | OBJ_ANVIL | Covered | H+0 | H+6.9 | 5.4 |
| B2-MECH | Seize | OBJ_ANVIL | Covered | H+0 | H+7.3 | 5.5 |
| B3-LIGHT | Seize | OBJ_SPIRE | Covered | H+0 | H+9.9 | 2.8 |
| B4-LIGHT | Seize | OBJ_FORD | Covered | H+0 | H+5.7 | 2.0 |
| B5-ARMD | Seize | OBJ_ANVIL | Covered | H+0 | H+6.7 | 5.5 |
| B6-RECCE | Seize | OBJ_SPIRE | Covered | H+1 | H+7.0 | 2.5 |

## COA 2 - ECONOMY OF FORCE
_Keep >=75% of the best mission score at the lowest cost._

| Unit | Task | Objective | Approach | Start | ETA | Route exposure |
|---|---|---|---|---|---|---|
| B1-MECH | Seize | OBJ_ANVIL | Covered | H+0 | H+6.9 | 5.4 |
| B2-MECH | Seize | OBJ_ANVIL | Covered | H+0 | H+7.3 | 5.5 |
| B3-LIGHT | Reserve (hold) | - | - | - | - | - |
| B4-LIGHT | Reserve (hold) | - | - | - | - | - |
| B5-ARMD | Seize | OBJ_ANVIL | Covered | H+0 | H+6.7 | 5.5 |
| B6-RECCE | Seize | OBJ_ANVIL | Covered | H+1 | H+6.4 | 5.0 |

## COA 3 - BALANCED
_Knee of the value/loss trade-off curve._

| Unit | Task | Objective | Approach | Start | ETA | Route exposure |
|---|---|---|---|---|---|---|
| B1-MECH | Seize | OBJ_ANVIL | Covered | H+0 | H+6.9 | 5.4 |
| B2-MECH | Seize | OBJ_ANVIL | Covered | H+0 | H+7.3 | 5.5 |
| B3-LIGHT | Reserve (hold) | - | - | - | - | - |
| B4-LIGHT | Seize | OBJ_FORD | Covered | H+0 | H+5.7 | 2.0 |
| B5-ARMD | Seize | OBJ_ANVIL | Covered | H+0 | H+6.7 | 5.5 |
| B6-RECCE | Seize | OBJ_ANVIL | Covered | H+1 | H+6.4 | 5.0 |

## Surrogate model (ML pre-screen)
Random forest trained on 1444 war-gamed COAs; hold-out R² on 482: mission 0.98, loss 0.90, time 0.92.

Top drivers of outcome (feature importance):

1. `OBJ_ANVIL:force_ratio` (0.28)
2. `OBJ_ANVIL:firepower` (0.25)
3. `OBJ_FORD:mean_arrival` (0.08)
4. `held_in_reserve` (0.07)
5. `OBJ_SPIRE:mean_arrival` (0.06)
6. `OBJ_ANVIL:mean_arrival` (0.05)
