# Cycle 1 + 2 results (22 September – 8 October 2026)

All deltas are paired against the base model (Qwen3.5-9B, greedy) on the same questions; 95% intervals are question-bootstrap (2,000 draws, seed 42). Development set: 1,500 internal questions. Final benchmark: the locked 7,405 official HotpotQA dev questions, opened once per policy.

## Development set: arms vs base

Joint F1 is the headline. Grounded success requires answer exact match, valid nonempty citations to
observed sentences, and exposure to every gold supporting fact. Runtime failures count every recorded
execution error, not only tool-budget exhaustion.

| Arm | Updates | Runtime failures (base 181) | Joint F1 Δ | EM Δ | Answer F1 Δ | Support F1 Δ | Grounded Δ |
|---|---|---|---|---|---|---|---|
| A (critical-step OPSD) seed 42 | 32 | 316 | +8.81 [+6.95, +10.72] | +7.53 [+5.13, +10.07] | +5.29 [+2.91, +7.70] | +1.68 [-0.13, +3.34] | +21.07 [+18.47, +23.60] |
| A seed 7 | 43 | 457 | +6.23 [+4.01, +8.25] | +3.33 [+0.60, +6.07] | -0.19 [-2.93, +2.42] | -1.94 [-3.99, -0.04] | +23.47 [+20.53, +26.07] |
| A seed 123 | 39 | 333 | +6.32 [+4.62, +8.00] | +2.53 [+0.20, +4.80] | +1.01 [-1.31, +3.24] | +2.94 [+1.18, +4.67] | +18.07 [+15.60, +20.60] |
| D (random-step OPSD) seed 42 | 23 | 396 | +1.64 [-0.44, +3.71] | +3.53 [+0.80, +6.27] | -0.00 [-2.67, +2.63] | -6.49 [-8.51, -4.50] | +19.80 [+17.00, +22.67] |
| D seed 7 | 31 | 1424 | -34.42 [-36.12, -32.59] | -49.07 [-51.60, -46.47] | -57.31 [-59.76, -54.90] | -47.34 [-49.11, -45.58] | -22.93 [-25.20, -20.60] |
| D seed 123 | 21 | 36 | +8.27 [+6.45, +10.14] | +8.80 [+6.47, +11.20] | +7.60 [+5.40, +9.91] | +4.13 [+2.49, +5.73] | +15.80 [+13.20, +18.27] |
| C (DPO) seed 42 | 22 | 49 | +0.81 [-0.62, +2.30] | +1.33 [-0.67, +3.33] | +0.99 [-0.92, +2.94] | +0.23 [-1.38, +1.80] | +9.07 [+6.73, +11.27] |
| B (evidence-only, 2e-5, tripwire at 20) seed 42 | 20 | 42 | -29.06 [-30.90, -27.29] | -38.93 [-41.60, -36.13] | -39.01 [-41.56, -36.53] | -24.60 [-26.91, -22.38] | -24.47 [-26.67, -22.27] |
| B' (evidence-only, 5e-6) seed 42 | 32 | 87 | -1.27 [-2.25, -0.33] | -0.87 [-2.20, +0.47] | -0.17 [-1.41, +1.05] | -0.54 [-1.60, +0.52] | -2.60 [-4.00, -1.27] |
| S (SFT on verified corrections) seed 42 | 31 | 534 | +2.52 [+1.01, +4.08] | -5.47 [-7.33, -3.53] | -7.74 [-9.61, -5.78] | -2.61 [-4.38, -0.87] | +11.47 [+9.27, +13.67] |
| R (rejection-sampling self-training) seed 42 | 32 | 502 | -3.28 [-4.69, -1.83] | -5.93 [-7.93, -3.93] | -8.05 [-10.00, -6.22] | -7.93 [-9.51, -6.37] | +9.53 [+7.20, +11.80] |

Arm A across three seeds (dev joint F1 Δ): mean +7.12, sd 1.46; every seed's interval excludes zero.
Arm D across three seeds: +1.64, -34.42, +8.27 (mean -8.17; excluding the collapsed seed 7, mean +4.96).

## Arm-vs-arm paired contrasts on the development set (updated − baseline; scripts/contrast_evaluations.py)

| Contrast | Joint F1 Δ | EM Δ | Support F1 Δ | Grounded Δ |
|---|---|---|---|---|
| A − D, seed 42 | +7.18 [+5.67, +8.63] | +4.00 [+2.13, +6.00] | +8.18 [+6.61, +9.74] | +1.27 [-0.87, +3.33] |
| A − D, seed 7 (D collapsed) | +40.66 [+38.60, +42.63] | +52.40 [+49.67, +54.80] | +45.41 [+43.34, +47.44] | +46.40 [+43.80, +48.87] |
| A − D, seed 123 | -1.96 [-3.56, -0.33] | -6.27 [-8.40, -4.20] | -1.19 [-2.59, +0.17] | +2.27 [+0.13, +4.40] |
| A − S (SFT), seed 42 | +6.29 [+4.26, +8.35] | +13.00 [+10.60, +15.47] | +4.29 [+2.23, +6.37] | +9.60 [+7.07, +12.13] |
| A − R (rejection sampling), seed 42 | +12.09 [+10.26, +14.10] | +13.47 [+10.93, +16.00] | +9.61 [+7.78, +11.49] | +11.53 [+8.80, +14.13] |
| A − C (DPO), seed 42 | +8.01 [+6.22, +9.66] | +6.20 [+3.87, +8.33] | +1.45 [-0.20, +3.02] | +12.00 [+9.67, +14.33] |
| A − B′ (evidence-only 5e-6), seed 42 | +10.08 [+8.26, +11.94] | +8.40 [+6.00, +10.87] | +2.22 [+0.52, +3.91] | +23.67 [+21.07, +26.13] |
| A continued to update 65 − A at update 32 (seed 42) | -1.17 [-2.57, +0.34] | +0.00 [-1.93, +1.87] | -0.94 [-2.35, +0.57] | -4.33 [-6.47, -2.27] |

## Arm A continued from update 32 to 5,000 questions (seed 42, development set)

| Checkpoint | Runtime failures | Joint F1 | Joint F1 Δ vs base | EM Δ | Support F1 Δ | Grounded Δ |
|---|---|---|---|---|---|---|
| step-032 | 316 | 44.94 | +8.81 [+6.95, +10.72] | +7.53 [+5.13, +10.07] | +1.68 [-0.13, +3.34] | +21.07 [+18.47, +23.60] |
| step-048 | 113 | 41.32 | +5.19 [+3.58, +6.90] | +8.53 [+6.33, +10.80] | -0.37 [-1.80, +1.09] | +5.53 [+3.07, +7.93] |
| step-064 | 268 | 43.45 | +7.33 [+5.67, +8.98] | +7.53 [+5.33, +9.73] | +0.68 [-0.73, +2.14] | +16.40 [+14.07, +18.67] |
| step-065 | 281 | 43.77 | +7.64 [+5.98, +9.33] | +7.53 [+5.27, +9.80] | +0.74 [-0.71, +2.23] | +16.73 [+14.27, +19.13] |

## Locked final benchmark (7,405 official HotpotQA dev questions, paired vs base)

Base model: EM 38.56, answer F1 51.10, support F1 47.20, joint F1 30.27, grounded 21.51; runtime failures 993.

| Policy | Runtime failures | Joint F1 | Joint F1 Δ | EM Δ | Answer F1 Δ | Support F1 Δ | Grounded Δ |
|---|---|---|---|---|---|---|---|
| A seed 42 (update 32) | 1655 | 38.56 | +8.30 [+7.41, +9.11] | +7.27 [+6.23, +8.37] | +5.79 [+4.73, +6.87] | +1.91 [+1.12, +2.69] | +15.98 [+14.85, +17.08] |
| A seed 7 (update 43) | 2414 | 36.54 | +6.28 [+5.32, +7.21] | +4.13 [+2.93, +5.36] | +0.88 [-0.30, +2.08] | -1.86 [-2.72, -0.98] | +16.80 [+15.60, +17.97] |
| A seed 123 (update 39) | 1859 | 35.16 | +4.90 [+4.12, +5.64] | +1.96 [+0.99, +2.93] | +0.62 [-0.35, +1.61] | +1.85 [+1.06, +2.66] | +12.03 [+11.05, +13.05] |

Arm A on the final benchmark across three seeds: joint F1 Δ mean +6.49 (sd 1.71); EM Δ mean +4.45 (sd 2.67).
