**Self-Evolve Search — round 2: fixed method and controls**

Updated 26 September 2026. Round 2 ran unattended from 21 September 21:59 to 25 September 20:37
Singapore time on the same local RTX 5090 (training seed 42): a memorization gate, then four arms of
2,500 collection questions each, each evaluated on the same 1,500-question internal-development set.
Round 1 is documented in [`self_evolve_search_results.md`](self_evolve_search_results.md); the diagnosis
that led to round 2 is in [`docs/UPGRADE_PLAN.md`](docs/UPGRADE_PLAN.md); aggregate numbers are in
[`results/round2_summary.json`](results/round2_summary.json).

**Finding.** With the round-1 confounds removed, the verified critical-step method improves the
restricted student by **+8.8 points joint F1 [+7.0, +10.7] and +21.1 points grounded success
[+18.5, +23.6]** over the original model (round 1: −0.01 [−0.47, +0.44]). Three matched controls each
remove one ingredient and each loses the gain: random step selection keeps the grounding gain but not
the answer gain; DPO on the same verified pairs learns little; evidence-only teacher guidance collapses
the policy into answering without retrieving. The gain therefore comes from the combination of
diagnosed critical steps, verified corrections, and distribution matching against a correction-guided
teacher, not from data volume or privileged context alone.

**What changed since round 1.** The recipe is the same — same model in all roles, same top-2/top-8
permissions, same diagnose → propose → replay → verify loop, same action-token JSD objective, LoRA
rank 16. Five things that were silently wrong were fixed:

| Problem in round 1 | Evidence | Fix |
| --- | --- | --- |
| Guidance JSON used spaced separators; the model emits compact JSON, so most of the JSD gradient taught whitespace | 0.04% of base replies spaced → 99.4% after training | Guidance in the model's dialect; loss restricted to content tokens (`opsd.content_token_mask`) |
| Corrections used entities the student could not have seen | 80% of search corrections; 25% contained the gold answer | Informational-legality filter (`repair.information_leak`); finish answers must be spans of read sentences |
| The same-model teacher, shown the retrieved evidence, prefers to `finish`; the student copies it | memorization gate: toward 11% / away 70% | Teacher sees the verified correction only (`repair.private_guidance`); premature-finish tripwire in every serving check |
| Student BM25 top-2 ranked an entity's stubs above its article | 34% of failed questions had both gold documents reachable | Exact/disambiguated title tiers before BM25 for both roles (`retrieval.Library`), permissions unchanged (52% reachable) |
| Too little content signal per update | 8 fresh records at 5e-6 | 2 fresh + 6 replayed records per update (reuse ≤ 4 within a 4-update window) at 2e-5, chosen by a memorization gate |

**Roles and evaluation** are as in round 1: original published Qwen3.5-9B as the unchanged baseline;
student = current checkpoint with adapter, top-2 search; teacher/critic = same current checkpoint,
top-8 retrieval; eight tool calls plus a final answer. Every arm evaluates the base model itself at
update 0 on the 1,500 development questions and pairs its final checkpoint against that, with 2,000
question-bootstrap samples (seed 42).

**Memorization gate** (`scripts/memorization_check.py`: 30 steps on the 192 round-1 records, then
greedy actions on the same prefixes; a run may start only if actions move toward the corrections):

| Guidance | Learning rate | Moved toward | Moved away | Verdict |
| --- | ---: | ---: | ---: | --- |
| retrieved evidence (round-1 style) | 2e-5 | 10.9% | 70.3% | fails; drifts to `finish` |
| retrieved evidence | 5e-5 | 26.0% | 49.0% | fails; 43% invalid JSON |
| verified correction only | 2e-5 | 44.8% | 24.5% | passes; action mix stable |
| verified correction only, 103 legal records | 2e-5 | 53.4% | 19.4% | passes |

**Arms.** All share model, data, question order, budget, learning rate, and evaluation; each control
changes exactly one ingredient of arm A.

| Arm | Step selection | Verification | Teacher guidance | Objective | Question |
| --- | --- | --- | --- | --- | --- |
| A — the method | model's diagnostic ranking | replay screen + 3-seed paired confirmation | verified correction | OPSD (JSD β 0.5) | — |
| D — random step | seeded random steps, same proposal and replay budget | as A | verified correction | OPSD | does the diagnosis matter? |
| C — DPO | as A | as A | correction (chosen) vs original (rejected) | DPO β 0.1, base reference | does the objective matter? |
| B — plain privileged context | seeded random step, no diagnosis | none | retrieved evidence | OPSD | is the repair machinery contributing over plain OPSD? |

**Development results.** Restricted student alone, 1,500 questions. Percentages; changes in
percentage points with paired 95% intervals. Base model: answer EM 51.33, answer F1 60.13,
supporting-fact F1 49.77, joint F1 36.13, grounded success 24.47; 166 questions exhausted the tool
budget.

| Arm | Updates | Verified records | Answer EM | Answer F1 | Supporting-fact F1 | Joint F1 | Grounded success | Budget exhausted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A. verified critical step** | 32 | 70 / 2,500 | **58.87 (+7.53 [+5.13, +10.07])** | **65.42 (+5.29 [+2.91, +7.70])** | 51.45 (+1.68 [−0.13, +3.34]) | **44.94 (+8.81 [+6.95, +10.72])** | **45.53 (+21.07 [+18.47, +23.60])** | 316 |
| D. random step | 23 | 53 / 2,500 | 54.87 (+3.53 [+0.80, +6.27]) | 60.13 (−0.00 [−2.67, +2.63]) | 43.28 (−6.49 [−8.51, −4.50]) | 37.76 (+1.64 [−0.44, +3.71]) | 44.27 (+19.80 [+17.00, +22.67]) | 396 |
| C. DPO | 22 | 50 / 2,500 | 52.67 (+1.33 [−0.67, +3.33]) | 61.12 (+0.99 [−0.92, +2.94]) | 50.00 (+0.23 [−1.38, +1.80]) | 36.93 (+0.81 [−0.62, +2.30]) | 33.53 (+9.07 [+6.73, +11.27]) | 46 |
| B. evidence only | 20 (tripwire) | 46 / 89 | 12.40 (−38.93 [−41.60, −36.13]) | 21.12 (−39.01) | 25.17 (−24.60) | 7.06 (−29.06 [−30.90, −27.29]) | 0.00 (−24.47) | 0 |

Paired contrasts between arms on the same questions (arm A minus the other arm):

| Contrast | Answer EM | Joint F1 | Supporting-fact F1 | Grounded success |
| --- | ---: | ---: | ---: | ---: |
| A − D (random step) | +4.00 [+2.13, +6.00] | **+7.18 [+5.67, +8.63]** | +8.18 [+6.61, +9.74] | +1.27 [−0.87, +3.33] |
| A − C (DPO) | +6.20 [+3.87, +8.33] | **+8.01 [+6.22, +9.66]** | +1.45 [−0.20, +3.02] | +12.00 [+9.67, +14.33] |
| A − B (evidence only) | +46.47 | +37.87 | +26.28 | +45.53 |

**Reading.**

- *Step selection carries the answer gain (D).* Verified corrections at random steps teach the policy
  to read before finishing as well as diagnosed ones do (grounded success within 1.3 points of A) but
  not what to search for or cite: supporting-fact F1 falls 6.5 points below base and joint F1 stays
  within noise of base. Random-step training also exhausts the tool budget most often (396 questions):
  the policy searches more without searching better.
- *The objective matters (C).* DPO on identical pairs learned the preferences early (margin 2.6,
  reward accuracy 1.0 at update 5) and then oscillated around zero margin from update 14 on. Its final
  policy is the most conservative (46 budget exhaustions) and gains only in grounding. Matching the
  guided teacher's full next-token distribution is a stronger signal than a pairwise preference on one
  sampled action.
- *Evidence-only guidance is harmful (B).* A teacher that sees the answer's evidence prefers to finish;
  the student copies it. Premature-finish share in the serving check rose 25% (update 1) → 75%
  (update 19) → 100% (update 20), where the tripwire stopped the run. Evaluated at that checkpoint
  (`scripts/evaluate_checkpoint.sh`), every one of the 1,500 development trajectories is a single
  `finish` with no retrieval. Round 1 used this guidance at 5e-6 and did not collapse only because most
  of its gradient went into whitespace.
- *The signal is small but sufficient.* Verified repairs are 2.8% of attempted questions in arm A
  (70 records, about 560 record-uses with replay). A few hundred well-chosen action distributions move
  the policy by 8.8 points — and a formatting confound could swamp that in round 1.
- *Where the headroom is.* Arm A exhausts the tool budget on 316 questions (base 166); part of its
  remaining failures are budget rather than judgement. Malformed-citation failures fell from 12 to 0.

**Limitations.** One training seed per arm. The development split also served to pick the learning
rate for the cycle; the 7,405-question final benchmark and the fixed repair benchmark are still
untouched. Self-evolution here still relies on dataset-based verification signals (replay against
gold answers) and on a privileged role of the same model. A parsing fix to `repair.diagnose` (skip
non-object entries in a malformed teacher reply) landed mid-run for arms A and D; it changes no
protocol decision, and each run's `source-changes.json` records the rebinding of its frozen hashes.

**Reproducing the cycle.** `scripts/run_cycle.sh` runs tests → memorization gate (writes the chosen
learning rate into the four arm configs) → arm A (`config/round1b.json`) → D
(`config/armD-random-step.json`) → C (`config/armC-dpo.json`) → B (`config/armB-evidence-opsd.json`,
`max_updates` matched to arm A), skipping finished steps and resuming each arm from its saved state.
`scripts/launch_cycle.cmd` starts it from Windows in a detached WSL window (edit its repository path
first); `runs/cycle/KILL` stops it and `runs/cycle/PAUSE` holds it between runs;
`scripts/cycle_status.py` prints progress, yields, losses, and development deltas.
`scripts/learnability_check.py` reproduces the retrieval-reachability analysis behind the title tiers.

**What remains.** A second seed of arms A and D; the locked final evaluation of arm A's step-32
adapter; the `repair_mode: unverified` ablation (diagnosis without replay confirmation; implemented,
no config); inspection of arm A's budget exhaustions; and the 200-update budget with a further round
under the updated teacher.
