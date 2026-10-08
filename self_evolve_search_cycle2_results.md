**Self-Evolve Search — cycle 2: seeds, same-data baselines, a fair evidence-only control, a longer run, and the locked test set**

Updated 8 October 2026. Round 2 (the method and its first three controls, one seed) is documented in [`self_evolve_search_round2_results.md`](self_evolve_search_round2_results.md); round 1 in [`self_evolve_search_results.md`](self_evolve_search_results.md); the plan in [`docs/UPGRADE_PLAN.md`](docs/UPGRADE_PLAN.md).

## Cycle 2

Cycle 2 ran unattended from 28 September 22:02 to 8 October 08:22 Singapore time
(`scripts/run_cycle2.sh` via `scripts/launch_cycle2.cmd`; step log `runs/cycle2/cycle.log`). It asked
the questions a reviewer would ask of the round-2 result: does it survive a second and third training
seed; does it transfer to the locked test set; do cheaper objectives on the same data reproduce it; does
the evidence-only control still fail when it is given a learning rate at which it does not collapse; and
what happens with twice the data. All numbers below are in `results/cycle2_results.json` and
`results/cycle2_results.md`; arm-vs-arm contrasts are in `results/contrasts/` (made with
`scripts/contrast_evaluations.py`, the same paired question-bootstrap as `scripts/compare_evaluation.py`).

## The method on the locked test set, three seeds

The final benchmark is the 7,405 official HotpotQA development questions
(`data/splits/final_eval.jsonl`, `scripts/make_final_split.py`), evaluated once per policy by
`scripts/evaluate_policy.sh` and never used for tuning. Base model: EM 38.56, answer F1 51.10,
support F1 47.20, joint F1 30.27, grounded success 21.51, with 993 tool-budget failures.

| Arm A checkpoint | Joint F1 | Joint F1 Δ | EM Δ | Answer F1 Δ | Support F1 Δ | Grounded Δ | Budget failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed 42, update 32 | 38.56 | **+8.30** [+7.41, +9.11] | +7.27 [+6.23, +8.37] | +5.79 [+4.73, +6.87] | +1.91 [+1.12, +2.69] | +15.98 [+14.85, +17.08] | 1,655 |
| seed 7, update 43 | 36.54 | **+6.28** [+5.32, +7.21] | +4.13 [+2.93, +5.36] | +0.88 [−0.30, +2.08] | −1.86 [−2.72, −0.98] | +16.80 [+15.60, +17.97] | 2,414 |
| seed 123, update 39 | 35.16 | **+4.90** [+4.12, +5.64] | +1.96 [+0.99, +2.93] | +0.62 [−0.35, +1.61] | +1.85 [+1.06, +2.66] | +12.03 [+11.05, +13.05] | 1,859 |

Every seed improves joint F1 and grounded success on the test set with intervals well clear of zero
(mean joint F1 Δ +6.49, sd 1.71 across seeds; mean EM Δ +4.45, sd 2.67). The size of the gain depends
on the seed: seed 42 is the strongest on every metric, seeds 7 and 123 gain mostly through grounding and
joint F1 with small or null answer-F1 change, and seed 7 loses support F1. All three trained policies
exhaust the tool budget far more often than the base model (1,655–2,414 vs 993 of 7,405 questions),
which is the clearest remaining inefficiency of the method: it searches more, and sometimes searches
past the budget, rather than searching better on every question.

The development-set gains per seed are +8.81 [+6.95, +10.72], +6.23 [+4.01, +8.25] and +6.32 [+4.62, +8.00]
joint F1 (mean +7.12, sd 1.46), each with a premature-finish share of zero on the serving checks
throughout training; the development ordering of the seeds is the same as on the test set.

## Random-step selection: the same ingredient, less reliable

Arm D (verified teacher correction at a seeded random step instead of the diagnosed critical step,
everything else identical) was run with the same three seeds:

| Arm D | Updates | Joint F1 Δ vs base | Support F1 Δ | Budget failures | A − D (paired, joint F1) |
| --- | --- | --- | --- | --- | --- |
| seed 42 | 23 | +1.64 [−0.44, +3.71] | −6.49 | 396 | +7.18 [+5.67, +8.63] |
| seed 7 | 31 | −34.42 [−36.12, −32.59] (collapsed) | −47.34 | 1,424 | +40.66 |
| seed 123 | 21 | **+8.27** [+6.45, +10.14] | +4.13 | 36 | −1.96 [−3.56, −0.33] |

Seed 123 of the random-step control is as good as the best arm-A seed — better on EM (+8.80) and with
the fewest budget failures of any policy — while seed 42 is within noise of base and seed 7 collapsed
from update 25 into reading a fabricated document id (`doc_1`) on every question, a failure the
premature-finish tripwire does not catch. So the round-2 reading that "step selection carries the
answer gain" was a one-seed artefact. What the three seeds show instead is that the shared ingredient of
A and D — a verified, replay-screened teacher correction distilled by matching action-token
distributions — produces the gain, and that choosing the diagnosed critical step makes the outcome
reliable: arm A is positive on 3/3 seeds with no collapse (dev joint F1 +6.2 to +8.8), arm D is
positive on 1/3 (+1.6 / collapsed / +8.3; mean −8.2, or +5.0 excluding the collapse). The claim this supports is
about variance, not mean: critical-step selection is a stabiliser, and the controls that remove the
verified correction itself are the ones that lose the gain outright.

## Objectives and baselines on the same records (seed 42, development set)

| Arm | What it changes | Joint F1 Δ vs base | EM Δ | Grounded Δ | A − arm (joint F1) |
| --- | --- | --- | --- | --- | --- |
| S | SFT cross-entropy on arm A's verified corrections (same records, no distribution matching) | +2.52 [+1.01, +4.08] | −5.47 | +11.47 | +6.29 [+4.26, +8.35] |
| C | DPO (β 0.1) on the same pairs | +0.81 [−0.62, +2.30] | +1.33 | +9.07 | +8.01 [+6.22, +9.66] |
| R | rejection-sampling self-training: SFT on the student's own grounded-successful steps, no teacher | −3.28 [−4.69, −1.83] | −5.93 | +9.53 | +12.09 [+10.26, +14.10] |
| B′ | evidence-only teacher guidance, OPSD at 5e-6 (no collapse; premature-finish share stayed in the base range) | −1.27 [−2.25, −0.33] | −0.87 | −2.60 | +10.08 [+8.26, +11.94] |
| B | evidence-only guidance at 2e-5 (tripwire at update 20) | −29.06 | −38.93 | −24.47 | — |

Three things fall out. Supervised fine-tuning on the identical verified corrections recovers less
than a third of the joint-F1 gain and loses answer accuracy, so the gain is not "any training on these
records"; the full-vocabulary distribution match against the corrected teacher matters. Self-training
without a teacher (R) is harmful: imitating one's own successes reinforces grounding but costs answers.
And the fair-rate evidence-only control settles the question round 2 left open: given a learning rate
at which it trains stably for 32 updates, evidence-only guidance still yields no gain (a small loss),
so the correction content — not the privileged context — is what the teacher contributes.

## Longer training (seed 42, update 32 → 65, questions 2,500–5,000)

| Checkpoint | Joint F1 | Joint F1 Δ vs base | Grounded Δ | Budget failures |
| --- | --- | --- | --- | --- |
| update 32 (reported) | 44.94 | +8.81 [+6.95, +10.72] | +21.07 | 316 |
| update 48 | 41.32 | +5.19 [+3.58, +6.90] | +5.53 | 113 |
| update 64 | 43.45 | +7.33 [+5.67, +8.98] | +16.40 | 268 |
| update 65 (final) | 43.77 | +7.64 [+5.98, +9.33] | +16.73 | 281 |

Update 65 against update 32, paired: joint F1 −1.17 [−2.57, +0.34], EM 0.00, grounded −4.33
[−6.47, −2.27]. Doubling the data gives no further gain and the policy oscillates — it traded most of
its grounding for finishing within budget around update 48 and then recovered — so the update-32
checkpoint is the one to report, and the scaling claim is "the gain saturates by ~2,500 questions in
one round", not "more data helps". A second round with the updated teacher (Phase 3) remains the untested
route to more.

## Caveats that remain

Three seeds is enough to show that the effect reproduces and that its size varies by roughly ±2 points
joint F1; it is not enough for a tight estimate of the mean. The test-set gains are smaller than the
development gains for two of three seeds. Verification still depends on dataset labels (replay against
gold answers), and the base-model development evaluation was copied, not re-run, into each new run
(greedy decoding is deterministic; the copy and its hash rebinding are recorded in each run's
`baseline-evaluation-source.json`). The random-step collapse shows the serving check needs an
illegal-read tripwire in addition to the premature-finish one; it was added after the cycle
(`illegal_read_share` in `src/self_evolve_search/opsd.py`, used by `scripts/check_adapter_serving.py`) and was not
active in any reported run. `scripts/launch_cycle2.cmd` carries the Windows/WSL paths of the recorded machine as
`<repository root>` placeholders.

## Operational record

Cycle 2 needed six interventions, none losing training: a report-file race on the Windows-backed
mount (`reporting.py` made fully tolerant), three refusals of `compare_evaluation.py` on copied base
evaluations carrying the pre-fix protocol hash (rebound with provenance notes; `run_cycle2.sh` now
rebinds at copy time), a Windows-update reboot that killed the worker during the last final-benchmark
evaluation (resumed from question 7,227 of 7,405), and the continuation run's own hash refusal at
update 48. Each run resumed from its saved state; `runs/cycle2/cycle.log` is the step-level record.
