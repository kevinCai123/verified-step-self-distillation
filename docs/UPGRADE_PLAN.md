# Upgrade plan after round 1

21 September 2026. Written from the saved round-1 artifacts (`runs/round1-seed42`) without any new model calls.

## Why round 1 showed no improvement

Round 1 completed cleanly (5,000 questions, 198 verified repairs, 24 OPSD updates) and every
paired 95% interval on the 500-question development set included zero. The saved artifacts show
three compounding causes. None of them is an engineering failure; all three are in what the loss
was pointed at.

**1. The gradient mostly taught JSON whitespace.** `private_guidance()` serialised the verified
correction with `json.dumps` defaults, i.e. `{"action": "search", "query": ...}` with spaces,
while the model natively writes the compact dialect shown in `SCHEMA`. The guidance-primed teacher
therefore disagreed with the student at every colon and comma of the action, and the full-vocabulary
JSD pushed the student toward spaced JSON. Student samples during training went from 0/8 spaced at
update 1 to 8/8 from update 12 on; on the development set the base model emitted spaced JSON in
1 of 2,656 replies, the update-24 model in 2,640 of 2,640, and output tokens per question rose 16%.
The fall in mean JSD from 0.061 to 0.036 tracks this transition, not search quality.

**2. The corrections were oracle hints, not a learnable strategy.** Of the 182 search corrections
trained on, 146 (80%) contain a capitalised term or number that appears nowhere in the student's
context at that step, and 45 (25%) contain a term of the gold answer itself (`Lordi`, `Knebworth`,
`My Cousin Rachel`, `Nikolaus Harnoncourt`). The privileged teacher's advantage was knowing the
bridge entity from top-8 evidence and writing it into the query. Distilling that trains per-question
memorisation of facts the student cannot derive, which cannot transfer to disjoint questions.
Tool legality was enforced; informational legality was not.

**3. The remaining content signal was tiny.** 24 optimizer steps of 8 records, roughly 25 action
tokens each, with one to three tokens of real content disagreement per record: enough to flip a
feature present in every record (formatting), not enough to shift a policy. The step-24 adapter did
not reproduce the corrections even on its own training prompts. Learning rate 5e-6 is also low for
LoRA. Yield (4%) capped the run at 24 of the planned 200 updates.

Secondary losses in the repair funnel: 360 of 439 diagnosis failures were truncated JSON
(`max_tokens=512`); 852 failures (23%) were dropped as "ambiguous alias", which removed nearly all
`finish`-step repairs (1 of 192); and 75% of trajectories count as failed under grounded success even
though answer EM is 52%.

Development-set changes are consistent with a randomly perturbed policy: 359 of 500 trajectories
changed (216 at the first action), answer EM improved on 16 and regressed on 15.

## Phase 0 — fixes in this repository (done)

| File | Change |
| --- | --- |
| `src/self_evolve_search/repair.py` | Guidance and evidence lines use compact separators (`compact()`). New `information_leak()` rejects replacement queries that introduce capitalised terms or numbers not visible to the student and `finish` answers that are not spans of read sentences; `novel_terms()` records the strict view for audit. Diagnosis prompt states the rule and asks for strategy fixes; `max_tokens` 1024 with prompt budget 7168. |
| `src/self_evolve_search/opsd.py` | `value_character_mask()` / `content_token_mask()` label the value tokens of an action; `jsd_per_token()`; `jsd_loss(..., weights)` averages over content tokens only (falls back to all tokens if a sample has none). |
| `scripts/train_update.py` | `--loss-tokens content|all`, `--learning-rate`, `--batch-size`, `--allowed-policies` (replay window). Every sample logs content vs structural JSD, the structural share and the five most divergent tokens; the checkpoint status carries the means. |
| `scripts/run_experiment.py` | Run directory and configuration from `SELF_EVOLVE_RUN` / `SELF_EVOLVE_CONFIG`; reads learning rate, batch size, `fresh_records_per_update`, `replay_window_updates`, `max_record_reuse`, `loss_tokens`, evaluation file/size/interval, `bootstrap` and `start_cursor` from the configuration. Fresh collection lands in `batches/step-N/fresh/`; the composed training batch (`records.json`) adds records replayed from the last W batches within their reuse allowance (`replay-usage.json`). Defaults reproduce round-1 behaviour. |
| `src/self_evolve_search/persistence.py` | `validate_training_batch(..., allowed_policies)` accepts records from the replay window and still rejects anything stale. |
| `src/self_evolve_search/reporting.py`, `scripts/freeze_sources.py`, `scripts/check_adapter_serving.py` | Run-directory aware; the serving check probes the batch that was just trained. |
| `config/round1b.json` | Settings for the next run: lr 2e-5, content-token loss, 4 fresh + up to 4 replayed records per update (window 3, reuse 3), no pilot bootstrap, questions 0–1,499 of round 1, evaluation on all 1,500 internal-development questions every 25 updates. |
| `tests/` | Legality filter, compact guidance, value mask, weighted loss, replay-window composition; controller test adapted to the new layout. |
| `scripts/memorization_check.py` | Phase-1 learnability of the objective (below). |
| `scripts/learnability_check.py` | Phase-1 learnability of the task under student permissions (below); `--top-k` simulates a wider student, `--engine library` uses the implemented retrieval. |
| `src/self_evolve_search/retrieval.py` | **Title tiers** (added after the Phase-1 result below): `Library.search` returns articles whose title equals the query, then titles equal to the query plus a parenthesised disambiguator (shortest first), then the BM25 ranking; every tier is ordered independently of `top_k`, so the student's top-2 is still the prefix of the teacher's top-8. Case variants are looked up through the existing title index. `Library(path, title_tier=False)` reproduces round-1 retrieval. |
| `scripts/collect_batch.py`, `scripts/run_experiment.py` | `--repair-mode`, `--step-selection`, `--objective`, `--dpo-beta` threaded from the configuration file. |
| `src/self_evolve_search/repair.py` (arms) | `repair(..., mode, step_selection, seed)`: `verified` (the method), `unverified` (top-ranked legal proposal accepted without replay), `evidence-only` (no diagnosis; seeded random step with evidence-only guidance); `step_selection='random'` asks the diagnoser for replacements at seeded random steps. Records carry `verification`, `guidance_kind`, `repair_mode`. |
| `scripts/train_update.py` (arms) | `--objective dpo`: replacement over original action, reference = base model with the adapter disabled, logs margins and reward accuracy. |
| `config/armB-evidence-opsd.json`, `config/armC-dpo.json`, `config/armD-random-step.json` | Phase-2 arm settings; each arm runs in its own `SELF_EVOLVE_RUN` directory. |

Every new run re-evaluates its own step-000 baseline, so the retrieval change never mixes with round-1 numbers.

Reusing a record for a few updates is safe for OPSD because the on-policy part of the loss is the
freshly sampled action; the record only supplies the pre-step state and the verified guidance.

Run the tests before anything else:

```bash
.venv-data/bin/python -m pytest tests/test_protocol.py tests/test_continuation.py -q
.venv-train/bin/python -m pytest tests/test_opsd.py tests/test_tokenization.py -q
```

## Phase 1 — two cheap checks before new collection

**Learnability of the task — done, 21 September.** `scripts/learnability_check.py` takes a seeded
sample of failed round-1 trajectories and asks, with BM25 alone, whether the gold supporting
documents can be reached from student-visible terms: hop 1 from queries built from the question
(entity phrases, two-entity combination, the whole question), hop 2 from entity phrases in the gold
document reached in hop 1. Results on 250 of the 3,732 failures (seed 42; ±6 pp):

| Retrieval | Both gold documents reachable | Bridge | Comparison |
| --- | --- | --- | --- |
| BM25 top-2 (student permissions) | **34.0%** | 29.7% | 49.1% |
| BM25 top-5 | 50.0% | 44.6% | 69.1% |
| Exact/disambiguated title match first, BM25 fill, top-2 | **50.8%** | | |

| **Implemented `Library.search` with title tiers, top-2** (`--engine library`) | **52.4%** | 42.6% | 87.3% |

Files: `runs/learnability-check.json`, `runs/learnability-check-top5.json`,
`runs/learnability-check-titletier.json`, `runs/learnability-check-library.json` (+ per-question
`.progress.jsonl`). Against the round-1 retrieval the implemented tiers gain 46 questions and lose none. The candidate queries are
templated, so these are lower bounds (the student itself had retrieved both gold documents in 22% of
the sampled failures, and the templates reproduce that in 77% of those cases).

The cause is visible in the index: with `bm25(search,3,1)` a prominent entity's own stubs outrank its
article. `Cocteau Twins` returns the discography and three album pages before the band (rank 5);
`Philippine Airlines` does not surface the airline in the top 5; `Terrence Malick` is rank 2. No legal
query fixes that at top-2, which is exactly why round 1's teacher had to smuggle bridge entities into
the query. **Done:** `retrieval.Library.search` now has the title tiers for both roles with top-2 /
top-8 unchanged; with the real index `Cocteau Twins`, `Philippine Airlines`, `The Who` (previously an
empty result: every term is a stopword) and `Monster (2003 film)` all rank first. This makes "search
the name you just saw" a reliable, learnable move without changing the permission asymmetry. Widening
the student to top-5 would buy the same reachability but weaken the student/teacher contrast.

Among the sampled failures where the student had retrieved both gold documents (56), it read both in
40 and still answered correctly in only 11: those are answer-synthesis and citation failures, the
part of the funnel a `finish`-step repair could address if near-miss answers were not excluded.

Applying the new informational-legality filter retroactively to round 1's 192 trained records keeps
103 (54%): 89 search corrections leaked entities, 0 read/finish corrections did.

**Learnability of the objective.** With the inference server stopped:

```bash
source scripts/runtime_env.sh
HF_HUB_OFFLINE=1 .venv-train/bin/python scripts/memorization_check.py \
    --model-path "$(python3 -c 'import json;print(json.load(open("config/local.json"))["model_path"])')" \
    --output runs/memorization-check --steps 30 --learning-rate 2e-5 --loss-tokens content
```

It trains a fresh LoRA on the 192 existing records (guidance rewritten compactly) and reports whether
greedy actions on those prefixes move toward the verified replacements (`status.json`: `before`,
`after`, `moved_toward_replacement`, `verdict`). Add `--only-informationally-legal` to see how many
records survive the new filter and whether those alone are learnable. If nothing moves, raise the
learning rate or steps and repeat; do not collect new data until this passes.

## Phase 2 — controlled comparison at small scale

Same 2,500 questions (round-1 positions 0–2,499) for every arm, 2 fresh + 6 replayed records per
update (window 4, reuse 4), evaluation on the full 1,500-question internal-development set (paired
CI about ±1.2 pp) at updates 0, 1, every 50 and the end. Budget per arm from round-1 rates
(27 s per collected question, 7 s per evaluated question, 2 min per update): roughly 19 h of
collection, 6–9 h of evaluation and 2–4 h of training, so about a day per arm. With the legality
filter roughly halving round 1's 4% yield, 2,500 questions give in the order of 60 fresh records,
i.e. 25–30 updates per arm at 2 fresh per update; the first update needs 8 fresh records.

**Unattended driver:** `scripts/run_cycle.sh` runs tests → memorization gate (2e-5, then 5e-5 if
nothing moves; the chosen rate is written into the four arm configs) → arm A → D → C → B (arm B's
`max_updates` is matched to arm A's final count), skipping finished steps and resuming arms from
their own state. `scripts/cycle_status.py` prints progress, yields, last losses and dev-set deltas.

```bash
cd <repository root>
setsid nohup bash scripts/run_cycle.sh > runs/cycle-main.log 2>&1 &
.venv-data/bin/python scripts/cycle_status.py
```

A behavioural probe on the base model's frozen failures from the locked repair benchmark (query
length, entity focus, read-before-search) would be far more sensitive than end-task accuracy and is
still to be written.

| Arm | What it tests | Run |
| --- | --- | --- |
| A. Verified-step OPSD with informational legality | the method | `SELF_EVOLVE_RUN=runs/round1b-seed42 SELF_EVOLVE_CONFIG=config/round1b.json bash scripts/run_experiment.sh > runs/round1b-main.log 2>&1` |
| B. Ordinary privileged-context OPSD (no diagnosis, no verification; seeded random step, evidence-only guidance) | is the repair machinery contributing over plain OPSD | `SELF_EVOLVE_RUN=runs/armB-seed42 SELF_EVOLVE_CONFIG=config/armB-evidence-opsd.json …`; set `max_updates` to arm A's final update count, since nearly every failed trajectory yields a record |
| C. DPO on replacement-vs-original pairs (the CSO recipe) | denser gradient than JSD on a self-sample | `SELF_EVOLVE_RUN=runs/armC-seed42 SELF_EVOLVE_CONFIG=config/armC-dpo.json …` |
| D. Random-step repair, same proposal and replay budget | does the diagnostic ranking help | `SELF_EVOLVE_RUN=runs/armD-seed42 SELF_EVOLVE_CONFIG=config/armD-random-step.json …` |

A cheaper ablation of verification alone is `repair_mode: unverified` (diagnosis and proposals as in
arm A, top-ranked legal proposal accepted without replay); it is implemented but has no config file.

Parts that could not be exercised without a GPU or the model and should be watched on their first
update: the DPO branch of `train_update.py` (check `samples.json` margins are finite and
`reward_accuracy` is reported), the forced-step diagnosis prompt of arm D (check
`forced_steps` in the saved repair evidence match the steps the model answered for), and the
`disable_adapter()` reference pass. Everything else new is covered by the tests (21 torch-free, 4
with torch).

Watch `mean_structural_share_of_jsd` in each checkpoint's `status.json`; with the fixes it should be
small and stable. Watch `novel_terms` in the accepted records; the leak rate should be near zero.

## Results of the cycle (22–25 September)

**Memorization gate.** With the teacher shown the retrieved evidence (the round-1 guidance) the
30-step check failed at both learning rates: greedy actions moved toward the verified corrections on
11% of the 192 prefixes and away on 70% (2e-5), 26% / 49% (5e-5), and the policy drifted toward
answering `finish`. With correction-only guidance (`private_guidance` default) it passed at 2e-5:
toward 45%, away 24%, action-type mix unchanged (`runs/memorization-correction-2e-5`); on the 103
informationally legal records alone, toward 53%, away 19%. The four arms therefore trained at 2e-5
with correction-only guidance, content-token loss, 2 fresh + 6 replayed records per update, and the
premature-finish tripwire in the serving check.

**Internal development split, 1,500 questions, paired against the base model** (bootstrap 95%
intervals over questions, seed 42; `runs/<arm>/evaluations/step-*/comparison.json`):

| Arm | Updates | Verified records | Answer EM | Joint F1 | Support F1 | Grounded success | Budget exhausted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base model | — | — | 51.3% | 36.1% | 49.8% | 24.5% | 166 |
| **A. verified critical-step OPSD** | 32 | 70 / 2,500 | **58.9% (+7.5 [+5.1, +10.1])** | **44.9% (+8.8 [+7.0, +10.7])** | 51.5% (+1.7 [−0.1, +3.3]) | **45.5% (+21.1 [+18.5, +23.6])** | 316 |
| D. random-step OPSD (same budget) | 23 | 53 / 2,500 | 54.9% (+3.5 [+0.8, +6.3]) | 37.8% (+1.6 [−0.4, +3.7]) | 43.3% (−6.5 [−8.5, −4.5]) | 44.3% (+19.8 [+17.0, +22.7]) | 396 |
| C. DPO on the same verified pairs | 22 | 50 / 2,500 | 52.7% (+1.3 [−0.7, +3.3]) | 36.9% (+0.8 [−0.6, +2.3]) | 50.0% (+0.2 [−1.4, +1.8]) | 33.5% (+9.1 [+6.7, +11.3]) | 46 |
| B. evidence-only privileged-context OPSD | 20 (stopped by tripwire) | 46 / 89 | 12.4% (−38.9) | 7.1% (−29.1) | 25.2% (−24.6) | 0.0% (−24.5) | 0 |

Paired contrasts between arms on the same questions (arm A minus the other arm, joint F1):
A − D **+7.2 pp [+5.7, +8.6]**, A − C **+8.0 pp [+6.2, +9.7]**, A − B +37.9 pp. On grounded success
A − D is +1.3 pp [−0.9, +3.3] — the two arms reach the evidence equally often — while on support F1
A − D is +8.2 pp [+6.6, +9.7] and on answer EM +4.0 pp [+2.1, +6.0].

Reading:

- **The method works once the confounds are removed.** Arm A is the same recipe as round 1 (verified
  critical-step repair, action-token full-vocabulary JSD, LoRA r=16) and moves joint F1 by +8.8 pp
  where round 1 moved it by −0.01 pp [−0.47, +0.44]. Nothing in the objective changed; what changed
  is that the guidance is in the model's own JSON dialect, corrections may not use entities the
  student cannot see, the teacher no longer sees evidence, the retrieval has title tiers, and the
  learning rate is the one the memorization gate accepted.
- **The diagnostic step selection carries the answer-quality gain (arm D).** Verified corrections at
  seeded random steps teach "read before you finish" just as well (grounded success +19.8 vs +21.1)
  but not *what* to search for or cite: support F1 falls 6.5 pp and joint F1 stays within noise of
  base. Part of D's support loss is budget exhaustion (396 vs 316 questions), so random-step training
  makes the policy search more without searching better.
- **The distribution-matching objective matters (arm C).** DPO on identical data learned the
  preferences early (margin 2.6, accuracy 1.0 at update 5) and then oscillated around zero margin
  (updates 14–22); the final policy is the most conservative of all (46 budget exhaustions) and gains
  only in grounded success. Dense pairwise preference on a single sampled action is a weaker signal
  than matching the guided teacher's full next-token distribution.
- **Showing the teacher the evidence is actively harmful (arm B).** With evidence-only guidance the
  same-model teacher prefers to finish, and the student copies that: premature-finish share in the
  serving check rose from 25% (update 1) to 75% (update 19) and 100% (update 20), at which point the
  tripwire stopped the run. `scripts/evaluate_checkpoint.sh` evaluated the update-20 adapter: every
  one of the 1,500 dev trajectories is a single `finish` with no retrieval (answer EM 12.4%, grounded
  success 0%). Round 1 used this guidance at 5e-6 and did not collapse only because most of its
  gradient went into JSON whitespace. This is the failure mode the gate and the tripwire exist for.
- **Cost of the signal.** Verified repairs are rare: 2.8% of attempted questions in arm A (70 records,
  ~560 record-uses with replay). The gain comes from a few hundred well-chosen action distributions,
  not from data volume, which is why the round-1 formatting confound could swamp it.

Caveats: one seed per arm; the development split is the same 1,500 questions used to choose the
learning rate for the whole cycle (the final evaluation in `docs/EXPERIMENT_PLAN.md` is still
untouched); arm A exhausts the tool budget on twice as many questions as the base model (316 vs
166) and the remaining headroom is partly there; the `repair.py` diagnosis-parsing fix landed mid-run
for arms A and D (recorded in each run's `source-changes.json`; it touches no protocol decision,
only the handling of a malformed teacher reply), so their protocol hashes were rebound to compare
with their own step-0 baselines.

Operational notes from the cycle: the unattended driver was interrupted five times (a WSL session
ending, a malformed diagnosis reply, two comparison refusals after the hash change, a report-file
race on the Windows mount, and a user pause), and every arm resumed from its saved state with no
lost updates; `scripts/launch_cycle.cmd` (Windows) with the `runs/cycle/KILL` and `runs/cycle/PAUSE`
flags is the supported way to stop and resume.

## Phase 3 — scale the winner

Only the arm whose effect exceeds its interval proceeds to the 200-update budget and to round 2 with
the updated teacher. The development gate in `docs/EXPERIMENT_PLAN.md` is unchanged.
