**Self-Evolve Search — experiment and results**

Updated 18 September 2026. Round 1 finished on 17 September at 19:09 Singapore time: **5,000 questions, 24 OPSD updates, and 500 development evaluations per checkpoint**. Execution used one local RTX 5090 and training seed 42.

**Finding:** the pipeline works locally, but this run does not demonstrate clear self-evolution. Answer accuracy was nearly unchanged; joint F1 increased slightly and grounded success decreased. All reported 95% intervals include zero.

**What we tested.** Can Qwen3.5-9B improve its restricted search decisions using its own trajectory diagnoses, broader retrieval access, verified corrections, and OPSD?

| Role | Model | Access |
| --- | --- | --- |
| Original baseline | Untouched published Qwen3.5-9B | Top-2 documents per search |
| Student | Current Qwen3.5-9B checkpoint with LoRA adapter | Top-2 documents per search |
| Teacher / critic | **Same current checkpoint as student** | Top-8 retrieval; private evidence for diagnosis and scoring |

Teacher and student share the updated checkpoint after every batch. The teacher is not permanently frozen. The original baseline remains unchanged. Both roles use `search`, `read`, and `finish`, with **eight tool calls plus a final-answer action**. Reads require previously retrieved IDs. Top-2/top-8 controls retrieval width, not trajectory depth.

**Training cycle.**

1. Generate a student trajectory and a privileged investigation on the same question.
2. Diagnose a failed trajectory: rank up to three suspect steps and propose up to two legal replacements per step.
3. Replay the original and replacement from the same pre-step state under the remaining student budget. After deterministic screening, accept a correction only when it succeeds on at least 2/3 fresh paired replays and the original succeeds on at most 1/3.
4. Collect eight verified records and apply one OPSD update. Sample fresh student actions and match the same model’s detached, evidence-conditioned predictions on those action tokens. Repeat using the updated model.

Settings: LoRA rank 16, learning rate 5e-6, batch size 8, BF16 base weights, and full-vocabulary JSD loss with beta 0.5. The run stopped at the 5,000-question limit after 24 updates; its other limit was 200 updates. The 192 training records contained 6,819 sampled action tokens. The separate compatibility update is excluded from the formal update count.

HotpotQA answers and supporting facts verify replay outcomes; they are not directly provided to the teacher. This avoids a stronger external model while retaining dataset-based supervision and retrieved evidence.

**Relation to TrajDebug and CSO.** We retain the mechanism **failed trajectory → suspect step → replacement → verified replay → learning**. The tasks and evaluation targets differ:

| Approach | Task / benchmark | Main target |
| --- | --- | --- |
| [TrajDebug](https://arxiv.org/abs/2608.06346) | Long tool-use/coding trajectories; TrajErrBench from Tau2Bench and SWE-Bench Pro | Locate the critical error responsible for failure |
| [CSO](https://arxiv.org/abs/2602.03412) | GAIA-Text-103 and XBench-DeepSearch | Verify consequential action replacements and learn through DPO |
| Our experiment | HotpotQA through local document search | Learn better restricted actions through same-model repair and OPSD |

Here a “bug” is an agent decision error, such as a poor query, wrong document choice, or premature answer. Our training improves next-action behavior as its objective; it does **not explicitly train faulty-step-index prediction**. HotpotQA has no gold faulty-step annotations, so a successful repair does not identify the unique or earliest mistake. We have not established improved diagnosis accuracy or long-trajectory debugging ability.

**Data and benchmark coverage.** Retrieval uses 5.23 million local Wikipedia introductory documents with SQLite BM25 search. No web search or external model API is used during the experiment. Splits are fixed with seed 42 and checked for question overlap.

| Partition | Size | Status |
| --- | --- | --- |
| Round-1 collection | 5,000, including the 500-question pilot | Completed |
| Development monitor | 500 from a separate 1,500-question pool | Original, update 1, and update 24 evaluated |
| Locked repair benchmark | 500 | Not run |
| Reserved final task benchmark | 7,405 official fullwiki development questions | Not run |
| Round-2 collection | 5,000 disjoint questions | Not run |

**Final-checkpoint metrics.** These are results on the same **500-question internal-development set**, not the reserved final benchmark. Both baseline and trained student run alone with top-2 retrieval and no teacher or repair assistance. Scores are percentages; changes and intervals are percentage points.

| Metric | Original | Update 24 | Change | Paired 95% interval |
| --- | --- | --- | --- | --- |
| Answer exact match | 52.80 | 53.00 | +0.20 | [-2.00, +2.40] |
| Answer F1 | 61.626 | 61.621 | -0.004 | [-2.09, +2.05] |
| Supporting-fact F1 | 50.29 | 50.45 | +0.16 | [-1.45, +1.76] |
| **Joint F1 (primary)** | 37.44 | 38.08 | +0.63 | [-0.91, +2.14] |
| Grounded success | 27.60 | 26.60 | -1.00 | [-3.20, +1.00] |

Intervals use 2,000 paired question-bootstrap samples. Joint F1 combines answer and supporting-sentence citation quality. Grounded success additionally requires reading all annotated supporting sentences, valid nonempty observed citations, and successful completion. Uncertainty across training seeds is not measured.

Answer EM improved on **16 questions and regressed on 15**. Grounded success improved on **12 and regressed on 17**. Execution errors decreased from **67 to 53**, but this did not yield a clear correctness gain. In the separate training-side pilot, original-model accuracy was 54.8% with restricted retrieval and 59.0% with privileged retrieval; that is a permissions comparison, not a learned improvement.

**Repair results.**

- **198 verified repairs** from 5,000 questions: **3.96% yield**.
- Training-pool **Repair@3 = 198 / 3,732 failed trajectories = 5.31%**. This pools changing checkpoints; it is not a locked-benchmark score or localization accuracy.
- **192 records trained on**; six pilot repairs were discarded after the first update.
- Trained correction types: **182 search, 9 read, 1 final answer**. The signal was dominated by search corrections.
- Repair@1, a random-step control, and fixed failure-bank evaluation remain unmeasured.

**Cost and checks.** Collection used **86,181 model calls** and **37.97 hours** of recorded question processing, including diagnosis and replay. Training subprocesses took **41.99 minutes**, including loading and checkpoint I/O. All 24 checkpoints passed recorded reload checks; 17 tests passed. The report audit verified question completeness, training/evaluation separation, and the saved evaluation means.

**What remains.** Before claiming a learning benefit, run the fixed repair benchmark and matched-cost comparisons: ordinary OPSD versus verified-step OPSD, restricted versus privileged critics, and ranked versus random step selection. Additional training seeds, a frozen-teacher comparison, the reserved final benchmark, and a second evolution round are pending. The plan conditions round 2 on development improvement, which this run has not clearly established.

**Saved evidence.** Aggregate measurements are in [results/round1_summary.json](results/round1_summary.json); model, data, and publication provenance are in [results/provenance.json](results/provenance.json). Raw trajectories and model checkpoints are not included in this initial code release. The local final adapter is stored at `runs/round1-seed42/checkpoints/step-024`. Regenerate this report with `scripts/write_round1_report.py`. No additional experiment was run for this revision.
