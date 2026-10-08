# Verified Step Self-Distillation

A Qwen3.5-9B experiment that learns from its own failed local-search trajectories using the same model with broader retrieval access.

**[Cycle-2 results: seeds, baselines, test set](self_evolve_search_cycle2_results.md)** · [Round-2 results](self_evolve_search_round2_results.md) · [Round-1 results](self_evolve_search_results.md) · [Aggregate metrics: cycle 2](results/cycle2_results.json), [round 2](results/round2_summary.json), [round 1](results/round1_summary.json) · [Protocol](docs/EXPERIMENT_PLAN.md) · [Round-1 diagnosis and upgrade plan](docs/UPGRADE_PLAN.md)

The core loop is:

```text
Failed trajectory -> suspect step -> replace one action -> replay and verify
                  -> action-only OPSD update -> collect with the updated model
```

The student searches up to two documents per query; the teacher/critic searches up to eight. They use the **same current model checkpoint**. A corrected branch continues from the saved pre-step state under the student's remaining budget. Only verified decision points contribute to training. The original model is retained as the evaluation baseline.

This adapts CSO-style verified step repair to same-model self-distillation. It is an experimental adaptation of OPSD, not a reproduction of the complete CSO or TrajDebug benchmarks. HotpotQA answer/support annotations provide verification signals; the teacher does not receive those labels directly.

**Cycle-2 result (28 September – 8 October 2026): the gain reproduces and transfers**

Cycle 2 re-ran the method with two more training seeds, ran the random-step control with the same three seeds, added three same-data baselines, continued the seed-42 run to twice the data, and opened the locked final benchmark (the 7,405 official HotpotQA dev questions, once per policy). Paired against the original model on that test set:

| Policy | Answer EM | Joint F1 | Grounded success | Budget failures |
| --- | ---: | ---: | ---: | ---: |
| Original model | 38.56% | 30.27% | 21.51% | 993 |
| **A. seed 42, update 32** | **45.82% (+7.27 [+6.23, +8.37])** | **38.56% (+8.30 [+7.41, +9.11])** | **37.49% (+15.98)** | 1,655 |
| A. seed 7, update 43 | 42.69% (+4.13 [+2.93, +5.36]) | 36.54% (+6.28 [+5.32, +7.21]) | 38.31% (+16.80) | 2,414 |
| A. seed 123, update 39 | 40.51% (+1.96 [+0.99, +2.93]) | 35.16% (+4.90 [+4.12, +5.64]) | 33.54% (+12.03) | 1,859 |

**The method improves joint F1 and grounded success on the held-out test set in all three seeds** (mean +6.5 joint F1, sd 1.7); the size of the gain depends on the seed. On the development set, same-data baselines do not reproduce it: supervised fine-tuning on the identical verified corrections +2.5 joint F1, DPO +0.8, rejection-sampling self-training −3.3, evidence-only guidance at a stable learning rate −1.3 (A − baseline contrasts +6.3 to +12.1, all intervals clear of zero). The random-step control reproduces it in one seed of three (+1.6 / collapsed / +8.3), so the diagnosed critical step is what makes the outcome reliable (3/3 positive, no collapse) rather than what makes it large. Continuing training to 5,000 questions gives no further gain (update 65 vs update 32: −1.2 joint F1 [−2.6, +0.3]). Trained policies exhaust the tool budget two to three times as often as the original model — the clearest remaining inefficiency. See the cycle-2 report for every arm, the paired contrasts (`results/contrasts/`, `scripts/contrast_evaluations.py`) and the caveats.

**Round-2 result (21–25 September 2026)**

Round 1 measured no clear gain. Its diagnosis (`docs/UPGRADE_PLAN.md`) found the guidance JSON in a dialect the model rarely emits (the gradient taught whitespace), corrections that named entities the student could not see, a teacher that preferred to finish when shown the evidence, and a BM25 top-2 retrieval ceiling. Round 2 addressed those, gated the learning rate on a memorization check, and ran the method with three controls on one RTX 5090. Arms A/D/C each completed 2,500 collection questions; B stopped after 89. All were evaluated on the same 1,500 development questions, paired against the original model. The development set was also used for tuning; the locked final benchmark was opened in cycle 2 (above).

| Arm | Updates | Answer EM | Joint F1 | Grounded success |
| --- | ---: | ---: | ---: | ---: |
| Original model | — | 51.33% | 36.13% | 24.47% |
| **A. verified critical-step OPSD (the method)** | 32 | **58.87% (+7.53 [+5.13, +10.07])** | **44.94% (+8.81 [+6.95, +10.72])** | **45.53% (+21.07 [+18.47, +23.60])** |
| D. same, random step instead of diagnosed | 23 | 54.87% (+3.53) | 37.76% (+1.64 [−0.44, +3.71]) | 44.27% (+19.80) |
| C. same repair procedure, DPO instead of OPSD | 22 | 52.67% (+1.33) | 36.93% (+0.81 [−0.62, +2.30]) | 33.53% (+9.07) |
| B. plain privileged-context OPSD (evidence-only guidance) | 20, stopped by tripwire | 12.40% (−38.93) | 7.06% (−29.06) | 0.00% (−24.47) |

**Arm A shows a clear development gain in this seed.** It exceeds random-step training (A − D: +7.2 joint F1 [+5.7, +8.6]) and DPO (A − C: +8.0 [+6.2, +9.7]). The evidence-only control collapses into answering without retrieving, but also removes diagnosis and verification, so it does not isolate guidance alone. On-policy records and update counts differ across arms. Single seed per arm in this round; cycle 2 (above) adds two seeds of A and D and the locked final benchmark. See the round-2 report for the gate, all metrics, contrasts, and limitations.

**Round-1 result**

Completed on one local RTX 5090: 5,000 collection questions, 198 verified repairs, 192 training records, and 24 OPSD updates. Evaluation uses the restricted student alone on a fixed 500-question internal-development set.

| Metric | Original model | After training |
| --- | ---: | ---: |
| Answer exact match | 52.80% | 53.00% |
| Joint F1 | 37.44% | 38.08% |
| Grounded success | 27.60% | 26.60% |

**Round 1 established no clear learning improvement.** Its joint-F1 change is +0.63 percentage points with a paired 95% interval of [-0.91, +2.14]. Round 2 and its controls are reported above. The 7,405-question final benchmark and additional training seeds are reported under cycle 2; the fixed repair benchmark remains pending. See the round-1 results report for all historical metrics and limitations.

**Setup**

Tested with Linux/WSL2, Python 3.12, and an NVIDIA GPU. The recorded run used a 32 GB RTX 5090. Training peaked at 22.77 GiB of allocated PyTorch memory on the observed short prefixes; full 8K training contexts were not validated.

Install `uv`, then create the three pinned environments from the repository root:

```bash
bash scripts/bootstrap.sh
cp config/local.example.json config/local.json
mkdir -p runs
```

The bootstrap accepts `uv` on PATH or at `.tools/uv`; set `SEARCH_PYTHON` if Python 3.12 has a different executable path. Download the recorded model revision to a local folder:

```bash
.venv-rollout/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    local_dir="models/Qwen3.5-9B",
)
PY
```

Edit `config/local.json` and set `model_path` to the **absolute Linux/WSL path** of that folder. Keep the index at `data/index/wiki.sqlite`. The actual local configuration is ignored by Git.

**Prepare HotpotQA and the local index**

```bash
source scripts/runtime_env.sh
.venv-data/bin/python scripts/download_data.py
.venv-data/bin/python scripts/prepare_splits.py
.venv-data/bin/python scripts/build_index.py \
  --archive data/raw/wiki-abstracts.tar.bz2 \
  --output data/index/wiki.sqlite
```

The downloader defaults to the recorded HotpotQA revision and verifies the corpus archive. Acquisition needs internet access. Agent execution uses a local SQLite index and a loopback vLLM endpoint; its application-level network guard is not an OS firewall. The corpus has 5,233,329 introductory documents, preserving supporting-sentence IDs.

**Run the pilot and training**

Start serving in one terminal:

```bash
bash scripts/serve.sh
```

After the server is ready, use another terminal in the same repository:

```bash
bash scripts/run_local.sh > runs/local-run.log 2>&1
```

This runs the 500-question repair pilot, stops the inference server, and performs one training compatibility update. It requires enough verified records to form the subsequent eight-record batches. After the compatibility check succeeds:

```bash
bash scripts/run_experiment.sh >> runs/round1-main.log 2>&1
```

For round 2, `bash scripts/run_cycle.sh` runs the whole cycle (tests, memorization gate, arms A/D/C/B) unattended with resume points; `scripts/cycle_status.py` reports it. Details in the round-2 report.

The round-1 runner alternates fresh repair collection and OPSD, restores optimizer state, saves checkpoints, and resumes completed questions. It stops at 5,000 questions or 200 updates and evaluates the final checkpoint. The original 500-question pilot counts toward the question limit. The compatibility checkpoint and first formal update are separate chains; they are not counted as two successive updates.

This code implements the recorded seed-42 protocol; some settings are fixed in scripts as well as `config/experiment.json`. Keep the two consistent when creating a new experiment. Do not modify frozen run settings while resuming. Details are in [the execution notes](docs/ROUND1_EXECUTION.md). Runtime progress is written to the ignored `RESULTS.md`; the published results snapshot is `self_evolve_search_results.md`.

**Checks and report generation**

After environment setup, copying the local configuration, and data acquisition:

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest tests -q
.venv-data/bin/python scripts/write_round1_report.py
```

Tests cover tool permissions, replay isolation, metric agreement, action-token alignment, detached teacher gradients, checkpoint continuity, and paired evaluation. The official-evaluator test needs `data/raw/hotpot_evaluate_v1.py` from the downloader; no model inference is performed by the tests.

The report generator can use the bundled aggregate results. `scripts/audit_round1_results.py` additionally requires the original local run artifacts. Model weights, raw datasets, trajectories, environments, and checkpoints are excluded from this publication.

**Code map**

- `src/self_evolve_search/`: agent, retrieval, verification, repair, and OPSD objective.
- `scripts/`: acquisition, rollout, training, evaluation, and reporting.
- `config/`: protocol settings and local configuration example.
- `tests/`: protocol and training-logic checks.
- `results/`: aggregate measurements and provenance (`round1_summary.json`, `round2_summary.json`).

**References**

[HotpotQA](https://hotpotqa.github.io/) · [CSO](https://arxiv.org/abs/2602.03412) · [OPSD](https://arxiv.org/abs/2601.18734) · [TrajDebug](https://arxiv.org/abs/2608.06346)
