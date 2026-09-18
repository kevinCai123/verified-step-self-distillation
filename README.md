# Verified Step Self-Distillation

A Qwen3.5-9B experiment that learns from its own failed local-search trajectories using the same model with broader retrieval access.

**[Read the experiment and results](self_evolve_search_results.md)** · [Aggregate metrics](results/round1_summary.json) · [Protocol](docs/EXPERIMENT_PLAN.md)

The core loop is:

```text
Failed trajectory -> suspect step -> replace one action -> replay and verify
                  -> action-only OPSD update -> collect with the updated model
```

The student searches up to two documents per query; the teacher/critic searches up to eight. They use the **same current model checkpoint**. A corrected branch continues from the saved pre-step state under the student's remaining budget. Only verified decision points contribute to training. The original model is retained as the evaluation baseline.

This adapts CSO-style verified step repair to same-model self-distillation. It is an experimental adaptation of OPSD, not a reproduction of the complete CSO or TrajDebug benchmarks. HotpotQA answer/support annotations provide verification signals; the teacher does not receive those labels directly.

**Round-1 result**

Completed on one local RTX 5090: 5,000 collection questions, 198 verified repairs, 192 training records, and 24 OPSD updates. Evaluation uses the restricted student alone on a fixed 500-question internal-development set.

| Metric | Original model | After training |
| --- | ---: | ---: |
| Answer exact match | 52.80% | 53.00% |
| Joint F1 | 37.44% | 38.08% |
| Grounded success | 27.60% | 26.60% |

**No clear learning improvement is established.** The joint-F1 change is +0.63 percentage points with a paired 95% interval of [-0.91, +2.14]. The 7,405-question final benchmark, fixed repair benchmark, controls, additional training seeds, and round 2 remain pending. See the results report for all metrics and limitations.

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

The round-1 runner alternates fresh repair collection and OPSD, restores optimizer state, saves checkpoints, and resumes completed questions. It stops at 5,000 questions or 200 updates and evaluates the final checkpoint. The original 500-question pilot counts toward the question limit. The compatibility checkpoint and first formal update are separate chains; they are not counted as two successive updates.

This code implements the recorded seed-42 protocol; some settings are fixed in scripts as well as `config/experiment.json`. Keep the two consistent when creating a new experiment. Do not modify frozen run settings while resuming. Details are in [the execution notes](docs/ROUND1_EXECUTION.md). Runtime progress is written to the ignored `RESULTS.md`; the published results snapshot is `self_evolve_search_results.md`.

**Checks and report generation**

After environment setup, copying the local configuration, and data acquisition:

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest tests -q
.venv-data/bin/python scripts/write_round1_report.py
```

Tests cover tool permissions, replay isolation, metric agreement, action-token alignment, detached teacher gradients, checkpoint continuity, and paired evaluation. The official-evaluator test needs `data/raw/hotpot_evaluate_v1.py` from the downloader; no model inference is performed by the tests.

The report generator can use the bundled aggregate results. `scripts/audit_round1_results.py` additionally requires the original local run artifacts. Model weights, raw datasets, trajectories, environments, and checkpoints are excluded from this initial publication.

**Code map**

- `src/self_evolve_search/`: agent, retrieval, verification, repair, and OPSD objective.
- `scripts/`: acquisition, rollout, training, evaluation, and reporting.
- `config/`: protocol settings and local configuration example.
- `tests/`: protocol and training-logic checks.
- `results/`: aggregate measurements and provenance.

**References**

[HotpotQA](https://hotpotqa.github.io/) · [CSO](https://arxiv.org/abs/2602.03412) · [OPSD](https://arxiv.org/abs/2601.18734) · [TrajDebug](https://arxiv.org/abs/2608.06346)
