**Verified critical-step self-evolution: Qwen3.5-9B, HotpotQA, and OPSD**

15 September 2026 · Approved protocol snapshot; see self_evolve_search_results.md for completed results

Proposed project: `verified-step-self-distillation`

**Main idea:** the model generates a failed trajectory, uses broader retrieval permissions to identify and correct a consequential action, verifies the correction by replaying with restricted tools, and learns specifically at that decision point through OPSD. The updated model repeats the process on new questions.

This preserves CSO's verified step-repair mechanism. The same Qwen3.5-9B supplies the diagnosis and correction, and OPSD supplies the learning objective. CSO originally combines critical-step verification with a process reward model, expert alternatives, and preference learning. [CSO paper](https://arxiv.org/abs/2602.03412).

**1. Fixed setup**

| Item | Choice |
|---|---|
| Model | `Qwen/Qwen3.5-9B`, post-trained checkpoint, text-only |
| Task | Multi-hop question answering through local document search |
| Question source | HotpotQA |
| Actual training data | Self-generated failed trajectories and replay-verified step corrections |
| Post-training | OPSD applied at verified decision points, with LoRA |
| Additional signals | Retrieved documents and deterministic checks against dataset annotations |

The student searches top 2 results per query; the privileged role searches top 8. Both access the same fixed Wikipedia corpus with `search`, `read`, and `finish`. Reads require previously retrieved IDs. Each investigation/trajectory permits 8 tool calls, 8K context, and 2,048 generated tokens, capped at 512 per response. Keep thinking disabled. The backend enforces permissions.

The privileged role sees the failed trajectory and its own retrieved evidence. It does not receive gold answers or supporting-fact labels. A separate verifier uses those labels. No stronger teacher or separate learned reward model is required.

**2. Change the dataset into a trajectory-repair dataset**

Keep HotpotQA as the task source: it provides answers and sentence-level supporting facts for verification. These annotations do **not** identify faulty actions in a model-generated trajectory; we must construct those records ourselves. [HotpotQA](https://hotpotqa.github.io/).

| Partition | Planned content |
|---|---|
| Local pilot | 500 fixed training questions for building and checking the repair pipeline |
| Round-1 training | 5,000 training questions; learn from the verified repair records they produce |
| Internal development | 1,500 separate training questions; use a fixed 500 for frequent checks |
| Locked repair benchmark | 500 separate training questions; freeze the original model's failed trajectories for comparing diagnosis and repair |
| Final task benchmark | All 7,405 official HotpotQA fullwiki development questions; no tuning |
| Round-2 training | A new, disjoint 5,000-question batch |

Freeze IDs with seed 42, remove question overlap, and exclude the 24 previously inspected pilot questions. The local pilot is part of the round-1 pool; all other listed question partitions are disjoint. Official development is our final research evaluation set, not the hidden leaderboard test set.

Build the search index from the official Wikipedia introductory paragraphs, preserving titles and sentence IDs. The corpus is fixed independently of evaluation support labels. [Corpus release](https://hotpotqa.github.io/wiki-readme.html).

Each repair record stores: question ID, model revision, original trajectory, selected step, exact pre-step state, original action, replacement action, private teacher evidence, both replay outcomes/seeds, and remaining budgets. Store all rejected proposals too. Only the pre-step student history becomes the student training prompt; diagnosis, correction, and privileged evidence remain teacher-only inputs.

**3. Locate and verify the critical step**

A “step” is one generated tool action or final-answer action. Target failures such as a poor query, wrong document choice, missed bridge entity, repeated search, premature finish, or incorrect answer synthesis.

For each failed student trajectory:

1. **Diagnose:** the same model with expanded retrieval ranks up to 3 suspect steps and explains each briefly.
2. **Propose:** generate at most 2 replacement actions per step, using the student's legal tools. A private teacher document ID cannot be read directly by the student.
3. **Replay:** restore the exact prefix, exposed documents, and remaining budgets. Replace only that action, then let the restricted student generate the continuation. Also replay the original action from the same state.
4. **Screen:** compare the candidates with deterministic continuations. Select the best apparent recovery, breaking ties by the diagnostic ranking.
5. **Confirm:** repeat that replacement and original-action branch on 3 fresh, paired continuation seeds. Accept only if the repaired branch succeeds on at least 2/3 and the original succeeds on at most 1/3.
6. **Record:** retain at most one confirmed repair point per trajectory for the initial experiment. If none passes, keep the failure in the yield denominator and skip its training update.

Success for repair verification requires a correct normalized answer, exposure of all annotated supporting sentences through student tools, and valid citations to observed sentences. Report official citation F1 separately. An ambiguous answer alias is excluded from training rather than automatically labeled wrong.

The correction consumes the student's original remaining budget; replay never grants extra calls or copies private teacher documents into student state. The confirmation thresholds are experimental filters, not proof of certainty. A successful intervention establishes a **verified repair point**, which need not be the unique or earliest mistake.

**4. Integrate the correction with OPSD**

For each batch, use identical current weights for the student, critic, and privileged teacher:

1. Generate fresh failures and verified repair records with the procedure above.
2. At each accepted pre-step state, sample a **new student action** at temperature 0.7 using only its restricted history.
3. Construct the teacher scoring context from that history plus the retrieved evidence and the verified replacement as guidance. Instruct it to choose a legal student action and respect the student's current evidence.
4. Compare teacher and student token distributions along the newly sampled student action. Apply OPSD loss **only to that action's tokens**.
5. Detach teacher probabilities and mask prompts, tool observations, and all other trajectory steps. Update the student, synchronize inference weights, and collect fresh data.

The teacher-written replacement is privileged guidance. The loss is evaluated on current student samples, retaining the on-policy component. The original failed action remains verification evidence, rather than becoming a preferred training target.

This is a proposed **adaptation of OPSD to verified decision points**, not an unchanged reproduction of the original method. Cap private guidance at 1,024 tokens, reserve context space, and verify that the actual scoring prompt retains the correction. [OPSD paper](https://arxiv.org/abs/2601.18734), [implementation](https://github.com/siyan-zhao/OPSD).

Initial settings: BF16; LoRA rank 16; learning rate 5e-6; effective batch of 8 verified decision points; generalized JSD beta 0.5 over the full vocabulary; gradient checkpointing; gradient norm cap 1.0. Stop at 200 optimizer updates or 5,000 attempted original questions, whichever comes first. Count all diagnostic and replay compute separately. Check internal development every 50 updates.

Validate target-token alignment, finite gradients, and checkpoint save/reload before the main server run. Local inference success does not establish training compatibility.

**5. Metrics must cover repair and learned performance**

| Metric | Definition and purpose |
|---|---|
| **Repair@1 / Repair@3** | Fraction of the fixed failure bank with a confirmed recovery when examining the first 1 / 3 ranked steps, with at most 2 alternatives per step |
| **Paired replay gain** | Repaired success rate minus original-action success rate on fresh matched seeds; report unsuccessful candidates too |
| Legal correction rate | Student-executable proposals divided by all proposals |
| Training-data yield | Unique confirmed repair records divided by all original questions attempted |
| Repair cost | Model tokens, tool calls, and time per confirmed recovery, including failed attempts |
| **Final joint F1 — primary learning metric** | Official combined answer-and-support score of the trained restricted student |
| Answer EM/F1 and support F1 | Separate answer correctness from evidence selection |
| Student cost and failure rate | Test-time tokens, calls, invalid actions, and unfinished answers |

Repair@k measures the combined ability to locate and repair a useful step. Do not call it localization accuracy: natural failures have no unique gold step label. Compare against selecting random steps with the same proposal and replay budgets.

Use the official HotpotQA answer/support evaluator; do not substitute document-title recall for sentence-support F1. Count final task failures in the denominator. [Official evaluator](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py).

At final task evaluation, run the student independently with top-2 retrieval: no critic or repair assistance. Report three training seeds and paired question-level 95% confidence intervals. A repair rate increase alone does not establish learned improvement.

**6. Essential comparisons and execution**

| Comparison | Question answered |
|---|---|
| Original restricted model vs proposed trained student | Did the model learn? |
| Ordinary evidence-conditioned OPSD vs verified-step OPSD | Does focusing on verified corrections improve learning? |
| Verified-step OPSD with restricted-only critic vs privileged critic | Do broader permissions improve self-teaching? |
| Ranked-step repair vs random-step repair on the frozen failure bank | Does step selection help at equal search cost? |

Use the same test-time student limits and report training-question exposure, accepted records, updates, and total generation/training cost. Account for the additional replay work when comparing methods. Select settings/checkpoints on internal development; freeze the protocol before opening either locked benchmark.

Execution: build the independent environment and index; validate the 500-question local repair pilot; verify OPSD backward/update/reload; train round 1; then attempt round 2 only if internal-development results improve. Round 2 uses the updated model for diagnosis, correction, and learning on new questions. Compare with retaining the original teacher under matched budgets before attributing continued gains to teacher refresh.

The intended result is **better held-out restricted performance from self-generated, verified step corrections**, with a second round testing continued improvement. The combination needs empirical validation and comparison with related action-specific self-distillation work before any novelty claim. [EviSD](https://arxiv.org/abs/2608.01359).

Execution was subsequently authorized. self_evolve_search_results.md records completed work and remaining experiments.
