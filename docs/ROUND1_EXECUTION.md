# Round-1 execution

The local run uses `scripts/run_experiment.sh`. It stops at 200 updates or the existing 5,000-question round-1 limit. The completed 500-question pilot counts toward that limit.

1. Train the first formal update from the pinned base and save its AdamW state using the same eight verified pilot records used by the compatibility test. Keep the compatibility artifact separate. The formal adapter is not bit-identical to that earlier checkpoint, so it begins its own consistent checkpoint/optimizer chain. The other six base-policy repairs become stale after this update and are discarded.
2. Serve each saved adapter through vLLM, retaining the original base model. Check that the adapter changes token probabilities on eight fixed training prompts. The base model remains available for baseline evaluation.
3. Finish the fixed 500-question internal-development baseline and evaluate update 1. Evaluate again every 50 updates and at the end. These labels are used only for monitoring, never for training or critic prompts. The locked repair benchmark and final evaluation remain unopened.
4. Collect on new round-1 questions from position 500 onward until eight verified repairs are available. Student, privileged role, diagnosis, and replays use the same current checkpoint. Each record contains its checkpoint fingerprint. Stop collection immediately when the batch is full.
5. Stop the inference server, load the current adapter and saved optimizer, sample fresh student actions, and apply one action-only OPSD update. Save adapter, optimizer, samples, and a completion manifest. Then serve the new adapter and repeat on unseen questions.

The single GPU alternates inference and training. Rollouts use vLLM's BF16 adapter execution; fresh-action sampling and full-vocabulary scoring use Transformers/PEFT with BF16 base weights and FP32 trainable adapters. This preserves checkpoint identity but is not a claim of bit-identical logits across the two engines. Adapter effects are checked in the actual rollout server. Initial training reconstruction and the completed compatibility update are separate evidence; only the formal run's checkpoint chain is counted in its update total.

Each question is saved atomically. The committed question cursor advances only after its batch's checkpoint is complete. A restart recovers saved questions and completed updates without repeating the optimizer step. A partial batch left at the question limit is reported and not used for an undersized update. A process lock prevents duplicate controllers; a checkpoint lock prevents duplicate trainers.

After a restart, run:

```bash
cd /absolute/path/to/verified-step-self-distillation
bash scripts/run_experiment.sh > runs/round1-main.log 2>&1
```

Use the same code and configuration to resume. A fingerprint mismatch stops the run rather than silently mixing protocols. The live Markdown includes baseline, update count, fresh-collection progress, and paired internal-development comparisons. A successful pipeline or an early positive score does not establish a research result: the full comparisons and held-out evaluation remain required.
