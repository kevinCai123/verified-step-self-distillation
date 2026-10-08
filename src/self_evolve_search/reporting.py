import json
import os
from datetime import datetime, timezone
from pathlib import Path

def write_report(root, pilot_dir=None):
    """Regenerate RESULTS.md. The driver calls this every few seconds while collectors and evaluators are
    rewriting their summaries atomically; on the Windows-backed mount a rename can make a file briefly
    unreadable (arm C, 24 September; arm B′, 3 October). A report is never worth stopping a run for,
    so any transient read error skips this refresh instead of propagating."""
    try: _write_report(root, pilot_dir)
    except (OSError, ValueError): pass

def _write_report(root, pilot_dir=None):
    root=Path(root).resolve()
    lines=['# Self-evolution experiment results','',f'Updated: {datetime.now(timezone.utc).isoformat()}', '', '**Scope:** Qwen3.5-9B, local HotpotQA fullwiki search, same-model critical-step repair, and action-only OPSD.', '', f'Repository: `{root}`', '', '## Setup', '', '- Fresh Git repository and separate data, rollout, and training environments created.', '- Official Wikipedia archive downloaded and verified against MD5 `01edf64cd120ecc03a2745352779514c`.', '- Question partitions: 500 pilot, 5,000 round 1, 5,000 round 2, 1,500 internal development, and 500 locked repair questions. Pilot is a subset of round 1.', '- 7,405 fullwiki development questions reserved for final evaluation; final answers/support labels have not been used.', '- Configured SSH server `konnext-server` was unreachable (connection timeout). No server training has run.', '']
    index=root/'data/index/wiki.manifest.json'
    if index.exists():
        meta=json.loads(index.read_text()); lines += [f"- Full local index: **{meta['documents']:,} documents**, built in {meta['elapsed_seconds']:.1f} seconds."]
    else: lines += ['- Full corpus index: building.']
    serving=root/'runs/serving-smoke.json'
    if serving.exists():
        lines += ['- Fresh-environment Qwen3.5-9B BF16 serving produced a valid JSON search action on the local RTX 5090 (32 GB). This checks inference only, not completed-task accuracy.']
    execution=root/'runs/execution.json'
    if execution.exists():
        job=json.loads(execution.read_text())
        lines += [f"- Local job stage: **{job['stage']}**; last stage change: {job['updated']}."]
    coverage=root/'runs/coverage.json'
    if coverage.exists():
        c=json.loads(coverage.read_text()); lines += [f"- Pilot supporting-sentence mapping coverage: {c['matched_facts']}/{c['facts']} ({c['coverage']:.1%})."]
    tests=root/'runs/tests.xml'
    if tests.exists():
        import xml.etree.ElementTree as ET
        suites=ET.parse(tests).getroot()
        totals={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','failures','errors')}
        lines += [f"- Recorded tests: {totals['tests']} tests, {totals['failures']} failures, {totals['errors']} errors (`runs/tests.xml`)."]
    lines += ['', '## Pilot observations', '']
    pilot_dir=Path(pilot_dir) if pilot_dir else root/'runs/pilot500'
    summary=pilot_dir/'summary.json'
    if summary.exists():
        s=json.loads(summary.read_text())
        lines += [f"Completed **{s['questions']} / 500** training-side questions. These are exploratory pilot results, not held-out benchmark results.", '', '| Role | Answer EM | Answer F1 | Support F1 | Joint F1 | Grounded success |', '|---|---:|---:|---:|---:|---:|']
        for role in ('student','teacher'):
            m=s[role]; lines += [f"| {role} | {m['answer_em']:.1%} | {m['answer_f1']:.1%} | {m['support_f1']:.1%} | {m['joint_f1']:.1%} | {m['grounded_success']:.1%} |"]
        lines += ['',f"- Confirmed repairs: **{s['confirmed_repairs']}**, from {s['failed_student_trajectories']} failed student trajectories.", f"- Repair@3: {s['repair_at_3']:.1%}; verified-data yield per original question: {s['training_yield']:.1%}.", f"- Legal, distinct replacement proposals: {s['legal_proposals']}/{s['candidate_proposals']}.", f"- Model calls: {s['model_calls']:,}; input tokens: {s['input_tokens']:,}; output tokens: {s['output_tokens']:,}.", '- Every accepted repair must pass at least 2/3 fresh edited replays while the original branch passes at most 1/3, under the same remaining student budgets.', '- Repair@1 and random-step comparison have not been measured. Partial-overlap answer aliases are conservatively excluded from training.', '']
    else: lines += ['No model pilot results yet.', '']
    if summary.exists():
        lines += [f"- Tool invocations including diagnosis/replays: {s['tool_calls']:,}; measured question-processing time: {s['elapsed_seconds']/60:.1f} minutes."]
        if s['paired_replay_gain'] is not None:
            lines += [f"- Mean paired replay gain: {s['paired_replay_gain']:.1%}, over {s['confirmation_candidates']} screened candidates sent to confirmation (includes confirmation failures; excludes proposals that failed screening)."]
        if s['seconds_per_confirmed_repair'] is not None:
            lines += [f"- Total processing cost per confirmed repair: {s['seconds_per_confirmed_repair']/60:.1f} minutes."]
    audit=root/'runs/recovery-audit.json'
    if audit.exists():
        a=json.loads(audit.read_text())
        lines += ['', '## Restart recovery audit', '', f"- All {a['saved_questions']} expected question files are present; {a['verified_repairs']} training records match their saved replay confirmations. Temporary partial question files: {a['temporary_question_files']}.", '- The first training attempt failed before any optimizer update because Transformers returned a BatchEncoding object instead of an integer token list. Fixed by explicitly requesting token IDs; the original failure is preserved in `runs/opsd-smoke-attempt01/`.', f"- The same return-type issue affected the old prompt-length guard. Actual maximum server prompt length was {a['actual_max_prompt_tokens']:,} tokens; ordinary prompts above 6,144: {a['ordinary_trace_prompts_above_6144']}; diagnoses above 7,552: {a['diagnoses_above_7552']}. Thus this guard bug did not change budget decisions in the saved pilot.", '- Completed pilot files retain their original protocol hash and were not regenerated or modified during recovery. Future collection uses the corrected guard and a new protocol hash.']
        if a.get('verified_replacement_action_types'):
            lines += [f"- Verified replacement action types: `{json.dumps(a['verified_replacement_action_types'],sort_keys=True)}`. Rechecked paired seeds, original pre-step states and restricted replay budgets for every accepted repair.", f"- Partial-overlap answers excluded from training: {a['partial_overlap_excluded']}. Most student runtime failures were exhausted tool budgets ({a['student_errors'].get('ValueError: Tool budget exhausted',0)} trajectories)."]
    lines += ['', '## OPSD training', '']
    smoke=root/'runs/opsd-smoke/status.json'
    if smoke.exists():
        t=json.loads(smoke.read_text())
        lines += [f"- Actual optimizer updates: **{t.get('optimizer_updates',0)}**.", f"- Compatibility test completed: **{t.get('complete',False)}**."]
        if 'error' in t: lines += [f"- Failure: `{t['error']}`."]
        if 'actual_batch' in t: lines += [f"- Verified decision points in this test: {t['actual_batch']}."]
        if 'reload_equal' in t: lines += [f"- Saved/reloaded adapter tensors equal: {t['reload_equal']}."]
        if 'gradient_norm' in t: lines += [f"- Gradient norm before clipping: {t['gradient_norm']:.6g}; changed adapter tensors: {t['changed_adapter_tensors']}; peak allocated CUDA memory: {t['peak_cuda_allocated_gib']:.2f} GiB."]
        if 'losses' in t: lines += [f"- Mean action-token JSD loss before this update: {sum(t['losses'])/len(t['losses']):.6g}."]
        samples=root/'runs/opsd-smoke/samples.json'
        if t.get('complete') and samples.exists():
            examples=json.loads(samples.read_text())
            lengths=[x['prefix_tokens'] for x in examples]
            lines += [f"- Compatibility batch used student prefixes of {min(lengths)}–{max(lengths)} tokens. The observed memory result does not establish capacity for 8K training contexts."]
    else: lines += ['The training compatibility check has not run yet; it requires a verified correction and a free GPU.']
    main=root/os.environ.get('SELF_EVOLVE_RUN','runs/round1-seed42')
    if main.exists():
        run_config=json.loads((main/'config.json').read_text()) if (main/'config.json').exists() else {}
        lines += ['', f"## Continuous run `{main.name}`", '', 'The implemented runner alternates fresh repair collection and action-only OPSD, restores AdamW state, fingerprints each checkpoint, and resumes saved questions/updates.']
        if run_config:
            lines += [f"- Limits: {run_config.get('max_updates')} updates or {run_config.get('max_questions'):,} original questions (cursor starts at {run_config.get('start_cursor',500)}). Learning rate {run_config.get('learning_rate')}, loss tokens `{run_config.get('loss_tokens','all')}`, {run_config.get('fresh_records_per_update',run_config.get('batch_size',8))} fresh records per update with replay window {run_config.get('replay_window_updates',0)} (max reuse {run_config.get('max_record_reuse',1)}), evaluation on {run_config.get('evaluation_questions',500)} questions."]
        state_path=main/'state.json'
        if state_path.exists():
            state=json.loads(state_path.read_text())
            lines += [f"- Stage: **{state['phase']}**.", f"- Formal updates completed: **{state['updates']}**; committed original-question cursor: **{state['cursor']} / 5,000**.", '- The compatibility test and the reconstructed first formal update are separate artifacts; they are not added together as two sequential learning updates.']
            optimizer_check=main/'checkpoints/step-001/optimizer-resume-check.json'
            if optimizer_check.exists():
                check=json.loads(optimizer_check.read_text())
                lines += [f"- Optimizer restoration check: {check['optimizer_parameter_states']} parameter states restored on CPU with matching names, shapes, update counters and finite moments.", '- The first formal update starts a separate chain from the base model. Its adapter is not bit-identical to the earlier compatibility checkpoint; only the formal chain is used for subsequent collection and training.']
            pending_folder=main/f"batches/step-{state['updates']+1:03}"
            pending=next((p for p in (pending_folder/'summary.json',pending_folder/'fresh/summary.json') if p.exists()),None)
            if pending and state['updates']:
                # The collector rewrites this summary atomically while the report is being generated; on the
                # Windows-backed mount the rename can make it briefly unreadable (arm C stopped on exactly this
                # on 24 September). A missing or half-written summary only costs the report one line.
                try: batch=json.loads(pending.read_text())
                except (OSError, ValueError): batch=None
                if batch:
                    lines += [f"- Current fresh batch: {batch['verified']} verified repairs from {batch['attempted']} questions; next question position: {batch['next_cursor']}."]
                    if batch.get('replayed'): lines += [f"  Composed training batch: {batch['fresh']} fresh + {batch['replayed']} replayed records (from updates {batch['replayed_from_steps']})."]
            if 'error' in state: lines += [f"- Last recorded stop: `{state['error']}`."]
        for evaluation in sorted((main/'evaluations').glob('step-*/summary.json')):
            score=json.loads(evaluation.read_text())
            lines += [f"- Internal development {evaluation.parent.name}: {score['questions']} / {score['expected_questions']} questions; answer EM {score['answer_em']:.1%}, joint F1 {score['joint_f1']:.1%}; complete: {score['complete']}."]
            comparison=evaluation.parent/'comparison.json'
            if comparison.exists():
                delta=json.loads(comparison.read_text())['metrics']['joint_f1']
                lines += [f"  Joint-F1 change: {100*delta['delta']:+.2f} percentage points; paired 95% interval [{100*delta['paired_ci95'][0]:+.2f}, {100*delta['paired_ci95'][1]:+.2f}]."]
        lines += ['- Evaluation runs the restricted student alone on the fixed internal-development set. Partial evaluations are progress indicators. Full three-seed comparisons, ablations, the locked repair benchmark, final task evaluation and a second evolution round remain pending.']
    else:
        lines += ['', 'The planned 200-update training run, three-seed comparison, final evaluation, and second evolution round have not been completed. A local optimizer update is only a compatibility result, not evidence of learned improvement.']
    lines += ['', '## Evidence', '', '- `data/raw/*.manifest.json`: data source URLs, sizes, and SHA-256 hashes.', '- `data/splits/manifest.json`: fixed IDs, exclusions, and split hashes.', '- `requirements-*.lock`: resolved independent environments.', '- `runs/pilot500/`: original/teacher trajectories, ranked corrections, screen/confirmation replays, aggregate metrics, and verified step records.', '- `runs/recovery-audit.json`: saved-file hashes, completeness checks and replay/token-budget audit.', '- `runs/opsd-smoke-attempt01/`: preserved initial training failure.', '- `runs/opsd-smoke/`: compatibility status, action samples and saved adapter.', '- `runs/round1-seed42/`: continuous-run configuration, cursor, checkpoint chain, optimizer states, fresh batches and evaluations.', '', 'No stronger external teacher or model judge is used. Agent actions are parsed JSON, with backend-enforced retrieval permissions. Network blocking is application-level, not a separately verified OS sandbox.']
    content='\n'.join(lines)+'\n'
    (root/'RESULTS.md').write_text(content,encoding='utf-8')
    local=root/'config/local.json'
    if local.exists():
        export=json.loads(local.read_text()).get('report_export_path')
        if export:
            destination=Path(export); destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(content,encoding='utf-8')
