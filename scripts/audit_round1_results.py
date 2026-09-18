"""Read saved round-1 artifacts and export report statistics; no model calls."""
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/round1-seed42'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def question_files(folder):
    return sorted(p for p in folder.glob('*.json') if re.fullmatch('[a-f0-9]{24}', p.stem))


def main():
    manifest = read(ROOT / 'data/splits/manifest.json')
    splits = manifest['splits']
    report = {'audited_at': datetime.now(timezone.utc).isoformat(),
              'state': read(RUN / 'state.json'),
              'splits': {k: {a: b for a, b in v.items() if a != 'ids'} for k, v in splits.items()},
              'excluded_questions': len(manifest['excluded_ids']),
              'final_reserved_questions': len(manifest['final_ids'])}
    assert report['state']['complete'] and report['state']['cursor'] == 5000
    expected = set(splits['round1']['ids'])
    seen, accepted = set(), {}
    counters = Counter()
    costs = Counter()
    kinds, ranks, positions = Counter(), Counter(), Counter()
    role_metrics = {r: Counter() for r in ('student', 'teacher')}
    errors = {r: Counter() for r in role_metrics}
    confirmation_gains = []
    folders = [ROOT / 'runs/pilot500'] + [p for p in sorted((RUN / 'batches').glob('step-*')) if p.name != 'step-001']
    for folder in folders:
        count = 0
        for path in question_files(folder):
            item = read(path)
            task = item['student']['task_id']
            assert task == path.stem and task not in seen and task in expected
            assert item['teacher']['task_id'] == task
            assert item['student']['top_k'] == 2 and item['teacher']['top_k'] == 8
            if folder.name != 'pilot500':
                assert item['policy_id'] == read(folder / 'summary.json')['policy_id']
            seen.add(task)
            count += 1
            for role in role_metrics:
                for key, value in item[role]['metrics'].items():
                    role_metrics[role][key] += float(value)
                if item[role]['error']:
                    errors[role][item[role]['error']] += 1
            repair = item['repair']
            counters['failed_student_trajectories'] += not item['student']['metrics']['grounded_success']
            counters['ambiguous_answer_exclusions'] += repair.get('ambiguous_answer_excluded', False)
            counters['diagnosis_errors'] += bool(repair.get('diagnosis_error'))
            counters['proposals'] += len(repair['attempts'])
            counters['legal_proposals'] += sum(bool(a['legal']) for a in repair['attempts'])
            selected = repair.get('selected')
            if selected:
                counters['candidates_sent_to_confirmation'] += 1
                confirmation_gains.append((selected['edited_wins'] - selected['original_wins']) / 3)
            record = repair.get('confirmed')
            if record:
                assert record['task_id'] == task and record['edited_wins'] >= 2 and record['original_wins'] <= 1
                accepted[task] = record
                counters['confirmed_repairs'] += 1
                kinds[record['replacement']['action']] += 1
                ranks[str(record['rank'])] += 1
                positions[str(record['step'] + 1)] += 1
            costs['model_calls'] += len(item['calls'])
            costs['tool_calls'] += len(item['tool_calls'])
            costs['question_processing_seconds'] += item['elapsed_seconds']
            costs['input_tokens'] += sum(c['usage']['prompt_tokens'] for c in item['calls'])
            costs['output_tokens'] += sum(c['usage']['completion_tokens'] for c in item['calls'])
        print(json.dumps({'audited_folder': folder.name, 'questions': count, 'total': len(seen)}), flush=True)
    assert seen == expected
    counters['questions'] = len(seen)
    report['collection'] = dict(counters)
    report['collection'].update(
        confirmed_yield=counters['confirmed_repairs'] / len(seen),
        repair_at3_training_failures=counters['confirmed_repairs'] / counters['failed_student_trajectories'],
        legal_proposal_rate=counters['legal_proposals'] / counters['proposals'],
        mean_confirmation_gain=sum(confirmation_gains) / len(confirmation_gains),
        replacement_types=dict(kinds), diagnostic_ranks=dict(ranks), step_positions_one_based=dict(positions),
        costs=dict(costs),
        seconds_per_confirmed_repair=costs['question_processing_seconds'] / counters['confirmed_repairs'],
        role_metrics_over_changing_checkpoints={r: {k: v / len(seen) for k, v in m.items()} for r, m in role_metrics.items()},
        role_errors={r: dict(v) for r, v in errors.items()})

    statuses, samples, used = [], [], set()
    used_types = Counter()
    for step in range(1, 25):
        status = read(RUN / f'checkpoints/step-{step:03}/status.json')
        assert status['complete'] and status['reload_equal'] and status['step'] == step
        assert status['optimizer_restored'] == (step > 1)
        records = read(RUN / f'batches/step-{step:03}/records.json')
        assert len(records) == 8
        for record in records:
            task = record['task_id']
            assert task in accepted and task not in used
            assert record['replacement'] == accepted[task]['replacement']
            assert record['policy_id'] == status['source_policy']
            used.add(task)
            used_types[record['replacement']['action']] += 1
        statuses.append(status)
        samples.extend(read(RUN / f'checkpoints/step-{step:03}/samples.json'))
    report['training'] = {
        'updates': len(statuses), 'used_records': len(used),
        'accepted_but_unused': len(set(accepted) - used), 'used_replacement_types': dict(used_types),
        'subprocess_seconds_including_loading_and_checkpoint_io': sum(s['elapsed_seconds'] for s in statuses),
        'peak_allocated_cuda_gib': max(s['peak_cuda_allocated_gib'] for s in statuses),
        'trainable_parameters': statuses[0]['trainable_parameters'],
        'student_prefix_tokens_range': [min(s['prefix_tokens'] for s in samples), max(s['prefix_tokens'] for s in samples)],
        'teacher_prefix_tokens_range': [min(s['teacher_prefix_tokens'] for s in samples), max(s['teacher_prefix_tokens'] for s in samples)],
        'sampled_action_tokens_total': sum(s['target_tokens'] for s in samples),
        'first_update_mean_jsd': statuses[0]['mean_jsd'], 'last_update_mean_jsd': statuses[-1]['mean_jsd'],
        'all_checkpoints_reload_equal': True, 'optimizer_restored_updates': 23}

    evaluations, eval_rows = {}, {}
    for step in (0, 1, 24):
        folder = RUN / f'evaluations/step-{step:03}'
        summary, config = read(folder / 'summary.json'), read(folder / 'config.json')
        rows = {p.stem: read(p) for p in question_files(folder)}
        assert len(rows) == 500 and set(rows) == set(splits['dev_monitor']['ids'])
        assert not (set(rows) & seen)
        assert summary['complete']
        metrics = {k: sum(float(r['metrics'][k]) for r in rows.values()) / len(rows) for k in next(iter(rows.values()))['metrics']}
        for key in ('answer_em', 'answer_f1', 'support_f1', 'joint_f1', 'grounded_success'):
            assert math.isclose(metrics[key], summary[key], abs_tol=1e-12)
        evaluations[str(step)] = dict(summary, extra_metrics=metrics, config=config,
            input_tokens=sum(c['usage']['prompt_tokens'] for r in rows.values() for c in r['calls']),
            output_tokens=sum(c['usage']['completion_tokens'] for r in rows.values() for c in r['calls']),
            error_types=dict(Counter(r['error'] for r in rows.values() if r['error'])))
        eval_rows[step] = rows
    assert len({e['config']['questions_sha256'] for e in evaluations.values()}) == 1
    assert len({e['config']['protocol_sha256'] for e in evaluations.values()}) == 1
    report['evaluations'] = evaluations
    report['paired_question_changes'] = {}
    for metric in ('answer_em', 'grounded_success', 'joint_f1'):
        transitions = Counter()
        for task in eval_rows[0]:
            before, after = eval_rows[0][task]['metrics'][metric], eval_rows[24][task]['metrics'][metric]
            transitions['improved' if after > before else 'regressed' if after < before else 'unchanged'] += 1
        report['paired_question_changes'][metric] = dict(transitions)
    report['final_comparison'] = read(RUN / 'evaluations/step-024/comparison.json')
    report['pilot_summary'] = read(ROOT / 'runs/pilot500/summary.json')
    report['audit_checks'] = {'round1_ids_complete_unique': True, 'dev_ids_match_and_disjoint': True,
                              'evaluation_protocols_match': True, 'saved_evaluation_means_match': True,
                              'all_used_repairs_confirmed_and_unique': True}
    (RUN / 'report-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    brief = {k: v for k, v in report.items() if k in ('splits', 'collection', 'training', 'paired_question_changes', 'audit_checks')}
    print(json.dumps(brief, indent=2), flush=True)


if __name__ == '__main__':
    main()
