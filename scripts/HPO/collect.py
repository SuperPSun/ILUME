"""Collect native per-task validation metrics without model selection."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from _common import ROOT, jobs_for, relative
from _configs import read_yaml
from common.io import atomic_json, sha256_file

METRICS = ('mae', 'rmse', 'r2', 'normalized_mae')


def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def metric_row(base, task, metrics, **details):
    row = {**base, 'task': task, **details}
    for name in METRICS:
        row[name] = metrics.get(name)
    present = [row[name] for name in METRICS if row[name] is not None]
    row['metric_status'] = ('missing' if not present else
                            'nonfinite' if any(not math.isfinite(v) for v in present) else 'available')
    row['unavailable_metrics'] = ','.join(name for name in METRICS if row[name] is None)
    return row


def completed_validation(job, fold):
    # Failed evaluation directories remain present, but cannot enter reports as completed.
    matches = sorted(job.output.glob(f'valid*/fold{fold}/hpo_evaluation.json'))
    for marker in reversed(matches):
        saved = load(marker)
        if all((marker.parent / name).is_file() and sha256_file(marker.parent / name) == digest
               for name, digest in saved.get('files', {}).items()) and saved.get('files'):
            return load(marker.parent / 'summary.json')
    return {}


def collect_jobs(jobs):
    results, rows = [], []
    for job in jobs:
        state = load(job.output / 'hpo_run.json')
        failure = load(ROOT / 'outputs/HPO/failures' / job.branch / (job.trial + '.json'))
        raw = read_yaml(job.config1 or job.config3)
        base = {'branch': job.branch, 'trial_id': job.trial,
                'split': 'stage1' if job.config1 else raw['data']['seed'],
                'training_seed': raw['seed'] if job.config1 else raw['training']['seed'],
                'status': state.get('status', failure.get('status', 'not_run')),
                'error': state.get('error', failure.get('error')),
                'config': relative(job.config1 or job.config3),
                'stage2_config': relative(job.config2) if job.config2 else None,
                'stage2_output': relative(job.stage2_output) if job.stage2_output else None,
                'elapsed_seconds': state.get('elapsed_seconds'),
                'caches': state.get('caches', {})}
        results.append(base)
        if job.config1:
            from stage1.regression import REGRESSION_TASKS
            attempt = ROOT / state.get('artifact_output', relative(job.output / 'attempt1'))
            summary = load(attempt / 'summary.json')
            for task in REGRESSION_TASKS:
                detail = summary.get('tasks', {}).get(task, {})
                rows.append(metric_row(base, task, detail.get('validation', {}), domain='stage1', fold=None,
                    checkpoint=relative(attempt / 'tasks' / task / 'regression_head.pt'),
                    parameter_count=detail.get('parameter_count'), actual_seed=detail.get('seed')))
            continue
        if job.config2:
            metrics_path = job.stage2_output / 'metrics.jsonl'
            history = [json.loads(line) for line in metrics_path.read_text().splitlines() if line] if metrics_path.exists() else []
            metrics = history[-1].get('validation_normalized_mae', {}) if history else {}
            for task in read_yaml(job.config2)['data']['tasks']:
                rows.append(metric_row(base, task, {'normalized_mae': metrics.get(task)}, domain='stage2', fold=None,
                    checkpoint=relative(job.stage2_output / 'stage2_final.pt'),
                    recorded_epoch=history[-1].get('epoch') if history else None,
                    expected_epoch=read_yaml(job.config2)['training']['epochs']))
        for fold in job.folds:
            validation = completed_validation(job, fold)
            train = job.output if job.branch == 'HoME' else job.output / 'train'
            final = load(train / f'fold{fold}' / 'three_phase_final.json')
            plan = load(train / f'fold{fold}' / 'resolved_training_plan.json')
            for task, config in raw['tasks'].items():
                if not config.get('enabled', True):
                    continue
                rows.append(metric_row(base, task, validation.get('tasks', {}).get(task, {}),
                    domain='experimental', fold=fold, actual_seed=base['training_seed']+fold,
                    checkpoint=relative(train / f'fold{fold}' / 'three_phase_final.pt'),
                    gate_diagnostics=validation.get('gate_diagnostics', {}).get(task),
                    parameter_counts=plan.get('parameter_counts')))
            simulation = (final.get('simulation_validation') or {}).get('simulation_tasks', {})
            for task in ('simulation/heat_of_vaporization', 'simulation/thermal_expansion'):
                rows.append(metric_row(base, task, simulation.get(task, {}),
                    domain='simulation_fold', fold=fold, actual_seed=base['training_seed']+fold,
                    checkpoint=relative(train / f'fold{fold}' / 'three_phase_final.pt')))
    return results, rows


def final_statistics(rows):
    # First average folds within each split, then summarize splits; neither is independent replication.
    groups = {}
    for row in rows:
        if row['branch'] != 'final':
            continue
        groups.setdefault((row['domain'], row['task']), []).append(row)
    summary = []
    for (domain, task), values in sorted(groups.items()):
        item = {'domain': domain, 'task': task, 'splits': {}}
        for split in sorted({r['split'] for r in values}):
            selected = [r for r in values if r['split'] == split]
            expected = 1 if domain == 'stage2' else 5
            stats = {'complete': len(selected) == expected and all(r['metric_status'] == 'available' and r['status'] == 'completed' for r in selected)}
            for metric in METRICS:
                numbers = [r[metric] for r in selected if r[metric] is not None and math.isfinite(r[metric])]
                stats[metric] = ({'mean': statistics.mean(numbers), 'std': statistics.stdev(numbers) if len(numbers)>1 else 0.,
                                  'min': min(numbers), 'max': max(numbers)} if len(numbers)==expected and stats['complete'] else None)
            item['splits'][str(split)] = stats
        for metric in METRICS:
            numbers = [s[metric]['mean'] for s in item['splits'].values() if s[metric] is not None]
            item['cross_split_'+metric] = ({'mean': statistics.mean(numbers), 'std': statistics.stdev(numbers),
                                           'min': min(numbers), 'max': max(numbers)} if len(numbers)==5 else None)
        summary.append(item)
    return summary


def csv_report(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in row.items()})


def main():
    parser = argparse.ArgumentParser(description='Collect per-task native validation results; never select a winner.')
    parser.add_argument('--branch', choices=('all','stage1','HoME','stage2','stage3','final'), default='all')
    parser.add_argument('--config-root', default='configs/HPO')
    parser.add_argument('--output', default='outputs/HPO/reports/'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    branches = ('stage1','HoME','stage2','stage3','final') if args.branch=='all' else (args.branch,)
    jobs=[]
    for branch in branches:
        if (Path(args.config_root) / branch).exists():
            jobs.extend(jobs_for(branch, Path(args.config_root)))
    output = Path(args.output)
    if not output.resolve().is_relative_to(ROOT / 'outputs/HPO'):
        parser.error('Reports must be under outputs/HPO')
    output.mkdir(parents=True, exist_ok=False)
    results, rows = collect_jobs(jobs)
    atomic_json(output/'results.json', {'runs': results, 'task_metrics': rows,
        'final_statistics': final_statistics(rows),
        'note': '人工选择；五套 split 有重叠，不是五次独立外部测试。未计算测试指标。'})
    csv_report(output/'results.csv', results)
    csv_report(output/'task_metrics.csv', rows)
    home = [{**r, 'grouping': r['trial_id'].split('_')[0], 'capacity': r['trial_id'].split('_')[1]}
            for r in rows if r['branch']=='HoME']
    csv_report(output/'home_grid.csv', home)
    print(f'Reports: {output}; no model selected.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
