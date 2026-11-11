from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.common import baseline_report as report
from benchmarks.common.config import load_benchmark_config
from benchmarks.common.data import BenchmarkTask, load_split
from stage2.registry import orbital_audit_columns, ORBITAL_SOURCE_FILE_BY_ROLE
from benchmarks.common.features import feature_schema
from common.identity import semantic_identity, validate_semantic_identity
from common.io import sha256_file
from common.reporting import comparison_identity, write_prediction_csv


@pytest.fixture
def baseline_sources(tmp_path, monkeypatch):
    models = ('mlp', 'ilbert', 'aionopedia', 'ecfp_xgboost')
    changed, completed = 'experiment/electrical_conductivity', 'experiment/self_diffusion_coefficient'
    simulations = ('simulation/thermal_expansion', 'simulation/homo', 'simulation/lumo')
    original = Path.cwd()
    configs = {str(p.relative_to(original)): load_benchmark_config(p) for p in
               list((original / 'configs/benchmarks/FLftv4_3407').glob('*.yaml'))
               + list((original / 'configs/benchmarks/FLftv4_3407/completion').glob('*.yaml'))
               + list((original / 'configs/benchmarks/catalog_v4_3407').glob('*.yaml'))}
    for name, config in configs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config.to_dict()))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(report, 'BASELINE_MODELS', models)
    monkeypatch.setattr(report, 'FLFT_22_RETRAIN_TASKS', (changed,))
    monkeypatch.setattr(report, 'FLFT_22_COMPLETION_TASKS', (completed,))
    monkeypatch.setattr(report, 'load_benchmark_config', lambda path: configs[str(Path(path).relative_to(tmp_path))])
    registry = {task: SimpleNamespace(enabled=True, to_dict=lambda task=task: {'task_id': task}) for task in (changed, completed)}
    authority = SimpleNamespace(data=SimpleNamespace(stage3_dir=Path('data/3407/stage3'), task_catalog=Path('data/3407/task_catalog.csv')))
    monkeypatch.setattr(report, 'load_stage3_config', lambda _: authority)
    monkeypatch.setattr(report, 'resolve_task_registry', lambda _: registry)

    def resolve(config, domain, task, fold):
        base = Path('data/3407') / task.replace('/', '__')
        paths = [base / f'fold{f}.csv' for f in range(1, 6)] if domain == 'stage3' else [base / 'train.csv']
        return BenchmarkTask(domain, task, ('solute',), ('temperature_K',), ('value',), orbital_audit_columns(task),
            tuple(p for p in paths if p.stem != f'fold{fold}'),
            (base / f'fold{fold}.csv',) if fold else (base / 'valid.csv',), base / 'test.csv', fold,
            'transport', {'task_id': task, 'condition_columns': ['temperature_K'], 'baseline_protocol': config.protocol})

    monkeypatch.setattr(report, 'resolve_task', resolve)
    for domain, tasks in [('stage3', (changed, completed)), ('simulation', simulations)]:
        for task in tasks:
            base = tmp_path / 'data/3407' / task.replace('/', '__')
            base.mkdir(parents=True)
            names = [f'fold{f}.csv' for f in range(1, 6)] if domain == 'stage3' else ['train.csv', 'valid.csv']
            if task != completed:
                names.append('test.csv')
            for index, name in enumerate(names):
                (base / name).write_text(f'solute,temperature_K,value,ion_role,provenance_source_file,provenance_source_row\nC[NH3+],300,{index},cation,{ORBITAL_SOURCE_FILE_BY_ROLE["cation"]},2\nC[NH3+],300,{index + .5},cation,{ORBITAL_SOURCE_FILE_BY_ROLE["cation"]},3\n')
    baseline = tmp_path / 'outputs/benchmarks/FLftv4_3407'
    records, moves, weights, predictions = [], [], [], []
    for model in models:
        for domain, tasks in [('stage3', (changed, completed)), ('simulation', simulations)]:
            for task_id in tasks:
                config, _, protocol = report._task_config(tmp_path, model, domain, task_id, Path('authority.yaml'))
                root = baseline / ('retrain_22' if protocol == report.FLFT_22_PROTOCOL and model != 'ilbert' else '') / model / domain / task_id.replace('/', '__')
                if protocol == report.XGBOOST_LIGHTWEIGHT_PROTOCOL:
                    root = baseline / 'lightweight/ecfp_xgboost_500_d4' / domain / task_id.replace('/', '__')
                record = {'model': model, 'domain': domain, 'task': task_id, 'errors': [], 'status': 'passed', 'folds': [], 'evaluations': []}
                manifests = []
                for fold in (range(1, 6) if domain == 'stage3' else [None]):
                    task = resolve(config, domain, task_id, fold)
                    run = root / (f'fold{fold}' if fold else 'train') / 'attempt-001'
                    run.mkdir(parents=True)
                    weight = run / 'model.pt'
                    weight.write_bytes(b'test-only-model')
                    weights.append(weight)
                    payload = {'benchmark_model': model, 'domain': domain, 'task_id': task_id, 'fold': fold,
                               'registry': task.registry_payload, 'model': config.model, 'training': config.training,
                               'seed': config.seed, 'source_hashes': {split: [sha256_file(p) for p in paths]
                                    for split, paths in [('train', task.train_paths), ('valid', task.valid_paths)]}}
                    if model in ('mlp', 'ecfp_xgboost'):
                        payload['feature'] = feature_schema(config.features).to_dict()
                    if model == 'aionopedia':
                        payload.update(task={'benchmark': domain, 'task_id': task_id, 'fold': fold, 'registry_payload': task.registry_payload}, effective_seed=config.seed + int(fold or 1) - 1)
                    identity = semantic_identity('benchmark.aionopedia-training.v1' if model == 'aionopedia' else 'benchmark.training.v1', payload)
                    integrity = {'model.pt': {'size': weight.stat().st_size, 'sha256': sha256_file(weight)}}
                    manifest = {'training_identity': identity, 'integrity': integrity}
                    (run / 'checkpoint.json').write_text(json.dumps(manifest))
                    (run / 'metadata.json').write_text(json.dumps({'status': 'completed', 'semantic_identity': identity}))
                    (run / 'run_config.yaml').write_text(yaml.safe_dump(config.to_dict()))
                    old = run.relative_to(tmp_path).as_posix()
                    if protocol == report.FLFT_22_COMPLETION_PROTOCOL:
                        old = old.replace('FLftv4_3407/', 'FLftv4_3407/completion_22/')
                        p = run / 'checkpoint.json'
                        moves.append({'source': old + '/checkpoint.json', 'destination': p.relative_to(tmp_path).as_posix(), 'size': p.stat().st_size, 'sha256': sha256_file(p)})
                    elif protocol == 'catalog_v4_3407':
                        old = old.replace('FLftv4_3407/', 'v4_3407/')
                    record['folds'].append({'fold': fold, 'run': old, 'training_identity_hash': identity['hash']})
                    manifests.append((old, run, manifest))
                for split in ('valid', 'test'):
                    for fold in (range(1, 6) if split == 'valid' and domain == 'stage3' else [None]):
                        task = resolve(config, domain, task_id, fold or (1 if domain == 'stage3' else None))
                        task = replace(task, fold=fold)
                        raw = load_split(task, split)
                        ensemble = domain == 'stage3' and split == 'test'
                        selected = [manifests[fold - 1]] if fold else manifests
                        run = root / (f'evaluate_valid_fold{fold}' if fold else f'evaluate_{split}') / 'attempt-001'
                        run.mkdir(parents=True)
                        rows = []
                        for index, target in enumerate(raw.targets[:, 0]):
                            row = {'source_row': index + 2, 'solute': 'C[NH3+]', 'temperature_K': 300., 'target': float(target)}
                            row.update(raw.audit_rows[index])
                            if fold:
                                row['source_fold'] = fold
                            row['prediction_ensemble' if ensemble else 'prediction'] = float(target) + .2
                            if ensemble:
                                row.update({f'prediction_fold{f}': float(target) + .2 for f in range(1, 6)})
                            rows.append(row)
                        entries = []
                        if rows:
                            name = 'predictions/' + task_id.replace('/', '__') + '.csv'
                            entry = write_prediction_csv(run / name, rows, list(rows[0]))
                            entry.update(path=name, task=task_id)
                            entries.append(entry)
                            predictions.append(run / name)
                        source = task.test_path if split == 'test' else task.valid_paths[0]
                        source_sha = sha256_file(source) if source.exists() else None
                        if domain == 'stage3':
                            sources = {f'{task_id}:fold{f}': sha256_file(task.test_path.parent / f'fold{f}.csv') for f in range(1, 6)}
                            if split == 'test':
                                sources[task_id + ':test'] = source_sha
                        else:
                            sources = {task_id + ':train': sha256_file(task.train_paths[0]), task_id + ':evaluation': source_sha}
                        comparison = comparison_identity('stage3_property' if domain == 'stage3' else 'simulation_property', split=split, expected=[task_id], sources=sources, normalization={}, folds=[1,2,3,4,5] if ensemble else [fold] if fold else [], ensemble=ensemble)
                        identity = semantic_identity('benchmark.evaluation.v1', {'benchmark_model': model, 'domain': domain, 'task': task_id, 'split': split, 'fold': fold, 'ensemble_folds': ensemble, 'evaluation_source_sha256': source_sha, 'checkpoints': [{'checkpoint_manifest_sha256': sha256_file(p / 'checkpoint.json'), 'training_identity': m['training_identity'], 'model_integrity': m['integrity']} for _, p, m in selected]})
                        meta = {'stage': 'benchmark', 'operation': 'evaluate', 'status': 'completed', 'semantic_identity': identity, 'provenance': {'checkpoints': [old for old, _, _ in selected]}}
                        summary = {'benchmark': domain, 'task': task_id, 'split': split, 'fold': fold, 'model_selector': 'final_training_state', 'checkpoint_epoch': None, 'reporting': {'schema_version': 1, 'model_id': model, 'benchmark': 'stage3_property' if domain == 'stage3' else 'simulation_property', 'comparison_identity': comparison, 'protocol': {'ensemble': ensemble, 'fold': fold, 'expected_tasks': [task_id]}, 'predictions': entries}}
                        (run / 'metadata.json').write_text(json.dumps(meta))
                        (run / 'summary.json').write_text(json.dumps(summary))
                        record['evaluations'].append({'split': split, 'fold': fold, 'rows': len(rows), **({'prediction_sha256': entries[0]['sha256']} if rows else {})})
                records.append(record)
    receipts, mapping = tmp_path / 'audit.json', tmp_path / 'moves.json'
    receipts.write_text(json.dumps({'records': records}))
    mapping.write_text(json.dumps({'moves': moves}))
    settings = {'protocol': report.PROTOCOL, 'baseline_root': baseline.relative_to(tmp_path).as_posix(), 'authority_config': 'authority.yaml', 'source_audits': [{'path': 'audit.json', 'sha256': sha256_file(receipts)}], 'path_maps': [{'path': 'moves.json', 'sha256': sha256_file(mapping)}]}
    config = tmp_path / 'report.yaml'
    config.write_text(yaml.safe_dump(settings))
    return SimpleNamespace(repository=tmp_path, config=config, output=tmp_path/'report', weights=weights, predictions=predictions, receipts=receipts, mapping=mapping, settings=settings)


def test_baseline_report_assembles_complete_sources_without_rewriting(baseline_sources):
    f = baseline_sources
    original = {p: p.read_bytes() for p in f.repository.rglob('checkpoint.json')}
    payload = report.publish_baseline_report(f.config, f.output, f.repository)
    assert payload['coverage']['training_jobs'] == 52
    assert payload['coverage']['evaluation_runs'] == 72
    assert payload['coverage']['experimental_no_test'] == ['experiment/self_diffusion_coefficient']
    assert len(payload['sections']['stage3_validation']['tasks']) == 8
    assert all(row['tasks'] == 3 for row in payload['sections']['simulation_test']['overall'])
    assert all(row['tasks'] == 1 for row in payload['sections']['stage3_test']['overall'])
    assert all(row['mae'] == pytest.approx(.2) for row in payload['sections']['stage3_validation']['tasks'])
    assert all(row['rank'] == 1 for row in payload['sections']['stage3_validation']['tasks'])
    assert all(row['tasks'] == 2 for row in payload['sections']['stage3_validation']['overall'])
    assert json.loads((f.output/'metadata.json').read_text())['status'] == 'completed'
    validate_semantic_identity(payload['comparison_identity'])
    sources = json.loads((f.output/'source_manifest.json').read_text())
    validate_semantic_identity(sources)
    assert any('/completion_22/' in r['original_run'] for g in sources['payload']['groups'] for r in g['training'])
    assert all(p.read_bytes() == contents for p, contents in original.items())
    with pytest.raises(FileExistsError):
        report.publish_baseline_report(f.config, f.output, f.repository)


@pytest.mark.parametrize('damage', ['missing_weight', 'weight_sha', 'missing_fold', 'source_data', 'duplicate_row', 'ensemble', 'prediction_sha', 'protocol', 'audit_sha', 'path_map_sha'])
def test_baseline_report_rejects_incomplete_or_changed_sources_atomically(baseline_sources, damage):
    f = baseline_sources
    if damage == 'missing_weight':
        f.weights[0].unlink()
    elif damage == 'weight_sha':
        f.weights[0].write_bytes(b'corrupted-model')
    elif damage == 'missing_fold':
        f.weights[0].with_name('metadata.json').unlink()
    elif damage == 'source_data':
        p = next((f.repository/'data').rglob('fold1.csv'))
        p.write_text(p.read_text().replace('300,0', '300,9'))
    elif damage in ('duplicate_row', 'ensemble', 'prediction_sha'):
        p = next(p for p in f.predictions if ('evaluate_test' in str(p)) == (damage == 'ensemble'))
        rows = list(csv.DictReader(p.open()))
        if damage == 'prediction_sha':
            rows[0]['prediction'] = str(float(rows[0]['prediction']) + .1)
        elif damage == 'duplicate_row':
            rows[1] = rows[0].copy()
        else:
            rows[0]['prediction_ensemble'] = '1000'
        with p.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        s = p.parent.parent/'summary.json'
        summary = json.loads(s.read_text()); summary['reporting']['predictions'][0]['sha256'] = sha256_file(p)
        s.write_text(json.dumps(summary))
        if damage != 'prediction_sha':
            receipts = json.loads(f.receipts.read_text())
            for record in receipts['records']:
                if record['model'] == summary['reporting']['model_id'] and record['task'] == summary['task']:
                    for evaluation in record['evaluations']:
                        if evaluation['split'] == summary['split'] and evaluation['fold'] == summary['fold']:
                            evaluation['prediction_sha256'] = sha256_file(p)
            f.receipts.write_text(json.dumps(receipts))
            f.settings['source_audits'][0]['sha256'] = sha256_file(f.receipts)
            f.config.write_text(yaml.safe_dump(f.settings))
    elif damage == 'protocol':
        p = f.weights[0].with_name('checkpoint.json')
        value = json.loads(p.read_text())
        value['training_identity']['payload']['registry']['baseline_protocol'] = 'catalog_v4'
        value['training_identity'] = semantic_identity(value['training_identity']['type'], value['training_identity']['payload'])
        p.write_text(json.dumps(value))
        meta_path = p.with_name('metadata.json')
        meta = json.loads(meta_path.read_text()); meta['semantic_identity'] = value['training_identity']
        meta_path.write_text(json.dumps(meta))
        receipts = json.loads(f.receipts.read_text())
        receipts['records'][0]['folds'][0]['training_identity_hash'] = value['training_identity']['hash']
        f.receipts.write_text(json.dumps(receipts))
        f.settings['source_audits'][0]['sha256'] = sha256_file(f.receipts)
        f.config.write_text(yaml.safe_dump(f.settings))
    elif damage == 'audit_sha':
        f.receipts.write_text(f.receipts.read_text() + ' ')
    else:
        f.mapping.write_text(f.mapping.read_text() + ' ')
    with pytest.raises((ValueError, FileNotFoundError)):
        report.publish_baseline_report(f.config, f.output, f.repository)
    assert not f.output.exists()


def test_baseline_report_does_not_fall_back_to_retired_xgboost(baseline_sources):
    f = baseline_sources
    receipts = json.loads(f.receipts.read_text())
    for record in receipts['records']:
        if record['model'] == 'ecfp_xgboost':
            for train in record['folds']:
                train['run'] = train['run'].replace('lightweight/ecfp_xgboost_500_d4/', 'ecfp_xgboost/')
    f.receipts.write_text(json.dumps(receipts))
    f.settings['source_audits'][0]['sha256'] = sha256_file(f.receipts)
    f.config.write_text(yaml.safe_dump(f.settings))
    with pytest.raises(ValueError, match='Exactly one approved source group'):
        report.publish_baseline_report(f.config, f.output, f.repository)
    assert not f.output.exists()
    assert not list(f.repository.glob('.report-staging-*'))


def test_baseline_report_rejects_retired_task_roster(baseline_sources, monkeypatch):
    f = baseline_sources
    registry = report.resolve_task_registry(None)
    registry['experiment/equilibrium_pressure'] = SimpleNamespace(enabled=True)
    monkeypatch.setattr(report, 'resolve_task_registry', lambda _: registry)
    with pytest.raises(ValueError, match='exact 22-task registry'):
        report.publish_baseline_report(f.config, f.output, f.repository)
    assert not f.output.exists()
