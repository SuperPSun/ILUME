from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from rdkit import Chem

from benchmarks.common.config import load_benchmark_config, benchmark_config_from_dict
from benchmarks.common.data import BenchmarkTask, RawDataset, configured_tasks, resolve_task
from benchmarks.common.engine import TargetStats
from benchmarks.common.features import basic_molecular_statistics, feature_schema, FeaturePreprocessor
from benchmarks.common import reuse
from benchmarks.common.summary import _simulation_results
from common.identity import semantic_identity
from common.io import sha256_file
from common.reporting import comparison_identity


@pytest.mark.parametrize('model', ('mlp', 'ecfp_xgboost', 'dmpnn', 'molformer', 'ilbert', 'spmm', 'llasmol', 'aionopedia', 'iltransr', 'aifc'))
@pytest.mark.parametrize('split', ('42', '3407'))
def test_catalog_v4_keeps_model_recipe_and_binds_tasks(model, split):
    protocol = 'catalog_v4' if split == '42' else 'catalog_v4_3407'
    current = load_benchmark_config(f'configs/benchmarks/{protocol}/{model}.yaml')
    legacy = load_benchmark_config(f'configs/benchmarks/{model}.yaml')
    assert current.model == legacy.model
    assert current.training == legacy.training
    assert current.seed == legacy.seed
    assert len(configured_tasks(current, 'stage3')) == 24
    assert current.simulation.tasks == ('simulation/thermal_expansion', 'simulation/homo', 'simulation/lumo')
    assert 'protocol' not in legacy.to_dict()
    assert resolve_task(current, 'stage3', 'experiment/hydration', 1).slots == ('solute',)
    assert current.protocol == protocol
    assert current.data.data_root == Path(f'data/{split}')
    assert current.data.task_catalog == Path(f'data/{split}/task_catalog.csv')
    hydration = resolve_task(current, 'stage3', 'experiment/hydration', 1)
    assert hydration.registry_payload['baseline_protocol'] == protocol
    assert hydration.valid_paths[0] == Path(f'data/{split}/stage3/experiment/hydration/random/cv1/fold1.csv')
    invalid = current.to_dict()
    invalid['simulation']['tasks'] = list(legacy.simulation.tasks)
    with pytest.raises(ValueError):
        benchmark_config_from_dict(invalid)
    if split == '3407':
        for name in ('homo', 'lumo'):
            task = resolve_task(current, 'simulation', f'simulation/{name}', None)
            assert task.train_paths == (Path(f'data/stage1/properties/{name}/train.csv'),)
        thermal = resolve_task(current, 'simulation', 'simulation/thermal_expansion', None)
        assert thermal.train_paths == (Path('data/3407/stage2/thermal_expansion/train.csv'),)
        invalid = current.to_dict()
        invalid['data']['data_root'] = 'data/42'
        with pytest.raises(ValueError, match='data/3407'):
            benchmark_config_from_dict(invalid)


def test_hydration_views_and_water_activity_are_complete():
    from benchmarks.aifc.adapter import aifc_model_views
    from benchmarks.iltransr.adapter import iltransr_model_views
    from benchmarks.ilbert.adapter import ilbert_model_sequences
    from benchmarks.llasmol.adapter import llasmol_model_views
    from benchmarks.aionopedia.adapter import _prepare_split, SampleStats
    from benchmarks.aionopedia.model import MultiModalRegressor
    import torch

    config = load_benchmark_config('configs/benchmarks/catalog_v4/aionopedia.yaml')
    task = resolve_task(config, 'stage3', 'experiment/hydration', 1)
    for views in (aifc_model_views, iltransr_model_views, ilbert_model_sequences, llasmol_model_views):
        assert views(task, ('O',))[0] == ('O',)
    raw = RawDataset((('O',),), 1, np.array([[300.]]), np.array([[1.]]), ('tiny.csv:2',), ({},))
    prepared = _prepare_split(task, raw, None)
    assert prepared.graph_roles == (('O', '', ''),)
    assert prepared.prompts == ('molecule O temperature 300K',)
    assert prepared.topology == 4
    assert prepared.temperature.tolist() == pytest.approx([.3])
    water = resolve_task(config, 'stage3', 'experiment/water_activity_coefficient', 1)
    raw = RawDataset((('[Na+]', '[Cl-]'),), 2, np.array([[.2, 300., 100.]]), np.array([[1.]]), ('tiny.csv:2',), ({},))
    prepared = _prepare_split(water, raw, SampleStats(100., 1., True))
    assert 'water_mole_fraction 0.2' in prepared.prompts[0]
    assert prepared.extra_conditions['water_activity'].tolist() == pytest.approx([.2])
    assert prepared.active_conditions == ('water_activity', 'pressure')
    torch.manual_seed(7)
    model = MultiModalRegressor(torch.nn.Identity(), water_activity=True)
    assert 'water_activity' in model.condition_projectors
    assert 'water_activity' in model.condition_segments
    torch.manual_seed(7)
    plain = MultiModalRegressor(torch.nn.Identity())
    assert 'water_activity' not in plain.condition_projectors
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in plain.state_dict().items())


def test_3407_orbitals_read_shared_stage1_rows(tmp_path, monkeypatch):
    from benchmarks.common.data import load_split
    config = load_benchmark_config('configs/benchmarks/catalog_v4_3407/mlp.yaml')
    catalog = config.data.task_catalog.read_bytes()
    monkeypatch.chdir(tmp_path)
    config.data.task_catalog.parent.mkdir(parents=True)
    config.data.task_catalog.write_bytes(catalog)
    for name in ('homo', 'lumo'):
        task = resolve_task(config, 'simulation', f'simulation/{name}', None)
        fields = (*task.slots, *task.target_columns, *task.audit_columns)
        for source in (*task.train_paths, *task.valid_paths, task.test_path):
            source.parent.mkdir(parents=True, exist_ok=True)
            with source.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for value in (1., 3.):
                    writer.writerow({'SMILES':'C[NH3+]', task.target_columns[0]:value,
                        'ion_role':'cation', 'provenance_source_file':'simulation/simulated_HOMO+LUMO_PBE_TZVP_cations_structured.csv',
                        'provenance_source_row':'2'})
        assert task.train_paths[0].parts[:4] == ('data', 'stage1', 'properties', name)
        assert load_split(task, 'train').targets.tolist() == [[1.], [3.]]
    assert not Path('data/3407/stage1').exists()


@pytest.fixture
def reuse_source(tmp_path, monkeypatch):
    import common.outputs
    config = load_benchmark_config('configs/benchmarks/catalog_v4/mlp.yaml')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(common.outputs, 'REPOSITORY_ROOT', tmp_path)
    task_id = reuse.REUSE_TASKS[0]
    tasks = (task_id, *(f'experiment/new_{i}' for i in range(23)))
    monkeypatch.setattr(reuse, 'configured_tasks', lambda _config, domain: tasks if domain == 'stage3' else ())
    data = Path('data'); data.mkdir()
    for fold in range(1, 6):
        with (data / f'fold{fold}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle);writer.writerow(['cation', 'anion', 'temperature_K', 'value'])
            writer.writerow(['[Na+]', '[Cl-]', 300 + fold, float(fold)])
    def task(_config, _domain, name, fold):
        return BenchmarkTask('stage3', name, ('cation', 'anion'), ('temperature_K',), ('value',), (),
            tuple(data / f'fold{f}.csv' for f in range(1, 6) if f != fold), (data / f'fold{fold}.csv',),
            data / 'test.csv', fold, 'transport', {'task_id': name, 'baseline_protocol': 'catalog_v4'})
    monkeypatch.setattr(reuse, 'resolve_task', task)
    root = Path('source');scope = root / 'stage3' / task_id.replace('/', '__')
    for fold in range(1, 6):
        spec = task(config, 'stage3', task_id, fold)
        target = np.array([[f] for f in range(1, 6) if f != fold], dtype=float)
        stats = TargetStats.fit(target).to_dict()
        base = np.concatenate([basic_molecular_statistics(Chem.MolFromSmiles(s)) for s in ('[Na+]', '[Cl-]')])
        matrix = np.asarray([np.concatenate([base, [300 + f]]) for f in range(1, 6) if f != fold])
        identity = semantic_identity('benchmark.training.v1', {'benchmark_model': 'mlp', 'task_id': task_id, 'fold': fold,
            'registry': {'task_id': task_id}, 'source_hashes': {'train': [sha256_file(p) for p in spec.train_paths], 'valid': [sha256_file(p) for p in spec.valid_paths]},
            'target_statistics': stats, 'model': config.model, 'training': config.training, 'seed': config.seed,
            'feature': feature_schema(config.features).to_dict(), 'preprocessing': FeaturePreprocessor.fit(matrix).to_dict()})
        train = scope / f'fold{fold}/attempt-001';train.mkdir(parents=True)
        (train / 'model.pt').write_bytes(b'tiny-audited-weight')
        (train / 'checkpoint.json').write_text(json.dumps({'training_identity': identity, 'target_statistics': stats,
            'integrity': {'model.pt': {'sha256': sha256_file(train / 'model.pt'), 'size': (train / 'model.pt').stat().st_size}}}))
        (train / 'metadata.json').write_text(json.dumps({'status': 'completed', 'semantic_identity': identity}))
        (train / 'run_config.yaml').write_text(yaml.safe_dump(config.to_dict()))
        evaluation = scope / f'evaluate_valid_fold{fold}/attempt-001';pred = evaluation / 'predictions/x.csv';pred.parent.mkdir(parents=True)
        with pred.open('w', newline='') as handle:
            writer = csv.writer(handle);writer.writerow(['source_row','source_fold','cation','anion','temperature_K','target','prediction','absolute_error'])
            writer.writerow([2,fold,'[Na+]','[Cl-]',300+fold,fold,fold+.1,.1])
        comparison = comparison_identity('stage3_property', split='valid', expected=(task_id,), sources={f'{task_id}:fold{p.stem.removeprefix("fold")}': sha256_file(p) for p in (*spec.train_paths, *spec.valid_paths)}, normalization={}, folds=(fold,))
        checkpoint = json.loads((train / 'checkpoint.json').read_text())
        evaluation_id = semantic_identity('benchmark.evaluation.v1', {'benchmark_model': config.name,
            'domain': 'stage3', 'task': task_id, 'split': 'valid', 'fold': fold, 'ensemble_folds': False,
            'checkpoints': [{'training_identity': identity, 'checkpoint_manifest_sha256': sha256_file(train / 'checkpoint.json'),
                             'model_integrity': checkpoint['integrity']}],
            'evaluation_source_sha256': sha256_file(spec.valid_paths[0])})
        (evaluation / 'metadata.json').write_text(json.dumps({'status': 'completed', 'semantic_identity': evaluation_id}))
        (evaluation / 'summary.json').write_text(json.dumps({'task': task_id, 'split': 'valid', 'fold': fold,
            'targets': {}, 'reporting': {'model_id': 'mlp', 'comparison_identity': comparison, 'protocol': {'expected_tasks': [task_id], 'split': 'valid', 'fold': fold, 'folds': [1,2,3,4,5], 'ensemble': False, 'model_selector': 'final_training_state', 'checkpoint_epoch': None},
            'predictions': [{'path': 'predictions/x.csv', 'task': task_id, 'sha256': sha256_file(pred), 'rows': 1}]}}))
    Path('config.yaml').write_text(yaml.safe_dump(config.to_dict()))
    return config, root.resolve(), Path('target').resolve(), scope


def test_reuse_copy_provenance_and_tamper_rejection(reuse_source):
    config, source, output, scope = reuse_source
    original = {str(p): sha256_file(p) for p in source.rglob('*') if p.is_file()}
    path = reuse.prepare_reuse(config, 'config.yaml', source, output)
    data = reuse.validate_reuse(config, path, output)
    assert sum(p['action'] == 'reuse' for p in data['tasks'].values()) == 1
    assert sum(p['action'] == 'train' for p in data['tasks'].values()) == 23
    derived = output / 'stage3' / reuse.REUSE_TASKS[0].replace('/', '__') / 'evaluate_valid_fold1/attempt-001/summary.json'
    summary = json.loads(derived.read_text())
    assert summary['targets']['value']['mae'] == pytest.approx(.1)
    assert summary['reuse_provenance']['training_identities']
    assert original == {str(p): sha256_file(p) for p in source.rglob('*') if p.is_file()}
    reuse.prepare_reuse(config, 'config.yaml', source, output)
    assert original == {str(p): sha256_file(p) for p in source.rglob('*') if p.is_file()}
    unexpected = derived.parent.parent / 'attempt-002'
    unexpected.mkdir()
    (unexpected / 'metadata.json').write_text(json.dumps({'status':'completed'}))
    (unexpected / 'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='清单外'):
        reuse.validate_reuse(config, path, output)
    for file in unexpected.iterdir():
        file.unlink()
    unexpected.rmdir()
    weight = output / 'imported' / 'stage3' / reuse.REUSE_TASKS[0].replace('/', '__') / 'fold1/attempt-001/model.pt'
    weight.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='完整性'):
        reuse.validate_reuse(config, path, output)
    with pytest.raises(ValueError, match='完整性'):
        reuse.prepare_reuse(config, 'config.yaml', source, output)


def test_reuse_data_changes_and_missing_weights(reuse_source):
    config, source, output, scope = reuse_source
    weight = scope / 'fold1/attempt-001/model.pt'
    weight.unlink()
    path = reuse.prepare_reuse(config, 'config.yaml', source, output)
    plan = reuse.validate_reuse(config, path, output)['tasks'][reuse.REUSE_TASKS[0]]
    assert plan['action'] == 'train' and plan['audit_status'] == 'integrity_error'
    assert not (output / 'imported').exists()
    weight.write_bytes(b'tiny-audited-weight')
    checkpoint = scope / 'fold1/attempt-001/checkpoint.json'
    metadata = scope / 'fold1/attempt-001/metadata.json'
    original_checkpoint, original_metadata = checkpoint.read_text(), metadata.read_text()
    content = json.loads(original_checkpoint)
    identity = content['training_identity']
    identity['payload']['preprocessing']['scale'][0] += 1
    content['training_identity'] = semantic_identity(identity['type'], identity['payload'])
    checkpoint.write_text(json.dumps(content))
    metadata.write_text(json.dumps({'status':'completed','semantic_identity':content['training_identity']}))
    path = reuse.prepare_reuse(config, 'config.yaml', source, output)
    plan = json.loads(path.read_text())['payload']['tasks'][reuse.REUSE_TASKS[0]]
    assert plan['action'] == 'train' and '预处理变化' in plan['reason']
    checkpoint.write_text(original_checkpoint)
    metadata.write_text(original_metadata)
    with Path('data/fold1.csv').open('a') as handle:
        handle.write('[Na+],[Cl-],301,1\n')
    path = reuse.prepare_reuse(config, 'config.yaml', source, output)
    assert json.loads(path.read_text())['payload']['tasks'][reuse.REUSE_TASKS[0]]['action'] == 'train'
    assert not (output / 'imported').exists()
    with pytest.raises(ValueError):
        reuse.prepare_reuse(replace(config, name='aionopedia'), 'config.yaml', source, output)


def test_reuse_rejects_unrelated_evaluation_weights(reuse_source):
    config, source, output, scope = reuse_source
    metadata = scope / 'evaluate_valid_fold1/attempt-001/metadata.json'
    content = json.loads(metadata.read_text())
    identity = content['semantic_identity']
    identity['payload']['checkpoints'][0]['checkpoint_manifest_sha256'] = 'another-checkpoint'
    content['semantic_identity'] = semantic_identity(identity['type'], identity['payload'])
    metadata.write_text(json.dumps(content))
    path = reuse.prepare_reuse(config, 'config.yaml', source, output)
    plan = json.loads(path.read_text())['payload']['tasks'][reuse.REUSE_TASKS[0]]
    assert plan['action'] == 'train'
    assert '检查点不匹配' in plan['reason']


def test_catalog_evaluation_rejects_legacy_checkpoint(reuse_source, monkeypatch):
    from benchmarks.common import engine
    config, source, output, scope = reuse_source
    monkeypatch.setattr(engine, 'resolve_task', reuse.resolve_task)
    checkpoint = scope / 'fold1/attempt-001/checkpoint.json'
    content = json.loads(checkpoint.read_text())
    content.update(format_version=engine.BENCHMARK_CHECKPOINT_VERSION, kind=engine.BENCHMARK_CHECKPOINT_KIND)
    checkpoint.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='identity'):
        engine.evaluate_checkpoint(config, 'stage3', reuse.REUSE_TASKS[0], 1, checkpoint.parent, 'valid')
    with pytest.raises(ValueError, match='禁止复用'):
        reuse.validate_reuse(replace(config, protocol='legacy'), Path('unused'), output)


@pytest.mark.parametrize('domain', ('stage3', 'simulation'))
def test_3407_identity_rejects_42_even_with_identical_source_bytes(reuse_source, monkeypatch, domain):
    from benchmarks.common import engine
    from common.identity import require_compatible_identity
    config, source, output, scope = reuse_source
    task_id = reuse.REUSE_TASKS[0]
    def task(config, benchmark, task_id, fold):
        spec = reuse.resolve_task(config, 'stage3', task_id, 1)
        return replace(spec, benchmark=benchmark, fold=fold,
            registry_payload={**spec.registry_payload, 'baseline_protocol': config.protocol})
    monkeypatch.setattr(engine, 'resolve_task', task)
    fold = 1 if domain == 'stage3' else None
    old = engine.prepare_training(config, domain, task_id, fold)
    current = replace(config, protocol='catalog_v4_3407', data=replace(config.data,
        data_root=Path('data/3407'), task_catalog=Path('data/3407/task_catalog.csv')))
    new = engine.prepare_training(current, domain, task_id, fold)
    assert np.array_equal(new.train_features, old.train_features)
    assert new.source_hashes == old.source_hashes
    with pytest.raises(ValueError, match='identity'):
        require_compatible_identity(new.training_identity, old.training_identity, context='split')
    with pytest.raises(ValueError, match='catalog_v4'):
        reuse.prepare_reuse(current, 'config.yaml', source, output)
    with pytest.raises(ValueError, match='禁止复用'):
        reuse.validate_reuse(current, Path('unused'), output)


def test_3407_sweep_schedules_every_task_without_reuse(reuse_source, monkeypatch):
    from scripts.benchmarks import sweep
    config, source, output, scope = reuse_source
    config = replace(config, protocol='catalog_v4_3407', data=replace(config.data,
        data_root=Path('data/3407'), task_catalog=Path('data/3407/task_catalog.csv')))
    Path('config.yaml').write_text(yaml.safe_dump(config.to_dict()))
    simulations = config.simulation.tasks
    experimental_tasks = reuse.configured_tasks(config, 'stage3')
    monkeypatch.setattr(sweep, 'configured_tasks', lambda c, domain: experimental_tasks if domain == 'stage3' else simulations)
    monkeypatch.setattr(sweep, 'resolve_task', reuse.resolve_task)
    def schedule(**kwargs):
        assert len(kwargs['jobs']) == 123
        assert len(kwargs['ensembles']) == 24
        assert len({j.key for j in kwargs['jobs']}) == 123
        assert len([j for j in kwargs['jobs'] if j.kind == 'simulation']) == 3
        assert kwargs['state'].rows == []
        assert kwargs['state'].catalog_config.protocol == 'catalog_v4_3407'
    monkeypatch.setattr(sweep, '_schedule', schedule)
    monkeypatch.setattr('sys.argv', ['sweep.py','--config','config.yaml','--output','target'])
    sweep.main()
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['reporting']['data_protocol'] == 'catalog_v4_3407'


def test_catalog_sweep_assembles_imported_and_new_tasks(reuse_source, monkeypatch):
    from scripts.benchmarks import sweep
    config, source, output, scope = reuse_source
    path = reuse.prepare_reuse(config, 'config.yaml', source, output)
    tasks = reuse.configured_tasks(config, 'stage3')
    monkeypatch.setattr(sweep, 'configured_tasks', reuse.configured_tasks)
    monkeypatch.setattr(sweep, 'resolve_task', reuse.resolve_task)
    monkeypatch.setenv('ILUME_DISABLE_PROGRESS', '1')
    template = output / 'stage3' / tasks[0].replace('/', '__') / 'evaluate_valid_fold1/attempt-001/summary.json'
    original = template.read_text()
    def schedule(**kwargs):
        jobs = kwargs['jobs']
        assert len(jobs) == 23 * 5
        assert {job.task for job in jobs} == set(tasks[1:])
        assert set(kwargs['ensembles']) == set(tasks[1:])
        assert len(kwargs['state'].rows) == 5
        assert {r['status'] for r in kwargs['state'].rows} == {'REUSED'}
        for job in jobs:
            target = output / 'stage3' / job.task.replace('/', '__') / f'evaluate_valid_fold{job.fold}/attempt-001'
            target.mkdir(parents=True)
            summary = json.loads(original.replace(tasks[0], job.task))
            summary['reporting']['schema_version'] = 1
            (target / 'summary.json').write_text(json.dumps(summary))
            (target / 'metadata.json').write_text(json.dumps({'status':'completed'}))
    monkeypatch.setattr(sweep, '_schedule', schedule)
    monkeypatch.setattr('sys.argv', ['sweep.py', '--config','config.yaml','--output','target','--reuse-manifest',str(path.relative_to(Path.cwd()))])
    sweep.main()
    result = json.loads((output / 'summary.json').read_text())
    assert len(result['stage3_property_benchmark']['validation_five_fold']) == 24
    assert len(result['reporting']['source_runs']['stage3_validation']) == 24
    assert all(len(refs) == 5 for refs in result['reporting']['source_runs']['stage3_validation'].values())
    assert result['reporting']['benchmarks']['stage3_test']['protocol']['expected_tasks'] == []
    assert result['reporting']['data_protocol'] == 'catalog_v4'
    assert (source / 'stage3').exists()


def test_thermal_projection_keeps_protocols_separate():
    thermal = 'simulation/thermal_expansion'
    def candidate(tasks, model):
        metrics = {t: {'count': 2, 'mae': 1., 'normalized_mae': .5} for t in tasks}
        comparison = comparison_identity('simulation_property', split='valid', expected=tasks,
            sources={f'{t}:{kind}': t+kind for t in tasks for kind in ('train','evaluation','rows')}, normalization={t: {'scale': 2.} for t in tasks})
        report = {'benchmark': 'simulation_property', 'model_id': model, 'model_display_name': model,
            'protocol': {'expected_tasks': tasks, 'split': 'valid'}, 'comparison_identity': comparison}
        return SimpleNamespace(current=True, metadata={'status': 'completed', 'stage': 'stage3'},
            summary={'reporting': report, 'split': 'valid', 'tasks': metrics}, source_run=model)
    candidates = [candidate((thermal, 'simulation/homo', 'simulation/lumo'), 'baseline'), candidate(('simulation/heat_of_vaporization', thermal), 'ILUME')]
    leaders, rows = _simulation_results(candidates, 'valid', catalog_only=True)
    assert len(leaders) == 1 and leaders[0]['total_tasks'] == 3
    leaders, rows = _simulation_results(candidates, 'valid', thermal_only=True)
    assert len(leaders) == 2 and {r['task'] for r in rows} == {thermal}
    comparison = candidates[1].summary['reporting']['comparison_identity']['payload']
    comparison['sources'][thermal + ':evaluation'] = 'different'
    candidates[1].summary['reporting']['comparison_identity'] = semantic_identity('reporting.comparison.v1', comparison)
    with pytest.raises(ValueError, match='incompatible'):
        _simulation_results(candidates, 'valid', thermal_only=True)


@pytest.mark.parametrize('model', ('ecfp_xgboost', 'dmpnn'))
def test_reuse_checks_original_fp32_ensemble(reuse_source, model):
    config, source, output, scope = reuse_source
    config = replace(config, name=model)
    task_id = reuse.REUSE_TASKS[0]
    task = reuse.resolve_task(config, 'stage3', task_id, 1)
    Path('data/test.csv').write_text('cation,anion,temperature_K,value\n[Na+],[Cl-],310,1\n')
    evaluation = scope / 'evaluate_test/attempt-001'
    evaluation.mkdir(parents=True)
    pred = evaluation / 'pred.csv'
    values = np.array([-.13088992238044739, .048193730413913727, -.04878658801317215, .23740342259407043, .15656007826328278], dtype=np.float32)
    fields = ['source_row','cation','anion','temperature_K','target', *(f'prediction_fold{f}' for f in range(1,6)), 'prediction_ensemble']
    def write_prediction(value):
        with pred.open('w', newline='') as handle:
            w=csv.writer(handle);w.writerow(fields);w.writerow([2,'[Na+]','[Cl-]',310,1,*values,value])
        summary['reporting']['predictions'][0]['sha256'] = sha256_file(pred)
        (evaluation / 'summary.json').write_text(json.dumps(summary))
    sources = {f'{task_id}:fold{p.stem.removeprefix("fold")}': sha256_file(p) for p in (*task.train_paths,*task.valid_paths)}
    sources[task_id+':test'] = sha256_file(task.test_path)
    summary = {'task': task_id, 'split': 'test', 'ensemble': {}, 'reporting': {'model_id': model,
        'protocol': {'split':'test','expected_tasks':[task_id],'folds':[1,2,3,4,5],'ensemble':True},
        'comparison_identity': comparison_identity('stage3_property',split='test',expected=(task_id,),sources=sources,normalization={},folds=(1,2,3,4,5),ensemble=True),
        'predictions':[{'path':'pred.csv','task':task_id,'rows':1}]}}
    write_prediction(float(values.mean()))
    manifests = [json.loads((scope / f'fold{f}/attempt-001/checkpoint.json').read_text()) for f in range(1,6)]
    evaluation_id = semantic_identity('benchmark.evaluation.v1', {'benchmark_model': model,
        'domain':'stage3','task':task_id,'split':'test','fold':None,'ensemble_folds':True,
        'evaluation_source_sha256':sha256_file(task.test_path),
        'checkpoints':[{'training_identity':m['training_identity'],
            'checkpoint_manifest_sha256':sha256_file(scope / f'fold{f}/attempt-001/checkpoint.json'),
            'model_integrity':m['integrity']} for f,m in enumerate(manifests,1)]})
    (evaluation / 'metadata.json').write_text(json.dumps({'semantic_identity': evaluation_id}))
    result, _ = reuse._prediction(config, task, evaluation, 'test', manifests)
    assert result['ensemble']['targets']['value']['count'] == 1
    write_prediction(float(values.mean())+.01)
    with pytest.raises(ValueError, match='集成预测'):
        reuse._prediction(config, task, evaluation, 'test', manifests)


@pytest.fixture
def paper_source(tmp_path, monkeypatch):
    from benchmarks.common import paper
    from common.reporting import reporting_block, write_prediction_csv

    monkeypatch.chdir(tmp_path)
    tasks = ('experiment/hydration', 'experiment/speed_of_sound')
    specs = {task: SimpleNamespace(task_id=task, enabled=True, meta_group='solvation' if i == 0 else 'thermophysical',
                                  identity_columns=('solute',), condition_columns=('temperature_K',),
                                  target_column='value', split_strategy='random' if i == 0 else 'system')
             for i, task in enumerate(tasks)}
    config = SimpleNamespace(data=SimpleNamespace(seed=3407, stage3_dir=Path('data/3407/stage3')))
    monkeypatch.setattr(paper, 'load_stage3_config', lambda _: config)
    monkeypatch.setattr(paper, 'resolve_task_registry', lambda _: specs)
    def source(_config, spec, fold):
        return Path('data/3407') / spec.task_id / f'fold{fold}.csv'
    def test_source(_config, spec):
        return Path('data/3407') / spec.task_id / 'test.csv'
    def read_rows(_config, spec, folds):
        for fold in folds if folds is not None else (0,):
            path = source(config, spec, fold) if fold else test_source(config, spec)
            if not path.exists():
                continue
            with path.open(newline='') as handle:
                for number, row in enumerate(csv.DictReader(handle), 2):
                    yield fold, number, row
    monkeypatch.setattr(paper, 'source_path', source)
    monkeypatch.setattr(paper, 'test_path', test_source)
    monkeypatch.setattr(paper, 'iter_rows', read_rows)
    for spec in specs.values():
        for fold in range(0, 6):
            if fold == 0 and spec.task_id == tasks[0]:
                continue
            path = source(config, spec, fold) if fold else test_source(config, spec)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'solute,temperature_K,value\nCC,300,{fold}\nCC,300,{fold + .5}\n')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('data: {seed: 3407}\n')
    ilume, baselines = tmp_path / 'ilume', tmp_path / 'baselines'

    def write_run(root, summary, *, sweep=False, ilume=False):
        root.mkdir(parents=True, exist_ok=True)
        identity = semantic_identity('test.paper-run', {'run': root.relative_to(tmp_path).as_posix()})
        metadata = {'status': 'completed', 'stage': 'stage3' if ilume else 'benchmark',
                    'operation': 'sweep' if sweep else 'evaluate', 'semantic_identity': identity,
                    'provenance': {'reporting_schema_version': 1}}
        if ilume:
            protocol = summary['reporting']['protocol']
            folds = [protocol['fold']] if protocol['split'] == 'valid' else range(1, 6)
            metadata['semantic_identity'] = semantic_identity('stage3.evaluation', {
                'prepared_identity': 'same-prepared-data',
                'checkpoint_training_identities': [f'ilume-training-fold{f}' for f in folds],
                'selector': {'split': protocol['split'], 'fold': protocol['fold'], 'checkpoint_epoch': None,
                             'model_selector': 'three_phase_final', 'tasks': protocol['expected_tasks'],
                             'ensemble_folds': protocol['ensemble']}})
        if not sweep and not ilume:
            model, task = summary['reporting']['model_id'], summary['task']
            fingerprints, originals = [], []
            for fold in (summary['fold'],) if summary['split'] == 'valid' else range(1, 6):
                training = semantic_identity('benchmark.training.v1', {'benchmark_model': model, 'task_id': task,
                    'fold': fold, 'registry': {'baseline_protocol': 'catalog_v4_3407'}})
                train = root / f'checkpoint{fold}'
                train.mkdir()
                path = train / 'checkpoint.json'
                path.write_text(json.dumps({'training_identity': training}))
                originals.append(train.relative_to(tmp_path).as_posix())
                fingerprints.append({'training_identity': training, 'checkpoint_manifest_sha256': sha256_file(path)})
            metadata['semantic_identity'] = semantic_identity('benchmark.evaluation.v1', {
                'benchmark_model': model, 'domain': 'stage3', 'task': task,
                'split': summary['split'], 'fold': summary['fold'], 'checkpoints': fingerprints,
                'ensemble_folds': summary['split'] == 'test',
                'evaluation_source_sha256': sha256_file(source(config, specs[task], summary['fold'])
                    if summary['split'] == 'valid' else test_source(config, specs[task]))})
            metadata['provenance']['checkpoints'] = originals
        (root / 'metadata.json').write_text(json.dumps(metadata))
        (root / 'summary.json').write_text(json.dumps(summary))

    def evaluation(root, model, split, fold, expected):
        selected = tasks if model == 'ilume' and split == 'valid' else tuple(expected['payload']['expected'])
        manifests = []
        for task in selected:
            records = []
            for _, number, row in read_rows(config, specs[task], (fold,) if fold else None):
                actual = float(row['value'])
                error = (.1 if task == tasks[0] else 1.2) if model == 'ilume' else .5
                record = {'source_row': number, 'solute': 'CC', 'temperature_K': 300, 'target': actual}
                if fold:
                    record.update(source_fold=fold, prediction=actual + error)
                else:
                    record.update({f'prediction_fold{f}': actual + error for f in range(1, 6)})
                    record['prediction_ensemble'] = actual + error
                records.append(record)
            name = f"predictions/{task.replace('/', '__')}.csv"
            manifest = write_prediction_csv(root / name, records, list(records[0]))
            manifest.update(path=name, task=task)
            manifests.append(manifest)
        protocol = {'split': split, 'fold': fold or None, 'folds': list(range(1, 6)),
                    'ensemble': split == 'test', 'expected_tasks': list(selected)}
        report = reporting_block(model_id=model, model_display_name=model.upper(), benchmark='stage3_property',
                                 protocol=protocol, comparison=expected, study_id=model + '-study', predictions=manifests)
        result = {'split': split, 'fold': fold or None, 'task': selected[0] if model != 'ilume' else None,
                  'model_selector': 'three_phase_final' if model == 'ilume' else 'final_training_state',
                  'reporting': report}
        write_run(root, result, ilume=model == 'ilume')

    for split in ('valid', 'test'):
        references, comparison = paper._reference(config, specs, split)
        for fold in range(1, 6) if split == 'valid' else (0,):
            root = ilume / 'valid' / f'fold{fold}' if fold else ilume / 'test'
            evaluation(root, 'ilume', split, fold, comparison)
    for model in paper.BASELINE_MODELS:
        sections, mappings = {}, {}
        for split, label in (('valid', 'stage3_validation'), ('test', 'stage3_test')):
            references, comparison = paper._reference(config, specs, split)
            # 故意使用不同 scaler；共同尺度结果必须由原单位预测重算。
            payload = comparison['payload']
            comparison = comparison_identity('stage3_property', split=split, expected=payload['expected'],
                                             sources=payload['sources'], normalization={k: {'scale': 999} for k in payload['normalization']},
                                             folds=range(1, 6), ensemble=split == 'test')
            mappings[label] = {}
            for task in references:
                by_fold = {}
                for fold in range(1, 6) if split == 'valid' else (0,):
                    root = baselines / model / task.replace('/', '__') / f'{split}{fold}'
                    fragment = comparison_identity('stage3_property', split=split, expected=(task,),
                                                   sources={k: v for k, v in payload['sources'].items() if k.startswith(task + ':')},
                                                   normalization={}, folds=(fold,) if fold else range(1, 6), ensemble=split == 'test')
                    evaluation(root, model, split, fold, fragment)
                    by_fold[f'fold{fold}'] = root.relative_to(tmp_path).as_posix()
                mappings[label][task] = by_fold if split == 'valid' else by_fold['fold0']
            if split == 'test':
                mappings[label][tasks[0]] = (baselines / model / 'no_samples').relative_to(tmp_path).as_posix()
            sections[label] = {'protocol': {'expected_tasks': list(references)}, 'comparison_identity': comparison}
        summary = {'jobs': {'failed': 0}, 'reporting': {'schema_version': 1, 'model_id': model,
                   'model_display_name': model.upper(), 'study_id': model + '-study',
                   'data_protocol': 'catalog_v4_3407', 'benchmarks': sections, 'source_runs': mappings}}
        write_run(baselines / model, summary, sweep=True)
    return config_path, ilume, baselines, tmp_path / 'paper', tmp_path


def test_paper_summary_common_scales_complete_tasks_and_plots(paper_source):
    from benchmarks.common.paper import publish_paper_summary
    pytest.importorskip('matplotlib')
    config, ilume, baselines, output, repository = paper_source
    payload = publish_paper_summary(config, ilume, baselines, output, repository, with_test=True)
    valid = payload['sections']['valid']
    assert len(valid['overall']) == 11
    assert len(valid['folds']) == 110
    ilume_row = next(r for r in valid['overall'] if r['model_id'] == 'ilume')
    assert ilume_row['wins'] == 1
    assert ilume_row['mean_rank'] == 6
    assert [r['mae_reduction_percent'] for r in valid['gains']] == pytest.approx([80, -140])
    assert all(r['common_scale'] != 999 for r in valid['folds'])
    assert len(payload['sections']['test']['gains']) == 1  # hydration 无测试集。
    native_profile = [r for r in valid['performance_profile'] if r['model_id'] == 'ilume']
    assert native_profile[0]['task_fraction'] == .5
    assert native_profile[-1]['task_fraction'] == 1
    for name in ('overall', 'task_gains', 'performance_profile'):
        assert (output / 'valid' / f'{name}.pdf').read_bytes().startswith(b'%PDF')
        assert '<svg' in (output / 'valid' / f'{name}.svg').read_text()
    assert '负增益完整保留' in (output / 'overview.md').read_text()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert all(sha256_file(output / name) == entry['sha256'] for name, entry in manifest['files'].items())
    assert not any(repository.as_posix() in str(v) for v in payload['sources'])
    with pytest.raises(FileExistsError):
        publish_paper_summary(config, ilume, baselines, output, repository)


@pytest.mark.parametrize('damage', ('sha', 'row', 'target', 'source', 'missing_fold', 'protocol', 'checkpoint', 'ensemble', 'plot'))
def test_paper_summary_rejects_malformed_inputs_atomically(paper_source, damage, monkeypatch):
    from benchmarks.common.paper import publish_paper_summary
    config, ilume, baselines, output, repository = paper_source
    summary_path = ilume / ('test' if damage == 'ensemble' else 'valid/fold1') / 'summary.json'
    summary = json.loads(summary_path.read_text())
    manifest = summary['reporting']['predictions'][0]
    prediction = summary_path.parent / manifest['path']
    if damage in {'sha', 'row', 'target', 'ensemble'}:
        with prediction.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        if damage == 'row':
            rows[1]['source_row'] = rows[0]['source_row']  # 不允许重复来源行替代完整集合。
        elif damage == 'target':
            rows[0]['target'] = str(float(rows[0]['target']) + .1)
        elif damage == 'ensemble':
            rows[0]['prediction_ensemble'] = str(float(rows[0]['prediction_ensemble']) + .1)
        else:
            rows[0]['prediction'] = str(float(rows[0]['prediction']) + .1)
        with prediction.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        if damage != 'sha':
            manifest['sha256'] = sha256_file(prediction)
            summary_path.write_text(json.dumps(summary))
    elif damage == 'source':
        path = next((repository / 'data').rglob('fold1.csv'))
        path.write_text(path.read_text().replace('CC,300,', 'CC,301,'))
    elif damage == 'missing_fold':
        (ilume / 'valid/fold5/metadata.json').unlink()
    elif damage == 'protocol':
        path = baselines / 'aionopedia/summary.json'
        value = json.loads(path.read_text())
        value['reporting']['data_protocol'] = 'catalog_v4'
        path.write_text(json.dumps(value))
    elif damage == 'checkpoint':
        path = next((baselines / 'aionopedia').rglob('checkpoint.json'))
        path.write_text(path.read_text() + ' ')
    elif damage == 'plot':
        pytest.importorskip('matplotlib')
        import benchmarks.common.paper_plots as plots
        def fail(*_):
            raise ValueError('plot failure')
        monkeypatch.setattr(plots, 'write_paper_figures', fail)
    with pytest.raises((ValueError, FileNotFoundError)):
        publish_paper_summary(config, ilume, baselines, output, repository, with_test=damage == 'ensemble')
    assert not output.exists()
    assert not list(repository.glob('.paper-staging-*'))
