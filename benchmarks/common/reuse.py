"""catalog_v4 的只读来源审计、完整复制及可追溯评估派生。"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import shutil
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rdkit import Chem

from common.identity import semantic_identity, validate_semantic_identity
from common.io import atomic_json, sha256_file
from common.outputs import open_run_directory, repository_path, repository_relative
from common.reporting import REPORTING_SCHEMA_VERSION, comparison_identity, reporting_block
from stage3.data import canonicalize_smiles
from .data import configured_tasks, has_test_rows, load_split, resolve_task
from .features import FeaturePreprocessor, basic_molecular_statistics, feature_schema
from .metrics import target_metrics

REUSE_TASKS = tuple('experiment/' + name for name in (
    'self_diffusion_coefficient', 'thermal_conductivity', 'heat_capacity',
    'speed_of_sound', 'equilibrium_pressure', 'dynamic_relative_permittivity',
    'static_relative_permittivity', 'transfer_organic', 'pec50',
))


class ReuseMismatch(ValueError):
    """输入或配方变化，需要新训练。"""


def scientific_config(config: Any) -> dict[str, Any]:
    return {k: v for k, v in config.to_dict().items() if k not in {'display_name', 'runtime'}}


def config_identity(config: Any) -> dict[str, Any]:
    return semantic_identity('benchmark.catalog-v4-config.v1', scientific_config(config))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=65536)
def _canonical_prediction(value: str) -> str:
    return canonicalize_smiles(value, '复用预测分子身份')


def _equal(left: Any, right: Any, context: str) -> None:
    if json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
        raise ReuseMismatch(context)


def _stats(old: dict[str, Any], values: np.ndarray, context: str) -> None:
    mean = values.mean(axis=0) if values.shape[1] else np.empty(0)
    std = values.std(axis=0) if values.shape[1] else np.empty(0)
    std = np.where(std > 0, std, 1.0)
    for key, expected in (('mean', mean), ('scale', std)):
        actual = np.asarray(old[key], dtype=np.float64)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ReuseMismatch(context + '/' + key)


def _completed(root: Path, required: str) -> Path:
    runs = [p for p in sorted(root.glob('attempt-*'))
            if (p / required).is_file() and (p / 'metadata.json').is_file()
            and _json(p / 'metadata.json').get('status') == 'completed']
    if not runs:
        raise ReuseMismatch('缺少已完成运行: ' + repository_relative(root))
    return runs[-1]


def _check_file(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file() or path.stat().st_size != int(expected['size']) or sha256_file(path) != expected['sha256']:
        raise ValueError('产物完整性异常: ' + repository_relative(path))


def _within(root: Path, name: str) -> Path:
    p = (root / name).resolve()
    if not p.is_relative_to(root.resolve()) or Path(name).is_absolute():
        raise ValueError('产物路径越界')
    return p


@lru_cache(maxsize=256)
def _asset_sha(path: str, size: int, modified_ns: int) -> str:
    return sha256_file(Path(path))


@lru_cache(maxsize=128)
def _asset_candidates(root: str, name: str) -> tuple[Path, ...]:
    return tuple(Path(root).rglob(Path(name).name))


def _audit_assets(config: Any, payload: dict[str, Any]) -> None:
    roots = []
    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, str) and value.startswith('artifacts/'):
            path = repository_path(value)
            roots.append(path if path.is_dir() else path.parent)
    collect(config.model)
    if config.name == 'molformer':
        from huggingface_hub import snapshot_download
        roots.append(Path(snapshot_download(repo_id=config.model['repository'], revision=config.model['revision'], local_files_only=True)))
    snapshot = payload.get('upstream_assets', payload.get('pretrained_snapshot', {}))
    records = []
    def walk(value: Any) -> None:
        if not isinstance(value, dict):
            return
        if {'path', 'sha256', 'size'} <= value.keys():
            records.append((value['path'], value))
        for key, item in value.items():
            if key == 'files' and isinstance(item, dict):
                records.extend(item.items())
            else:
                walk(item)
    walk(snapshot)
    for name, expected in records:
        candidates = [repository_path(name)] if name.startswith('artifacts/') else [root / suffix for root in roots for suffix in (name, Path(name).name)]
        if not name.startswith('artifacts/'):
            candidates.extend(p for root in set(roots) for p in _asset_candidates(str(root), name))
        paths = [p for p in candidates if p.is_file() and p.stat().st_size == int(expected['size'])]
        if not paths:
            raise ValueError('预训练资产缺失或大小错误: ' + name)
        if not any(_asset_sha(str(p), p.stat().st_size, p.stat().st_mtime_ns) == expected['sha256'] for p in paths):
            raise ValueError('预训练资产SHA错误: ' + name)


def _source_map(config: Any) -> dict[str, str | None]:
    paths = set()
    for task_id in configured_tasks(config, 'stage3'):
        task = resolve_task(config, 'stage3', task_id, 1)
        paths.update((*task.train_paths, *task.valid_paths, task.test_path))
    for task_id in configured_tasks(config, 'simulation'):
        task = resolve_task(config, 'simulation', task_id, None)
        paths.update((*task.train_paths, *task.valid_paths, task.test_path))
    return {p.as_posix(): sha256_file(p) if p.is_file() else None for p in sorted(paths)}


def _audit_train(config: Any, task: Any, run: Path) -> dict[str, Any]:
    manifest = _json(run / 'checkpoint.json')
    identity = manifest['training_identity']
    validate_semantic_identity(identity)
    metadata = _json(run / 'metadata.json')
    _equal(metadata['semantic_identity'], identity, '训练元数据身份')
    old = identity['payload']
    for key, value in (('benchmark_model', config.name), ('task_id', task.task_id),
                       ('fold', task.fold), ('seed', config.seed),
                       ('model', config.model), ('training', config.training)):
        _equal(old.get(key), value, '训练配方变化/' + key)
    saved = yaml.safe_load((run / 'run_config.yaml').read_text())
    for key in ('model', 'training', 'seed', 'features', 'environment', 'runtime'):
        _equal(saved.get(key), config.to_dict().get(key), '冻结配置变化/' + key)
    expected_registry = dict(task.registry_payload)
    for key in ('baseline_protocol', 'condition_width', 'role_policy'):
        expected_registry.pop(key, None)
    _equal(old['registry'], expected_registry, '任务合同变化')
    hashes = old['source_hashes']
    for label, paths in (('train', task.train_paths), ('valid', task.valid_paths)):
        _equal(hashes[label], [sha256_file(p) for p in paths], label + '来源变化')
    if not manifest.get('integrity'):
        raise ValueError('模型缺少完整性清单')
    for name, expected in manifest['integrity'].items():
        _check_file(_within(run, name), expected)
    _audit_assets(config, old)
    train = load_split(task, 'train')
    _stats(old['target_statistics'], train.targets, '目标归一化变化')
    _equal(manifest['target_statistics'], old['target_statistics'], 'checkpoint scaler变化')
    if config.name in {'mlp', 'ecfp_xgboost'}:
        _equal(old['feature'], feature_schema(config.features).to_dict(), '特征合同变化')
        if config.name == 'mlp':
            @lru_cache(None)
            def features(smiles: str) -> np.ndarray:
                return basic_molecular_statistics(Chem.MolFromSmiles(smiles))
            values = np.asarray([np.concatenate([*(features(s) for s in cs), train.conditions[i]])
                                 for i, cs in enumerate(train.components)])
            _equal(old['preprocessing'], FeaturePreprocessor.fit(values).to_dict(), '特征预处理变化')
    else:
        module = importlib.import_module(f'benchmarks.{config.name}.adapter')
        key = 'graph_contract' if config.name == 'dmpnn' else 'input_contract'
        symbol = 'DMPNN_GRAPH_CONTRACT' if config.name == 'dmpnn' else config.name.upper() + '_INPUT_CONTRACT'
        _equal(old[key], getattr(module, symbol), '输入预处理合同变化')
        if config.environment:
            lock = sha256_file(config.environment.lock)
            original_lock = old.get('environment_lock_sha256', metadata.get('provenance', {}).get('environment_lock_sha256'))
            if original_lock != lock:
                raise ReuseMismatch('独立环境锁变化')
        if config.name == 'ilbert':
            _equal(old['condition_contract'], {'columns': list(task.condition_columns), 'transform': 'raw_physical_units'}, 'ILBERT条件合同变化')
        elif config.name == 'iltransr':
            population, digest = hashlib.sha256(), old['source_hashes']['condition_population_sha256']
            population.update('\t'.join(task.condition_columns).encode())
            current = {p.name: p for p in {*task.train_paths, *task.valid_paths, task.test_path}}
            rows = []
            for original in old['source_hashes']['condition_population_sources']:
                population.update(original.encode())
                path = current[Path(original).name]
                if not path.exists():
                    population.update(b'<missing-empty-split>')
                    continue
                with path.open(newline='', encoding='utf-8-sig') as handle:
                    for number, row in enumerate(csv.DictReader(handle), 2):
                        value = [float(row[k]) for k in task.condition_columns]
                        rows.append(value)
                        population.update(str(number).encode())
                        population.update(np.asarray(value, dtype=np.float64).tobytes())
            if population.hexdigest() != digest:
                raise ReuseMismatch('ILTransR条件统计总体变化')
            stats = module.ConditionStats.fit(task.condition_columns, np.asarray(rows).reshape(len(rows), len(task.condition_columns)), digest)
            _equal(old['condition_statistics'], asdict(stats), 'ILTransR条件统计变化')
            _equal(old['recipe'], module.resolve_iltransr_recipe(config, task.task_id), 'ILTransR任务配方变化')
        elif config.name == 'aifc':
            _equal(old['condition_statistics'], asdict(module.ConditionStats.fit(task.condition_columns, train.conditions)), 'AIFC条件统计变化')
        else:
            _stats(old['condition_statistics'], train.conditions, '条件归一化变化')
    return manifest


def _prediction(config: Any, task: Any, run: Path, split: str, manifests: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = _json(run / 'metadata.json')['semantic_identity']
    validate_semantic_identity(evaluation)
    payload = evaluation['payload']
    for key, value in (('benchmark_model', config.name), ('domain', 'stage3'), ('task', task.task_id),
                       ('split', split), ('fold', None if split == 'test' else task.fold), ('ensemble_folds', split == 'test')):
        _equal(payload[key], value, '原评估身份变化/' + key)
    fingerprints = []
    for manifest in manifests:
        train = _completed(run.parents[1] / f"fold{manifest['training_identity']['payload']['fold']}", 'checkpoint.json')
        fingerprints.append({'training_identity': manifest['training_identity'],
                             'checkpoint_manifest_sha256': sha256_file(train / 'checkpoint.json'),
                             'model_integrity': manifest['integrity']})
    _equal(payload['checkpoints'], fingerprints, '原评估与训练检查点不匹配')
    summary = _json(run / 'summary.json')
    report = summary['reporting']
    validate_semantic_identity(report['comparison_identity'])
    _equal(report['model_id'], config.name, '预测模型变化')
    _equal(summary['task'], task.task_id, '预测任务变化')
    _equal(summary['split'], split, '预测split变化')
    if split == 'valid':
        _equal(summary['fold'], task.fold, '预测fold变化')
    source = task.test_path if split == 'test' else task.valid_paths[0]
    _equal(payload['evaluation_source_sha256'], sha256_file(source) if source.exists() else None, '原评估来源变化')
    sources = report['comparison_identity']['payload']['sources']
    if split == 'test' and sources.get(task.task_id + ':test') != (sha256_file(source) if source.exists() else None):
        raise ReuseMismatch('测试来源变化')
    for p in (*task.train_paths, *task.valid_paths):
        key = f'{task.task_id}:fold{p.stem.removeprefix("fold")}'
        if sources.get(key) != sha256_file(p):
            raise ReuseMismatch('评估来源变化')
    artifact, = report['predictions']
    pred_path = _within(run, artifact['path'])
    if sha256_file(pred_path) != artifact['sha256']:
        raise ValueError('预测SHA异常')
    with pred_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    raw = load_split(task, split)
    if len(rows) != len(raw) or len(rows) != artifact['rows']:
        raise ReuseMismatch('预测行覆盖变化')
    preds = []
    for i, (row, cs, source_row) in enumerate(zip(rows, raw.components, raw.source_rows, strict=True)):
        if tuple(_canonical_prediction(row[k]) for k in task.slots) != cs or int(row['source_row']) != int(source_row.rsplit(':', 1)[1]):
            raise ReuseMismatch('预测行顺序或分子身份变化')
        if split == 'valid' and int(row['source_fold']) != task.fold:
            raise ReuseMismatch('预测折变化')
        if [float(row[k]) for k in task.condition_columns] != raw.conditions[i].tolist() or float(row['target']) != float(raw.targets[i, 0]):
            raise ReuseMismatch('预测条件或目标变化')
        if split == 'test':
            values = [float(row[f'prediction_fold{f}']) for f in config.stage3.folds]
            dtype = np.float32 if config.name in {'ecfp_xgboost', 'dmpnn'} else np.float64
            if not np.isclose(np.asarray(values, dtype=dtype).mean(), float(row['prediction_ensemble']), rtol=1e-12, atol=1e-12):
                raise ValueError('集成预测异常')
            preds.append(float(row['prediction_ensemble']))
        else:
            preds.append(float(row['prediction']))
    scales = [m['target_statistics']['scale'][0] for m in manifests]
    scale = float(np.mean(scales))
    metrics = target_metrics(np.asarray(preds).reshape(-1, 1), raw.targets, task.target_columns, (scale,))
    comparison_sources = {f'{task.task_id}:fold{p.stem.removeprefix("fold")}': sha256_file(p) for p in (*task.train_paths, *task.valid_paths)}
    if split == 'test':
        comparison_sources[task.task_id + ':test'] = sha256_file(source) if source.exists() else None
    normalization = {f'{task.task_id}:fold{f}': {'scale': float(value)} for f, value in zip(config.stage3.folds if split == 'test' else (task.fold,), scales, strict=True)}
    comparison = comparison_identity('stage3_property', split=split, expected=(task.task_id,), sources=comparison_sources, normalization=normalization, folds=config.stage3.folds if split == 'test' else (task.fold,), ensemble=split == 'test')
    derived = {k: v for k, v in summary.items() if k not in {'reporting', 'input_audit'}}
    if split == 'test':
        derived['ensemble'] = {'targets': metrics}
    else:
        derived['targets'] = metrics
    study = semantic_identity('benchmark.reporting-study.v1', {'model': config.name, 'config': scientific_config(config)})['hash']
    derived['reporting'] = reporting_block(model_id=config.name, model_display_name=config.display_name, benchmark='stage3_property', protocol=report['protocol'], comparison=comparison, study_id=f'{config.name}-{study}', predictions=[artifact])
    return derived, {'path': repository_relative(pred_path), 'sha256': sha256_file(pred_path)}


def prepare_reuse(config: Any, config_path: str, source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if config.protocol != 'catalog_v4' or config.name == 'aionopedia':
        raise ValueError('只允许catalog_v4的九个非AIonopedia模型复用')
    if source.resolve() == output.resolve() or output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError('复用源与目标必须独立')
    if (output / 'metadata.json').exists() and _json(output / 'metadata.json').get('operation') == 'sweep':
        raise ValueError('已启动正式sweep的根目录不能重新准备复用')
    tasks = configured_tasks(config, 'stage3')
    if len(tasks) != 24:
        raise ValueError('catalog_v4必须覆盖24个实验任务')
    data = _source_map(config)
    plans = {}
    verified_files = {}
    evaluations = {}
    for task_id in tasks:
        if task_id not in REUSE_TASKS:
            plans[task_id] = {'action': 'train', 'reason': '新任务或已确认的数据变化'}
            continue
        task_dir = source / 'stage3' / task_id.replace('/', '__')
        try:
            training = []
            for fold in config.stage3.folds:
                task = resolve_task(config, 'stage3', task_id, fold)
                training.append(_audit_train(config, task, _completed(task_dir / f'fold{fold}', 'checkpoint.json')))
            ev = []
            for fold in config.stage3.folds:
                task = resolve_task(config, 'stage3', task_id, fold)
                run = _completed(task_dir / f'evaluate_valid_fold{fold}', 'summary.json')
                derived, prediction = _prediction(config, task, run, 'valid', [training[fold - 1]])
                ev.append((f'evaluate_valid_fold{fold}', run, derived, prediction))
            if has_test_rows(task):
                run = _completed(task_dir / 'evaluate_test', 'summary.json')
                derived, prediction = _prediction(config, task, run, 'test', training)
                ev.append(('evaluate_test', run, derived, prediction))
            files = {}
            for p in sorted(task_dir.rglob('*')):
                if p.is_symlink():
                    raise ValueError('复用来源禁止符号链接')
                if p.is_file():
                    files[p.relative_to(task_dir).as_posix()] = {'size': p.stat().st_size, 'sha256': sha256_file(p)}
            plans[task_id] = {'action': 'reuse', 'source': repository_relative(task_dir), 'training_identities': [m['training_identity'] for m in training]}
            verified_files[task_id] = files
            evaluations[task_id] = ev
        except ReuseMismatch as error:
            plans[task_id] = {'action': 'train', 'reason': str(error)}
        except (ValueError, KeyError, OSError) as error:
            plans[task_id] = {'action': 'train', 'audit_status': 'integrity_error', 'reason': str(error)}
    if _source_map(config) != data:
        raise ValueError('审计过程中当前数据发生变化')
    count = sum(len(fs) for fs in verified_files.values())
    size = sum(v['size'] for fs in verified_files.values() for v in fs.values())
    print(json.dumps({'copy_files': count, 'copy_bytes': size, 'destination': repository_relative(output / 'imported'), 'conflicts': 'same SHA reuse; different SHA fail'}, ensure_ascii=False), flush=True)
    if shutil.disk_usage(output.parent if output.parent.exists() else source).free < size:
        raise OSError('复制目标空间不足')
    # 先检查所有目标冲突，之后才开始复制。
    for task_id, files in verified_files.items():
        dest = output / 'imported' / 'stage3' / task_id.replace('/', '__')
        if not dest.resolve().is_relative_to(output.resolve()):
            raise ValueError('复制目标路径越界')
        for name, expected in files.items():
            target = _within(dest, name)
            if (dest / name).is_symlink():
                raise ValueError('复制目标禁止符号链接')
            if target.exists():
                _check_file(target, expected)
    identity = semantic_identity('benchmark.sweep.v1', {'config': scientific_config(config), 'benchmark': 'all'})
    data_metadata = str(config.data.data_root / 'stage3' / 'metadata.json')
    root_run = open_run_directory(stage='benchmark', operation='reuse_prepare', config_path=config_path, config_payload=config.to_dict(), semantic_identity=identity, output=repository_relative(output), seed=config.seed, reusable=output.exists(), data_metadata=data_metadata, details={'reporting_schema_version': REPORTING_SCHEMA_VERSION})
    migration_root = output / 'migrations'
    number = max([int(p.name.split('-')[1]) for p in migration_root.glob('attempt-*')], default=0) + 1
    audit = migration_root / f'attempt-{number:03d}'
    audit.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for task_id, files in verified_files.items():
        origin = repository_path(plans[task_id]['source'])
        dest = output / 'imported' / 'stage3' / task_id.replace('/', '__')
        for name, expected in files.items():
            target = _within(dest, name)
            _check_file(_within(origin, name), expected)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.copying')
                shutil.copy2(origin / name, temporary)
                _check_file(temporary, expected)
                temporary.replace(target)
            _check_file(target, expected)
            inventory[repository_relative(target)] = expected
        for label, original_run, derived, prediction in evaluations[task_id]:
            provenance = {'source_evaluation': repository_relative(dest / original_run.relative_to(origin)), 'source_prediction_sha256': prediction['sha256'], 'training_identities': plans[task_id]['training_identities']}
            ev_id = semantic_identity('benchmark.reused-evaluation.v1', {'config': config_identity(config), 'comparison': derived['reporting']['comparison_identity'], 'provenance': provenance})
            evaluation_root = output / 'stage3' / task_id.replace('/', '__') / label
            attempts = sorted(evaluation_root.glob('attempt-*'))
            run_root = attempts[-1] if attempts else evaluation_root / 'attempt-001'
            if attempts and _json(run_root / 'metadata.json').get('data_metadata') != data_metadata:
                run_root = evaluation_root / f'attempt-{int(run_root.name.split("-")[1]) + 1:03d}'
            plans[task_id].setdefault('evaluations', {})[label] = repository_relative(run_root)
            if run_root.exists():
                _equal(_json(run_root / 'metadata.json')['semantic_identity'], ev_id, '派生评估冲突')
                _equal(_json(run_root / 'summary.json'), {**derived, 'reuse_provenance': provenance}, '派生汇总冲突')
            else:
                run = open_run_directory(stage='benchmark', operation='evaluate', config_path=config_path, config_payload=config.to_dict(), semantic_identity=ev_id, output=repository_relative(run_root), seed=config.seed, data_metadata=data_metadata, details={'reporting_schema_version': REPORTING_SCHEMA_VERSION})
                pred_dest = run_root / derived['reporting']['predictions'][0]['path']
                pred_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repository_path(prediction['path']), pred_dest)
                run.complete({**derived, 'reuse_provenance': provenance})
            for p in run_root.rglob('*'):
                if p.is_file():
                    inventory[repository_relative(p)] = {'size': p.stat().st_size, 'sha256': sha256_file(p)}
    manifest = semantic_identity('benchmark.catalog-v4-reuse.v1', {'config': config_identity(config), 'data_sources': data, 'source': repository_relative(source), 'output': repository_relative(output), 'tasks': plans, 'files': inventory})
    atomic_json(audit / 'manifest.json', manifest)
    totals = {'reuse_tasks': sum(p['action'] == 'reuse' for p in plans.values()), 'train_tasks': sum(p['action'] == 'train' for p in plans.values()), 'errors': sum(p.get('audit_status') == 'integrity_error' for p in plans.values()), 'copy_files': count, 'copy_bytes': size, 'reuse_manifest': repository_relative(audit / 'manifest.json')}
    atomic_json(audit / 'summary.json', totals)
    root_run.complete(totals)
    return audit / 'manifest.json'


def validate_reuse(config: Any, path: Path, output: Path) -> dict[str, Any]:
    if config.protocol != 'catalog_v4' or config.name == 'aionopedia':
        raise ValueError('该协议或模型禁止复用')
    manifest = _json(path)
    validate_semantic_identity(manifest)
    if manifest['type'] != 'benchmark.catalog-v4-reuse.v1':
        raise ValueError('复用清单类型错误')
    payload = manifest['payload']
    _equal(payload['config'], config_identity(config), '复用配置变化')
    _equal(payload['output'], repository_relative(output), '复用目标变化')
    _equal(payload['data_sources'], _source_map(config), '复用数据变化')
    if set(payload['tasks']) != set(configured_tasks(config, 'stage3')):
        raise ValueError('复用任务覆盖不完整')
    for task_id, plan in payload['tasks'].items():
        if plan['action'] not in {'reuse', 'train'}:
            raise ValueError('复用清单存在完整性异常: ' + task_id)
    for name, expected in payload['files'].items():
        p = repository_path(name)
        if not p.is_relative_to(output.resolve()):
            raise ValueError('复用文件不属于目标根目录')
        _check_file(p, expected)
    for task_id, plan in payload['tasks'].items():
        if plan['action'] != 'reuse':
            continue
        if task_id not in REUSE_TASKS or len(plan.get('training_identities', [])) != 5:
            raise ValueError('复用训练身份覆盖不完整')
        imported = output / 'imported' / 'stage3' / task_id.replace('/', '__')
        for fold, identity in zip(config.stage3.folds, plan['training_identities'], strict=True):
            validate_semantic_identity(identity)
            train = _completed(imported / f'fold{fold}', 'checkpoint.json')
            manifest = _audit_train(config, resolve_task(config, 'stage3', task_id, fold), train)
            _equal(manifest['training_identity'], identity, '复用训练身份映射变化')
            required = {repository_relative(train / 'checkpoint.json'), *(repository_relative(_within(train, name)) for name in manifest['integrity'])}
            if not required <= payload['files'].keys():
                raise ValueError('复用权重清单覆盖不完整')
        expected_labels = {f'evaluate_valid_fold{f}' for f in config.stage3.folds}
        if has_test_rows(resolve_task(config, 'stage3', task_id, 1)):
            expected_labels.add('evaluate_test')
        if set(plan.get('evaluations', {})) != expected_labels:
            raise ValueError('复用评估映射覆盖不完整')
        for label, name in plan['evaluations'].items():
            run = repository_path(name)
            if not run.is_relative_to(output.resolve() / 'stage3' / task_id.replace('/', '__')):
                raise ValueError('复用评估映射路径错误')
            if repository_relative(run / 'summary.json') not in payload['files']:
                raise ValueError('复用评估清单覆盖不完整')
            if _completed(output / 'stage3' / task_id.replace('/', '__') / label, 'summary.json') != run:
                raise ValueError('复用评估目录出现清单外的已完成尝试')
            derived_id = _json(run / 'metadata.json')['semantic_identity']
            validate_semantic_identity(derived_id)
            _equal(derived_id['payload']['config'], config_identity(config), '派生评估配置变化')
            _equal(derived_id['payload']['provenance']['training_identities'], plan['training_identities'], '派生评估来源变化')
    return payload


def validate_completed_run(config: Any, run: Path, required: str) -> None:
    """新协议的普通完成作业必须仍然满足配置、来源与文件完整性。"""
    saved = yaml.safe_load((run / 'run_config.yaml').read_text())
    _equal({k: v for k, v in saved.items() if k not in {'display_name', 'runtime'}}, scientific_config(config), '已完成作业配置变化')
    metadata = _json(run / 'metadata.json')
    identity = metadata['semantic_identity']
    validate_semantic_identity(identity)
    details = metadata['provenance']
    task = resolve_task(config, details['benchmark'], details['task'], details.get('fold') or (1 if details['benchmark'] == 'stage3' else None))
    if required == 'checkpoint.json':
        manifest = _json(run / required)
        _equal(manifest['training_identity'], identity, '已完成训练身份变化')
        payload = identity['payload']
        hashes = payload['source_hashes']
        for label, paths in (('train', task.train_paths), ('valid', task.valid_paths)):
            _equal(hashes[label], [sha256_file(p) for p in paths], '已完成训练数据变化')
        if config.name == 'iltransr':
            from benchmarks.iltransr.adapter import _condition_population
            _, population_hash = _condition_population(task)
            _equal(hashes['condition_population_sha256'], population_hash, '已完成ILTransR条件总体变化')
        for name, expected in manifest['integrity'].items():
            _check_file(_within(run, name), expected)
    else:
        payload = identity['payload']
        source = task.test_path if details['split'] == 'test' else task.valid_paths[0]
        _equal(payload['evaluation_source_sha256'], sha256_file(source) if source.exists() else None, '已完成评估数据变化')
        for original, fingerprint in zip(details['checkpoints'], payload['checkpoints'], strict=True):
            checkpoint = repository_path(original)
            if sha256_file(checkpoint / 'checkpoint.json') != fingerprint['checkpoint_manifest_sha256']:
                raise ValueError('已完成评估的检查点清单变化')
            for name, expected in fingerprint['model_integrity'].items():
                _check_file(_within(checkpoint, name), expected)
        for item in _json(run / 'summary.json')['reporting']['predictions']:
            path = _within(run, item['path'])
            if sha256_file(path) != item['sha256']:
                raise ValueError('已完成预测SHA变化')
