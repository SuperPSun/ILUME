"""Behavioral checks for static research recipes, cache integrity and local scheduling."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import time

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/HPO'))
from _configs import (generate_initial, generate_stage2, generate_stage3, generate_final,
                      read_yaml, write_yaml, validate_pair, stage3_recipes, GROUP_VARIANTS, fingerprint)
from _common import Job, run_queue
from collect import final_statistics, metric_row
from common.artifact_cache import artifact_cache, artifact_lock
from common.identity import semantic_identity
from common.io import sha256_file
from stage2.home_config import load_home_recipe, home_recipe_from_dict, effective_training_seed
from stage3.config import load_stage3_config, stage3_config_from_dict

import weakopt
import weakopt_joint
import weakopt_shared


@pytest.fixture
def weakopt_campaign(tmp_path, monkeypatch):
    raw = deepcopy(read_yaml(ROOT / 'configs/v4/stage3/optimized_3407.yaml'))
    monkeypatch.setattr(weakopt, 'ROOT', tmp_path)
    monkeypatch.setattr(weakopt, 'CONFIGS', tmp_path / 'configs/HPO/WeakOpt')
    monkeypatch.setattr(weakopt, 'OUTPUT', tmp_path / 'outputs/HPO/WeakOpt')
    monkeypatch.setattr(weakopt, 'BASE', tmp_path / 'base.yaml')
    for name in ('stage1_encoder', 'stage2_final'):
        raw['initialization'][name] = name + '.pt'
        (tmp_path / raw['initialization'][name]).write_bytes(b'frozen weights')
    raw['data']['artifacts_dir'] = 'prepared/artifacts'
    raw['preparation']['data_metadata_path'] = 'prepared/audit.json'
    (tmp_path / 'prepared/artifacts').mkdir(parents=True)
    (tmp_path / 'prepared/artifacts/metadata.json').write_text('{}')
    (tmp_path / 'prepared/audit.json').write_text('{}')
    write_yaml(weakopt.BASE, raw)
    snapshot = {'sources': {}, 'folds': [], 'comparison_identity': {}}
    monkeypatch.setattr(weakopt, 'baseline_snapshot', lambda: snapshot)
    monkeypatch.setattr(weakopt, 'validate_data_sources', lambda *a: None)
    weakopt.generate()
    return raw


def test_weakopt_frozen_recipes_limit_and_tamper_rejection(weakopt_campaign, monkeypatch):
    manifest, base = weakopt.load_campaign()
    assert len(manifest['trials']) == 24
    assert base == weakopt_campaign
    assert read_yaml(weakopt.CONFIGS / 'trials/021.yaml')['tasks']['experiment/static_relative_permittivity']['phase3_private_epochs'] == 1
    weakopt.generate()
    assert len(weakopt.load_campaign()[0]['trials']) == 24
    proposal = weakopt.CONFIGS / 'proposal.yaml'
    write_yaml(proposal, [{'description': 'duplicate', 'overrides': manifest['trials'][0]['overrides']}])
    with pytest.raises(ValueError, match='Duplicate'):
        weakopt.generate(proposal)
    proposal = weakopt.CONFIGS / 'new-proposal.yaml'
    write_yaml(proposal, [{'description': 'new', 'overrides': {'experiment/viscosity': {'phase3_private_epochs': 8}}}])
    monkeypatch.setattr(weakopt, 'LIMIT', 24)
    with pytest.raises(ValueError, match='budget'):
        weakopt.generate(proposal)
    changed = deepcopy(base)
    changed['model']['global_experts'] += 1
    with pytest.raises(ValueError, match='whitelist'):
        weakopt.validate_recipe(base, changed)
    changed = deepcopy(base)
    changed['tasks']['experiment/viscosity']['model_overrides']['private_hidden_ratio'] *= 2
    with pytest.raises(ValueError, match='whitelist'):
        weakopt.validate_recipe(base, changed)
    candidate = weakopt.CONFIGS / 'trials/001.yaml'
    candidate.write_text(candidate.read_text() + '\n# tampered\n')
    with pytest.raises(ValueError, match='modified'):
        weakopt.load_campaign()


def test_weakopt_appends_new_configs_without_rebinding_folds(weakopt_campaign):
    before = sha256_file(weakopt.CONFIGS / 'trials/001.yaml')
    proposal = weakopt.CONFIGS / 'proposal.yaml'
    write_yaml(proposal, [{'description': 'combined task budgets', 'overrides': {
        'experiment/viscosity': {'phase3_private_epochs': 8},
        'experiment/static_relative_permittivity': {'phase3_private_epochs': 2}}}])
    weakopt.generate(proposal)
    manifest, _ = weakopt.load_campaign()
    assert manifest['trials'][-1]['id'] == '025'
    assert sha256_file(weakopt.CONFIGS / 'trials/001.yaml') == before
    (weakopt.ROOT / 'stage2_final.pt').write_bytes(b'altered')
    with pytest.raises(ValueError, match='SHA256'):
        weakopt.load_campaign()


@pytest.mark.parametrize('device', ['cuda:2', 'cuda:3'])
def test_weakopt_gpu_limits_resume_and_cross_trial_rejection(weakopt_campaign, monkeypatch, device):
    assert weakopt.devices_and_folds('cuda:2,cuda:3', 2, [1, 3]) == ('cuda:2', 'cuda:3')
    assert weakopt.devices_and_folds('cuda:2,cuda:3', 6, [1, 3]) == ('cuda:2', 'cuda:3')
    assert weakopt.devices_and_folds(device, 1, [1]) == (device,)
    for devices, maximum, folds in [('cuda:0', 1, [1]), ('cuda:1', 1, [1]),
                                    ('cuda:2', 4, [1]), ('cuda:2,cuda:3', 7, [1]),
                                    ('cuda:2,cuda:2', 2, [1]), ('cuda:2', 1, [1, 1])]:
        with pytest.raises(ValueError):
            weakopt.devices_and_folds(devices, maximum, folds)
    trial = weakopt.load_campaign()[0]['trials'][0]
    updated = []
    evaluated = []
    def fake_native(script, arguments, log, assigned_device):
        assert script == 'stage3/train' and '--resume' in arguments
        assert arguments[arguments.index('--max-parallel') + 1] == 1
        assert arguments[arguments.index('--devices') + 1] == assigned_device == device
        requested = arguments[arguments.index('--fold') + 1:arguments.index('--output')]
        root = weakopt.ROOT / arguments[arguments.index('--output') + 1]
        for fold in requested:
            artifact = root / f'fold{fold}/three_phase_final.pt'
            if not artifact.exists():
                artifact.parent.mkdir(parents=True)
                artifact.write_text('complete')
                updated.append(fold)
    monkeypatch.setattr(weakopt, 'native', fake_native)
    monkeypatch.setattr(weakopt, 'evaluate_fold', lambda job, fold, device, resume: evaluated.append(fold))
    weakopt.run_trial(trial, [1, 3], device)
    weakopt.run_trial(trial, [1, 2, 3, 4, 5], device)
    assert updated == [1, 3, 2, 4, 5]
    assert evaluated == [1, 3, 1, 2, 3, 4, 5]
    altered = dict(trial, recipe_hash='another recipe')
    with pytest.raises(ValueError, match='another frozen configuration'):
        weakopt.run_trial(altered, [1], device)


@pytest.mark.parametrize('legacy_running', [0, 1, 2])
def test_weakopt_gpu_slot_limit_and_existing_controller(tmp_path, monkeypatch, legacy_running):
    monkeypatch.setattr(weakopt, 'OUTPUT', tmp_path)
    limit = 3 - legacy_running
    barrier = threading.Barrier(limit)
    mutex = threading.Lock()
    active = 0
    peak = 0
    def work(_):
        nonlocal active, peak
        with weakopt.gpu_slot('cuda:2'):
            with mutex:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=10)
            time.sleep(.01)
            with mutex:
                active -= 1
    with ExitStack() as previous:
        for filename in ('cuda-2.lock', 'cuda-2-slot2.lock')[:legacy_running]:
            previous.enter_context(artifact_lock(tmp_path / 'locks' / filename))
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(work, range(6)))
    assert peak == limit and active == 0


def test_weakopt_same_fold_ranking_ties_and_win_priority():
    baseline = {'folds': [{'task': task, 'fold': fold, 'model_id': model, 'mae': value,
                           'normalized_mae': value, 'common_scale': 1.0}
        for task in ('a', 'b') for fold in range(1, 6)
        for model, value in [('ilume', 1.1)] + [(f'baseline{i}', 1.0) for i in range(10)]]}
    tied = [{'task': task, 'fold': fold, 'mae': 1.0, 'normalized_mae': 1.0}
            for task in ('a', 'b') for fold in (1, 3)]
    assert weakopt.score_rows(tied, baseline, [1, 3])['wins'] == 2
    with pytest.raises(ValueError, match='coverage'):
        weakopt.score_rows(tied, baseline, [1, 2, 3, 4, 5])
    candidate = [dict(r, mae=0.9 if r['task'] == 'a' else 10., normalized_mae=0.9 if r['task'] == 'a' else 10.) for r in tied]
    score = weakopt.score_rows(candidate, baseline, [1, 3])
    assert score['wins'] == 1 and score['macro_normalized_mae'] > 1.1
    fewer_wins = dict(score, wins=0, macro_normalized_mae=1.01, id='001')
    ranked = sorted([fewer_wins, dict(score, id='002')], key=lambda r: (-r['wins'], r['macro_normalized_mae'], r['id']))
    assert ranked[0]['id'] == '002'


def test_weakopt_watch_fills_free_slots_without_duplicate_or_failed_work(weakopt_campaign, monkeypatch):
    from types import SimpleNamespace
    for fold in (1, 3):
        marker = weakopt.OUTPUT / f'trials/001/valid/fold{fold}/hpo_evaluation.json'
        marker.parent.mkdir(parents=True)
        marker.write_text('{}')
    status = weakopt.OUTPUT / 'trials/003/status.json'
    status.parent.mkdir(parents=True)
    status.write_text('{"status":"failed"}')
    commands = []
    processes = []
    def spawn(command, **kwargs):
        assert command[command.index('--max-parallel') + 1] == '1'
        assert command[command.index('--fold') + 1:command.index('--devices')] == ['1', '3']
        process = SimpleNamespace(pid=100 + len(commands), code=None)
        process.poll = lambda: process.code
        commands.append(command)
        processes.append(process)
        return process
    monkeypatch.setattr(weakopt.subprocess, 'Popen', spawn)
    children, failed = {}, {'004'}
    with ExitStack() as locks:
        for name in ('cuda-2.lock', 'cuda-2-slot2.lock', 'cuda-3.lock', 'trial-002.lock'):
            locks.enter_context(artifact_lock(weakopt.OUTPUT / 'locks' / name))
        record = weakopt.fill_slots(children, failed)
        assert [(r['id'], r['device']) for r in record['launched']] == [
            ('005', 'cuda:2'), ('006', 'cuda:3'), ('007', 'cuda:3')]
        assert weakopt.fill_slots(children, failed)['launched'] == []
        processes[0].code = 1
        record = weakopt.fill_slots(children, failed)
        assert record['launched'][0]['id'] == '008' and '005' in failed
    assert len(commands) == 4


def test_weakopt_baseline_reader_requests_validation_only(monkeypatch):
    import benchmarks.common.paper as paper
    seen = []
    def fake_reader(*args, with_test):
        seen.append(with_test)
        assert not with_test
        return {'sources': {}, 'sections': {'valid': {'folds': []}}, 'comparison_identity': {}}, {}
    monkeypatch.setattr(paper, 'build_paper_summary', fake_reader)
    assert weakopt.baseline_snapshot()['folds'] == [] and seen == [False]


def test_weakopt_validation_artifacts_and_identity_are_required(weakopt_campaign, monkeypatch):
    import stage3.evaluate as evaluation
    trial = weakopt.load_campaign()[0]['trials'][0]
    baseline = {'folds': [{'task': task, 'fold': 1, 'model_id': 'ilume', 'count': 1, 'common_scale': 2.0,
                           'source_run': 'reference-valid'}
                          for task in weakopt_campaign['tasks']]}
    assert weakopt.validation_rows(trial, [1], baseline) is None
    root = weakopt.OUTPUT / 'trials/001/valid/fold1'
    root.mkdir(parents=True)
    manifests = []
    for index, task in enumerate(weakopt_campaign['tasks']):
        path = root / f'prediction{index}.csv'
        path.write_text('target,prediction\n1,0.5\n')
        manifests.append({'task': task, 'path': path.name, 'sha256': sha256_file(path)})
    comparison = semantic_identity('reporting.comparison.v1', {'source': 'same data'})
    summary = {'tasks': {task: {'count': 1, 'mae': .5} for task in weakopt_campaign['tasks']},
               'reporting': {'predictions': manifests, 'comparison_identity': comparison}}
    reference = weakopt.ROOT / 'reference-valid'
    reference.mkdir()
    (reference / 'summary.json').write_text(json.dumps(summary))
    (root / 'summary.json').write_text(json.dumps(summary))
    (root / 'metadata.json').write_text('{}')
    identity = {'frozen': 'first configuration'}
    marker = root / 'hpo_evaluation.json'
    files = {p.name: sha256_file(p) for p in root.iterdir() if p.is_file()}
    marker.write_text(json.dumps({'identity': identity, 'files': files}))
    monkeypatch.setattr(evaluation, 'resolve_stage3_evaluation_identity', lambda *a, **kw: identity)
    rows = weakopt.validation_rows(trial, [1], baseline)
    assert len(rows) == 24 and all(row['normalized_mae'] == .25 for row in rows)
    monkeypatch.setattr(evaluation, 'resolve_stage3_evaluation_identity', lambda *a, **kw: {'frozen': 'second configuration'})
    with pytest.raises(ValueError, match='identity'):
        weakopt.validation_rows(trial, [1], baseline)
    monkeypatch.setattr(evaluation, 'resolve_stage3_evaluation_identity', lambda *a, **kw: identity)
    (root / 'prediction0.csv').write_text('target,prediction\n1,0\n')
    with pytest.raises(ValueError, match='SHA256'):
        weakopt.validation_rows(trial, [1], baseline)


def test_weakopt_finalize_requires_complete_validation_winner(weakopt_campaign, monkeypatch):
    folder = weakopt.OUTPUT / 'reports/test'
    folder.mkdir(parents=True)
    (folder / 'ranking.json').write_text('{}')
    seen = []
    def fake_report(folds):
        seen.append(tuple(folds))
        return {'ranked': [{'id': '002', 'wins': 15}]}, folder
    monkeypatch.setattr(weakopt, 'report', fake_report)
    with pytest.raises(ValueError, match='highest ranked'):
        weakopt.finalize('001')
    weakopt.finalize('002')
    assert seen == [(1, 2, 3, 4, 5)] * 2
    assert json.loads((weakopt.OUTPUT / 'selection.json').read_text())['id'] == '002'
    with pytest.raises(FileExistsError, match='locked'):
        weakopt.finalize('002')


@pytest.fixture
def joint_campaign(weakopt_campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(weakopt_joint, 'ROOT', tmp_path)
    monkeypatch.setattr(weakopt_joint, 'CONFIGS', weakopt.CONFIGS / 'Joint')
    monkeypatch.setattr(weakopt_joint, 'OUTPUT', weakopt.OUTPUT / 'Joint')
    monkeypatch.setattr(weakopt_joint, 'BASE2', tmp_path / 'stage2.yaml')
    s2 = read_yaml(ROOT / 'configs/v4/stage2/optimized_3407.yaml')
    s2['data']['artifacts_dir'] = 'prepared/artifacts'
    s2['preparation']['data_metadata_path'] = 'prepared/audit.json'
    write_yaml(weakopt_joint.BASE2, s2)
    proposals = [{'description': f'dropout {v}', 'stage2': {'epochs': 5},
                  'stage3': {'experiment/viscosity': {'model_overrides': {'private_dropout': v}}}}
                 for v in (.05, 0.)]
    proposal = tmp_path / 'proposal.yaml'
    write_yaml(proposal, proposals)
    weakopt_joint.generate(proposal)
    return proposals


def test_joint_generation_deduplicates_sources_and_rejects_changes(joint_campaign, tmp_path):
    manifest, s2, s3 = weakopt_joint.load()
    assert len(manifest['stage2']) == 1 and len(manifest['trials']) == 2
    assert {t['stage2_id'] for t in manifest['trials']} == {'S001'}
    assert weakopt_joint.normalized_stage3(read_yaml(weakopt_joint.CONFIGS / 'trials/001.yaml'), s3)['initialization'] == s3['initialization']
    duplicate = tmp_path / 'duplicate.yaml'
    write_yaml(duplicate, joint_campaign)
    with pytest.raises(ValueError, match='Duplicate'):
        weakopt_joint.generate(duplicate)
    too_many = tmp_path / 'too-many.yaml'
    write_yaml(too_many, joint_campaign * 40)
    with pytest.raises(ValueError, match='80-pair'):
        weakopt_joint.generate(too_many)
    with pytest.raises(ValueError, match='Only Stage2'):
        weakopt_joint.stage2_recipe(s2, {'batch_size': 512})
    path = weakopt_joint.CONFIGS / 'stage2/S001.yaml'
    changed = read_yaml(path)
    changed['home']['model']['global_experts'] += 1
    path.write_text(yaml.safe_dump(changed))
    manifest['stage2'][0].update(sha256=sha256_file(path), recipe_hash=fingerprint(changed))
    (weakopt_joint.CONFIGS / 'campaign.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='protected'):
        weakopt_joint.load()


def test_joint_stage3_resume_binds_exact_stage2_source(joint_campaign, monkeypatch):
    manifest, _, _ = weakopt_joint.load()
    trial = manifest['trials'][0]
    source = weakopt_joint.OUTPUT / 'stage2/S001/train'
    source.mkdir(parents=True)
    for name in ('stage2_final.pt', 'stage2_final.json'):
        (source / name).write_bytes(b'frozen source')
    calls = []
    monkeypatch.setattr(weakopt_joint, 'native', lambda script, args, log, device: calls.append((script, args, device)))
    evaluated = []
    monkeypatch.setattr(weakopt_joint, 'evaluate_fold', lambda job, fold, device, resume: evaluated.append(fold))
    weakopt_joint.run_trial(trial, [1, 3], 'cuda:2')
    weakopt_joint.run_trial(trial, [2, 4, 5], 'cuda:3')
    assert evaluated == [1, 3, 2, 4, 5]
    assert all(args[args.index('--max-parallel') + 1] == 1 and '--resume' in args for _, args, _ in calls)
    assert [device for _, _, device in calls] == ['cuda:2', 'cuda:3']
    (source / 'stage2_final.pt').write_bytes(b'changed source')
    with pytest.raises(ValueError, match='another configuration or Stage2 source'):
        weakopt_joint.run_trial(trial, [1], 'cuda:2')
    assert len(calls) == 2


def test_joint_finalize_requires_validation_winner_and_preserves_lock(joint_campaign, monkeypatch):
    folder = weakopt_joint.OUTPUT / 'reports/test'
    folder.mkdir(parents=True)
    (folder / 'ranking.json').write_text('{}')
    seen = []
    def fake_report(folds):
        seen.append(tuple(folds))
        return {'ranked': [{'id': '002', 'wins': 10}]}, folder
    monkeypatch.setattr(weakopt_joint, 'report', fake_report)
    with pytest.raises(ValueError, match='highest ranked'):
        weakopt_joint.finalize('001')
    weakopt_joint.finalize('002')
    assert seen == [(1, 2, 3, 4, 5)] * 2
    with pytest.raises(FileExistsError, match='locked'):
        weakopt_joint.finalize('002')


@pytest.fixture
def shared_campaign(weakopt_campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(weakopt_joint, 'ROOT', tmp_path)
    monkeypatch.setattr(weakopt_joint, 'CONFIGS', weakopt.CONFIGS / 'Joint')
    monkeypatch.setattr(weakopt_joint, 'OUTPUT', weakopt.OUTPUT / 'Joint')
    monkeypatch.setattr(weakopt_joint, 'load', lambda: (*weakopt.load_campaign(), None))
    monkeypatch.setattr(weakopt_shared, 'ROOT', tmp_path)
    monkeypatch.setattr(weakopt_shared, 'CONFIGS', weakopt.CONFIGS / 'Shared')
    monkeypatch.setattr(weakopt_shared, 'OUTPUT', weakopt.OUTPUT / 'Shared')
    monkeypatch.setattr(weakopt_shared, 'BASE', tmp_path / 'reference-068.yaml')
    raw = read_yaml(ROOT / 'configs/HPO/WeakOpt/Joint/trials/068.yaml')
    for key in ('data', 'preparation', 'initialization'):
        raw[key] = deepcopy(weakopt_campaign[key])
    write_yaml(weakopt_shared.BASE, raw)
    (tmp_path / 'stage2_final.json').write_text('{}')
    write_yaml(weakopt_joint.CONFIGS / 'stage2/S030.yaml', {})
    snapshot = weakopt_joint.OUTPUT / 'reference/baselines.json'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps({'sources': {}, 'folds': []}))
    weakopt_shared.generate()
    return raw


def test_shared_owner_whitelist_sources_and_duplicate_budget(shared_campaign, tmp_path):
    manifest, base = weakopt_shared.load()
    assert len(manifest['trials']) == 16
    weakopt_shared.generate()
    changed = deepcopy(base)
    changed['training']['three_phase']['global']['lr'] *= 1.2
    weakopt_shared.validate_recipe(base, changed)
    for path in [('model', 'global_experts'), ('training', 'max_grad_norm')]:
        invalid = deepcopy(changed)
        invalid[path[0]][path[1]] += 1
        with pytest.raises(ValueError, match='whitelist'):
            weakopt_shared.validate_recipe(base, invalid)
    invalid = deepcopy(changed)
    invalid['training']['three_phase']['global']['warmup_ratio'] = .1
    with pytest.raises(ValueError, match='whitelist'):
        weakopt_shared.validate_recipe(base, invalid)
    duplicate = tmp_path / 'duplicate.yaml'
    write_yaml(duplicate, [manifest['trials'][0]['overrides']])
    with pytest.raises(ValueError, match='Duplicate'):
        weakopt_shared.generate(duplicate)
    too_many = tmp_path / 'too-many.yaml'
    write_yaml(too_many, [manifest['trials'][0]['overrides']] * 33)
    with pytest.raises(ValueError, match='48-candidate'):
        weakopt_shared.generate(too_many)
    (tmp_path / 'stage2_final.pt').write_text('corrupted')
    with pytest.raises(ValueError, match='SHA256'):
        weakopt_shared.load()


def test_shared_promotion_budget_and_lock_are_persistent(shared_campaign):
    trials = weakopt_shared.load()[0]['trials']
    weakopt_shared.reserve(trials, [1, 3])
    with pytest.raises(ValueError, match='completed screening'):
        weakopt_shared.reserve(trials[:1], [2, 4, 5])
    for t in trials[:9]:
        for fold in (1, 3):
            marker = weakopt_shared.OUTPUT / f"trials/{t['id']}/valid/fold{fold}/hpo_evaluation.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text('{}')
    weakopt_shared.reserve(trials[:8], [2, 4, 5])
    before = (weakopt_shared.OUTPUT / 'budget.json').read_bytes()
    weakopt_shared.reserve(trials[:8], [2, 4, 5])
    assert (weakopt_shared.OUTPUT / 'budget.json').read_bytes() == before
    with pytest.raises(ValueError, match='eight-candidate'):
        weakopt_shared.reserve(trials[8:9], [2, 4, 5])
    assert (weakopt_shared.OUTPUT / 'budget.json').read_bytes() == before
    (weakopt_shared.OUTPUT / 'selection.json').write_text('{}')
    with pytest.raises(ValueError, match='locked'):
        weakopt_shared.reserve(trials[:1], [1, 3])
    with pytest.raises(ValueError, match='locked'):
        weakopt_shared.generate()


def test_shared_trial_explicit_paths_preserve_source_binding(shared_campaign, monkeypatch, tmp_path):
    trial = weakopt_shared.load()[0]['trials'][0]
    calls = []
    monkeypatch.setattr(weakopt_joint, 'native', lambda script, args, log, device: calls.append(args))
    monkeypatch.setattr(weakopt_joint, 'evaluate_fold', lambda *args, **kwargs: None)
    weakopt_joint.run_trial(trial, [1, 3], 'cuda:2', configs=weakopt_shared.CONFIGS, output=weakopt_shared.OUTPUT)
    assert calls[0][calls[0].index('--output') + 1].startswith('outputs/HPO/WeakOpt/Shared/')
    assert calls[0][calls[0].index('--max-parallel') + 1] == 1
    (tmp_path / 'stage2_final.pt').write_text('different source')
    with pytest.raises(ValueError, match='another configuration or Stage2 source'):
        weakopt_joint.run_trial(trial, [2, 4, 5], 'cuda:3', configs=weakopt_shared.CONFIGS, output=weakopt_shared.OUTPUT)
    assert len(calls) == 1


def test_all_static_rounds_and_manual_binding(tmp_path):
    root = tmp_path / 'configs'
    generate_initial(root)
    assert len(list((root/'stage1').glob('*.yaml'))) == 100
    assert len(list((root/'HoME/stage2').glob('*.yaml'))) == len(list((root/'HoME/stage3').glob('*.yaml'))) == 100
    pairs = {p.stem for p in (root/'HoME/stage2').glob('*.yaml')}
    assert pairs == {f'{g}_C{i:02}' for g in GROUP_VARIANTS for i in range(20)}
    source = root/'selected.yaml'
    write_yaml(source, {'home': None})
    with pytest.raises(ValueError, match='Manual selection'):
        generate_stage2(source, root)
    source.write_text('home: G4_C05\n')
    generate_stage2(source, root)
    recipes = [read_yaml(p) for p in (root/'stage2').glob('*.yaml')]
    assert len(recipes)==20 and {r['training']['epochs'] for r in recipes}=={5,10,15,20}
    source.write_text(f'home: G4_C05\nstage2_config: {root}/stage2/007.yaml\nstage2_checkpoint: outputs/HPO/stage2/007/train/stage2_final.pt\n')
    generate_stage3(source, root)
    assert len(list((root/'stage3').glob('*.yaml')))==50
    for path in (root/'stage3').glob('*.yaml'):
        stage3_config_from_dict(read_yaml(path))
    source.write_text(source.read_text()+f'stage3_config: {root}/stage3/000.yaml\n')
    generate_final(source, root)
    assert len(list((root/'final').glob('*/stage3.yaml')))==5
    for path in (root/'final').glob('*/stage3.yaml'):
        value=read_yaml(path)
        assert value['training']['seed']==42
        assert value['data']['seed']==int(path.parent.name)
        assert value['initialization']['stage2_final']==f'outputs/HPO/final/{path.parent.name}/stage2/stage2_final.pt'
    source.write_text('home: G1_C05\n')
    with pytest.raises(ValueError, match='different manual selection'):
        generate_stage2(source, root)


def test_transfer_budget_private_and_structure_boundaries():
    a=read_yaml(ROOT/'configs/v4/stage2/base.yaml')
    b=read_yaml(ROOT/'configs/v4/stage3/base.yaml')
    b['groups']['solvation']['phase2']['epochs']=12
    b['tasks']['experiment/pec50']['model_overrides']['private_hidden_ratio']=.5
    b['tasks']['experiment/pec50']['model_overrides']['private_dropout']=.1
    validate_pair(a,b)
    for key in ('global_experts','private_experts','dropout','expert_hidden_ratio'):
        changed=deepcopy(b)
        changed['model'][key]=changed['model'][key]+1 if key.endswith('experts') else .2
        with pytest.raises(ValueError):
            validate_pair(a,changed)
    changed=deepcopy(b)
    changed['groups']['solvation']['experts']+=1
    with pytest.raises(ValueError,match='architecture mismatch'):
        validate_pair(a,changed)
    changed=deepcopy(b)
    changed['tasks']['experiment/gas_solubility']['meta_group']='thermophysical'
    with pytest.raises(ValueError,match='interaction topology'):
        validate_pair(a,changed)


def test_seed_epoch_defaults_and_private_overrides():
    original=load_home_recipe(ROOT/'configs/v4/stage2/base.yaml')
    assert 'seed' not in original.stage2.to_dict()['training']
    assert 'data_metadata_path' not in original.stage2.to_dict()['preparation']
    value=original.to_dict()
    value['training'].update(seed=23,epochs=5)
    recipe=home_recipe_from_dict(value)
    assert effective_training_seed(recipe.stage2)==23
    assert effective_training_seed(replace(recipe.stage2,data=replace(recipe.stage2.data,seed=90756)))==23
    assert recipe.stage2_epochs==5
    for bad in (0,-1,True,1.5):
        changed=deepcopy(value);changed['training']['epochs']=bad
        with pytest.raises(ValueError): home_recipe_from_dict(changed)
    for bad in (-1,True,1.5):
        changed=deepcopy(value);changed['training']['seed']=bad
        with pytest.raises(ValueError): home_recipe_from_dict(changed)
    base=read_yaml(ROOT/'configs/v4/stage3/base.yaml')
    parsed=stage3_config_from_dict(base)
    assert 'data_metadata_path' not in parsed.to_dict()['preparation']
    for changed in stage3_recipes(base):
        config=stage3_config_from_dict(changed)
        for task in config.tasks:
            original_recipe=parsed.resolved_private_recipe(task)
            actual=config.resolved_private_recipe(task)
            for phase in (2,3):
                if getattr(original_recipe,f'phase{phase}_epochs')==0:
                    assert getattr(actual,f'phase{phase}_epochs')==0


def test_cache_single_writer_integrity_and_dry_run(tmp_path):
    identity=semantic_identity('test.prepare', {'split':42,'source':'frozen',
        'registry': {'task': {'primary_slots': ('solute',), 'condition_columns': ('temperature',)}}})
    path=tmp_path/'cache'
    calls=[]
    def build(staging):
        calls.append(1)
        time.sleep(.01)
        (staging/'data').write_text('prepared')
    assert artifact_cache(path,identity,build,dry_run=True)=='CREATE'
    assert not path.exists() and not calls
    with ThreadPoolExecutor(max_workers=4) as pool:
        decisions=list(pool.map(lambda _: artifact_cache(path,identity,build),range(10)))
    assert len(calls)==1 and decisions.count('CREATE')==1
    assert artifact_cache(path,identity,None,dry_run=True)=='REUSE'
    saved_identity = json.loads((path/'complete.json').read_text())['identity']
    assert artifact_cache(path,saved_identity,None,dry_run=True)=='REUSE'
    changed=semantic_identity('test.prepare', {'split':42,'source':'frozen',
        'registry': {'task': {'primary_slots': ('solvent',), 'condition_columns': ('temperature',)}}})
    assert artifact_cache(path,changed,None,dry_run=True)=='INVALID'
    other=semantic_identity('test.prepare',{'split':123,'source':'frozen'})
    assert artifact_cache(path,other,None,dry_run=True)=='INVALID'
    with pytest.raises(ValueError): artifact_cache(path,other,build)
    (path/'data').write_text('damaged')
    assert artifact_cache(path,identity,None,dry_run=True)=='INVALID'
    with pytest.raises(ValueError): artifact_cache(path,identity,build)


@pytest.mark.parametrize('maximum', [2, 4])
def test_queue_device_slots_global_limit_and_failures(tmp_path, maximum):
    lock=threading.Lock();active=set();peak=[];events=[]
    jobs=[Job(str(i),'test',tmp_path/str(i)) for i in range(8)]
    def work(job,device):
        with lock:
            assert device not in active
            active.add(device);peak.append(len(active));events.append((job.trial,'stage2'))
        time.sleep(.005)
        with lock:
            events.append((job.trial,'stage3'));active.remove(device)
        if job.trial=='3': raise RuntimeError('failed trial')
    assert run_queue(jobs,('cuda:0','cuda:1','cuda:2','cuda:3'),maximum,work,record_failures=False)==['3']
    assert max(peak)==maximum
    for job in jobs:
        assert events.index((job.trial,'stage2'))<events.index((job.trial,'stage3'))


@pytest.mark.parametrize('devices,maximum', [
    (('cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'), 1),
    (('cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'), 7),
    (('cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'), 8),
    (('cuda:2', 'cuda:3'), 6),
])
def test_queue_shares_gpus_with_balanced_slots(tmp_path, devices, maximum):
    jobs = [Job(str(i), 'test', tmp_path / str(i)) for i in range(maximum * 2)]
    barrier = threading.Barrier(maximum)
    lock = threading.Lock()
    active = {device: 0 for device in devices}
    snapshots = []
    def work(job, device):
        with lock:
            active[device] += 1
            if sum(active.values()) == maximum:
                snapshots.append(dict(active))
        barrier.wait(timeout=10)
        with lock:
            active[device] -= 1
        barrier.wait(timeout=10)
    assert run_queue(jobs, devices, maximum, work, record_failures=False) == []
    assert len(snapshots) == 2
    for snapshot in snapshots:
        assert sum(snapshot.values()) == maximum
        assert max(snapshot.values()) - min(snapshot.values()) <= 1
    assert not any(active.values())


def test_collection_does_not_hide_missing_or_choose_split():
    assert metric_row({},'task',{})['metric_status']=='missing'
    assert metric_row({},'task',{'mae':float('nan')})['metric_status']=='nonfinite'
    rows=[]
    for split in (42,123,1234,3407,90756):
        for fold in range(1,6):
            rows.append(metric_row({'branch':'final','split':split,'status':'completed'},'task',
                                   {'mae':float(fold)},domain='experimental',fold=fold))
    result=final_statistics(rows)[0]
    assert result['cross_split_mae']['mean']==3
    assert 'winner' not in result
    rows[0]['mae']=None
    result=final_statistics(rows)[0]
    assert result['cross_split_mae'] is None


def test_collection_reads_nested_simulation_validation(tmp_path, monkeypatch):
    import collect
    monkeypatch.setattr(collect, 'ROOT', tmp_path)
    monkeypatch.setattr(collect, 'relative', lambda path: str(path))
    config = tmp_path / 'stage3.yaml'
    write_yaml(config, {'data': {'seed': 42}, 'training': {'seed': 42}, 'tasks': {}})
    output = tmp_path / 'trial'
    fold = output / 'train/fold1'
    fold.mkdir(parents=True)
    tasks = {
        'simulation/heat_of_vaporization': {'count': 12, 'normalized_mae': .25},
        'simulation/thermal_expansion': {'count': 8, 'normalized_mae': .5},
    }
    (fold / 'three_phase_final.json').write_text(json.dumps({
        'simulation_validation': {'simulation_tasks': tasks},
    }))
    job = Job('000', 'stage3', output, config3=config, folds=(1,))
    _, rows = collect.collect_jobs([job])
    assert len(rows) == 2
    for row in rows:
        assert row['domain'] == 'simulation_fold'
        assert row['metric_status'] == 'available'
        assert row['normalized_mae'] == tasks[row['task']]['normalized_mae']


def test_controller_dependency_order_resume_and_failed_stage2(tmp_path, monkeypatch):
    import _common as runner
    import stage3.home
    import stage3.evaluate
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    a=read_yaml(ROOT/'configs/HPO/HoME/stage2/G0_C05.yaml')
    b=read_yaml(ROOT/'configs/HPO/HoME/stage3/G0_C05.yaml')
    ca,cb=tmp_path/'configs/a.yaml',tmp_path/'configs/b.yaml'
    write_yaml(ca,a);write_yaml(cb,b)
    identity=semantic_identity('test.prepare',{'split':42})
    events=[]
    monkeypatch.setattr(runner,'cache_identity',lambda *args:(None,identity))
    monkeypatch.setattr(runner,'prepare',lambda stage,*args,**kw:(events.append('prepare_'+stage) or identity,'REUSE'))
    monkeypatch.setattr(runner,'validate_stage2_final',lambda *args:events.append('validate_stage2'))
    monkeypatch.setattr(stage3.home,'load_source',lambda *args:events.append('load_stage2_source'))
    eval_identity=semantic_identity('test.evaluation',{'source':'stage2'})
    monkeypatch.setattr(stage3.evaluate,'resolve_stage3_evaluation_identity',lambda *args,**kw:eval_identity)
    fail=False
    def native(script,arguments,log,device,**kwargs):
        events.append(script)
        path=tmp_path/arguments[arguments.index('--output')+1]
        if script=='stage2/train':
            if fail: raise RuntimeError('broken Stage2')
            path.mkdir(parents=True)
            (path/'stage2_final.pt').write_text('complete final')
        elif script=='stage3/evaluate':
            fold=arguments[arguments.index('--fold')+1]
            folder=path/f'fold{fold}'
            folder.mkdir(parents=True)
            (folder/'summary.json').write_text('{}')
    monkeypatch.setattr(runner,'native',native)
    job=Job('G0_C05','HoME',tmp_path/'outputs/HPO/HoME/stage3/G0_C05',
            config2=ca,config3=cb,stage2_output=tmp_path/'outputs/HPO/HoME/stage2/G0_C05')
    runner.run_job(job,'cuda:0')
    assert events.index('stage2/train')<events.index('load_stage2_source')<events.index('prepare_stage3')<events.index('stage3/train')<events.index('stage3/evaluate')
    events.clear()
    runner.run_job(job,'cuda:0',resume=True)
    assert 'stage2/train' not in events and 'stage3/evaluate' not in events
    assert json.loads((job.output/'hpo_run.json').read_text())['status']=='completed'
    # A failed dependency cannot launch any Stage3 work and keeps a failed record.
    fail=True
    failed_job=replace(job,trial='failed',output=tmp_path/'outputs/HPO/failed',stage2_output=tmp_path/'outputs/HPO/failed_source')
    events.clear()
    with pytest.raises(RuntimeError,match='broken Stage2'):
        runner.run_job(failed_job,'cuda:0')
    assert 'stage3/train' not in events and 'stage3/evaluate' not in events
    assert json.loads((failed_job.output/'hpo_run.json').read_text())['status']=='failed'


def test_dry_run_executes_no_subprocess_and_writes_no_cache(tmp_path,monkeypatch):
    import _common as runner
    from types import SimpleNamespace
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    a=tmp_path/'configs/stage2.yaml';b=tmp_path/'configs/stage3.yaml'
    write_yaml(a,read_yaml(ROOT/'configs/HPO/HoME/stage2/G0_C05.yaml'))
    write_yaml(b,read_yaml(ROOT/'configs/HPO/HoME/stage3/G0_C05.yaml'))
    identity=semantic_identity('test.prepare',{'split':42})
    cfg=SimpleNamespace(data=SimpleNamespace(artifacts_dir=tmp_path/'outputs/HPO/cache/stage2/42/artifacts'))
    monkeypatch.setattr(runner,'cache_identity',lambda *args:(cfg,identity))
    def forbidden(*args,**kwargs): raise AssertionError('Dry Run launched a subprocess')
    monkeypatch.setattr(runner.subprocess,'run',forbidden)
    monkeypatch.setattr(runner.subprocess,'Popen',forbidden)
    job=Job('G0_C05','HoME',tmp_path/'outputs/HPO/HoME/stage3/G0_C05',config2=a,config3=b,
            stage2_output=tmp_path/'outputs/HPO/HoME/stage2/G0_C05')
    runner.run_job(job,'cuda:2',dry_run=True)
    assert not (tmp_path/'outputs').exists()


def test_native_prepare_audit_can_live_inside_run(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    import stage2.entity_cache
    spec=importlib.util.spec_from_file_location('hpo_prepare_entry',ROOT/'scripts/stage2/prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    recipe=load_home_recipe(ROOT/'configs/v4/stage2/base.yaml')
    features=tmp_path/'features';features.mkdir()
    (features/'metadata.json').write_text(json.dumps({'semantic':{'identities':{'feature':semantic_identity('test.feature',{})}}}))
    config=replace(recipe.stage2,data=replace(recipe.stage2.data,pretrain_artifacts_dir=features),
                   training=replace(recipe.stage2.training,device='cpu'))
    recipe=replace(recipe,stage2=config)
    cfg=tmp_path/'config.yaml';write_yaml(cfg,recipe.to_dict())
    output=tmp_path/'prepare'
    dataset=SimpleNamespace(split_path=lambda *args:tmp_path/'source.csv',resource_manifest_path=lambda *args:None)
    registry=SimpleNamespace(tasks=[SimpleNamespace(dataset=dataset)])
    monkeypatch.setattr(module,'load_stage2_config',lambda *args:config)
    monkeypatch.setattr(module,'load_home_recipe',lambda *args:recipe)
    monkeypatch.setattr(type(config),'resolved_registry',lambda *args:registry)
    monkeypatch.setattr(module,'load_stage2_registry',lambda *args,**kw:registry)
    monkeypatch.setattr(module,'build_stage2_data_identity',lambda *args:semantic_identity('test.data',{}))
    events=[]
    def open_run(**kwargs):
        assert not output.exists()
        output.mkdir();events.append('open')
        return SimpleNamespace(artifacts=output/'artifacts',complete=lambda _:events.append('complete'),fail=lambda:None)
    def write_audit(*args,**kwargs):
        assert output.exists() and events==['open']
        Path(kwargs['output_path']).write_text('{}');events.append('audit')
    monkeypatch.setattr(module,'open_run_directory',open_run)
    monkeypatch.setattr(module,'write_data_identity',write_audit)
    monkeypatch.setattr(module,'repository_relative',str)
    monkeypatch.setattr(module,'prepare_stage2_data',lambda _:events.append('prepare') or {})
    monkeypatch.setattr(stage2.entity_cache,'prepare_frozen_entities',lambda _:None)
    monkeypatch.setattr(sys,'argv',['prepare.py','--config',str(cfg),'--output',str(output),
                                  '--data-metadata-output',str(output/'data_metadata.json')])
    module.main()
    assert events==['open','audit','prepare','complete']


def test_queue_interrupt_cancels_unstarted_work(tmp_path):
    import _common as runner
    jobs=[Job(str(i),'test',tmp_path/str(i)) for i in range(30)]
    seen=[]
    def work(job,device):
        seen.append(job.trial)
        if job.trial=='0': raise KeyboardInterrupt
        time.sleep(.02)
    with pytest.raises(KeyboardInterrupt):
        run_queue(jobs,('cuda:0',),1,work,record_failures=False)
    assert len(seen)<len(jobs) and runner.STOP.is_set()
