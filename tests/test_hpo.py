"""Behavioral checks for static research recipes, cache integrity and local scheduling."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import time

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/HPO'))
from _configs import (generate_initial, generate_stage2, generate_stage3, generate_final,
                      read_yaml, write_yaml, validate_pair, stage3_recipes, GROUP_VARIANTS)
from _common import Job, run_queue
from collect import final_statistics, metric_row
from common.artifact_cache import artifact_cache
from common.identity import semantic_identity
from stage2.home_config import load_home_recipe, home_recipe_from_dict, effective_training_seed
from stage3.config import load_stage3_config, stage3_config_from_dict


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
    identity=semantic_identity('test.prepare', {'split':42,'source':'frozen'})
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
    other=semantic_identity('test.prepare',{'split':123,'source':'frozen'})
    assert artifact_cache(path,other,None,dry_run=True)=='INVALID'
    with pytest.raises(ValueError): artifact_cache(path,other,build)
    (path/'data').write_text('damaged')
    assert artifact_cache(path,identity,None,dry_run=True)=='INVALID'
    with pytest.raises(ValueError): artifact_cache(path,identity,build)


def test_queue_device_slots_global_limit_and_failures(tmp_path):
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
    assert run_queue(jobs,('cuda:0','cuda:1','cuda:2','cuda:3'),2,work,record_failures=False)==['3']
    assert max(peak)==2
    for job in jobs:
        assert events.index((job.trial,'stage2'))<events.index((job.trial,'stage3'))


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
