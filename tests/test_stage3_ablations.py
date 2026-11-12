"""Scientific configs, input identities, and statistical boundaries of the active ablations."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from ablations.entity_home.config import load_research_config, research_config_from_dict
from ablations.entity_home.report import paired_cluster_interval, comparison_rows, stratify
from benchmarks.common.config import load_benchmark_config
from stage2.home_config import load_home_recipe
from stage3.config import load_stage3_config
from stage1.config import load_config as load_stage1_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'configs/ablations/v4'


def test_current_token_ablation_configs_preserve_base_and_retire_removed_experiments():
    configs = ROOT/'configs/ablations/token_attention_3407'
    base1 = yaml.safe_load((ROOT/'configs/v4/stage1/base-100h.yaml').read_text())
    base2 = load_home_recipe(ROOT/'configs/v4/stage2/base.yaml')
    base3 = load_stage3_config(ROOT/'configs/v4/stage3/base.yaml')
    assert len(list(configs.rglob('*.yaml'))) == 31
    assert len(base3.tasks) == 22 and base3.model.expert_architecture == 'entity_token_attention_v1'
    for path in configs.rglob('stage1.yaml'):
        actual = yaml.safe_load(path.read_text())
        expected = deepcopy(base1)
        name = path.parent.name
        removed = {'no_unimol': ('unimol',), 'no_electronic': ('electronic',),
                   'no_partial_charge': ('partial_charge',), 'no_alignment': ('alignment',),
                   'no_descriptor_loss': ('descriptor',), 'no_stage1': (),
                   'no_smiles': ('smiles', 'alignment'), 'no_graph': ('atom', 'bond', 'partial_charge', 'alignment')}[name]
        for loss in removed:
            expected['loss']['lambda_' + loss] = 0.
        if name in {'no_smiles', 'no_graph'}:
            expected['ablation'] = {'variant': name}
        assert actual == expected
        from stage1.config import config_from_dict
        config = load_stage1_config(path)
        assert config_from_dict(config.to_dict()) == config
    for path in configs.rglob('stage2.yaml'):
        recipe = load_home_recipe(path)
        downstream = load_research_config(path.with_name('stage3.yaml'))
        assert recipe.stage3 == base2.stage3 and recipe.stage2.training == base2.stage2.training
        assert recipe.stage2.data.seed == 3407
        assert recipe.stage2.initialization.checkpoint == downstream.initialization.stage1_encoder
        assert recipe.stage2.data.artifacts_dir == downstream.initialization.simulation_artifacts_dir
    for path in configs.rglob('stage3.yaml'):
        config = load_research_config(path)
        assert config.tasks == base3.tasks and config.groups == base3.groups
        assert config.training == base3.training and config.model == base3.model
        assert research_config_from_dict(config.to_dict()) == config
        assert config.runtime_backend.final_kind == 'ilume_stage3_token_home_ablation_final_v1'
    mlp = load_benchmark_config(configs/'core/single_task_mlp.yaml')
    assert mlp.name == 'ilume_stage3_single_task_mlp_entity_token_v4'
    assert mlp.data.stage3_authority_config == Path('configs/v4/stage3/base.yaml')
    assert mlp.training['max_epochs'] == 10 and not mlp.simulation.enabled
    for variant in ('uniform_gate', 'chemberta2_mtr', 'chemberta2_mlm'):
        bad = yaml.safe_load((configs/'core/no_group/stage3.yaml').read_text())
        bad['ablation']['variant'] = variant
        with pytest.raises(ValueError, match='Unknown'):
            research_config_from_dict(bad)
        assert not any((ROOT/'configs/ablations').glob(f'*/core/{variant}/*.yaml'))


@pytest.mark.parametrize('config_root,authority', [
    ('v4', 'base'), ('v4_3407', 'optimized_3407'),
])
def test_registered_ablation_contracts_are_paired_and_fixed_final(config_root, authority):
    configs=ROOT/'configs/ablations'/config_root
    base1=yaml.safe_load((ROOT/'configs/v4/stage1/base.yaml').read_text())
    # Existing ablations retain their pre-ADR-0112 full217 descriptor recipe.
    base1['descriptor'].pop('excluded_names', None)
    base1['descriptor'].pop('value_policy', None)
    # Historical suites do not filter the Stage1 training index.
    base1['training'].pop('max_graph_nodes', None)
    # These suites also retain the pre-ADR-0116 residual D-MPNN architecture.
    base1['model'].pop('graph_encoder_type')
    base1['model'].pop('graph_feedforward_dim')
    base1['model']['graph_message_mode']='residual_blocks'
    base1['auxiliary'].pop('graph_topology_cache')
    base1['training'].pop('dgt_query_chunk_size')
    base1['training'].pop('dgt_attention_checkpointing')
    base1['training'].pop('dgt_attention_compile')
    # Historical suites retain the global batch used before the Base reduction.
    base1['training']['batch_size']=512
    base2=load_home_recipe(ROOT/f'configs/v4/stage2/{authority}.yaml')
    base3=load_stage3_config(ROOT/f'configs/v4/stage3/{authority}.yaml')
    if config_root == 'v4':
        # The split42 suite remains paired with its pre-promotion G0_C05 recipe.
        base2=load_home_recipe(ROOT/'configs/HPO/HoME/stage2/G0_C05.yaml')
        base2=replace(base2, stage2=replace(base2.stage2,
            training=replace(base2.stage2.training, seed=None)))
        historical=load_stage3_config(ROOT/'configs/HPO/HoME/stage3/G0_C05.yaml')
        base3=replace(historical, training=replace(historical.training, seed=None),
            data=replace(historical.data, artifacts_dir=Path('outputs/v4/stage3/base/prepare/artifacts')))
    assert len(list(configs.rglob('*.yaml')))==35
    cancelled=list((configs/'capacity').rglob('*.yaml')) if config_root=='v4_3407' else []
    assert len(list(configs.rglob('*.yaml')))-len(cancelled)==(31 if config_root=='v4_3407' else 35)
    assert not (configs/'repeats').exists()
    for name,removed in {'no_smiles':('smiles','alignment'), 'no_graph':('atom','bond','partial_charge','alignment')}.items():
        config=load_stage1_config(configs/f'core/{name}/stage1.yaml')
        reference=deepcopy(base1)
        reference['ablation']={'variant':name}
        for key in removed:
            reference['loss']['lambda_'+key]=0.
        assert yaml.safe_load((configs/f'core/{name}/stage1.yaml').read_text())==reference
        assert config.ablation.variant==name and config.training.epochs==10
        recipe=load_home_recipe(configs/f'core/{name}/stage2.yaml')
        downstream=load_research_config(configs/f'core/{name}/stage3.yaml')
        assert recipe.stage2.initialization.checkpoint==downstream.initialization.stage1_encoder
        assert downstream.ablation.variant==name
    expected={'no_unimol':'lambda_unimol','no_electronic':'lambda_electronic',
              'no_partial_charge':'lambda_partial_charge','no_alignment':'lambda_alignment','no_descriptor_loss':'lambda_descriptor'}
    for name,key in expected.items():
        changed=yaml.safe_load((configs/f'auxiliary/{name}/stage1.yaml').read_text())
        reference=deepcopy(base1);reference['loss'][key]=0.
        assert changed==reference
        config=load_research_config(configs/f'auxiliary/{name}/stage3.yaml')
        assert config.ablation.supervision_removed==key
    for path in configs.rglob('stage2.yaml'):
        recipe=load_home_recipe(path)
        assert recipe.stage2.data.data_root==base2.stage2.data.data_root
        assert recipe.stage2.data.seed==base2.stage2.data.seed
        assert recipe.stage2.training==base2.stage2.training
        assert recipe.stage3==base2.stage3
        downstream=load_research_config(path.with_name('stage3.yaml'))
        assert recipe.stage2.data.artifacts_dir==downstream.initialization.simulation_artifacts_dir
        assert recipe.stage2.initialization.checkpoint==downstream.initialization.stage1_encoder
    for path in configs.rglob('stage3.yaml'):
        if path in cancelled:
            with pytest.raises(ValueError,match='Capacity controls for data/3407 were cancelled'):
                load_research_config(path)
            continue
        config=load_research_config(path)
        assert config.data.seed==base3.data.seed and config.tasks==base3.tasks and config.groups==base3.groups
        assert config.data.stage3_dir==base3.data.stage3_dir
        assert config.data.task_catalog==base3.data.task_catalog
        assert config.training==base3.training
        assert config.model==base3.model and config.training.sampling_mode=='raw'
        assert config.data.split_strategies['experiment/hydration']=='random'
        assert config.initialization.simulation_artifacts_dir is not None
        if config.runtime_backend and config.ablation.zero_update:
            assert config.initialization.stage2_final is None
            recipe=load_home_recipe(config.ablation.stage2_config)
            assert (recipe.stage2.training.seed or recipe.stage2.data.seed)==(config.training.seed or config.data.seed)
    mlp=load_benchmark_config(configs/'core/single_task_mlp.yaml')
    assert mlp.name=='ilume_stage3_single_task_mlp_entity_v4' and not mlp.simulation.enabled
    assert mlp.training['model_selection']=='final_training_state' and mlp.training['max_epochs']==10
    assert mlp.seed==42
    assert mlp.data.stage3_authority_config==Path(f'configs/v4/stage3/{authority}.yaml')
    assert mlp.data.stage3_prepared_artifacts==base3.data.artifacts_dir
    bad=yaml.safe_load((configs/'core/no_stage2/stage3.yaml').read_text())
    bad['initialization']['stage2_final']='outputs/accidental_trained.pt'
    with pytest.raises(ValueError):
        research_config_from_dict(bad)


def test_entity_mlp_sweep_accepts_explicit_gpu_slots(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts.benchmarks import sweep
    config=load_benchmark_config(ROOT/'configs/ablations/v4_3407/core/single_task_mlp.yaml')
    monkeypatch.setattr(sweep, 'load_benchmark_config', lambda _:config)
    monkeypatch.setattr(sweep, 'configured_tasks', lambda c, domain:('experiment/density',) if domain=='stage3' else ())
    monkeypatch.setattr(sweep, 'ensure_benchmark_environment', lambda _:None)
    monkeypatch.setattr(sweep, 'open_run_directory', lambda **kw:SimpleNamespace(root=tmp_path, complete=lambda result:None, fail=lambda:None))
    monkeypatch.setattr(sweep, '_aggregate', lambda *a, **kw:{})
    scheduled=[]
    def schedule(**kw):
        assert kw['devices']==('cuda:0','cuda:1')
        assert len(kw['jobs'])==5 and kw['max_workers']==2
        scheduled.extend(kw['jobs'])
    monkeypatch.setattr(sweep, '_schedule', schedule)
    monkeypatch.setattr('sys.argv', ['sweep.py','--config','config.yaml','--output','unused',
        '--benchmark','stage3','--max-workers','2','--devices','cuda:0,cuda:1'])
    sweep.main()
    assert len(scheduled)==5


def test_ablation_summary_uses_selected_authority(monkeypatch):
    from scripts.benchmarks import summarize
    from ablations.entity_home import report
    authority='configs/v4/stage3/optimized_3407.yaml'
    monkeypatch.setattr(summarize, 'publish_summary', lambda *a, **kw:{'snapshot':'validated'})
    calls=[]
    def analyze(payload, config, output, **kw):
        assert payload=={'snapshot':'validated'}
        assert config==ROOT/authority and kw['bootstrap']
        assert kw['base_run']=='I4'
        calls.append(config)
    monkeypatch.setattr(report, 'write_analysis', analyze)
    monkeypatch.setattr('sys.argv', ['summarize.py','--input','outputs/ablations/v4_3407',
        '--output','summary_ablation_cli_test','--ablation-analysis-output','analysis_ablation_cli_test',
        '--ablation-authority-config',authority,'--ablation-bootstrap','--ablation-base-run','I4'])
    summarize.main()
    assert calls==[ROOT/authority]


def test_paired_bootstrap_keeps_whole_systems_and_rejects_mismatched_rows():
    base=[1,1,1,1];other=[2,2,3,3];groups=['A','A','B','B']
    first=paired_cluster_interval(base,other,groups)
    assert first==paired_cluster_interval(base,other,groups)
    assert first['systems']==2 and first['delta_mae']==1.5 and first['ci95']==[1.,2.]
    assert paired_cluster_interval(base,base,groups)['ci95']==[0.,0.]
    with pytest.raises(ValueError,match='coverage'):
        paired_cluster_interval(base,other[:-1],groups)
    with pytest.raises(ValueError,match='coverage'):
        paired_cluster_interval(base,[1,float('nan'),1,1],groups)


def test_stage2_costs_keep_type_coverage_separate_from_accuracy(tmp_path):
    import json
    from ablations.entity_home.report import training_costs
    (tmp_path/'summary.json').write_text(json.dumps({'kind':'ilume_stage2_token_home_final_v4',
        'fixed_final_epoch':2,'training_identity':{'hash':'source'}}))
    counts = {'GROUP:solvation':{'ionic':{'samples':0,'entity_occurrences':0,'tasks':[],'optimizer_steps':0},
                               'neutral':{'samples':3,'entity_occurrences':6,'tasks':['simulation/transfer_organic'],'optimizer_steps':1}}}
    rows = [{'epoch':epoch,'train_seconds':2.,'validation_seconds':1.,'checkpoint_seconds':1.,'epoch_seconds':4.,
             'home_parameter_count':100,'owner_parameter_counts':{'GROUP:solvation':100},
             'train_peak_allocated_bytes':1024,'entity_type_training':counts} for epoch in (1,2)]
    log=tmp_path/'performance.jsonl'
    log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result=training_costs([tmp_path])[0]
    assert result['stage']=='stage2' and result['gpu_hours']==pytest.approx(8/3600)
    assert result['entity_type_training']['GROUP:solvation']['ionic']['optimizer_steps']==0
    assert result['entity_type_training']['GROUP:solvation']['neutral']['optimizer_steps']==2
    assert 'normalized_mae' not in result
    log.write_text(json.dumps(rows[1])+'\n')
    with pytest.raises(ValueError,match='complete epoch'):
        training_costs([tmp_path])


@pytest.mark.parametrize('variant', ['no_smiles', 'no_graph'])
def test_single_view_supervision_and_encoder_gradients(tiny_samples, variant):
    from dataclasses import replace
    from stage1.auxiliary import empty_auxiliary_targets
    from stage1.model import build_stage1_model
    from stage1.masking import MultimodalPacker, MultimodalMasker
    from stage1.config import config_from_dict
    from stage1.gradient_audit import audit_gradient_norms
    vocabulary,samples=tiny_samples
    config=load_stage1_config(CONFIGS/f'core/{variant}/stage1.yaml')
    config=replace(config,model=replace(config.model,d_model=32,n_heads=4,smiles_layers=1,
        graph_depth=2,feedforward_dim=64,dropout=0.))
    prepared=[]
    for sample in samples:
        aux=empty_auxiliary_targets()
        aux.update(unimol=torch.ones(768),unimol_valid=torch.tensor(True),
                   electronic=torch.ones(13),electronic_valid=torch.ones(13,dtype=torch.bool),
                   partial_charge=torch.ones(len(sample['atom_categorical'])),
                   partial_charge_valid=torch.ones(len(sample['atom_categorical']),dtype=torch.bool))
        prepared.append({**{key:value for key,value in sample.items() if key!='fingerprints'},'auxiliary_targets':aux})
    batch=MultimodalPacker(vocabulary)(prepared)
    batch=MultimodalMasker(vocabulary,config.masking,42).apply(batch,evaluation=True)
    model=build_stage1_model(config,vocabulary).eval()
    output=model(batch)
    expected={'descriptor','unimol','electronic'} | ({'atom','bond','partial_charge'} if variant=='no_smiles' else {'smiles'})
    assert set(output.losses)==expected
    audit=audit_gradient_norms(model,batch,config,torch.device('cpu'),amp_enabled=False,amp_dtype=torch.bfloat16)
    assert 'alignment' not in audit['coverage']
    output.loss.backward()
    surviving=model.graph_encoder if variant=='no_smiles' else model.smiles_encoder
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in surviving.parameters())
    if variant=='no_smiles':
        assert model.partial_charge_head.weight.grad.abs().sum()>0
    raw=config.to_dict();raw['loss']['lambda_alignment']=.1
    with pytest.raises(ValueError,match='dependent supervision'):
        config_from_dict(raw)


@pytest.mark.parametrize('authority_name', ['base', 'optimized_3407'])
def test_comparison_requires_complete_base_and_reports_task_group_size(authority_name, tmp_path):
    import json
    from ablations.entity_home.report import write_analysis
    authority_path=ROOT/f'configs/v4/stage3/{authority_name}.yaml'
    authority=load_stage3_config(authority_path)
    rows=[]
    for run,model,value in [('base','ILUME',.2),('no_group','ILUME v4 no_group',.3)]:
        sources = []
        for fold in range(1, 6):
            path = tmp_path / run / f'fold{fold}'
            path.mkdir(parents=True)
            (path / 'summary.json').write_text(json.dumps({'split':'valid',
                'reporting':{'benchmark':'stage3_property','protocol':{'fold':fold}},
                'tasks':{task:{'count':1,'normalized_mae':value} for task in authority.tasks}}))
            sources.append(str(path.relative_to(tmp_path)))
        rows.extend({'run':run,'model':model,'task':task,'normalized_mae_mean':value,
                     'mae_mean':value,'rmse_mean':value,'r2_mean':.5,'normalized_mae_std':.01,
                     'source_run':';'.join(sources)}
                    for task in authority.tasks)
    snapshot={'metrics':{'stage3_validation':rows,'stage3_test':[]},
              'leaderboards':{'stage3_validation':[{'run':'base','model':'ILUME'}],'stage3_test':[]}}
    result=comparison_rows(snapshot,authority,repository_root=tmp_path)
    assert all(row['fold_losses']==5 for row in result if row['run']=='no_group')
    summaries=stratify(result)
    macro=next(row for row in summaries if row['run']=='no_group' and row['stratum']=='all')
    assert macro['tasks']==len(authority.tasks) and macro['loss']==len(authority.tasks)
    assert macro['delta_macro_normalized_mae']==pytest.approx(.1)
    assert len([row for row in summaries if row['run']=='no_group' and row['stratum']=='group'])==len({task.meta_group for task in authority.tasks.values()})
    assert {row['label'] for row in summaries if row['stratum']=='topology'} == {
        'ionic_liquid','ionic_liquid_partner','solute_solvent','single_molecule'}
    write_analysis(snapshot, authority_path, tmp_path/'analysis')
    analysis=json.loads((tmp_path/'analysis/analysis.json').read_text())
    assert analysis['protocol']['split_seed']==authority.data.seed
    snapshot['leaderboards']['stage3_validation'].append({'run':'no_group','model':'ILUME I4'})
    paired = comparison_rows(snapshot, authority, base_run='no_group', repository_root=tmp_path)
    assert all(row['base_run']=='no_group' for row in paired)
    assert all(row['delta_normalized_mae']==pytest.approx(-.1) for row in paired if row['run']=='base')
    assert all(row['fold_wins']==5 for row in paired if row['run']=='base')
    with pytest.raises(ValueError, match='explicit comparable Base'):
        comparison_rows(snapshot, authority, base_run='missing')
    (tmp_path / 'no_group/fold5/summary.json').unlink()
    with pytest.raises(FileNotFoundError):
        comparison_rows(snapshot, authority, base_run='no_group', repository_root=tmp_path)
    snapshot['metrics']['stage3_validation']=rows[1:]
    with pytest.raises(ValueError,match=f'{len(authority.tasks)}-task'):
        comparison_rows(snapshot,authority)


@pytest.mark.parametrize('suite', ['v4', 'token_attention_3407'])
def test_entity_mlp_fixed_final_state_reloads_with_worsening_validation(monkeypatch, tmp_path, suite):
    from dataclasses import replace
    from ablations.stage3_single_task_mlp import adapter
    from ablations.stage3_single_task_mlp.model import Stage3SingleTaskMLP
    from common.identity import tensor_state_hash
    config=load_benchmark_config(ROOT/'configs/ablations'/suite/'core/single_task_mlp.yaml')
    config=replace(config,training={**config.training,'max_epochs':3,'batch_size':2})
    model=Stage3SingleTaskMLP(2490,(1024,512),0.)
    features=torch.randn(3,2490);targets=torch.zeros(3)
    calls=0
    def worsening(model,features,**kwargs):
        nonlocal calls
        calls+=1
        return torch.full((len(features),),float(calls))
    monkeypatch.setattr(adapter,'_predict_normalized',worsening)
    history,state,epoch,_=adapter._run_training_epochs(model,features,targets,features,targets,config,
        training_seed=42,device=torch.device('cpu'),use_bf16=False,reporter=None)
    assert epoch==3 and history[-1]['valid_normalized_mae']>history[0]['valid_normalized_mae']
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in state.items())
    path=tmp_path/'mlp.pt';torch.save(state,path)
    reloaded=Stage3SingleTaskMLP(2490,(1024,512),0.)
    reloaded.load_state_dict(torch.load(path,weights_only=True),strict=True)
    assert tensor_state_hash(adapter._state_namespace(config),state)==tensor_state_hash(adapter._state_namespace(config),reloaded.state_dict())
    assert adapter._checkpoint_kind(config)==('ilume_stage3_single_task_mlp_model_entity_token_v4' if suite=='token_attention_3407' else 'ilume_stage3_single_task_mlp_model_entity_v4')
