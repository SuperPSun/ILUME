"""现役消融的科研配置、输入身份及统计边界。"""
from copy import deepcopy
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


@pytest.mark.parametrize('config_root,authority', [
    ('v4', 'base'), ('v4_3407', 'optimized_3407'),
])
def test_registered_ablation_contracts_are_paired_and_fixed_final(config_root, authority):
    configs=ROOT/'configs/ablations'/config_root
    base1=yaml.safe_load((ROOT/'configs/v4/stage1/base.yaml').read_text())
    base2=load_home_recipe(ROOT/f'configs/v4/stage2/{authority}.yaml')
    base3=load_stage3_config(ROOT/f'configs/v4/stage3/{authority}.yaml')
    assert len(list(configs.rglob('*.yaml')))==36
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
        calls.append(config)
    monkeypatch.setattr(report, 'write_analysis', analyze)
    monkeypatch.setattr('sys.argv', ['summarize.py','--input','outputs/ablations/v4_3407',
        '--output','summary_ablation_cli_test','--ablation-analysis-output','analysis_ablation_cli_test',
        '--ablation-authority-config',authority,'--ablation-bootstrap'])
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
        rows.extend({'run':run,'model':model,'task':task,'normalized_mae_mean':value,
                     'mae_mean':value,'rmse_mean':value,'r2_mean':.5,'normalized_mae_std':.01}
                    for task in authority.tasks)
    snapshot={'metrics':{'stage3_validation':rows,'stage3_test':[]},
              'leaderboards':{'stage3_validation':[{'run':'base','model':'ILUME'}],'stage3_test':[]}}
    result=comparison_rows(snapshot,authority)
    summaries=stratify(result)
    macro=next(row for row in summaries if row['run']=='no_group' and row['stratum']=='all')
    assert macro['tasks']==24 and macro['loss']==24
    assert macro['delta_macro_normalized_mae']==pytest.approx(.1)
    assert len([row for row in summaries if row['run']=='no_group' and row['stratum']=='group'])==len({task.meta_group for task in authority.tasks.values()})
    write_analysis(snapshot, authority_path, tmp_path/'analysis')
    analysis=json.loads((tmp_path/'analysis/analysis.json').read_text())
    assert analysis['protocol']['split_seed']==authority.data.seed
    snapshot['metrics']['stage3_validation']=rows[1:]
    with pytest.raises(ValueError,match='24-task'):
        comparison_rows(snapshot,authority)


def test_entity_mlp_fixed_final_state_reloads_with_worsening_validation(monkeypatch, tmp_path):
    from dataclasses import replace
    from ablations.stage3_single_task_mlp import adapter
    from ablations.stage3_single_task_mlp.model import Stage3SingleTaskMLP
    from common.identity import tensor_state_hash
    config=load_benchmark_config(CONFIGS/'core/single_task_mlp.yaml')
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
    assert adapter._checkpoint_kind(config)=='ilume_stage3_single_task_mlp_model_entity_v4'
