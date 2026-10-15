"""从已校验的 v4 汇总生成独立消融分析，不改变现役快照结构。"""
import csv
import json
import math
from pathlib import Path

import numpy as np

from common.io import atomic_json
from stage3.config import load_stage3_config


def paired_cluster_interval(base, ablated, groups, *, seed=42, repeats=10000):
    """固定预测上的体系聚类配对区间；不把 CV 折当独立重复。"""
    base, ablated = np.asarray(base, dtype=float), np.asarray(ablated, dtype=float)
    if base.shape != ablated.shape or base.shape != (len(groups),) or not np.isfinite(base).all() or not np.isfinite(ablated).all():
        raise ValueError('Paired errors/groups must have identical finite row coverage')
    buckets = {}
    for difference, group in zip(ablated-base, groups, strict=True):
        total, count = buckets.get(group, (0.,0))
        buckets[group] = (total+float(difference),count+1)
    if not buckets:
        raise ValueError('Paired bootstrap has no systems')
    totals, counts = np.asarray(list(buckets.values())).T
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0,len(totals),len(totals))
        values.append(float(totals[indices].sum()/counts[indices].sum()))
    low, high = np.quantile(values,[.025,.975])
    return {'delta_mae': float((ablated-base).mean()), 'ci95': [float(low),float(high)],
            'systems':len(totals),'rows':len(base),'seed':seed,'repeats':repeats,
            'interpretation':'conditional_on_fixed_predictions','estimator':'pooled_row_mae'}


def comparison_rows(snapshot, authority, *, base_run=None):
    results = []
    for split, section, suffix in [('valid','stage3_validation','_mean'),('test','stage3_test','')]:
        rows = snapshot['metrics'][section]
        if not rows:
            continue
        leaders = snapshot['leaderboards'][section]
        baselines = [leader['run'] for leader in leaders if leader['model']=='ILUME' and (base_run is None or leader['run']==base_run)]
        if len(baselines)!=1:
            raise ValueError('Ablation analysis requires one explicit comparable Base per split/seed')
        baseline = {row['task']:row for row in rows if row['run']==baselines[0]}
        if split=='valid' and set(baseline)!=set(authority.tasks):
            raise ValueError('Ablation validation requires the complete 24-task Base')
        for run in sorted({row['run'] for row in rows}):
            current = {row['task']:row for row in rows if row['run']==run}
            if set(current)!=set(baseline):
                raise ValueError('Ablation/Base task sets differ')
            for task, row in current.items():
                recipe = authority.tasks[task]
                reference = baseline[task]
                value, base = float(row['normalized_mae'+suffix]),float(reference['normalized_mae'+suffix])
                if not math.isfinite(value) or not math.isfinite(base):
                    raise ValueError('Non-finite ablation MAE')
                results.append({'split':split,'run':run,'model':row['model'],'base_run':baselines[0],
                    'task':task,'group':recipe.meta_group,'size_class':recipe.size_class,
                    **{metric:row[metric+suffix] for metric in ('mae','rmse','r2','normalized_mae')},
                    'base_normalized_mae':base,'delta_normalized_mae':value-base,
                    'relative_change':(value-base)/base if base else None,
                    'outcome':'win' if value<base else 'loss' if value>base else 'tie',
                    'fold_std':row.get('normalized_mae_std') if split=='valid' else None})
    if not results:
        raise ValueError('No complete experimental ablation results')
    return results


def stratify(rows):
    buckets = {}
    for row in rows:
        for kind,label in [('all','24_experimental' if row['split']=='valid' else 'test_eligible'),('group',row['group']),('size_class',row['size_class'])]:
            buckets.setdefault((row['split'],row['run'],kind,label),[]).append(row)
    return [{'split':split,'run':run,'stratum':kind,'label':label,'tasks':len(values),
             'macro_normalized_mae':sum(row['normalized_mae'] for row in values)/len(values),
             'delta_macro_normalized_mae':sum(row['delta_normalized_mae'] for row in values)/len(values),
             **{outcome:sum(row['outcome']==outcome for row in values) for outcome in ('win','loss','tie')}}
            for (split,run,kind,label),values in sorted(buckets.items())]


def training_costs(input_roots, repository_root=None):
    result=[]
    roots = {Path(root) for root in input_roots}
    if repository_root is not None:
        for root in tuple(roots):
            for metadata_path in root.rglob('metadata.json'):
                metadata = json.loads(metadata_path.read_text())
                source = metadata.get('provenance', {}).get('checkpoint_dir')
                if metadata.get('operation') == 'evaluate' and source:
                    source = (Path(repository_root) / source).resolve()
                    if not source.is_relative_to(Path(repository_root).resolve()):
                        raise ValueError('Training-cost source escapes repository')
                    roots.add(source)
    paths = sorted({path for root in roots for path in root.rglob('summary.json')})
    for path in paths:
        summary=json.loads(path.read_text())
        if 'training_cost' not in summary:
            continue
        plan_path=path.parent/'resolved_training_plan.json'
        if not plan_path.is_file():
            manifest_path = path.parent / 'checkpoint.json'
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get('kind') == 'ilume_stage3_single_task_mlp_model_entity_v4':
                    cost = summary['training_cost']
                    payload = manifest['training_identity']['payload']
                    result.append({'ablation':'single_task_mlp','fold':payload['fold'],'task':payload['task_id'],
                        **cost,'gpu_hours':cost['gpu_seconds']/3600,
                        'owner_updates':{'PRIVATE':cost['optimizer_updates']},
                        'sample_exposures':{payload['task_id']:cost['sample_exposures']}})
            continue
        plan=json.loads(plan_path.read_text())
        exposures, updates = {}, {}
        for log in path.parent.rglob('diagnostics.jsonl'):
            diagnostics=[json.loads(line) for line in log.read_text().splitlines() if line]
            for row in diagnostics:
                for task,values in row.get('epoch_gradient_stats',{}).get('tasks',{}).items():
                    exposures[task]=exposures.get(task,0)+values['samples']
            if diagnostics:
                for owner,count in diagnostics[-1].get('owner_updates',{}).items():
                    updates[owner]=updates.get(owner,0)+count
        cost=summary['training_cost']
        result.append({'fold':summary['fold'],'ablation':plan.get('ablation',{}).get('variant','base'),
                       'training_seed':plan.get('training_seed',42),**cost,
                       'gpu_hours':cost['gpu_seconds']/3600,
                       'owner_parameters':plan['parameter_counts'].get('by_owner'),
                       'owner_updates':updates,'sample_exposures':exposures})
    return result


def write_analysis(snapshot, authority_config, output, *, base_run=None, input_roots=(), repository_root=None, bootstrap=False):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Ablation analysis output already exists')
    authority=load_stage3_config(authority_config)
    rows=comparison_rows(snapshot,authority,base_run=base_run)
    strata=stratify(rows)
    costs=training_costs(input_roots, repository_root)
    intervals = bootstrap_rows(snapshot, rows, authority, Path(repository_root)) if bootstrap else []
    output.mkdir(parents=True)
    atomic_json(output/'analysis.json',{'schema_version':1,'protocol':{'split_seed':42,'folds':[1,2,3,4,5],
        'selection':'fixed_final_state','fold_std':'split_variation_not_independent_repeats'},
        'tasks':rows,'strata':strata,'training_costs':costs,'paired_bootstrap':intervals})
    for name,values in [('task_differences',rows),('stratified_differences',strata)]:
        with (output/(name+'.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (output/'README.md').write_text('本报告只比较通过现役完整性校验的实验结果。差值为消融减 Base，负值表示改善。\n\n五折标准差反映划分差异；删专家包含容量及迁移参数变化；阶段比较包含训练量变化；独立 MLP 包含架构及预算变化。模拟结果保持独立榜单。\n')
    return rows


def prediction_rows(source_runs, task, split, repository_root):
    """Read SHA-bound CSVs, including benchmark aggregate source-run references."""
    from common.io import sha256_file
    result = {}
    for relative in source_runs.split(';'):
        root = (repository_root/relative).resolve()
        if not root.is_relative_to(repository_root.resolve()):
            raise ValueError('Prediction run escapes repository')
        summary = json.loads((root/'summary.json').read_text())
        reporting = summary['reporting']
        if 'source_runs' in reporting:
            references = reporting['source_runs']['stage3_validation' if split=='valid' else 'stage3_test'][task]
            sources = references.values() if isinstance(references,dict) else (references,)
            for source in sources:
                nested = prediction_rows(source,task,split,repository_root)
                if result.keys() & nested.keys():
                    raise ValueError('Duplicate paired prediction row')
                result.update(nested)
            continue
        manifests = [value for value in reporting['predictions'] if value['task']==task]
        if len(manifests)!=1:
            raise ValueError('Paired predictions require one task manifest')
        manifest=manifests[0]
        path=(root/manifest['path']).resolve()
        if not path.is_relative_to(root) or sha256_file(path)!=manifest['sha256']:
            raise ValueError('Paired prediction SHA mismatch')
        with path.open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        if len(rows)!=manifest['rows']:
            raise ValueError('Paired prediction coverage mismatch')
        for row in rows:
            key=(int(row.get('source_fold') or 0),int(row['source_row']))
            if key in result:
                raise ValueError('Duplicate paired prediction row')
            row['prediction']=float(row.get('prediction_ensemble',row.get('prediction')))
            row['target']=float(row['target'])
            result[key]=row
    return result


def bootstrap_rows(snapshot, comparisons, authority, repository_root):
    from stage3.data import resolve_task_registry, canonicalize_smiles
    specs=resolve_task_registry(authority)
    sources={(split,row['run'],row['task']):row['source_run']
             for split,section in [('valid','stage3_validation'),('test','stage3_test')]
             for row in snapshot['metrics'][section]}
    cache={}
    def get(split,run,task):
        key=(split,run,task)
        if key not in cache:
            cache[key]=prediction_rows(sources[key],task,split,repository_root)
        return cache[key]
    intervals=[]
    for comparison in comparisons:
        split,run,task=comparison['split'],comparison['run'],comparison['task']
        if run==comparison['base_run']:
            continue
        base=get(split,comparison['base_run'],task); ablated=get(split,run,task)
        if base.keys()!=ablated.keys():
            raise ValueError('Bootstrap sources/folds/rows differ')
        spec=specs[task]
        columns=spec.primary_slots if spec.split_strategy=='il' else spec.identity_columns
        groups,base_errors,ablation_errors=[],[],[]
        for key in sorted(base):
            first,second=base[key],ablated[key]
            if not math.isclose(first['target'],second['target'],rel_tol=1e-6,abs_tol=1e-6):
                raise ValueError('Bootstrap targets differ')
            for column in spec.identity_columns:
                if canonicalize_smiles(first[column],task)!=canonicalize_smiles(second[column],task):
                    raise ValueError('Bootstrap molecular sources differ')
            groups.append(tuple(canonicalize_smiles(first[column],task) for column in columns))
            base_errors.append(abs(first['prediction']-first['target']))
            ablation_errors.append(abs(second['prediction']-second['target']))
        intervals.append({'split':split,'run':run,'task':task,
                          **paired_cluster_interval(base_errors,ablation_errors,groups)})
    return intervals
