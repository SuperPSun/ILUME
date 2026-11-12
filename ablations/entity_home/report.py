"""Build independent ablation analyses from verified v4 summaries without changing the active snapshot schema."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from common.io import atomic_json
from stage3.config import load_stage3_config


def paired_cluster_interval(base, ablated, groups, *, seed=42, repeats=10000):
    """Compute system-clustered paired intervals for fixed predictions; CV folds are not independent repetitions."""
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


def comparison_rows(snapshot, authority, *, base_run=None, repository_root=None):
    results = []
    fold_cache = {}
    def folds(row):
        source = row.get('source_run')
        if repository_root is None or not source:
            return None
        if source not in fold_cache:
            values = {}
            for relative in source.split(';'):
                path = (Path(repository_root) / relative).resolve()
                if not path.is_relative_to(Path(repository_root).resolve()):
                    raise ValueError('Ablation fold source escapes repository')
                summary = json.loads((path / 'summary.json').read_text())
                reporting = summary['reporting']
                if reporting.get('benchmark') != 'stage3_property':
                    fold_cache[source] = None
                    return None
                fold = reporting['protocol']['fold']
                if summary['split'] != 'valid' or fold in values:
                    raise ValueError('Ablation fold source is not unique validation')
                values[fold] = summary['tasks']
            if set(values) != set(range(1, 6)):
                raise ValueError('Ablation comparison requires all five validation folds')
            fold_cache[source] = values
        return fold_cache[source]
    for split, section, suffix in [('valid','stage3_validation','_mean'),('test','stage3_test','')]:
        rows = snapshot['metrics'][section]
        if not rows:
            continue
        leaders = snapshot['leaderboards'][section]
        baselines = [leader['run'] for leader in leaders
                     if (leader['run']==base_run if base_run is not None else leader['model']=='ILUME')]
        if len(baselines)!=1:
            raise ValueError('Ablation analysis requires one explicit comparable Base per split/seed')
        baseline = {row['task']:row for row in rows if row['run']==baselines[0]}
        if split=='valid' and set(baseline)!=set(authority.tasks):
            raise ValueError(f'Ablation validation requires the complete {len(authority.tasks)}-task Base')
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
                fold_counts = {'fold_wins': None, 'fold_losses': None, 'fold_ties': None}
                if split == 'valid':
                    first, second = folds(reference), folds(row)
                    if first is not None and second is not None:
                        pairs = [(first[f][task], second[f][task]) for f in range(1, 6)]
                        if any(a['count'] != b['count'] or a['count'] <= 0
                               or not math.isfinite(a['normalized_mae']) or not math.isfinite(b['normalized_mae'])
                               for a, b in pairs):
                            raise ValueError('Ablation fold coverage or metric mismatch')
                        if (not math.isclose(sum(a['normalized_mae'] for a, _ in pairs)/5, base, rel_tol=1e-9, abs_tol=1e-12)
                                or not math.isclose(sum(b['normalized_mae'] for _, b in pairs)/5, value, rel_tol=1e-9, abs_tol=1e-12)):
                            raise ValueError('Ablation fold metrics differ from verified snapshot')
                        fold_counts = {'fold_wins': sum(b['normalized_mae'] < a['normalized_mae'] for a, b in pairs),
                                       'fold_losses': sum(b['normalized_mae'] > a['normalized_mae'] for a, b in pairs),
                                       'fold_ties': sum(b['normalized_mae'] == a['normalized_mae'] for a, b in pairs)}
                topology = ('ionic_liquid_partner' if recipe.partner_mode == 'interaction' else 'ionic_liquid') if len(recipe.primary_slots) == 2 else (
                    'solute_solvent' if recipe.partner_mode == 'interaction' else 'single_molecule')
                results.append({'split':split,'run':run,'model':row['model'],'base_run':baselines[0],
                    'task':task,'group':recipe.meta_group,'size_class':recipe.size_class,'topology':topology,
                    **{metric:row[metric+suffix] for metric in ('mae','rmse','r2','normalized_mae')},
                    'base_normalized_mae':base,'delta_normalized_mae':value-base,
                    'relative_change':(value-base)/base if base else None,
                    'outcome':'win' if value<base else 'loss' if value>base else 'tie',
                    'fold_std':row.get('normalized_mae_std') if split=='valid' else None, **fold_counts})
    if not results:
        raise ValueError('No complete experimental ablation results')
    return results


def stratify(rows):
    buckets = {}
    for row in rows:
        for kind,label in [('all','experimental' if row['split']=='valid' else 'test_eligible'),('group',row['group']),('size_class',row['size_class']),('topology',row['topology'])]:
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
        if summary.get('kind') in {'ilume_stage2_token_home_final_v4', 'ilume_stage2_entity_home_final_v4'}:
            performance = [json.loads(line) for line in (path.parent/'performance.jsonl').read_text().splitlines() if line]
            if [row['epoch'] for row in performance] != list(range(1, summary['fixed_final_epoch']+1)):
                raise ValueError('Stage2 cost report requires complete epoch performance logs')
            gpu_seconds = sum(row['epoch_seconds'] for row in performance if row['train_peak_allocated_bytes'] > 0)
            entry = {'stage':'stage2','training_identity':summary['training_identity']['hash'],
                     'source_run':str(path.parent.relative_to(repository_root)) if repository_root is not None else path.parent.name,
                     'epochs':len(performance),'gpu_hours':gpu_seconds/3600,
                     'timing_scope':'epoch_wall_clock_including_validation_and_checkpoint',
                     'home_parameters':performance[-1]['home_parameter_count'],
                     'owner_parameters':performance[-1].get('owner_parameter_counts'),
                     'peak_allocated_bytes':max(row['train_peak_allocated_bytes'] for row in performance),
                     **{field:sum(row[field] for row in performance) for field in
                        ('train_seconds','validation_seconds','checkpoint_seconds','epoch_seconds')}}
            if 'entity_type_training' in performance[0]:
                entry['entity_type_training'] = {
                    owner:{kind:{'samples_per_epoch':counts['samples'],
                                 'entity_occurrences_per_epoch':counts['entity_occurrences'],'tasks':counts['tasks'],
                                 'optimizer_steps':sum(row['entity_type_training'][owner][kind]['optimizer_steps'] for row in performance)}
                           for kind,counts in kinds.items()}
                    for owner,kinds in performance[0]['entity_type_training'].items()}
            result.append(entry)
            continue
        if 'training_cost' not in summary:
            continue
        plan_path=path.parent/'resolved_training_plan.json'
        if not plan_path.is_file():
            manifest_path = path.parent / 'checkpoint.json'
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get('kind') in {'ilume_stage3_single_task_mlp_model_entity_v4', 'ilume_stage3_single_task_mlp_model_entity_token_v4'}:
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
    rows=comparison_rows(snapshot,authority,base_run=base_run,repository_root=repository_root)
    strata=stratify(rows)
    costs=training_costs(input_roots, repository_root)
    intervals = bootstrap_rows(snapshot, rows, authority, Path(repository_root)) if bootstrap else []
    output.mkdir(parents=True)
    atomic_json(output/'analysis.json',{'schema_version':1,'protocol':{'split_seed':authority.data.seed,'folds':[1,2,3,4,5],
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
