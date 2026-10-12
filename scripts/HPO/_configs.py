"""Deterministic full YAML recipes; generation never selects a model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPLITS = (42, 123, 1234, 3407, 90756)
GROUP_VARIANTS = {
    'G0': {},
    'G1': {'transport': 'thermophysical'},
    'G2': {'electrochemical': 'phase_stability'},
    'G3': {'dielectric_optical': 'thermophysical'},
    'G4': {'transport': 'thermophysical', 'electrochemical': 'phase_stability',
           'dielectric_optical': 'thermophysical'},
}


def read_yaml(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_yaml(path, value):
    path = Path(path)
    if path.exists():
        if read_yaml(path) != value:
            raise FileExistsError(f'Configuration is frozen; use a new directory: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def bind_cache(stage, raw, split=42):
    raw['data']['seed'] = split
    raw['data']['artifacts_dir'] = f'outputs/HPO/cache/{stage}/{split}/artifacts'
    raw['preparation']['data_metadata_path'] = f'outputs/HPO/cache/{stage}/{split}/data_metadata.json'
    raw['training']['seed'] = 42
    if stage == 'stage2':
        raw['data']['data_root'] = f'data/{split}'
        raw['data']['task_catalog_path'] = f'data/{split}/task_catalog.csv'
    else:
        raw['data']['stage3_dir'] = f'data/{split}/stage3'
        raw['data']['task_catalog'] = f'data/{split}/task_catalog.csv'
        raw['preparation']['cache_dir'] = 'outputs/HPO/cache/stage3/entities'
        raw['initialization']['simulation_artifacts_dir'] = f'outputs/HPO/cache/stage2/{split}/artifacts'
        raw['training']['device'] = 'cuda'
    return raw


def validate_pair(s2, s3):
    from stage2.home_config import home_recipe_from_dict
    from stage3.config import stage3_config_from_dict
    from stage3.home import validate_transfer_architecture
    a, b = home_recipe_from_dict(s2), stage3_config_from_dict(s3)
    validate_transfer_architecture(a.stage3.model, b.model, a.stage3.groups, b.groups)
    # Interaction ownership must remain available in the two transferred groups.
    partner = {task['meta_group'] for task in s3['tasks'].values()
               if task.get('enabled', True) and task.get('partner_mode') == 'interaction'}
    if ('thermophysical' in partner or 'solvation' not in partner
            or not {'thermophysical', 'solvation'} <= {t['meta_group'] for t in s3['tasks'].values()}):
        raise ValueError('Transferred GROUP interaction topology differs from Stage2')


def stage1_recipes():
    base = read_yaml(ROOT / 'configs/v4/stage1/regression_heads.yaml')
    recipes = [deepcopy(base)]
    for lr, epochs in [(lr, ep) for lr in (1e-5, 1e-4, 1e-3) for ep in (5, 10, 30)] + [(3e-4, 20)]:
        value = deepcopy(base)
        value.update(learning_rate=lr, epochs=epochs, predictor={'type': 'linear'},
                     predictor_overrides={'partial_atomic_charge': {'type': 'linear'}})
        recipes.append(value)
    training = ((1e-4, 10, 0., .01), (3e-5, 20, .1, .01), (3e-4, 5, 0., 0.),
                (1e-4, 30, .2, .01), (1e-5, 50, .3, .1))
    for kind in ('mlp', 'residual_mlp'):
        for scale in (.5, 1., 2.):
            for depth in (1, 2, 3):
                for lr, epochs, dropout, wd in training:
                    value = deepcopy(base)
                    head = lambda width: {'type': kind, 'hidden_dims': [int(width * scale / 2**i) for i in range(depth)],
                                           'activation': 'gelu', 'dropout': dropout}
                    value.update(learning_rate=lr, epochs=epochs, weight_decay=wd, predictor=head(512),
                                 predictor_overrides={'partial_atomic_charge': head(256)})
                    if value != base:
                        recipes.append(value)
    assert len(recipes) == 100 and len({fingerprint(p) for p in recipes}) == 100
    return recipes


def generate_initial(destination=ROOT / 'configs/HPO'):
    from stage1.regression import load_regression_config
    for i, raw in enumerate(stage1_recipes()):
        path = destination / 'stage1' / f'{i:03}.yaml'
        write_yaml(path, raw)
        load_regression_config(path)
    s2base = read_yaml(ROOT / 'configs/v4/stage2/base.yaml')
    s3base = read_yaml(ROOT / 'configs/v4/stage3/base.yaml')
    for gid, mapping in GROUP_VARIANTS.items():
        for ci in range(20):
            s2, s3 = bind_cache('stage2', deepcopy(s2base)), bind_cache('stage3', deepcopy(s3base))
            count, level = ci // 5 + 1, ci % 5
            for model in (s2['home']['model'], s3['model']):
                model['global_experts'] = count
            for raw in (s2['home'], s3):
                for group in raw['groups'].values():
                    if level == 1:
                        group['experts'] = max(1, group['experts'] - 1)
                    elif level == 2:
                        group['experts'] += 1
                    elif level in (3, 4):
                        group['expert_hidden_ratio'] *= .5 if level == 3 else 2.
            for task in s3['tasks'].values():
                task['meta_group'] = mapping.get(task['meta_group'], task['meta_group'])
            # Identical Stage2 recipes across G share one canonical capacity run.
            s3['initialization']['stage2_final'] = f'outputs/HPO/HoME/stage2/G0_C{ci:02}/stage2_final.pt'
            validate_pair(s2, s3)
            trial = f'{gid}_C{ci:02}'
            write_yaml(destination / 'HoME/stage2' / f'{trial}.yaml', s2)
            write_yaml(destination / 'HoME/stage3' / f'{trial}.yaml', s3)
    write_yaml(destination / 'selected.example.yaml', {
        'home': None, 'stage2_config': None, 'stage2_checkpoint': None, 'stage3_config': None})


def selected(path, required):
    value = read_yaml(path)
    if not isinstance(value, dict) or any(not value.get(key) for key in required):
        raise ValueError(f'Manual selection required: {", ".join(required)}')
    return value


def generation_binding(selection, required, destination):
    value = {key: selection[key] for key in required}
    a, b = chosen_home(selection, destination)
    value['home_recipes'] = [fingerprint(a), fingerprint(b)]
    for key in ('stage2_config', 'stage3_config'):
        if key in required:
            value[key + '_hash'] = fingerprint(read_yaml(selection[key]))
    if 'stage2_checkpoint' in required and Path(selection['stage2_checkpoint']).is_file():
        from common.io import sha256_file
        value['stage2_checkpoint_sha256'] = sha256_file(selection['stage2_checkpoint'])
    return value


def record_generation(directory, selection, required, destination):
    path = directory / 'generation.json'
    payload = generation_binding(selection, required, destination)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('Round is bound to a different manual selection; use a new config root')
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def chosen_home(selection, destination):
    trial = selection['home']
    if trial not in {f'{g}_C{i:02}' for g in GROUP_VARIANTS for i in range(20)}:
        raise ValueError('Unknown selected HoME ID')
    return (read_yaml(destination / 'HoME/stage2' / f'{trial}.yaml'),
            read_yaml(destination / 'HoME/stage3' / f'{trial}.yaml'))


def generate_stage2(selection_path, destination=ROOT / 'configs/HPO'):
    selection = selected(selection_path, ('home',))
    a, b = chosen_home(selection, destination)
    record_generation(destination / 'stage2', selection, ('home',), destination)
    for i, (epochs, lr) in enumerate((e, lr) for e in (5, 10, 15, 20)
                                    for lr in (3e-5, 6e-5, 1e-4, 2e-4, 3e-4)):
        s2, s3 = deepcopy(a), deepcopy(b)
        s2['training'].update(epochs=epochs, task_head_learning_rate=lr)
        s3['initialization']['stage2_final'] = f'outputs/HPO/stage2/{i:03}/train/stage2_final.pt'
        validate_pair(s2, s3)
        write_yaml(destination / 'stage2' / f'{i:03}.yaml', s2)
        write_yaml(destination / 'stage2/downstream' / f'{i:03}.yaml', s3)


def private_variant(base, lr=1., epochs=1., width=1., dropout=None):
    from stage3.config import stage3_config_from_dict
    config = stage3_config_from_dict(base)
    raw = deepcopy(base)
    for task, detail in raw['tasks'].items():
        recipe = config.resolved_private_recipe(task)
        detail['phase1_private_lr'] = recipe.phase1_lr * lr
        for phase in (1, 2, 3):
            budget = getattr(recipe, f'phase{phase}_epochs')
            detail[f'phase{phase}_private_epochs'] = max(1, round(budget * epochs)) if budget else 0
        overrides = detail.setdefault('model_overrides', {})
        for field in ('private_hidden_ratio', 'tower_hidden_ratio', 'film_hidden_ratio'):
            overrides[field] = getattr(recipe, field) * width
        overrides['private_dropout'] = recipe.private_dropout if dropout is None else dropout
    return raw


def stage3_recipes(base):
    recipes = [deepcopy(base)]
    global_budget = base['training']['three_phase']['global']
    for lr in (.9, 1.2, 1.6, 2.):
        for ep in (1., 4/3, 5/3):
            raw = deepcopy(base)
            raw['training']['three_phase']['global'].update(lr=global_budget['lr']*lr, epochs=round(global_budget['epochs']*ep))
            recipes.append(raw)
    for lr in (.95, 1.05, 1.1):
        for ep in (.5, 1., 1.5, 2.):
            raw = deepcopy(base)
            for group in raw['groups'].values():
                group['phase1']['lr'] *= lr
                group['phase2']['lr'] = group['phase1']['lr'] * raw['training']['three_phase']['phase1_min_lr_ratio']
                group['phase2']['epochs'] = max(1, round(group['phase2']['epochs']*ep))
            recipes.append(raw)
    for lr in (.5, .7, .9, 1.02, 1.1):
        for ep, width in ((.5, .5), (.75, 1.), (1., 1.5)):
            recipes.append(private_variant(base, lr=lr, epochs=ep, width=width))
    for dropout in (0., .05, .1, .15):
        recipes.append(private_variant(base, dropout=dropout))
    for wd in (0., .001, .03, .1):
        raw = deepcopy(base)
        raw['training']['weight_decay'] = wd
        recipes.append(raw)
    for lr, width, wd in ((.7, .75, .001), (.9, 1.25, .03)):
        raw = private_variant(base, lr=lr, width=width, dropout=.05)
        raw['training']['weight_decay'] = wd
        recipes.append(raw)
    assert len(recipes) == 50 and len({fingerprint(r) for r in recipes}) == 50
    return recipes


def generate_stage3(selection_path, destination=ROOT / 'configs/HPO'):
    selection = selected(selection_path, ('home', 'stage2_config', 'stage2_checkpoint'))
    a, base = chosen_home(selection, destination)
    chosen = read_yaml(selection['stage2_config'])
    validate_pair(chosen, base)
    # LR/E may differ; all other Stage2 fields must match the selected architecture.
    expected, actual = deepcopy(a), deepcopy(chosen)
    for raw in (expected, actual):
        for field in ('epochs', 'task_head_learning_rate'):
            raw['training'].pop(field)
    if expected != actual:
        raise ValueError('Selected Stage2 recipe does not belong to selected HoME architecture')
    record_generation(destination / 'stage3', selection, ('home', 'stage2_config', 'stage2_checkpoint'), destination)
    base['initialization']['stage2_final'] = selection['stage2_checkpoint']
    for i, raw in enumerate(stage3_recipes(base)):
        validate_pair(chosen, raw)
        write_yaml(destination / 'stage3' / f'{i:03}.yaml', raw)


def generate_final(selection_path, destination=ROOT / 'configs/HPO'):
    selection = selected(selection_path, ('home', 'stage2_config', 'stage2_checkpoint', 'stage3_config'))
    chosen2, chosen3 = read_yaml(selection['stage2_config']), read_yaml(selection['stage3_config'])
    home2, home3 = chosen_home(selection, destination)
    validate_pair(chosen2, chosen3)
    if chosen3['initialization']['stage2_final'] != selection['stage2_checkpoint']:
        raise ValueError('Selected Stage3 references a different Stage2 checkpoint')
    if {k: v['meta_group'] for k, v in chosen3['tasks'].items()} != {k: v['meta_group'] for k, v in home3['tasks'].items()}:
        raise ValueError('Selected Stage3 grouping differs from selected HoME')
    validate_pair(home2, chosen3)
    record_generation(destination / 'final', selection, ('home', 'stage2_config', 'stage2_checkpoint', 'stage3_config'), destination)
    for split in SPLITS:
        a, b = bind_cache('stage2', deepcopy(chosen2), split), bind_cache('stage3', deepcopy(chosen3), split)
        b['initialization']['stage2_final'] = f'outputs/HPO/final/{split}/stage2/stage2_final.pt'
        validate_pair(a, b)
        write_yaml(destination / 'final' / str(split) / 'stage2.yaml', a)
        write_yaml(destination / 'final' / str(split) / 'stage3.yaml', b)
