"""Small local queue around the public Stage command-line interfaces."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import threading
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common.artifact_cache import artifact_cache, artifact_lock
from common.io import atomic_json, sha256_file
STOP = threading.Event()
CHILDREN = set()
CHILD_LOCK = threading.Lock()

from _configs import (SPLITS, fingerprint, generate_final, generate_initial, generate_stage2,
                      generate_stage3, generation_binding, read_yaml, selected, validate_pair)


@dataclass
class Job:
    trial: str
    branch: str
    output: Path
    config1: Path | None = None
    config2: Path | None = None
    config3: Path | None = None
    stage2_output: Path | None = None
    folds: tuple = (1, 3)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def native(script, arguments, log, device, *, dry_run=False):
    command = [sys.executable, f'scripts/{script}.py', *map(str, arguments)]
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        handle.write('\n' + shlex.join(command) + '\n')
        handle.flush()
        if STOP.is_set():
            raise RuntimeError('HPO interrupted')
        with CHILD_LOCK:
            if STOP.is_set():
                raise RuntimeError('HPO interrupted')
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            CHILDREN.add(process)
        try:
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
        finally:
            with CHILD_LOCK:
                CHILDREN.discard(process)


def stop_children():
    STOP.set()
    with CHILD_LOCK:
        processes = list(CHILDREN)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def cache_identity(stage, config_path):
    from common.identity import semantic_identity
    from stage1.features import validate_stage1_feature_files
    if stage == 'stage2':
        from stage2.home_config import load_home_recipe
        from stage2.identity import build_stage2_data_identity
        from stage2.registry import load_stage2_registry
        config = load_home_recipe(config_path).stage2
        registry = config.resolved_registry(load_stage2_registry(config.data.task_catalog_path, task_ids=config.data.tasks))
        metadata = json.loads((config.data.pretrain_artifacts_dir / 'metadata.json').read_text())
        feature = metadata['semantic']['identities']['feature']
        prepared = build_stage2_data_identity(config, registry, feature)
        identity = semantic_identity('hpo.stage2.prepare.v1', {
            'data': prepared['hash'], 'split_seed': config.data.seed,
            'stage1_sha256': sha256_file(config.initialization.checkpoint),
            'feature_files': validate_stage1_feature_files(config.data.pretrain_artifacts_dir)})
    else:
        from stage3.config import load_stage3_config
        from stage3.data import resolve_task_registry, source_hashes
        from stage3.prepare import prepared_artifact_kind, prepared_artifact_version
        config = load_stage3_config(config_path)
        registry = resolve_task_registry(config)
        data = config.to_dict()['data']
        data.pop('artifacts_dir')
        features = json.loads((config.initialization.stage1_artifacts_dir / 'metadata.json').read_text())
        # Do not refit five-fold scalers or canonicalize the entire corpus to decide reuse.
        # Native prepared identity/normalization is published by prepare and checked by train.
        identity = semantic_identity('hpo.stage3.prepare.v1', {
            'kind': prepared_artifact_kind(config), 'version': prepared_artifact_version(config),
            'data': data, 'sources': source_hashes(config, registry),
            'registry': {task: spec.prepared_dict() for task, spec in sorted(registry.items())},
            'stage1_feature_identity': features['semantic']['identities']['feature']['hash'],
            'stage1_sha256': sha256_file(config.initialization.stage1_encoder),
            'feature_files': validate_stage1_feature_files(config.initialization.stage1_artifacts_dir)})
    # Match the JSON representation used by cache and trial manifests (tuple -> list).
    return config, json.loads(json.dumps(identity))


def prepare(stage, config_path, device, *, dry_run=False):
    config, identity = cache_identity(stage, config_path)
    root = config.data.artifacts_dir.parent
    if not root.resolve().is_relative_to((ROOT / 'outputs/HPO/cache').resolve()):
        raise ValueError('HPO prepare writes only to outputs/HPO/cache')
    def build(staging):
        run = staging / 'run'
        arguments = ['--config', relative(config_path), '--output', relative(run),
                     '--data-metadata-output', relative(run / 'data_metadata.json')]
        if stage == 'stage3':
            # The underlying entity bank is shared across splits, with one writer.
            with artifact_lock(ROOT / 'outputs/HPO/cache/stage3/entities.lock'):
                native(f'{stage}/prepare', arguments, staging / 'prepare.log', device)
        else:
            native(f'{stage}/prepare', arguments, staging / 'prepare.log', device)
        for item in run.iterdir():
            item.rename(staging / item.name)
        run.rmdir()
        # Only execution locators change when atomically publishing the run.
        for path in staging.glob('*.json'):
            path.write_text(path.read_text().replace(relative(run), relative(root)))
    decision = artifact_cache(root, identity, build, dry_run=dry_run)
    if dry_run:
        print(f'{decision} {stage} prepare {relative(root)}', flush=True)
        if decision == 'CREATE':
            native(f'{stage}/prepare', ['--config', relative(config_path), '--output', relative(root),
                   '--data-metadata-output', relative(root / 'data_metadata.json')], root / 'prepare.log', device, dry_run=True)
        elif decision == 'INVALID':
            raise ValueError(f'Invalid cache retained: {root}')
    return identity, decision


def frozen_run(path, identity, *, resume):
    record = path / 'hpo_identity.json'
    if record.exists():
        if json.loads(record.read_text()) != identity:
            raise ValueError(f'Existing output identity differs: {path}')
        if not resume:
            raise FileExistsError(f'Use --resume for existing output: {path}')
    elif path.exists():
        raise FileExistsError(f'Existing unbound output retained: {path}')
    else:
        path.mkdir(parents=True)
        atomic_json(record, identity)


def validate_stage2_final(path, config_path):
    from stage2.home_artifact import load_home_final
    from stage2.home_config import load_home_recipe
    from stage2.home_train import resolve_training_identity
    payload, _, _ = load_home_final(path / 'stage2_final.pt')
    # Child training uses fused CUDA AdamW, irrespective of the controller's device.
    expected = resolve_training_identity(load_home_recipe(config_path), payload['training_identity']['payload']['math_contract'])
    if payload['training_identity'] != expected:
        raise ValueError('Stage2 final belongs to a different full training recipe')


def train_stage2(job, device, resume):
    path = job.stage2_output
    recipe = read_yaml(job.config2)
    identity = {'recipe': fingerprint(recipe), 'prepare': cache_identity('stage2', job.config2)[1]}
    with artifact_lock(path.with_name(path.name + '.lock')):
        marker = path / 'hpo_stage2.json'
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise ValueError('Stage2 single-writer recipe identity differs')
        if (path / 'stage2_final.pt').exists():
            validate_stage2_final(path, job.config2)
            print(f'REUSE Stage2 final {relative(path)}', flush=True)
            return
        arguments = ['--config', relative(job.config2), '--output', relative(path), '--device', device]
        if path.exists():
            if not resume:
                raise FileExistsError(f'Incomplete Stage2 output; use --resume: {path}')
            arguments += ['--resume', relative(path)]
        native('stage2/train', arguments, path.parent / (path.name + '.log'), device)
        validate_stage2_final(path, job.config2)
        atomic_json(marker, identity)


def evaluate_fold(job, fold, device, resume):
    from stage3.config import load_stage3_config
    from stage3.evaluate import resolve_stage3_evaluation_identity
    config = load_stage3_config(job.config3)
    train = job.output / 'train' if job.branch != 'HoME' else job.output
    valid = job.output / 'valid'
    expected = resolve_stage3_evaluation_identity(config, train, split='valid', ensemble_folds=False, fold=fold)
    folder = valid / f'fold{fold}'
    marker = folder / 'hpo_evaluation.json'
    prior = sorted(job.output.glob(f'valid*/fold{fold}/hpo_evaluation.json'))
    if prior:
        if not resume:
            raise FileExistsError(f'Validation output exists: {prior[0].parent}')
        for completed in prior:
            saved = json.loads(completed.read_text())
            if saved['identity'] != expected:
                raise ValueError('Validation identity differs from current final model')
            for name, digest in saved['files'].items():
                if sha256_file(completed.parent / name) != digest:
                    raise ValueError('Validation artifact hash mismatch')
        return
    if folder.exists():
        if not resume:
            raise FileExistsError(f'Validation output exists: {folder}')
        # Evaluation has no partial resume. Keep failed output and restart in a new directory.
        valid = job.output / f'valid_attempt{sum(1 for _ in job.output.glob("valid_attempt*")) + 1}'
        folder, marker = valid / f'fold{fold}', valid / f'fold{fold}' / 'hpo_evaluation.json'
    native('stage3/evaluate', ['--config', relative(job.config3), '--checkpoint-dir', relative(train),
           '--split', 'valid', '--domain', 'experimental', '--fold', fold, '--output', relative(valid)],
           job.output / f'evaluate_fold{fold}.log', device)
    files = {p.relative_to(folder).as_posix(): sha256_file(p) for p in folder.rglob('*') if p.is_file()}
    atomic_json(marker, {'identity': expected, 'files': files})


def execute_job(job, device, resume=False, dry_run=False, checkpoint=None):
    paths = [p for p in (job.config1, job.config2, job.config3) if p]
    recipes = {relative(p): read_yaml(p) for p in paths}
    from stage1.regression import load_regression_config, regression_identity, regression_representation_identity
    dependencies, caches = {}, {}
    if job.config1:
        config = load_regression_config(job.config1)
        dependencies['source'] = regression_identity(config, checkpoint)
        bank_identity = regression_representation_identity(config, checkpoint)
        bank = ROOT / 'outputs/HPO/cache/stage1' / bank_identity['hash']
        caches['stage1'] = artifact_cache(bank, bank_identity, None, dry_run=True)
        print(f'{caches["stage1"]} stage1 representations {relative(bank)}', flush=True)
        if caches['stage1'] == 'INVALID':
            raise ValueError('Invalid Stage1 representation cache')
    else:
        if job.config2:
            dependencies['stage2_cache'], caches['stage2'] = prepare('stage2', job.config2, device, dry_run=dry_run)
        if job.config3:
            # Identity/data planning is independent of an as-yet-untrained Stage2 source.
            dependencies['stage3_cache'] = cache_identity('stage3', job.config3)[1]
            source = Path(read_yaml(job.config3)['initialization']['stage2_final'])
            if not job.config2:
                from stage3.home import load_source
                from stage3.config import load_stage3_config
                load_source(load_stage3_config(job.config3))
                dependencies['stage2_source_sha256'] = sha256_file(source)
    identity = {'recipes': recipes, 'dependencies': dependencies, 'folds': list(job.folds)}
    if dry_run:
        if job.config1:
            native('stage1/regression', ['--config', relative(job.config1), '--checkpoint', checkpoint,
                   '--output', relative(job.output / 'attempt1'), '--representation-cache', 'outputs/HPO/cache/stage1',
                   '--device', 'cuda:0'], job.output / 'train.log', device, dry_run=True)
        else:
            if job.config2:
                native('stage2/train', ['--config', relative(job.config2), '--output', relative(job.stage2_output),
                       '--device', device], job.output / 'stage2.log', device, dry_run=True)
            if job.config3:
                source = Path(read_yaml(job.config3)['initialization']['stage2_final'])
                if source.is_file():
                    from stage3.home import load_source
                    from stage3.config import load_stage3_config
                    load_source(load_stage3_config(job.config3))
                else:
                    print(f'PENDING Stage2 source validation: {source}', flush=True)
                prepare('stage3', job.config3, device, dry_run=True)
                native('stage3/train', ['--config', relative(job.config3), '--fold', *job.folds,
                       '--output', relative(job.output if job.branch == 'HoME' else job.output / 'train'),
                       '--devices', device, '--max-parallel', 1], job.output / 'train.log', device, dry_run=True)
                for fold in job.folds:
                    native('stage3/evaluate', ['--config', relative(job.config3), '--checkpoint-dir',
                           relative(job.output if job.branch == 'HoME' else job.output / 'train'), '--split', 'valid',
                           '--domain', 'experimental', '--fold', fold, '--output', relative(job.output / 'valid')],
                           job.output / 'evaluate.log', device, dry_run=True)
        return
    frozen_run(job.output, identity, resume=resume)
    record = job.output / 'hpo_run.json'
    started = time.time()
    state = {'trial_id': job.trial, 'branch': job.branch, 'configs': list(recipes), 'status': 'running',
             'training_seed': read_yaml(job.config1)['seed'] if job.config1 else 42, 'folds': list(job.folds), 'device': device, 'caches': caches,
             'stage2_output': relative(job.stage2_output) if job.stage2_output else None}
    atomic_json(record, state)
    try:
        if job.config1:
            from stage1.regression import load_regression_head
            attempts = sorted(job.output.glob('attempt*'))
            for attempt in attempts:
                summary = attempt / 'summary.json'
                if summary.exists():
                    result = json.loads(summary.read_text())
                    if result.get('kind') == 'ilume_stage1_regression_summary_v4':
                        if result['identity'] != dependencies['source']:
                            raise ValueError('Stage1 completion identity differs')
                        for task in result['tasks']:
                            load_regression_head(attempt / 'tasks' / task, checkpoint)
                        state['artifact_output'] = relative(attempt)
                        break
            else:
                attempt = job.output / f'attempt{len(attempts)+1}'
                native('stage1/regression', ['--config', relative(job.config1), '--checkpoint', checkpoint,
                       '--output', relative(attempt), '--representation-cache', 'outputs/HPO/cache/stage1',
                       '--device', 'cuda:0'], job.output / f'{attempt.name}.log', device)
                state['artifact_output'] = relative(attempt)
        else:
            if job.config2:
                train_stage2(job, device, resume)
            from stage3.config import load_stage3_config
            from stage3.home import load_source
            load_source(load_stage3_config(job.config3))
            _, caches['stage3'] = prepare('stage3', job.config3, device)
            train = job.output if job.branch == 'HoME' else job.output / 'train'
            arguments = ['--config', relative(job.config3), '--fold', *job.folds, '--output', relative(train),
                         '--devices', device, '--max-parallel', 1]
            if resume:
                arguments.append('--resume')
            native('stage3/train', arguments, job.output / 'stage3.log', device)
            for fold in job.folds:
                evaluate_fold(job, fold, device, resume)
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['elapsed_seconds'] = time.time() - started
        atomic_json(record, state)


def run_job(job, device, resume=False, dry_run=False, checkpoint=None):
    if dry_run:
        return execute_job(job, device, resume, dry_run, checkpoint)
    with artifact_lock(job.output.with_name(job.output.name + '.trial.lock')):
        return execute_job(job, device, resume, dry_run, checkpoint)


def run_queue(jobs, devices, max_parallel, worker, *, record_failures=True):
    STOP.clear()
    slots = queue.Queue()
    for index in range(max_parallel):
        slots.put(devices[index % len(devices)])
    def assigned(job):
        if STOP.is_set():
            raise RuntimeError('HPO interrupted')
        device = slots.get()
        try:
            return worker(job, device)
        finally:
            slots.put(device)
    failed = []
    pool = ThreadPoolExecutor(max_workers=max_parallel)
    futures = {}
    try:
        futures = {pool.submit(assigned, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                future.result()
            except Exception as error:
                failed.append(job.trial)
                if record_failures:
                    failure = ROOT / 'outputs/HPO/failures' / job.branch / (job.trial + '.json')
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(failure, {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
                print(f'FAILED {job.trial}: {error}', file=sys.stderr, flush=True)
            print(f'Progress {done}/{len(jobs)}: {job.trial}', flush=True)
    except KeyboardInterrupt:
        STOP.set()
        for future in futures:
            future.cancel()
        stop_children()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=STOP.is_set())
    return failed


def jobs_for(branch, config_root):
    root = ROOT / 'outputs/HPO'
    if branch == 'stage1':
        return [Job(p.stem, branch, root / branch / p.stem, config1=p) for p in sorted((config_root / branch).glob('*.yaml'))]
    if branch == 'HoME':
        jobs = []
        for p in sorted((config_root / 'HoME/stage2').glob('*.yaml')):
            q = config_root / 'HoME/stage3' / p.name
            a, b = read_yaml(p), read_yaml(q)
            validate_pair(a, b)
            source = Path(b['initialization']['stage2_final'])
            canonical = config_root / 'HoME/stage2' / (source.parent.name + '.yaml')
            if read_yaml(canonical) != a:
                raise ValueError('Paired Stage2 canonical recipe differs')
            jobs.append(Job(p.stem, branch, root / 'HoME/stage3' / p.stem,
                            config2=p, config3=q, stage2_output=ROOT / source.parent))
        return jobs
    if branch == 'stage2':
        return [Job(p.stem, branch, root / branch / p.stem / 'downstream', config2=p,
                    config3=config_root / 'stage2/downstream' / p.name,
                    stage2_output=root / branch / p.stem / 'train')
                for p in sorted((config_root / branch).glob('*.yaml'))]
    if branch == 'stage3':
        return [Job(p.stem, branch, root / branch / p.stem, config3=p)
                for p in sorted((config_root / branch).glob('*.yaml'))]
    return [Job(str(split), branch, root / 'final' / str(split) / 'stage3',
                config2=config_root / 'final' / str(split) / 'stage2.yaml',
                config3=config_root / 'final' / str(split) / 'stage3.yaml',
                stage2_output=root / 'final' / str(split) / 'stage2', folds=(1,2,3,4,5)) for split in SPLITS]


def main(branch):
    parser = argparse.ArgumentParser(description=f'Run manually specified {branch} recipes; no automatic selection.')
    pool = parser.add_mutually_exclusive_group()
    pool.add_argument('--device')
    pool.add_argument('--devices')
    parser.add_argument('--max-parallel', '--max_parallel', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--generate', action='store_true', help='Write static YAML only; never train.')
    parser.add_argument('--selected', default='configs/HPO/selected.yaml')
    parser.add_argument('--config-root', default='configs/HPO')
    if branch == 'stage1':
        parser.add_argument('--checkpoint', default='outputs/v4/stage1/base/train/last.pt')
    args = parser.parse_args()
    os.chdir(ROOT)
    config_root = Path(args.config_root).resolve()
    if not config_root.is_relative_to(ROOT / 'configs/HPO'):
        parser.error('--config-root must be under configs/HPO')
    if branch in ('stage2', 'stage3', 'final') and not Path(args.selected).resolve().is_relative_to(ROOT / 'configs/HPO'):
        parser.error('--selected must be under configs/HPO')
    if args.generate:
        if args.dry_run:
            parser.error('--generate and --dry-run are separate operations')
        if branch in ('stage1', 'HoME'):
            generate_initial(config_root)
        else:
            {'stage2': generate_stage2, 'stage3': generate_stage3, 'final': generate_final}[branch](args.selected, config_root)
        print(f'Static configurations generated: {branch}; training not started.')
        return 0
    devices = tuple((args.devices or args.device or 'cuda:0').split(','))
    if len(set(devices)) != len(devices) or any(not re.fullmatch(r'cuda:\d+', d) for d in devices):
        parser.error('Specify unique CUDA devices, e.g. cuda:0,cuda:1')
    maximum = args.max_parallel if args.max_parallel is not None else len(devices)
    if maximum < 1:
        parser.error('--max-parallel must be positive')
    if branch in ('stage2', 'stage3', 'final'):
        required = {'stage2': ('home',), 'stage3': ('home','stage2_config','stage2_checkpoint'),
                    'final': ('home','stage2_config','stage2_checkpoint','stage3_config')}[branch]
        selection = selected(args.selected, required)
        binding = config_root / branch / 'generation.json'
        if not binding.exists() or json.loads(binding.read_text()) != generation_binding(selection, required, config_root):
            parser.error('Configurations are missing or bound to a different manual selection; generate this round first')
    jobs = jobs_for(branch, config_root)
    expected_count = {'stage1':100,'HoME':100,'stage2':20,'stage3':50,'final':5}[branch]
    if len(jobs) != expected_count:
        parser.error(f'Incomplete configuration round: expected {expected_count}, found {len(jobs)}')
    if args.ids:
        requested = set(','.join(args.ids).split(','))
        unknown = requested - {job.trial for job in jobs}
        if unknown:
            parser.error(f'Unknown IDs: {sorted(unknown)}')
        jobs = [j for j in jobs if j.trial in requested]
    if not jobs:
        parser.error('No static configurations; generate the round after manual selection')
    print(f'{branch}: {len(jobs)} jobs, devices={devices}, max_parallel={maximum}', flush=True)
    try:
        failed = run_queue(jobs, devices, maximum,
                           lambda job, device: run_job(job, device, args.resume, args.dry_run,
                                                       getattr(args, 'checkpoint', None)), record_failures=not args.dry_run)
    except KeyboardInterrupt:
        print('HPO interrupted; child processes stopped and artifacts retained.', file=sys.stderr)
        return 130
    return 1 if failed else 0
