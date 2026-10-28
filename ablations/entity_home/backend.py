from dataclasses import asdict
import json
from pathlib import Path

import torch

from common.identity import require_compatible_identity, semantic_identity, tensor_state_hash
from common.io import sha256_file
from stage1.identity import metadata_identity as feature_identity, encoding_state_hash
from stage2.data import load_artifact_registry, Stage2EntityDataset
from stage2.home_artifact import full_state_hash
from stage2.home_config import load_home_recipe
from stage2.home_contract import transferable_state, state_hash
from stage2.home_train import build_model
from stage2.identity import build_stage2_data_identity, metadata_identity
from stage3.home import load_source, source_plan, validate_transfer_architecture
from stage3.model import GLOBAL
from .config import base_config
from .model import transform_model


class EntityHomeBackend:
    def __init__(self, spec):
        self.spec = spec

    def load_source(self, config):
        if not self.spec.zero_update:
            source = load_source(base_config(config))
            if config.frozen_encoder is not None:
                if source["_loaded_model"].stage2_config.frozen_encoder != config.frozen_encoder:
                    raise ValueError("Stage2/Stage3 external frozen encoder source mismatch")
                return source
            if self.spec.variant == 'auxiliary' and getattr(source['_loaded_model'].backbone.config.loss, self.spec.supervision_removed) != 0:
                raise ValueError('Auxiliary ablation source did not remove its declared supervision')
            source_spec = getattr(source['_loaded_model'].backbone.config, 'ablation', None)
            if self.spec.variant in {'no_smiles', 'no_graph'}:
                if source_spec is None or source_spec.variant != self.spec.variant:
                    raise ValueError('Single-view ablation source view mismatch')
            elif source_spec is not None:
                raise ValueError('This ablation requires a complete dual-view Stage1 source')
            if self.spec.variant == 'no_stage1':
                artifact = torch.load(config.initialization.stage1_encoder, map_location='cpu', weights_only=False)
                if artifact.get('kind') != 'ilume_stage1_random_encoder_v4':
                    raise ValueError('No-Stage1 ablation requires the audited random frozen encoder')
            return source
        recipe = load_home_recipe(self.spec.stage2_config)
        original = recipe.stage2
        if not original.is_entity_home or recipe.initialization != 'pretrained':
            raise ValueError('Zero-update source requires the paired frozen pretrained Stage1 recipe')
        if (original.initialization.checkpoint != config.initialization.stage1_encoder
                or original.data.pretrain_artifacts_dir != config.initialization.stage1_artifacts_dir
                or original.data.artifacts_dir != config.initialization.simulation_artifacts_dir):
            raise ValueError('Zero-update Stage2/Stage3 source paths differ')
        from stage2.home_config import effective_training_seed as source_seed
        from stage3.config import effective_training_seed as target_seed
        if source_seed(original) != target_seed(config):
            raise ValueError('Zero-update Stage2/Stage3 training seeds differ')
        registry = load_artifact_registry(original.data.artifacts_dir)
        original.validate_registry(registry)
        metadata = json.loads((original.data.artifacts_dir / 'metadata.json').read_text())
        features_root = original.data.pretrain_artifacts_dir
        features_meta = json.loads((features_root / 'metadata.json').read_text())
        features_identity = feature_identity(features_meta, 'feature', context='Stage1 features')
        data_identity = metadata_identity(metadata, 'data', context='Stage2 zero-update data')
        require_compatible_identity(build_stage2_data_identity(original, registry, features_identity), data_identity,
                                    context='Zero-update prepared simulation source')
        model, loaded = build_model(recipe, registry)
        if getattr(loaded.config, 'ablation', None) is not None:
            raise ValueError('Zero-update controls require the complete dual-view Stage1 source')
        validate_transfer_architecture(model.home.model_config, config.model, model.home.group_configs, config.groups)
        entities = Stage2EntityDataset(original.data.artifacts_dir)
        if (entities.frozen_entity_manifest is None or entities.frozen_entity_manifest['identity']['payload']['stage1_state_hash'] != encoding_state_hash(model.backbone)):
            raise ValueError('Zero-update frozen simulation entities differ from Stage1')
        shared = transferable_state(model.home)
        features = {name: json.loads((features_root / name).read_text()) for name in (
            'tokenizer.json', 'descriptor_schema.json', 'descriptor_scaler.json')}
        full_hash = full_state_hash(model.state_dict())
        identity = semantic_identity('ablation.stage2.zero-update.v1', {
            'recipe': recipe.to_dict(), 'stage2_data_identity': data_identity['hash'],
            'stage1_checkpoint_sha256': sha256_file(original.initialization.checkpoint),
            'stage1_feature_identity': features_identity['hash'], 'initial_state_hash': full_hash,
            'optimizer_updates': 0,
        })
        return {
            'kind': 'ilume_stage2_entity_home_zero_update_v1',
            'training_identity': identity, 'stage2_data_identity': data_identity,
            'stage1_feature_identity': features_identity, 'shared_state': shared,
            'shared_state_hash': state_hash(shared), 'full_model_state_hash': full_hash,
            'registry_hash': registry.registry_hash, 'catalog_sha256': registry.catalog_sha256,
            'recipe': recipe.to_dict(), 'scalers': metadata['scalers'],
            'stage1_backbone': model.backbone.state_dict(), 'stage1_config': loaded.config.to_dict(),
            'stage1_encoding_contract': {**model.model_contract, 'feature_generation_contract': features_meta['feature_generation_contract']},
            'feature_artifacts': features, '_loaded_model': model, '_vocabulary': loaded.vocabulary,
        }

    def source_plan(self, config, source, names):
        if not self.spec.zero_update:
            return source_plan(base_config(config), source, names)
        return {'mode': 'zero_update', 'optimizer_updates': 0,
                'source_initialization_identity': source['training_identity']['hash'],
                'source_data_identity': source['stage2_data_identity']['hash'],
                'source_full_model_state_hash': source['full_model_state_hash'],
                'transferred_state_hash': source['shared_state_hash'],
                'transferred_parameters': list(names),
                'simulation_training': {'tasks': [
                    'simulation/heat_of_vaporization', 'simulation/thermal_expansion']}}

    def transform_model(self, config, model, names):
        if self.spec.selector != 'final':
            return model, names
        from stage3.config import effective_training_seed
        model = transform_model(model, self.spec.variant, seed=effective_training_seed(config))
        retained = set(model.state_dict())
        return model, tuple(name for name in names if name in retained)

    def finalize_plan(self, config, model, plan):
        if self.spec.selector != 'final':
            raise ValueError('Phase snapshots are evaluation-only; train the paired Base')
        plan['ablation'] = {'contract_version': 1, **asdict(self.spec),
                            'initial_model_state_hash': tensor_state_hash('ablation.entity-home.initial-state.v1', model.state_dict())}
        if hasattr(model, 'capacity_match'):
            plan['ablation']['capacity_match'] = model.capacity_match
        global_hidden = model.l1_global_experts[0].layers[0].out_features if model.l1_global_experts else 0
        plan['model']['resolved_widths']['expert_hidden'] = global_hidden
        plan['model']['global_experts'] = len(model.l1_global_experts)
        if 'GLOBAL' in plan['phases']['phase1']['owners']:
            plan['phases']['phase1']['owners']['GLOBAL']['capacity'] = {
                'experts': len(model.l1_global_experts), 'expert_hidden': global_hidden,
                'expert_hidden_ratio': global_hidden / model.d_model,
            }
        if not model.parameters_for_owner(GLOBAL):
            plan['phases']['phase1']['owners'].pop('GLOBAL')
    @property
    def final_kind(self):
        return 'ilume_stage3_entity_home_ablation_final_v1'

    @property
    def model_selector(self):
        return 'three_phase_final' if self.spec.selector == 'final' else self.spec.selector + '_final'

    def artifact_path(self, root, fold, config):
        suffix = {'final': 'three_phase_final.pt', 'phase2': 'phase_2/stitched.pt'}
        if self.spec.selector == 'phase1':
            # Checkpoint epoch is determined by the fixed owner calendar, not metrics.
            suffix['phase1'] = f'phase_1/checkpoint_epoch_{config.training.three_phase.global_scope.epochs:05d}.pt'
        relative = Path(suffix[self.spec.selector])
        nested = Path(root) / f'fold{fold}' / relative
        return nested if nested.is_file() else Path(root) / relative

    def validate_plan(self, config, model, plan):
        expected = {'contract_version': 1, **asdict(self.spec),
                    'initial_model_state_hash': tensor_state_hash('ablation.entity-home.initial-state.v1', model.state_dict())}
        if hasattr(model, 'capacity_match'):
            expected['capacity_match'] = model.capacity_match
        if plan.get('ablation') != expected:
            raise ValueError('Stage3 ablation recipe or initial state mismatch')
        from stage3.home import resolve_training_identity
        from stage3.identity import build_stage3_training_identity
        require_compatible_identity(resolve_training_identity(config, plan['fold']),
                                    build_stage3_training_identity(plan), context='Ablation evaluation training recipe')

    def load_phase_model(self, config, prepared, path, fold, device):
        from common.training import canonical_json_sha256
        from stage3.home import build_model_and_store, resolve_training_identity
        from stage3.identity import build_stage3_training_identity, metadata_identity
        ordinary = base_config(config)
        model, store, _ = build_model_and_store(ordinary, prepared, fold=fold, device=device)
        artifact = torch.load(path, map_location='cpu', weights_only=False)
        selector = self.spec.selector
        kind = 'ilume_stage3_entity_home_three_phase_v4_' + ('checkpoint' if selector == 'phase1' else '2_stitched')
        plan = artifact.get('resolved_training_plan', {})
        if (artifact.get('kind') != kind or artifact.get('format_version') != 4
                or artifact.get('fold') != fold
                or plan.get('resolved_registry') != {task: spec.to_dict() for task, spec in model.task_specs.items()}
                or artifact.get('ownership_manifest') != model.ownership_manifest()
                or plan.get('prepared_identity') != metadata_identity(prepared['metadata'], 'prepared', context='Stage3 phase snapshot')['hash']
                or plan.get('normalization_hash') != canonical_json_sha256(artifact.get('normalization'))
                or artifact.get('stage1_encoder_identity') != plan.get('stage1_encoder_identity')
                or artifact.get('model_state_hash') != tensor_state_hash('stage3.entity-home-model-state.v4', artifact['model'])):
            raise ValueError('Stage3 phase snapshot contract mismatch')
        require_compatible_identity(build_stage3_training_identity(plan), artifact['training_identity'], context='Phase snapshot plan')
        require_compatible_identity(resolve_training_identity(ordinary, fold), artifact['training_identity'], context='Phase snapshot Base identity')
        if selector == 'phase1' and (artifact.get('phase') != 'phase_1' or artifact.get('completed_epoch') != config.training.three_phase.global_scope.epochs):
            raise ValueError('Phase1 evaluation requires the fixed last epoch')
        self.validate_phase_manifest(config, path, artifact)
        model.load_state_dict(artifact['model'], strict=True)
        return model.to(device).eval(), artifact, store

    def validate_phase_manifest(self, config, path, artifact):
        if self.spec.selector == 'phase1':
            return sha256_file(path)
        manifest_path = Path(path).with_suffix('.json')
        manifest = json.loads(manifest_path.read_text())
        plan = artifact['resolved_training_plan']
        if (manifest.get('artifact') != Path(path).name
                or manifest.get('artifact_sha256') != sha256_file(path)
                or manifest.get('kind') != artifact['kind'] or manifest.get('format_version') != 4
                or manifest.get('fold') != artifact['fold']
                or manifest.get('model_state_hash') != artifact['model_state_hash']
                or manifest.get('anchor_model_state_hash') != artifact.get('anchor_model_state_hash')
                or manifest.get('ownership_manifest') != artifact.get('ownership_manifest')
                or set(manifest.get('groups', {})) != set(plan['phases']['phase2']['branches'])):
            raise ValueError('Phase2 stitched manifest mismatch')
        require_compatible_identity(artifact['training_identity'], manifest['training_identity'], context='Phase2 manifest')
        return sha256_file(manifest_path)
