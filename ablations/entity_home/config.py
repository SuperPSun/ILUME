from . import stage1  # Register strict reloading of single-view sources.
from . import chemberta  # Register explicit external frozen sources.
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

from stage3.config import Stage3Config, Stage3SimulationTrainingConfig, load_stage3_config, stage3_config_from_dict


VARIANTS = {
    'no_stage1', 'no_stage2', 'no_global', 'no_group', 'no_private',
    'chemberta2_mtr', 'chemberta2_mlm', 'uniform_gate', 'no_smiles', 'no_graph', 'phase1', 'phase2', 'capacity_no_global', 'capacity_no_group',
    'capacity_match_global', 'capacity_match_group', 'auxiliary',
}


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    stage2_config: str | None = None
    supervision_removed: str | None = None

    def validate(self):
        if self.variant not in VARIANTS:
            raise ValueError('Unknown Entity-HoME ablation variant')
        targets = {'lambda_unimol', 'lambda_electronic', 'lambda_partial_charge', 'lambda_alignment', 'lambda_descriptor'}
        if (self.variant == 'auxiliary') != (self.supervision_removed in targets):
            raise ValueError('Auxiliary ablations require exactly one registered removed supervision')
        if self.variant != 'auxiliary' and self.supervision_removed is not None:
            raise ValueError('Only auxiliary ablations declare removed supervision')
        if self.zero_update != (self.stage2_config is not None):
            raise ValueError('Zero-update ablations require an explicit Stage2 initialization config')

    @property
    def zero_update(self):
        return self.variant == 'no_stage2' or self.variant.startswith('capacity_')

    @property
    def selector(self):
        return self.variant if self.variant in {'phase1', 'phase2'} else 'final'

    @property
    def display_name(self):
        if self.variant in chemberta.SOURCES:
            return 'ChemBERTa-2 ' + self.variant.rsplit('_', 1)[1].upper() + ' + HoME'
        if self.variant in {'no_smiles', 'no_graph'}:
            return 'ILUME v4 w/o ' + ('SMILES' if self.variant == 'no_smiles' else 'Molecular Graph')
        label = 'w/o ' + self.supervision_removed.removeprefix('lambda_') + ' supervision' if self.variant == 'auxiliary' else self.variant
        return 'ILUME v4 ' + label

    @property
    def model_id(self):
        return 'ilume_v4_' + (self.supervision_removed if self.variant == 'auxiliary' else self.variant)


@dataclass(frozen=True)
class AblationConfig(Stage3Config):
    ablation: AblationSpec | None = None

    @property
    def runtime_backend(self):
        if self.ablation is None:
            return None
        from .backend import EntityHomeBackend
        return EntityHomeBackend(self.ablation)

    def validate(self):
        super().validate()
        if self.ablation is None or not self.is_entity_home:
            raise ValueError('Research config requires an Entity-HoME ablation')
        self.ablation.validate()
        external = self.ablation.variant in chemberta.SOURCES
        if external != (self.frozen_encoder is not None):
            raise ValueError('ChemBERTa ablation requires its explicit frozen_encoder source')
        if external:
            chemberta.validate_spec(self.frozen_encoder)
            if self.frozen_encoder.variant != self.ablation.variant:
                raise ValueError('ChemBERTa ablation/source version mismatch')
        if self.data.seed == 3407 and self.ablation.variant.startswith('capacity_'):
            raise ValueError('Capacity controls for data/3407 were cancelled; compare fixed Base phase states instead')
        if self.ablation.zero_update:
            if self.initialization.home_mode != 'no_stage2' or self.initialization.stage2_final is not None:
                raise ValueError('Zero-update ablations forbid a trained Stage2 final')
        elif self.initialization.home_mode != 'trained':
            raise ValueError('This ablation requires a validated trained Stage2 final')
        if self.initialization.simulation_artifacts_dir is None:
            raise ValueError('Entity-HoME ablations retain both simulation tasks')


def base_config(config):
    return Stage3Config(**{field.name: getattr(config, field.name) for field in fields(Stage3Config)})


def research_config_from_dict(raw):
    raw = dict(raw)
    spec = AblationSpec(**raw.pop('ablation'))
    spec.validate()
    initialization = dict(raw['initialization'])
    # The ordinary parser still rejects unsupported simulation initialization.
    if spec.zero_update:
        raw['initialization'] = {**initialization, 'simulation_artifacts_dir': None}
        simulation = raw['training'].get('simulation')
        raw['training'] = {**raw['training'], 'simulation': None}
    ordinary = stage3_config_from_dict(raw)
    if spec.zero_update:
        ordinary = replace(ordinary,
            initialization=replace(ordinary.initialization, simulation_artifacts_dir=Path(initialization['simulation_artifacts_dir'])),
            training=replace(ordinary.training, simulation=Stage3SimulationTrainingConfig(**simulation)))
    config = AblationConfig(**{field.name: getattr(ordinary, field.name) for field in fields(Stage3Config)}, ablation=spec)
    config.validate()
    return config


def load_research_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    return research_config_from_dict(raw) if 'ablation' in raw else load_stage3_config(path)
