"""Identity-checked, frozen five-fold inference for previously unseen ion pairs."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from rdkit import rdBase

from common.entity_inputs import EntityInputs
from common.entity_roles import molecular_role
from common.identity import semantic_identity
from common.io import sha256_file
from stage2 import FrozenObjectSpec, load_frozen_stage1_entities
from .config import load_stage3_config
from .data import canonicalize_smiles, iter_rows
from .evaluate import _load_model, _validate_three_phase_manifest
from .prepare import load_prepared_stage3


TARGET_UNITS = {
    "electrical_conductivity_S/m_log10": "S/m",
    "self_diffusion_coefficient_10^-9*m^2/s_log10": "10^-9*m^2/s",
    "viscosity_mPa*s_log10": "mPa*s",
    "thermal_conductivity_W/m/K": "W/m/K",
    "density_g/cm^3": "g/cm^3",
    "heat_capacity_J/mol/K": "J/mol/K",
    "speed_of_sound_m/s": "m/s",
    "surface_tension_mN/m": "mN/m",
    "pressure_kPa_log10": "kPa",
    "glass_transition_temperature_K": "K",
    "melting_point_K": "K",
    "thermal_decomposition_temperature_K": "K",
    "dynamic_relative_permittivity_unitless": "unitless",
    "static_relative_permittivity_unitless": "unitless",
    "refractive_index_unitless": "unitless",
    "solvation_kcal/mol": "kcal/mol",
    "pEC50": "pEC50",
    "x_gas_unitless": "unitless",
    "water_activity_coefficient_unitless": "unitless",
    "enthalpy_of_vaporization_or_sublimation_kJ/mol": "kJ/mol",
    "anodic_potential_limit_V": "V",
    "cathodic_potential_limit_V": "V",
    "heat_of_vaporization_kJ/mol": "kJ/mol",
    "thermal_expansion_K^-1": "K^-1",
}


@dataclass(frozen=True)
class PropertyCapability:
    task_id: str
    target_column: str
    unit: str
    transform: str
    condition_columns: tuple[str, ...]
    requires_solute: bool
    mean: float
    scale: float
    condition_stats: dict[str, dict[str, float]]

    def to_native(self, value: float, unit: str) -> float:
        if unit != self.unit or not math.isfinite(value):
            raise ValueError(f"Invalid target unit or value for {self.task_id}: {unit}")
        if self.transform == "log10":
            if value <= 0:
                raise ValueError("Log10 property thresholds must be positive")
            return math.log10(value)
        return value

    def to_physical(self, value: float) -> float:
        result = 10.0 ** value if self.transform == "log10" else value
        if not math.isfinite(result):
            raise ValueError("Non-finite physical prediction")
        return result


@dataclass(frozen=True)
class PredictionQuery:
    cation: str
    anion: str
    conditions: dict[str, dict[str, float]]
    solutes: dict[str, str] = field(default_factory=dict)


class FrozenPropertyPredictor:
    """Ensemble in the original target space; expose physical values separately."""

    def __init__(self, config: Any, checkpoint_dir: str | Path, *, device: str = "cpu"):
        self.config = load_stage3_config(config) if isinstance(config, (str, Path)) else config
        config = self.config
        if not config.is_entity_home or config.runtime_backend is not None:
            raise ValueError("Frozen pair inference requires the formal Entity-HoME contract")
        self.device = torch.device(device)
        execution = {"device_type": self.device.type, "amp_dtype": config.training.amp_dtype,
            "torch_version": str(torch.__version__), "rdkit_version": rdBase.rdkitVersion,
            "cuda_matmul_precision": torch.backends.cuda.matmul.fp32_precision,
            "cudnn_conv_precision": torch.backends.cudnn.conv.fp32_precision}
        if self.device.type == "cuda":
            execution.update(gpu_model=torch.cuda.get_device_name(self.device),
                gpu_capability=list(torch.cuda.get_device_capability(self.device)))
        self.execution_identity = semantic_identity("stage3.pair-inference-execution.v1", execution)
        self.prepared = load_prepared_stage3(config)
        self.encoder = load_frozen_stage1_entities(config.initialization.stage1_encoder,
            config.initialization.stage1_artifacts_dir, device=device)
        self.models, self.normalizations = [], []
        self.simulation_source = None
        if config.initialization.simulation_artifacts_dir is not None:
            from .home import load_source
            self.simulation_source = load_source(config)
            self.simulation_source.pop("_loaded_model", None)
            self.simulation_source.pop("full_model_state", None)
        sources = {}
        for fold in range(1, 6):
            path = Path(checkpoint_dir) / f"fold{fold}" / "three_phase_final.pt"
            model, checkpoint, _ = _load_model(config, self.prepared, path, fold, 0,
                torch.device("cpu"), three_phase_final=True)
            manifest = _validate_three_phase_manifest(path, checkpoint, config)
            model.requires_grad_(False).eval()
            self.models.append(model)
            self.normalizations.append(checkpoint["normalization"])
            sources[f"fold{fold}"] = {"artifact": sha256_file(path), "manifest": manifest,
                "training_identity": checkpoint["training_identity"]["hash"]}
        self.identity = semantic_identity("stage3.frozen-pair-predictor.v1", {
            "folds": sources, "stage1": self.encoder.encoder_identity,
            "ensemble": "native_target_mean_v1", "amp_dtype": config.training.amp_dtype,
            "registry_hash": self.prepared["metadata"]["registry_hash"],
        })
        self.capabilities = {}
        for task, spec in self.models[0].task_specs.items():
            if not spec.enabled or tuple(spec.primary_slots) != ("cation", "anion"):
                continue
            target = spec.target_column
            if target not in TARGET_UNITS:
                raise ValueError(f"Unregistered inference unit: {target}")
            stats = [self.stats(fold, task) for fold in range(5)]
            self.capabilities[task] = PropertyCapability(task, target, TARGET_UNITS[target],
                "log10" if target.endswith("_log10") else "identity", tuple(spec.condition_columns),
                tuple(spec.partner_slots) == ("solute",),
                sum(s["target"]["mean"] for s in stats) / 5,
                sum(s["target"]["scale"] for s in stats) / 5,
                {name: {key: sum(s["conditions"][name][key] for s in stats) / 5
                    for key in ("mean", "scale")} for name in spec.condition_columns})
        self._entities: dict[tuple[str, str], torch.Tensor] = {}

    def stats(self, fold: int, task: str) -> dict[str, Any]:
        if task.startswith("simulation/"):
            stats = self.simulation_source["scalers"][task]
            target = self.models[fold].task_specs[task].target_column
            return {"target": stats["targets"][target], "conditions": stats["conditions"]}
        return self.normalizations[fold][task]

    def capability_table(self) -> list[dict[str, Any]]:
        return [asdict(self.capabilities[task]) for task in sorted(self.capabilities)]

    def reference_rows(self, task: str):
        """Yield development structures and conditions, never formal test rows."""
        if task not in self.capabilities:
            raise ValueError(f"Unsupported pair property: {task}")
        if task.startswith("simulation/"):
            import csv
            from stage2.registry import load_stage2_registry
            source = self.simulation_source
            registry = load_stage2_registry(self.config.data.task_catalog,
                task_ids=tuple(source["recipe"]["data"]["tasks"]))
            path = registry.by_id(task).dataset.split_path(Path(source["recipe"]["data"]["data_root"]), "train")
            with path.open(newline="", encoding="utf-8-sig") as handle:
                yield from csv.DictReader(handle)
        else:
            for _, _, row in iter_rows(self.config, self.prepared["registry"][task], tuple(range(1, 6))):
                yield row

    def _slots(self, pairs: Sequence[tuple[str, str]]) -> EntityInputs:
        missing = list(dict.fromkeys(pair for pair in pairs if pair not in self._entities))
        if missing:
            values, _ = self.encoder.encode_slots([FrozenObjectSpec("molecule", (pair,)) for pair in missing])
            self._entities.update((pair, value[0]) for pair, value in zip(missing, values, strict=True))
        values = torch.stack([self._entities[pair] for pair in pairs]).reshape(1, len(pairs), 1241)
        roles = torch.tensor([[{"cation": 0, "anion": 1, "neutral": 2}[role] for role, _ in pairs]])
        return EntityInputs.from_slots(values, roles).to(self.device)

    @torch.inference_mode()
    def predict(self, queries: Sequence[PredictionQuery]) -> list[dict[str, dict[str, Any]]]:
        results = [dict() for _ in queries]
        validated = []
        for query in queries:
            cation = canonicalize_smiles(query.cation, "inference cation")
            anion = canonicalize_smiles(query.anion, "inference anion")
            if molecular_role(cation) != "cation" or molecular_role(anion) != "anion":
                raise ValueError("Pair inference requires positive cation and negative anion")
            if not query.conditions or set(query.solutes) - set(query.conditions):
                raise ValueError("Inference requires properties and matching solutes")
            for task, conditions in query.conditions.items():
                if task not in self.capabilities:
                    raise ValueError(f"Unsupported pair property: {task}")
                cap = self.capabilities[task]
                if set(conditions) != set(cap.condition_columns) or any(not math.isfinite(float(v)) for v in conditions.values()):
                    raise ValueError(f"Missing, extra or non-finite conditions: {task}")
                if cap.requires_solute != (task in query.solutes):
                    raise ValueError(f"Solute requirement mismatch: {task}")
            validated.append((query, cation, anion))
        for fold, model in enumerate(self.models):
            model.to(self.device).eval()
            try:
                for index, (query, cation, anion) in enumerate(validated):
                    primary = self._slots((("cation", cation), ("anion", anion)))
                    for task, conditions in query.conditions.items():
                        stats = self.stats(fold, task)
                        tensor = torch.tensor([[(float(conditions[name]) - stats["conditions"][name]["mean"])
                            / stats["conditions"][name]["scale"] for name in self.capabilities[task].condition_columns]],
                            dtype=torch.float32, device=self.device)
                        partner = None
                        if task in query.solutes:
                            solute = canonicalize_smiles(query.solutes[task], "inference solute")
                            partner = self._slots(((molecular_role(solute), solute),))
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.config.training.amp_dtype == "bf16"):
                            prediction = model(task, primary, tensor, partner_embedding=partner).predictions
                        # Experimental evaluation denormalizes float32; simulation uses float64.
                        if task.startswith("simulation/"):
                            value = float(prediction.float().cpu().item()) * stats["target"]["scale"] + stats["target"]["mean"]
                        else:
                            value = float((prediction.float() * stats["target"]["scale"] + stats["target"]["mean"]).item())
                        if not math.isfinite(value):
                            raise ValueError(f"Non-finite frozen prediction: {task}")
                        results[index].setdefault(task, {"folds_native": []})["folds_native"].append(value)
            finally:
                model.to("cpu")
        for result in results:
            for task, values in result.items():
                cap = self.capabilities[task]
                mean = sum(values["folds_native"]) / 5
                std = math.sqrt(sum((v - mean) ** 2 for v in values["folds_native"]) / 5)
                values.update(native=mean, value=cap.to_physical(mean), std_native=std,
                    unit=cap.unit, transform=cap.transform, source=task.split("/", 1)[0])
        return results
