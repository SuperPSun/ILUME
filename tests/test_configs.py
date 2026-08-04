from __future__ import annotations

from pathlib import Path

import pytest

from ilume_pretrain.config import config_from_dict, load_config
from ilume_pretrain.sampler import coverage_epoch_plan


ROOT = Path(__file__).resolve().parents[1]
ABLATION_NAMES = (
    "reference",
    "descriptor_full",
    "descriptor_pruned",
    "descriptor_tokens_1",
    "descriptor_tokens_12",
    "fingerprint_none",
    "role_embedding_off",
    "graph_head_linear",
    "modality_dropout_off",
    "asymmetric_masking_off",
)


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        flattened = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(item, name))
        return flattened
    return {prefix: value}


def test_formal_profiles_use_five_coverage_epochs():
    base = load_config(ROOT / "configs/pretrain_base.yaml")
    large = load_config(ROOT / "configs/pretrain_large.yaml")
    xlarge = load_config(ROOT / "configs/pretrain_xlarge.yaml")

    assert (base.training.batch_size, base.training.epochs) == (256, 5)
    assert base.training.gradient_accumulation_steps == 1
    assert base.training.learning_rate == pytest.approx(1.0e-4)
    assert (large.training.batch_size, large.training.epochs) == (256, 5)
    assert large.training.gradient_accumulation_steps == 1
    assert large.training.learning_rate == pytest.approx(1.0e-4)
    assert (xlarge.training.batch_size, xlarge.training.epochs) == (128, 5)
    assert xlarge.training.gradient_accumulation_steps == 2
    assert xlarge.training.learning_rate == pytest.approx(1.0e-4)
    assert xlarge.model.gradient_checkpointing
    assert (
        xlarge.model.d_model,
        xlarge.model.n_heads,
        xlarge.model.smiles_layers,
        xlarge.model.fusion_layers,
        xlarge.model.graph_depth,
        xlarge.model.descriptor_hidden_dim,
        xlarge.model.feedforward_dim,
    ) == (640, 10, 10, 10, 7, 1280, 2560)
    assert xlarge.model.d_model % xlarge.model.n_heads == 0
    assert xlarge.model.d_model > large.model.d_model
    assert xlarge.model.smiles_layers > large.model.smiles_layers
    assert xlarge.model.fusion_layers > large.model.fusion_layers
    assert xlarge.data.artifacts_dir == base.data.artifacts_dir
    assert xlarge.training.output_dir == Path(
        "artifacts/training/pretrain_xlarge_bs128_acc2"
    )

    counts = (24908, 27907, 56532)
    base_plan = coverage_epoch_plan(counts, 256, 1)
    large_plan = coverage_epoch_plan(counts, 256, 1)
    xlarge_plan = coverage_epoch_plan(counts, 128, 2)
    assert (base_plan.steps_per_epoch, base_plan.draws_per_epoch) == (
        2209,
        565504,
    )
    assert base_plan.role_quotas == (254477, 254477, 56550)
    assert base_plan == large_plan == xlarge_plan
    assert base_plan.draws_per_epoch * base.training.epochs == 2827520
    assert large_plan.draws_per_epoch * large.training.epochs == 2827520
    assert xlarge_plan.draws_per_epoch * xlarge.training.epochs == 2827520


def test_pretrain_config_round_trips_from_checkpoint_dictionary():
    config = load_config(ROOT / "configs/pretrain_base.yaml")
    assert config_from_dict(config.to_dict()) == config


def test_ablation_configs_change_only_the_declared_factor():
    configs = {
        name: load_config(ROOT / f"configs/ablations/{name}.yaml")
        for name in ABLATION_NAMES
    }
    reference = _flatten(configs["reference"].to_dict())
    ignored = {"data.artifacts_dir", "training.output_dir"}
    expected = {
        "descriptor_full": {"descriptor.mode"},
        "descriptor_pruned": {"descriptor.mode"},
        "descriptor_tokens_1": {"descriptor.token_count"},
        "descriptor_tokens_12": {"descriptor.token_count"},
        "fingerprint_none": {
            "fingerprint.kind",
            "loss.lambda_fingerprint",
        },
        "role_embedding_off": {"model.role_embedding"},
        "graph_head_linear": {"model.graph_head"},
        "modality_dropout_off": {"masking.dropout_schedule"},
        "asymmetric_masking_off": {"masking.asymmetric_enabled"},
    }
    for name, allowed in expected.items():
        candidate = _flatten(configs[name].to_dict())
        changed = {
            key
            for key in reference
            if reference[key] != candidate[key] and key not in ignored
        }
        assert changed == allowed

    active_configs = [
        load_config(ROOT / "configs/pretrain_base.yaml"),
        load_config(ROOT / "configs/pretrain_large.yaml"),
        load_config(ROOT / "configs/pretrain_xlarge.yaml"),
        *configs.values(),
    ]
    output_dirs = {
        str(config.training.output_dir) for config in active_configs
    }
    assert len(output_dirs) == len(active_configs)
    for name in (
        "reference",
        "role_embedding_off",
        "graph_head_linear",
        "modality_dropout_off",
        "asymmetric_masking_off",
    ):
        assert configs[name].data.artifacts_dir == Path("artifacts/pretrain_base")


def test_step_based_archived_config_is_rejected():
    path = ROOT / "configs/archive/2026-07-24_pretrain_base.yaml"
    with pytest.raises(ValueError, match="max_steps"):
        load_config(path)

    for path in (
        ROOT / "configs/pretrain_base.yaml",
        ROOT / "configs/pretrain_large.yaml",
        ROOT / "configs/pretrain_xlarge.yaml",
        *(ROOT / "configs/ablations").glob("*.yaml"),
    ):
        assert "max_steps" not in path.read_text(encoding="utf-8")
