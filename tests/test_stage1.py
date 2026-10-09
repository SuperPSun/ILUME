from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from common.data_identity import write_data_identity
from common.identity import semantic_identity
from common.outputs import open_run_directory
import common.outputs as outputs_module
from stage1.config import (
    ArchitectureConfig,
    GLOBAL_RDKIT_STAGE1_CHECKPOINT_VERSION,
    STAGE1_CHECKPOINT_KIND,
    STAGE1_CHECKPOINT_VERSION,
    config_from_dict,
    load_config,
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PreparationConfig,
    PretrainConfig,
    TrainingConfig,
)
from stage1.data import (
    CORPUS_FORMAT_VERSION,
    CORPUS_KIND,
    GLOBAL_RDKIT_CORPUS_FORMAT_VERSION,
    PreparedCorpusDataset,
)
from stage1.descriptors import rdkit_descriptor_names, DescriptorSchema
from stage1.fingerprints import calculate_fingerprints
from stage1.features import IPC_SQUARE_OVERFLOW_LIMIT
import stage1.features as features_module
from stage1.graph import ATOM_FEATURE_NAMES, BOND_FEATURE_NAMES
from stage1.masking import (
    MultimodalPacker,
    mask_smiles_tokens,
    MultimodalCollator,
    MultimodalMasker,
)
from stage1.model import (
    LossStatistics,
    PretrainOutput,
    MultimodalPretrainModel,
    _weighted_component,
)
from stage1.prepare import prepare_corpus
import stage1.prepare as prepare_module
from stage1.tokenizer import SmilesTokenizer, ais_tokenize
from stage1.train import _DistributedContext, _global_training_losses, run_training
import stage1.train as train_module


def test_global_rdkit_v2_representation_and_losses(
    tiny_config, tiny_samples
) -> None:
    vocabulary, legacy_samples = tiny_samples
    config = replace(
        tiny_config,
        architecture=ArchitectureConfig(kind="global_rdkit_v2"),
        descriptor=DescriptorConfig(mode="full", token_count=1),
        model=replace(tiny_config.model, descriptor_blocks=2),
    )
    config.validate()
    samples = [
        {key: value for key, value in sample.items() if key != "fingerprints"}
        for sample in legacy_samples
    ]
    schema = DescriptorSchema.fit(
        np.stack([sample["descriptors"].numpy() for sample in samples]),
        rdkit_descriptor_names(),
        "full",
        1,
    )
    model = MultimodalPretrainModel(config, vocabulary, schema)
    packed = MultimodalPacker(vocabulary)(samples)
    encoded = model.encode_entity(packed)

    assert config.checkpoint_version == GLOBAL_RDKIT_STAGE1_CHECKPOINT_VERSION
    assert "fingerprint" not in config.to_dict()
    assert (model.token_dim, model.atom_dim, model.entity_dim) == (32, 32, 64)
    assert model.fusion.modality_embedding.num_embeddings == 4
    assert encoded.cls_embedding.shape == (3, 32)
    assert encoded.rdkit_embedding.shape == (3, 32)
    assert encoded.entity_embedding.shape == (3, 64)
    assert torch.equal(
        encoded.entity_embedding,
        torch.cat((encoded.cls_embedding, encoded.rdkit_embedding), dim=-1),
    )
    assert torch.equal(model.encode(packed), encoded.cls_embedding)
    assert not any("fingerprint" in name for name, _ in model.named_parameters())

    masked = MultimodalMasker(vocabulary, config.masking).apply(packed)
    output = model(masked)
    assert masked.masks.modality_dropped.shape == (3, 3)
    assert set(output.losses) == {"smiles", "descriptor", "atom", "bond"}
    assert set(output.logits) == {"smiles", "descriptor", "atom", "bond"}
    assert output.logits["descriptor"].shape == (3, 217)
    invalid = config.to_dict()
    invalid["fingerprint"] = {"kind": "none"}
    with pytest.raises(ValueError, match="forbids fingerprint"):
        config_from_dict(invalid)

# --- Configuration, identity, and checkpoint/resume contracts ---

ROOT = Path(__file__).resolve().parents[1]


def test_v4_residual_capacity_and_shared_compatibility(tiny_config, tiny_samples, monkeypatch):
    from types import SimpleNamespace
    from stage1.dual_view import DualViewEncoder, DualViewPretrainModel
    from stage1.identity import build_stage1_corpus_identity

    base = load_config(ROOT / "configs/v4/stage1/base.yaml")
    assert [base.loss.lambda_smiles, base.loss.lambda_atom, base.loss.lambda_bond,
            base.loss.lambda_alignment, base.loss.lambda_descriptor,
            base.loss.lambda_unimol, base.loss.lambda_electronic] == [.2, 1., 1., .1, .25, .5, .25]
    assert (base.training.gradient_audit_interval_steps, base.training.gradient_audit_batch_size) == (1000, 32)
    assert base.loss.lambda_partial_charge == .25
    assert (base.model.smiles_layers, base.model.graph_depth, base.model.graph_message_mode) == (12, 8, "residual_blocks")
    assert config_from_dict(base.to_dict()).to_dict() == base.to_dict()
    old = replace(base, model=replace(base.model, smiles_layers=8, graph_depth=6, graph_message_mode="shared"))
    assert "graph_message_mode" not in old.to_dict()["model"]
    with torch.device("meta"):
        vocabulary = SimpleNamespace(tokens=tuple(range(2048)))
        encoder = DualViewEncoder(base, vocabulary)
        training = DualViewPretrainModel(base, vocabulary)
        legacy = DualViewEncoder(old, vocabulary)
    assert sum(p.numel() for p in encoder.parameters()) == 60_725_419
    assert sum(p.numel() for p in training.parameters()) == 65_125_947 + 513
    assert sum(p.numel() for p in legacy.parameters()) == 29_466_795
    assert len(encoder.smiles_encoder.encoder.layers) == 12
    assert len(encoder.graph_encoder.blocks) == 8
    parameters = [p for block in encoder.graph_encoder.blocks for p in block.parameters()]
    assert len({id(p) for p in parameters}) == len(parameters)
    assert (encoder.entity_dim, encoder.atom_dim) == (1024, 512)
    assert set(encoder.fusion.state_dict()) == set(legacy.fusion.state_dict())
    audit = {"semantic": {"identities": {"source": semantic_identity("test.source", {"sources": {"molecules": "unchanged"}})}},
             "locator": {"files": {"molecules": "ignored"}}, "integrity": {"files": {"molecules": "ignored"}}}
    monkeypatch.setattr("stage1.auxiliary.electronic_source_contract", lambda _: {"train": "same electronic labels"})
    assert build_stage1_corpus_identity(base, audit) == build_stage1_corpus_identity(old, audit)
    invalid = base.to_dict()
    invalid["model"]["graph_message_mode"] = "unknown"
    with pytest.raises(ValueError, match="graph_message_mode"):
        config_from_dict(invalid)
    with pytest.raises(ValueError, match="v4"):
        replace(tiny_config, model=replace(tiny_config.model, graph_message_mode="residual_blocks")).validate()

    vocabulary, samples = tiny_samples
    tiny = replace(tiny_config, architecture=ArchitectureConfig("dual_view_v4"),
                   descriptor=DescriptorConfig(mode="full", token_count=1),
                   fingerprint=FingerprintConfig(),
                   model=replace(tiny_config.model, role_embedding=False),
                   masking=replace(tiny_config.masking, fusion_only_dropout=True, descriptor_dropout=0))
    explicit = tiny.to_dict()
    explicit["model"]["graph_message_mode"] = "shared"
    assert config_from_dict(explicit).to_dict() == tiny.to_dict()
    assert not any(name.startswith("gradient_audit") for name in tiny.to_dict()["training"])
    assert "lambda_partial_charge" not in tiny.to_dict()["loss"]
    assert "partial_charge_cache" not in tiny.to_dict()["auxiliary"]
    audited = replace(tiny, training=replace(tiny.training, gradient_audit_interval_steps=1))
    assert train_module._config_hash(audited) == train_module._config_hash(tiny)
    with pytest.raises(ValueError, match="dual_view_v4"):
        replace(tiny_config, training=replace(tiny_config.training, gradient_audit_interval_steps=1)).validate()
    for name, value in (("gradient_audit_interval_steps", -1), ("gradient_audit_batch_size", 0), ("gradient_audit_interval_steps", True)):
        with pytest.raises(ValueError, match=name):
            replace(tiny, training=replace(tiny.training, **{name: value})).validate()
    for value in (-1., float("nan")):
        with pytest.raises(ValueError, match="lambda_partial_charge"):
            replace(tiny, loss=replace(tiny.loss, lambda_partial_charge=value)).validate()
    with pytest.raises(ValueError, match="dual_view_v4"):
        replace(tiny_config, loss=replace(tiny_config.loss, lambda_partial_charge=.1)).validate()
    torch.manual_seed(42)
    implicit_model = DualViewEncoder(tiny, vocabulary).eval()
    torch.manual_seed(42)
    explicit_model = DualViewEncoder(config_from_dict(explicit), vocabulary).eval()
    assert all(torch.equal(value, explicit_model.state_dict()[key]) for key, value in implicit_model.state_dict().items())
    packed = MultimodalPacker(vocabulary)(samples)
    assert torch.equal(implicit_model.encode(packed), explicit_model.encode(packed))


@pytest.mark.parametrize("smiles", [("CCO",), ("[Na+]",), ("[Na+]", "CCO")])
def test_residual_graph_bonded_and_bondless_gradients(smiles):
    from stage1.encoders import DirectedMessagePassingEncoder
    from stage1.graph import featurize_mol, pack_graphs

    graph = pack_graphs([featurize_mol(Chem.MolFromSmiles(value)) for value in smiles])
    encoder = DirectedMessagePassingEncoder(16, 8, 0., message_mode="residual_blocks", feedforward_dim=64)
    atoms, bonds = encoder(graph, torch.zeros(len(graph.atom_categorical), dtype=torch.bool),
                           torch.zeros(len(graph.bond_categorical), dtype=torch.bool))
    assert atoms.shape == (len(graph.atom_categorical), 16)
    assert bonds.shape == (len(graph.bond_categorical), 16)
    assert torch.isfinite(atoms).all() and torch.isfinite(bonds).all()
    ((atoms * torch.arange(16)).sum() + (bonds * torch.arange(16)).sum()).backward()
    for block in encoder.blocks:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        if len(graph.bond_categorical):
            assert block.message_projection.weight.grad.abs().sum() > 0
        else:
            assert all(torch.count_nonzero(p.grad) == 0 for p in block.parameters())


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_v4_gradient_audit_reference_and_state_isolation(tiny_config, tiny_samples, device_name):
    import random
    from stage1.auxiliary import empty_auxiliary_targets
    from stage1.dual_view import DualViewPretrainModel
    from stage1.gradient_audit import AUDIT_LOSSES, audit_gradient_norms

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    amp_enabled = device.type == "cuda"
    config = replace(tiny_config, architecture=ArchitectureConfig("dual_view_v4"),
                     descriptor=DescriptorConfig(mode="full", token_count=1), fingerprint=FingerprintConfig(),
                     model=replace(tiny_config.model, role_embedding=False, dropout=.1, graph_message_mode="residual_blocks"),
                     masking=replace(tiny_config.masking, fusion_only_dropout=True, descriptor_dropout=0),
                     loss=replace(tiny_config.loss, lambda_descriptor=.5, lambda_unimol=.25, lambda_partial_charge=.1))
    vocabulary, samples = tiny_samples
    samples = [{**{k: v for k, v in s.items() if k != "fingerprints"}, "auxiliary_targets": empty_auxiliary_targets()} for s in samples]
    for sample in samples:
        sample["auxiliary_targets"]["unimol"] = torch.ones(768)
        sample["auxiliary_targets"]["unimol_valid"] = torch.tensor(True)
        sample["auxiliary_targets"]["electronic_valid"] = torch.ones(13, dtype=torch.bool)
        count = len(sample["atom_categorical"])
        sample["auxiliary_targets"]["partial_charge"] = torch.arange(count).float() / 10
        sample["auxiliary_targets"]["partial_charge_valid"] = torch.ones(count, dtype=torch.bool)
    probe = MultimodalMasker(vocabulary, config.masking, 400043).apply(MultimodalPacker(vocabulary)(samples), evaluation=True)
    model = DualViewPretrainModel(config, vocabulary).to(device).eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        reference = model(probe.to(device))
    assert reference.loss.item() == pytest.approx(sum(getattr(config.loss, f"lambda_{name}") * value.item() for name, value in reference.losses.items()))
    parameters = list(model.smiles_encoder.parameters()) + list(model.graph_encoder.parameters()) + list(model.fusion.parameters())
    expected = {}
    objectives = {**AUDIT_LOSSES, "partial_charge": "partial_charge"}
    for name, loss in objectives.items():
        gradients = torch.autograd.grad(reference.losses[loss], parameters, retain_graph=True, allow_unused=True)
        expected[name] = torch.cat([g.flatten() for g in gradients if g is not None]).norm().item()
    model.train()
    model.electronic_head.eval()  # Preserve heterogeneous module modes too.
    modes = [module.training for module in model.modules()]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)  # autograd.grad must leave existing .grad untouched.
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if amp_enabled else None
    row = audit_gradient_norms(model, probe, config, device, amp_enabled=amp_enabled, amp_dtype=torch.bfloat16)
    assert modes == [module.training for module in model.modules()]
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    if amp_enabled:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    for name, loss in objectives.items():
        assert row[f"{name}_grad_norm"] == pytest.approx(expected[name], rel=1e-5)
        assert row["weighted_grad_norms"][name] == pytest.approx(abs(getattr(config.loss, f"lambda_{loss}")) * expected[name], rel=1e-5)
        assert row["coverage"][name]["valid_molecules"] > 0
    missing = replace(probe, auxiliary_targets={**probe.auxiliary_targets, "electronic_valid": torch.zeros_like(probe.auxiliary_targets["electronic_valid"])})
    row = audit_gradient_norms(model, missing, config, device, amp_enabled=amp_enabled, amp_dtype=torch.bfloat16)
    assert row["electronic_grad_norm"] is None and row["weighted_grad_norms"]["electronic"] is None
    assert row["coverage"]["electronic"]["status"] == "no_valid_targets"
    missing_atoms = replace(probe, auxiliary_targets={**probe.auxiliary_targets, "partial_charge_valid": torch.zeros_like(probe.auxiliary_targets["partial_charge_valid"])})
    row = audit_gradient_norms(model, missing_atoms, config, device, amp_enabled=amp_enabled, amp_dtype=torch.bfloat16)
    assert row["partial_charge_grad_norm"] is None and row["coverage"]["partial_charge"]["status"] == "no_valid_targets"
    with torch.no_grad():
        model.electronic_head.weight.zero_()
    row = audit_gradient_norms(model, probe, config, device, amp_enabled=amp_enabled, amp_dtype=torch.bfloat16)
    assert row["electronic_grad_norm"] == 0 and row["coverage"]["electronic"]["status"] == "ok"
    # Atom supervision first averages within each molecule, then weights roles.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        model.eval()
        output = model(probe.to(device))
    packed = probe.to(device)
    atom_loss = F.smooth_l1_loss(output.logits["partial_charge"].float(), packed.auxiliary_targets["partial_charge"], reduction="none")
    molecule_losses = torch.stack([atom_loss[packed.graphs.atom_batch == i].mean() for i in range(len(samples))])
    weights = torch.as_tensor(config.loss.role_weights, device=device)[packed.roles]
    assert output.losses["partial_charge"].item() == pytest.approx(((molecule_losses * weights).sum() / weights.sum()).item())
    repeated = [*samples]
    charge = samples[0]["auxiliary_targets"]["partial_charge"]
    repeated[0] = {**samples[0], "auxiliary_targets": {**samples[0]["auxiliary_targets"],
                   "partial_charge": torch.stack([charge, charge + 1]),
                   "partial_charge_valid": torch.ones(2, len(charge), dtype=torch.bool)}}
    repeated_probe = replace(MultimodalPacker(vocabulary)(repeated), masks=probe.masks).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        repeated_output = model(repeated_probe)
    aux = repeated_probe.auxiliary_targets
    element_loss = F.smooth_l1_loss(repeated_output.logits["partial_charge"][aux["charge_atom_indices"]].float(), aux["partial_charge"], reduction="none")
    observation_loss = torch.stack([element_loss[aux["charge_observation_ids"] == i].mean() for i in range(len(aux["charge_observation_roles"]))])
    observation_weights = torch.as_tensor(config.loss.role_weights, device=device)[aux["charge_observation_roles"]]
    assert repeated_output.losses["partial_charge"].item() == pytest.approx(((observation_loss * observation_weights).sum() / observation_weights.sum()).item())
    malformed = [*samples]
    malformed[0] = {**samples[0], "auxiliary_targets": {**samples[0]["auxiliary_targets"], "partial_charge": torch.zeros(2)}}
    with pytest.raises(ValueError, match="real atom count"):
        MultimodalPacker(vocabulary)(malformed)


@pytest.mark.parametrize("override, expected", [(None, 1000), (0, 0), (17, 17), (-1, None)])
def test_stage1_cli_gradient_audit_override(tiny_config, tmp_path, monkeypatch, override, expected):
    import importlib.util
    import sys
    from types import SimpleNamespace

    script = Path(__file__).resolve().parents[1] / "scripts/stage1/train.py"
    spec = importlib.util.spec_from_file_location("stage1_train_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = replace(load_config(script.parents[2] / "configs/v4/stage1/base.yaml"),
                     training=replace(tiny_config.training, device="cpu", gradient_audit_interval_steps=1000))
    captured = []
    monkeypatch.setattr(module, "load_config", lambda _: config)
    monkeypatch.setattr(module, "resolve_stage1_training_identity", lambda _: {})
    run = SimpleNamespace(root=tmp_path, metadata={"attempt_id": "test"},
                          complete=lambda _: None, fail=lambda: None)
    monkeypatch.setattr(module, "open_run_directory", lambda **_: run)
    monkeypatch.setattr(module, "run_training", lambda effective, **_: captured.append(effective) or [])
    monkeypatch.setenv("WORLD_SIZE", "1")
    arguments = [str(script), "--config", "fixture.yaml", "--output", str(tmp_path)]
    if override is not None:
        arguments += ["--gradient-audit-interval-steps", str(override)]
    monkeypatch.setattr(sys, "argv", arguments)
    if expected is None:
        with pytest.raises(ValueError, match="gradient_audit_interval_steps"):
            module.main()
        assert not captured
    else:
        module.main()
        assert captured[0].training.gradient_audit_interval_steps == expected


def test_frozen_regression_affine_initialization_and_molecule_loss(tiny_config):
    from stage1.regression import initialize_regression_head, regression_loss, load_regression_config, regression_tasks
    from stage1.dual_view import DualViewPretrainModel
    config = replace(tiny_config, architecture=ArchitectureConfig("dual_view_v4"),
                     model=replace(tiny_config.model, role_embedding=False),
                     descriptor=DescriptorConfig(mode="full", token_count=1))
    vocabulary = SmilesTokenizer.fit(["CC"], backend="ais")
    torch.manual_seed(123)
    old = DualViewPretrainModel(config, vocabulary)
    torch.manual_seed(123)
    new = DualViewPretrainModel(replace(config, loss=replace(config.loss, lambda_partial_charge=.1)), vocabulary)
    assert not hasattr(old, "partial_charge_head")
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    state = {name: value.clone() for name, value in new.state_dict().items()}
    old_scaler, new_scaler = {"mean": -7., "scale": .4}, {"mean": -5., "scale": 2.}
    for task, width in (("HOMO_eV", 64), ("partial_atomic_charge", 32)):
        head = initialize_regression_head(state, task, old_scaler, new_scaler, torch.device("cpu"))
        inputs = torch.randn(3, width)
        original = new.electronic_head(inputs)[:, 0] if task == "HOMO_eV" else new.partial_charge_head(inputs).squeeze(-1)
        assert torch.allclose(original * .4 - 7, head(inputs).squeeze(-1) * 2 - 5, atol=1e-6, rtol=1e-6)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        head(inputs).sum().backward()
        optimizer.step()
        assert all(torch.equal(value, new.state_dict()[name]) for name, value in state.items())
    prediction = torch.tensor([.5, 1., 2.], requires_grad=True)
    labels = torch.zeros(3)
    ids, counts, roles = torch.tensor([0, 1, 1]), torch.tensor([1, 2]), torch.tensor([0, 2])
    expected = (2 * F.smooth_l1_loss(prediction[:1], labels[:1]) + F.smooth_l1_loss(prediction[1:], labels[1:])) / 3
    assert regression_loss(prediction, labels, ids, counts, roles, (2., 2., 1.)).item() == pytest.approx(expected.item())
    recipe = load_regression_config(ROOT / "configs/v4/stage1/regression_heads.yaml")
    assert (recipe.epochs, recipe.learning_rate, recipe.batch_size) == (10, 1e-4, 128)
    from stage1.predictors import build_predictor
    from stage1.regression import REGRESSION_TASKS
    heads = []
    for task in REGRESSION_TASKS:
        predictor = recipe.resolved_predictor(task)
        atomic = task == "partial_atomic_charge"
        width, hidden, count = (512, (256, 128), 164353) if atomic else (1024, (512, 256), 656385)
        assert (predictor.type, predictor.hidden_dims, predictor.activation, predictor.dropout) == ("mlp", hidden, "gelu", 0.)
        head = build_predictor(predictor, width)
        assert [(m.in_features, m.out_features) for m in head.modules() if isinstance(m, torch.nn.Linear)] == [
            (width, hidden[0]), (hidden[0], hidden[1]), (hidden[1], 1)]
        assert sum(p.numel() for p in head.parameters()) == count
        heads.append(head)
    parameters = [p for head in heads for p in head.parameters()]
    assert len({id(p) for p in parameters}) == len(parameters)
    assert regression_tasks(["LUMO_eV", "HOMO_eV"]) == ("HOMO_eV", "LUMO_eV")
    with pytest.raises(ValueError, match="unique"):
        regression_tasks(["HOMO_eV", "HOMO_eV"])


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_regression_predictor_config_structure_and_rng(tmp_path, device_name):
    import yaml
    from stage1.predictors import PredictorConfig, ResidualPredictorBlock, build_predictor, predictor_from_dict, predictor_rng
    from stage1.regression import RegressionConfig, initialize_regression_head, load_regression_config

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    linear = RegressionConfig()
    assert "predictor" not in linear.to_dict() and "predictor_overrides" not in linear.to_dict()
    mlp = PredictorConfig("mlp", (16, 8), "silu", .2)
    residual = PredictorConfig("residual_mlp", (16, 16), "relu", .1)
    config = replace(linear, predictor=mlp, predictor_overrides={"partial_atomic_charge": residual, "LUMO_eV": PredictorConfig()})
    yaml_path = tmp_path / "regression.yaml"
    yaml_path.write_text(yaml.safe_dump(config.to_dict()))
    assert load_regression_config(yaml_path).to_dict() == config.to_dict()
    assert config.resolved_predictor("HOMO_eV") == mlp
    assert config.resolved_predictor("LUMO_eV") == PredictorConfig()
    assert config.resolved_predictor("partial_atomic_charge") == residual
    for invalid in ({"type": "unknown"}, {"type": "mlp"},
                    {"type": "mlp", "hidden_dims": [True]}, {"type": "mlp", "hidden_dims": [0]},
                    {"type": "mlp", "hidden_dims": "512"}, {"type": "mlp", "hidden_dims": [16], "activation": "tanh"},
                    {"type": "mlp", "hidden_dims": [16], "dropout": float("nan")},
                    {"type": "mlp", "hidden_dims": [16], "dropout": 1.},
                    {"type": "linear", "hidden_dims": [16]}, {"type": "linear", "input_dim": 1024}):
        with pytest.raises(ValueError):
            predictor_from_dict(invalid)
    yaml_path.write_text(yaml.safe_dump({"predictor_overrides": {"not_a_target": {"type": "linear"}}}))
    with pytest.raises(ValueError, match="override target"):
        load_regression_config(yaml_path)
    # Overrides replace the full recipe, rather than inheriting MLP defaults.
    yaml_path.write_text(yaml.safe_dump({"predictor": mlp.to_dict(), "predictor_overrides": {"HOMO_eV": {"type": "residual_mlp"}}}))
    with pytest.raises(ValueError, match="hidden layer"):
        load_regression_config(yaml_path)

    state = {"electronic_head.weight": torch.randn(13, 1024), "electronic_head.bias": torch.randn(13),
             "partial_charge_head.weight": torch.randn(1, 512), "partial_charge_head.bias": torch.randn(1)}
    stats = {"mean": 0., "scale": 1.}
    for task, width in (("HOMO_eV", 1024), ("partial_atomic_charge", 512)):
        for recipe in (PredictorConfig(), mlp, residual):
            head = initialize_regression_head(state, task, stats, stats, device, recipe, seed=71)
            inputs = torch.ones(4, width, device=device)
            outputs = head.eval()(inputs)
            assert outputs.shape == (4, 1) and torch.isfinite(outputs).all()
            outputs.sum().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            if recipe.type == "linear":
                assert set(head.state_dict()) == {"weight", "bias"}
                continue
            repeated = initialize_regression_head(state, task, stats, stats, device, recipe, seed=71)
            assert all(torch.equal(value, repeated.state_dict()[name]) for name, value in head.state_dict().items())
            other = initialize_regression_head(state, task, stats, stats, device, recipe, seed=72)
            assert any(not torch.equal(value, other.state_dict()[name]) for name, value in head.state_dict().items())
            head.train()
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
            with predictor_rng(recipe, 71, device):
                left = head(inputs)
            with predictor_rng(recipe, 72, device):
                head(inputs)
            with predictor_rng(recipe, 71, device):
                right = head(inputs)
            assert torch.equal(left, right) and torch.equal(torch.get_rng_state(), cpu_rng)
            if cuda_rng is not None:
                assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    model = build_predictor(residual, 512).eval()
    assert isinstance(model[0].shortcut, torch.nn.Linear)
    assert isinstance(model[1].shortcut, torch.nn.Identity)
    assert all(not isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)) for module in model.modules())
    block = ResidualPredictorBlock(16, 16, torch.nn.GELU, 0.)
    values = torch.randn(2, 16)
    assert torch.equal(block(values), values + block.residual(values))


@pytest.mark.parametrize("graph_message_mode", ["shared", "residual_blocks"])
def test_dual_view_structure_only_losses_and_downstream_freeze(tiny_config, tiny_samples, monkeypatch, graph_message_mode):
    from stage1.auxiliary import empty_auxiliary_targets
    from stage1.dual_view import DualViewPretrainModel, molecule_loss_statistics, reduce_molecule_elements
    from stage1.masking import sample_modality_dropout
    from stage2.model import encode_object_entities

    config = replace(
        tiny_config, architecture=ArchitectureConfig("dual_view_v4"),
        descriptor=DescriptorConfig(mode="full", token_count=1),
        fingerprint=FingerprintConfig(),
        model=replace(tiny_config.model, role_embedding=False, graph_message_mode=graph_message_mode),
        masking=replace(tiny_config.masking, fusion_only_dropout=True, descriptor_dropout=0,
                        smiles_dropout=0.1, graph_dropout=0.1),
    )
    config.validate()
    assert config_from_dict(config.to_dict()).to_dict() == config.to_dict()
    vocabulary, samples = tiny_samples
    samples = [{**{key: value for key, value in sample.items() if key != "fingerprints"}, "auxiliary_targets": empty_auxiliary_targets()} for sample in samples]
    packed = MultimodalPacker(vocabulary)(samples)
    model = DualViewPretrainModel(config, vocabulary).eval()
    encoded = model.encode_entity(packed)
    altered = replace(packed, descriptors=torch.randn_like(packed.descriptors), roles=packed.roles.flip(0))
    assert torch.equal(model.encode_entity(altered).entity_embedding, encoded.entity_embedding)
    assert encoded.entity_embedding.shape == (3, 64)
    assert encoded.atom_states.shape[1] == 32
    assert not any("role_embedding" in key or "descriptor_encoder" in key for key in model.state_dict())

    choices = sample_modality_dropout(10000, config.masking, torch.Generator().manual_seed(42))
    assert not choices[:, :2].all(1).any()
    assert choices[:, 0].float().mean().item() == pytest.approx(0.1, abs=0.02)
    assert choices[:, 1].float().mean().item() == pytest.approx(0.1, abs=0.02)
    masked = MultimodalMasker(vocabulary, config.masking).apply(packed)
    output = model(masked)
    assert set(output.losses) == {"smiles", "atom", "bond", "alignment", "descriptor", "unimol", "electronic"}
    assert output.losses["unimol"].item() == output.losses["electronic"].item() == 0
    output.loss.backward()
    assert model.unimol_projector.weight.grad is not None
    assert model.electronic_head.weight.grad is not None

    values = torch.tensor([1., 3., 10., 20.], requires_grad=True)
    mean, valid = reduce_molecule_elements(values, torch.ones(4, dtype=torch.bool), torch.tensor([0, 0, 1, 2]), 3)
    stats = molecule_loss_statistics(mean, valid, packed.roles, torch.tensor([2., 2., 1.]))
    assert stats.mean().item() == pytest.approx((2 * 2 + 2 * 10 + 20) / 5)
    # DDP averages gradients: the local objective must divide by GLOBAL weights.
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op: tensor.add_(5))
    distributed = PretrainOutput(stats.mean(), {"descriptor": stats.mean()}, {"descriptor": stats}, {}, mean)
    context = _DistributedContext(rank=0, local_rank=0, world_size=2)
    total, _ = _global_training_losses(distributed, context, config)
    assert total.item() == pytest.approx(2 * 44 / 10)

    # Frozen Stage1 outputs stay detached while a downstream expert learns.
    model.zero_grad(set_to_none=True)
    baseline = {key: value.clone() for key, value in model.state_dict().items()}
    downstream = encode_object_entities(model, packed)
    assert not downstream.entity_embedding.requires_grad and not model.training
    expert = torch.nn.Linear(281, 64)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3)
    expert(downstream.entity_embedding).square().mean().backward()
    assert expert.weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in baseline.items())


@pytest.mark.parametrize("graph_message_mode", ["shared", "residual_blocks"])
def test_dual_view_teacher_electronics_export_and_epoch_resume(tmp_path, monkeypatch, graph_message_mode):
    from stage1.config import AuxiliaryConfig, MaskingConfig
    from stage1.auxiliary import ELECTRONIC_SOURCES, TeacherCache, prepare_teacher_cache, load_electronic_labels
    from stage1.model import load_stage1_model
    from stage1.identity import resolve_stage1_training_identity

    source = tmp_path / "stage1"
    source.mkdir()
    for name, molecules in {"cation": ["[Na+]", "[K+]", "C[NH3+]"], "anion": ["[Cl-]", "[Br-]", "C(=O)[O-]"], "molecule": ["O", "CC", "CCO"]}.items():
        _write_smiles(source / f"{name}.csv", molecules)
    for task, columns, _ in ELECTRONIC_SOURCES:
        root = tmp_path / "simulation" / task
        root.mkdir(parents=True)
        with (root / "train.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["SMILES", *columns])
            writer.writeheader()
            writer.writerow({"SMILES": "CCO", **{column: index + 1 for index, column in enumerate(columns)}})
        (root / "valid.csv").write_text("must not be read")
        (root / "test.csv").write_text("must not be read")
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"isolated fake teacher for contract tests only")
    config = PretrainConfig(
        architecture=ArchitectureConfig("dual_view_v4"),
        data=DataConfig(stage1_dir=source, artifacts_dir=tmp_path / "prepared", valid_fraction=0.4, max_smiles_tokens=64, shard_size=3),
        model=ModelConfig(d_model=16, n_heads=4, smiles_layers=1, graph_depth=2, feedforward_dim=32, dropout=0.1, role_embedding=False, fusion_layers=0, graph_message_mode=graph_message_mode),
        masking=MaskingConfig(fusion_only_dropout=True, descriptor_ratio=0., descriptor_dropout=0.),
        auxiliary=AuxiliaryConfig(simulation_dir=tmp_path / "simulation", teacher_cache=tmp_path / "cache", teacher_checkpoint=checkpoint, teacher_shard_size=2),
        training=TrainingConfig(batch_size=2, epochs=2, device="cpu", amp_dtype="none", num_workers=0, validation_interval_steps=100),
    )
    prepare_corpus(config)
    charge_valid = None
    if graph_message_mode == "residual_blocks":
        from stage1.partial_charge import PartialChargeCache, prepare_partial_charge
        molecule_list = ["[Na+]", "[K+]", "C[NH3+]", "[Cl-]", "[Br-]", "C(=O)[O-]", "O", "CC", "CCO"]
        charge_manifest, charge_valid = _write_charge_resources(tmp_path, molecule_list)
        original_config = config
        config = replace(config, loss=replace(config.loss, lambda_partial_charge=.1),
                         auxiliary=replace(config.auxiliary, partial_charge_manifest=charge_manifest,
                                           partial_charge_cache=tmp_path / "partial_charge"))
        corpus_metadata = json.loads((config.data.artifacts_dir / "metadata.json").read_text())
        before = {name: (config.data.artifacts_dir / name).read_bytes()
                  for name in ["metadata.json", *corpus_metadata["artifact_hashes"]]}
        with pytest.raises(FileNotFoundError):
            PartialChargeCache(config, corpus_metadata)
        sidecar = prepare_partial_charge(config)
        cache = PartialChargeCache(config, corpus_metadata)
        assert "CCC" not in cache.targets and "CCCC" not in cache.targets
        train_structures = set()
        with (config.data.artifacts_dir / "manifest.csv").open() as handle:
            train_structures = {r["canonical_smiles"] for r in csv.DictReader(handle) if r["split"] == "train"}
        from stage1.partial_charge import load_charge_rows
        raw_rows, _ = load_charge_rows(config)
        reference_atoms = np.concatenate([row["targets"] for key in sorted(train_structures) for row in raw_rows[key]])
        assert cache.scaler["mean"] == pytest.approx(reference_atoms.mean())
        assert cache.scaler["scale"] == pytest.approx(reference_atoms.std())
        assert sidecar["matched_molecules"] == 9 and sidecar["source_molecules"] == 10
        assert prepare_partial_charge(config) == sidecar
        assert all((config.data.artifacts_dir / name).read_bytes() == value for name, value in before.items())
        # Conflicting observations remain independent, without changing the corpus.
        charge_file = config.auxiliary.simulation_dir / "partial_atomic_charge/train.csv"
        contents, resource_manifest = charge_file.read_bytes(), charge_manifest.read_bytes()
        duplicate = charge_manifest.parent / "duplicate.mol2"
        duplicate.write_text((charge_manifest.parent / "mol7.mol2").read_text().replace("0.7", "5.0"))
        with charge_manifest.open("a", newline="") as handle:
            csv.writer(handle).writerow(["duplicate", duplicate.name, "mol2", duplicate.stat().st_size,
                                        hashlib.sha256(duplicate.read_bytes()).hexdigest(), "true"])
        with charge_file.open("a") as handle:
            handle.write("CC,duplicate\n")
            handle.write("CCO,mol7\n")
        duplicate_rows, duplicate_audit = load_charge_rows(config)
        assert len(duplicate_rows["CC"]) == 2
        assert [row["mol_id"] for row in duplicate_rows["CC"]] == ["mol7", "duplicate"]
        assert not np.array_equal(*(row["targets"] for row in duplicate_rows["CC"]))
        from stage1.regression import load_regression_rows
        skipped = [row for row in duplicate_audit if row["status"] == "skipped"]
        assert len(skipped) == 1 and skipped[0]["mol_id"] == "mol7"
        assert skipped[0]["canonical_smiles"] == "CCO" and skipped[0]["reason"] == "no_graph_isomorphism"
        assert skipped[0]["csv_line"] == len(duplicate_audit) + 1
        assert skipped[0]["structure_file"] == "mol7.mol2"
        regression_rows = load_regression_rows(config.auxiliary, config, "partial_atomic_charge", "train", audit_dir=tmp_path)
        assert [row["mol_id"] for row in regression_rows if row["canonical_smiles"] == "CC"] == ["mol7", "duplicate"]
        assert [row["mol_id"] for row in regression_rows if row["canonical_smiles"] == "CCO"] == [raw_rows["CCO"][0]["mol_id"]]
        assert json.loads((tmp_path / "partial_charge_train_mapping_audit.json").read_text()) == duplicate_audit
        resource_bytes = duplicate.read_bytes()
        duplicate.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="MOL2 size mismatch"):
            load_charge_rows(config)
        duplicate.write_bytes(resource_bytes)
        duplicate_config = replace(config, auxiliary=replace(config.auxiliary, partial_charge_cache=tmp_path / "duplicate_charge"))
        duplicate_sidecar = prepare_partial_charge(duplicate_config)
        duplicate_cache = PartialChargeCache(duplicate_config, corpus_metadata)
        assert duplicate_cache.targets["CC"].shape == (2, 2)
        assert duplicate_cache.targets["CCO"].shape == (1, 3)
        retained_train_atoms = np.concatenate([row["targets"] for key in sorted(train_structures) for row in duplicate_rows[key]])
        assert duplicate_cache.scaler["mean"] == pytest.approx(retained_train_atoms.mean())
        assert duplicate_cache.scaler["scale"] == pytest.approx(retained_train_atoms.std())
        assert duplicate_sidecar["source_observations"] == len(duplicate_audit) - 1
        assert duplicate_sidecar["skipped_observations"] == 1
        assert duplicate_sidecar["attempted_observations"] == len(duplicate_audit)
        assert json.loads((duplicate_config.auxiliary.partial_charge_cache / "mapping_audit.json").read_text()) == duplicate_audit
        with pytest.raises(ValueError, match="source/corpus"):
            PartialChargeCache(config, corpus_metadata)
        charge_file.write_bytes(contents)
        charge_manifest.write_bytes(resource_manifest)
        targets_path = config.auxiliary.partial_charge_cache / "targets.pt"
        original_targets = targets_path.read_bytes()
        targets_path.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="artifact hash"):
            PartialChargeCache(config, corpus_metadata)
        targets_path.write_bytes(original_targets)
    # Parallel execution preserves conformer coordinates, order and immutable resume.
    parallel_config = replace(config, auxiliary=replace(config.auxiliary, teacher_shard_size=4))
    def coordinate_teacher(molecules):
        # Uni-Mol featurization consumes float32 coordinates.
        return torch.from_numpy(np.stack([mol.GetConformer().GetPositions().astype(np.float32).sum(axis=0) for mol in molecules])).repeat(1, 256)
    serial_root = tmp_path / "serial_cache"
    serial = prepare_teacher_cache(parallel_config, teacher=coordinate_teacher, batch_size=2, output=serial_root)
    parallel_root = tmp_path / "parallel_cache"
    attempts = []
    def fail_after_shard(molecules):
        attempts.append(len(molecules))
        if len(attempts) == 3:
            raise RuntimeError("parallel audit interruption")
        return coordinate_teacher(molecules)
    with pytest.raises(RuntimeError, match="parallel audit interruption"):
        prepare_teacher_cache(parallel_config, teacher=fail_after_shard, batch_size=2, output=parallel_root)
    committed = (parallel_root / "shard_000000.pt").read_bytes()
    parallel = prepare_teacher_cache(parallel_config, teacher=coordinate_teacher, batch_size=2, workers=2, output=parallel_root)
    assert (parallel_root / "shard_000000.pt").read_bytes() == committed
    assert serial["identity"] == parallel["identity"]
    assert serial["valid"] == parallel["valid"] == 9
    for record in serial["shards"]:
        before = torch.load(serial_root / record["path"], weights_only=False)
        after = torch.load(parallel_root / record["path"], weights_only=False)
        assert before["smiles"] == after["smiles"] and before["failures"] == after["failures"]
        assert torch.equal(before["embeddings"], after["embeddings"])
        assert torch.equal(before["valid"], after["valid"])
    with pytest.raises(ValueError, match="workers must be positive"):
        prepare_teacher_cache(config, workers=0)
    # Exercise the real adapter's prepared-feature path without downloading a teacher.
    import sys
    from types import SimpleNamespace
    from stage1.auxiliary import UniMolTeacher
    feature_calls = []
    def featurize(molecule, **kwargs):
        feature_calls.append(Chem.MolToSmiles(Chem.RemoveHs(molecule)))
        if feature_calls[-1] == "CC":
            raise ValueError("unsupported teacher feature")
        return torch.tensor([molecule.GetNumAtoms()], dtype=torch.float32)
    class TeacherModel:
        def batch_collate_fn(self, examples):
            return {"features": torch.stack([item[0] for item in examples])}, None
        def __call__(self, *, features, return_repr):
            return {"cls_repr": features.repeat(1, 768)}
    adapter = object.__new__(UniMolTeacher)
    adapter.model, adapter.device = TeacherModel(), torch.device("cpu")
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "unimol_tools.data.conformer", SimpleNamespace(mol2unimolv2=featurize))
        feature_manifest = prepare_teacher_cache(config, teacher=adapter, batch_size=2, output=tmp_path / "feature_cache")
    assert len(feature_calls) == len(set(feature_calls)) == 9
    assert feature_manifest["valid"] == 8 and feature_manifest["failed"] == 1
    # Missing/failed 3D is masked, never removed from the molecular corpus.
    import stage1.auxiliary as auxiliary
    real_conformer = auxiliary.generate_conformer
    monkeypatch.setattr(auxiliary, "generate_conformer", lambda smiles, seed: (_ for _ in ()).throw(ValueError("audit_failure")) if smiles == "CC" else real_conformer(smiles, seed))
    calls = []
    def teacher(molecules):
        calls.extend(molecules)
        return torch.ones(len(molecules), 768)
    manifest = prepare_teacher_cache(config, teacher=teacher, batch_size=2)
    assert manifest["complete"] and manifest["attempted"] == manifest["unique_structures"] == 9
    assert manifest["failed"] == 1
    assert prepare_teacher_cache(config, teacher=lambda _: pytest.fail("completed cache reran teacher")) == manifest
    # A committed shard survives interruption; changing its source cannot resume.
    interrupted_root = tmp_path / "interrupted_cache"
    attempts = []
    def interrupted_teacher(molecules):
        attempts.append(len(molecules))
        if len(attempts) == 2:
            raise RuntimeError("planned teacher interruption")
        return torch.ones(len(molecules), 768)
    with pytest.raises(RuntimeError, match="planned teacher interruption"):
        prepare_teacher_cache(config, teacher=interrupted_teacher, batch_size=2, output=interrupted_root)
    committed = (interrupted_root / "shard_000000.pt").read_bytes()
    changed = replace(config, auxiliary=replace(config.auxiliary, teacher_version="different"))
    with pytest.raises(ValueError, match="resume recipe mismatch"):
        prepare_teacher_cache(changed, teacher=teacher, output=interrupted_root)
    resumed = prepare_teacher_cache(config, teacher=teacher, batch_size=2, output=interrupted_root)
    assert resumed["complete"] and resumed["attempted"] == 9
    assert (interrupted_root / "shard_000000.pt").read_bytes() == committed
    assert len(load_electronic_labels(config)) == 1
    metadata = json.loads((config.data.artifacts_dir / "metadata.json").read_text())
    cache = TeacherCache(config.auxiliary.teacher_cache, metadata, require_complete=True)
    from stage1.auxiliary import teacher_recipe
    alternate = replace(config, model=replace(config.model, graph_message_mode="residual_blocks" if graph_message_mode == "shared" else "shared"))
    assert teacher_recipe(alternate, metadata) == teacher_recipe(config, metadata)
    assert resolve_stage1_training_identity(alternate) != resolve_stage1_training_identity(config)
    dataset = PreparedCorpusDataset(config.data.artifacts_dir, "train")
    dataset.teacher_cache = cache
    assert all("canonical_smiles" in dataset[index] for index in range(len(dataset)))
    audited_config = replace(config, training=replace(config.training, gradient_audit_interval_steps=1))
    assert resolve_stage1_training_identity(audited_config) == resolve_stage1_training_identity(config)
    if charge_valid is not None:
        assert resolve_stage1_training_identity(config) != resolve_stage1_training_identity(original_config)
    changed_loss = replace(config, loss=replace(config.loss, lambda_descriptor=.5, lambda_unimol=.25))
    assert resolve_stage1_training_identity(changed_loss) != resolve_stage1_training_identity(config)

    output = tmp_path / "train"
    real_save = train_module._save_checkpoint
    def stop_after_epoch(*args, **kwargs):
        real_save(*args, **kwargs)
        if kwargs["completed_epoch"] == 1:
            raise RuntimeError("planned interruption")
    monkeypatch.setattr(train_module, "_save_checkpoint", stop_after_epoch)
    with pytest.raises(RuntimeError, match="planned interruption"):
        run_training(config, output_dir=output)
    monkeypatch.setattr(train_module, "_save_checkpoint", real_save)
    with pytest.raises(ValueError, match="identity"):
        run_training(alternate, output_dir=tmp_path / "incompatible", resume_from=output / "last.pt")
    with pytest.raises(ValueError, match="identity"):
        run_training(changed_loss, output_dir=tmp_path / "incompatible_loss", resume_from=output / "last.pt")
    run_training(config, output_dir=output, resume_from=output / "last.pt")
    encoder = load_stage1_model(output / "stage1_encoder.pt", config.data.artifacts_dir)
    assert encoder.model.entity_dim == 32 and encoder.model.atom_dim == 16
    assert all(key.startswith(("smiles_encoder.", "graph_encoder.", "fusion.")) for key in encoder.model.state_dict())
    exported = torch.load(output / "stage1_encoder.pt", weights_only=False)
    assert exported["fixed_final_epoch"] == 2
    assert exported["training_identity"] == resolve_stage1_training_identity(config)
    assert not any("head" in name for name in exported["model"])
    audited_output = tmp_path / "audited_train"
    # The same complete-epoch recovery path with audit enabled changes no updates.
    monkeypatch.setattr(train_module, "_save_checkpoint", stop_after_epoch)
    with pytest.raises(RuntimeError, match="planned interruption"):
        run_training(audited_config, output_dir=audited_output)
    committed_audits = (audited_output / "gradient_audit.jsonl").read_bytes()
    monkeypatch.setattr(train_module, "_save_checkpoint", real_save)
    run_training(audited_config, output_dir=audited_output, resume_from=audited_output / "last.pt")
    assert (audited_output / "gradient_audit.jsonl").read_bytes().startswith(committed_audits)
    assert (audited_output / "metrics.jsonl").read_bytes() == (output / "metrics.jsonl").read_bytes()
    from common.identity import tensor_state_hash
    original = torch.load(output / "last.pt", weights_only=False)
    audited = torch.load(audited_output / "last.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "scaler"):
        left, right = original[key], audited[key]
        if key == "optimizer":
            left = {**left, "state": {str(k): v for k, v in left["state"].items()}}
            right = {**right, "state": {str(k): v for k, v in right["state"].items()}}
        assert tensor_state_hash("test.audit-state", left) == tensor_state_hash("test.audit-state", right)
    rows = [json.loads(line) for line in (audited_output / "gradient_audit.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in rows] == list(range(1, original["global_step"] + 1))
    assert len({row["probe_hash"] for row in rows}) == len({row["mask_hash"] for row in rows}) == 1
    valid_dataset = PreparedCorpusDataset(config.data.artifacts_dir, "valid")
    valid_ids = {valid_dataset[i]["sample_id"] for i in range(len(valid_dataset))}
    assert all(set(row["probe_ids"]) <= valid_ids and len(row["probe_ids"]) == len(set(row["probe_ids"])) for row in rows)
    from stage1.gradient_audit import prepare_gradient_probe
    valid_dataset.teacher_cache = cache
    _, same_probe = prepare_gradient_probe(valid_dataset, vocabulary=encoder.vocabulary, config=audited_config)
    assert same_probe["probe_hash"] == rows[0]["probe_hash"]
    # Tiny runs exercise positive multiples without adding epoch-final audits.
    interval_output = tmp_path / "interval_train"
    interval_config = replace(audited_config, training=replace(audited_config.training, gradient_audit_interval_steps=4))
    run_training(interval_config, output_dir=interval_output)
    assert [json.loads(line)["global_step"] for line in (interval_output / "gradient_audit.jsonl").read_text().splitlines()] == [4]
    if graph_message_mode == "residual_blocks":
        for label, recipe in (("off", config), ("on", audited_config)):
            mp.spawn(_ddp_training_worker, args=(2, str(tmp_path / f"audit_{label}_init"), recipe,
                                                str(tmp_path / f"ddp_{label}"), None, False), nprocs=2, join=True)
        left = torch.load(tmp_path / "ddp_off/last.pt", weights_only=False)
        right = torch.load(tmp_path / "ddp_on/last.pt", weights_only=False)
        for key in ("model", "optimizer", "scheduler", "scaler"):
            a, b = left[key], right[key]
            if key == "optimizer":
                a = {**a, "state": {str(k): v for k, v in a["state"].items()}}
                b = {**b, "state": {str(k): v for k, v in b["state"].items()}}
            assert tensor_state_hash("test.audit-ddp", a) == tensor_state_hash("test.audit-ddp", b)
        assert (tmp_path / "ddp_off/metrics.jsonl").read_bytes() == (tmp_path / "ddp_on/metrics.jsonl").read_bytes()
        ddp_rows = [json.loads(line) for line in (tmp_path / "ddp_on/gradient_audit.jsonl").read_text().splitlines()]
        assert len(ddp_rows) == left["global_step"]  # Only rank0 appends.
        assert {row["probe_hash"] for row in ddp_rows} == {rows[0]["probe_hash"]}
        mp.spawn(_ddp_training_worker, args=(2, str(tmp_path / "audit_error_init"), audited_config,
                                            str(tmp_path / "ddp_error"), None, False, True), nprocs=2, join=True)
        assert all("partial_charge_grad_norm" in row for row in ddp_rows)
    # Independent post-training never reads test and does not change the source.
    from stage1.regression import RegressionConfig, REGRESSION_TASKS, regression_identity, run_regression, load_regression_head
    for task, columns, _ in ELECTRONIC_SOURCES:
        path = config.auxiliary.simulation_dir / task / "valid.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["SMILES", *columns])
            writer.writeheader()
            writer.writerow({"SMILES": "CCCC", **{column: index + 2 for index, column in enumerate(columns)}})
    if charge_valid is not None:
        with (config.auxiliary.simulation_dir / "partial_atomic_charge" / "valid.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["SMILES", "mol_id"])
            writer.writeheader()
            writer.writerows(charge_valid)
    regression_config = RegressionConfig(stage1_artifacts=config.data.artifacts_dir,
                        simulation_dir=config.auxiliary.simulation_dir,
                        partial_charge_manifest=config.auxiliary.partial_charge_manifest,
                        partial_charge_cache=config.auxiliary.partial_charge_cache,
                        epochs=2, batch_size=2, device="cpu", amp_dtype="none")
    selected = REGRESSION_TASKS if charge_valid is not None else ["HOMO_eV", "LUMO_eV"]
    if charge_valid is None:
        with pytest.raises(ValueError, match="no partial-charge head"):
            regression_identity(regression_config, output / "last.pt")
    source_bytes = (output / "last.pt").read_bytes()
    final_encoder = (output / "stage1_encoder.pt").read_bytes()
    heads = run_regression(regression_config, output / "last.pt", tmp_path / "regression", selected)
    from stage1.predictors import PredictorConfig
    assert "predictor" not in regression_config.to_dict()
    assert regression_identity(replace(regression_config, predictor=PredictorConfig()), output / "last.pt", selected) == heads["identity"]
    assert (output / "last.pt").read_bytes() == source_bytes
    assert (output / "stage1_encoder.pt").read_bytes() == final_encoder
    assert set(heads["tasks"]) == set(selected)
    for task in selected:
        head, head_manifest = load_regression_head(tmp_path / "regression/tasks" / task, output / "last.pt")
        assert head_manifest["format_version"] == 2
        assert head_manifest["predictor"] == {"type": "linear"}
        assert head_manifest["fixed_final_epoch"] == 2
        history = [json.loads(line) for line in (tmp_path / "regression/tasks" / task / "metrics.jsonl").read_text().splitlines()]
        assert head_manifest["validation"] == history[-1]["validation"]
        assert all(row["learning_rate"] == 1e-4 for row in history)
        assert head_manifest["updates"] > 0 and head_manifest["input_dim"] == (16 if task == "partial_atomic_charge" else 32)
        assert all(not p.requires_grad for p in head.parameters())
    if charge_valid is None:
        import stage1.regression as regression_module
        real_rows = regression_module.load_regression_rows
        long_key = "C" * 300
        def rows_with_overlength(*args, **kwargs):
            key = long_key if args[3] == "train" else long_key + "O"
            return real_rows(*args, **kwargs) + [{"canonical_smiles": key,
                "role_id": 2, "targets": np.asarray([1.e9])}]
        with monkeypatch.context() as patch:
            patch.setattr(regression_module, "load_regression_rows", rows_with_overlength)
            filtered = run_regression(regression_config, output / "last.pt", tmp_path / "filtered", selected)
        bank = torch.load(tmp_path / "filtered/representations.pt", weights_only=False)["representations"]
        assert long_key not in bank and long_key + "O" not in bank
        audit = json.loads((tmp_path / "filtered/input_filter_audit.json").read_text())
        assert len(audit["excluded"]) == 2 * len(selected)
        for task in selected:
            assert filtered["tasks"][task]["scaler"] == heads["tasks"][task]["scaler"]
            assert filtered["tasks"][task]["state_hash"] == heads["tasks"][task]["state_hash"]
            assert filtered["input_filter_counts"][task]["train"]["excluded"] == 1
        calls = []
        def worse_validation(*args, **kwargs):
            calls.append(1)
            return {"mae": float(len(calls)), "rmse": float(len(calls)), "molecules": 1}
        with monkeypatch.context() as patch:
            patch.setattr(regression_module, "evaluate_regression_head", worse_validation)
            reported = run_regression(regression_config, output / "last.pt", tmp_path / "reporting_only", selected)
        assert all(reported["tasks"][task]["state_hash"] == heads["tasks"][task]["state_hash"] for task in selected)
    else:
        import stage1.regression as regression_module
        from common.io import atomic_json, atomic_torch_save, sha256_file
        mixed = replace(regression_config, predictor=PredictorConfig("mlp", (16, 8), "gelu", .15),
                        predictor_overrides={"LUMO_eV": PredictorConfig(),
                                             "partial_atomic_charge": PredictorConfig("residual_mlp", (16, 16), "relu", .1)})
        mixed_tasks = ["HOMO_eV", "LUMO_eV", "partial_atomic_charge"]
        mixed_identity = regression_identity(mixed, output / "last.pt", mixed_tasks)
        assert mixed_identity != regression_identity(regression_config, output / "last.pt", mixed_tasks)
        assert regression_identity(replace(regression_config, predictor_overrides={"gap_eV": mixed.predictor}),
                                   output / "last.pt", mixed_tasks) == regression_identity(regression_config, output / "last.pt", mixed_tasks)
        forward = run_regression(mixed, output / "last.pt", tmp_path / "mixed", mixed_tasks)
        with monkeypatch.context() as patch:
            # Exercise reversed execution, not just the CLI's stable sorting.
            patch.setattr(regression_module, "regression_tasks", lambda tasks=None: tuple(tasks) if tasks is not None else REGRESSION_TASKS)
            reverse = run_regression(mixed, output / "last.pt", tmp_path / "mixed_reverse", list(reversed(mixed_tasks)))
        for task in mixed_tasks:
            assert forward["tasks"][task]["state_hash"] == reverse["tasks"][task]["state_hash"]
            assert forward["tasks"][task]["validation"] == reverse["tasks"][task]["validation"]
            loaded, loaded_manifest = load_regression_head(tmp_path / "mixed/tasks" / task, output / "last.pt")
            recipe = mixed.resolved_predictor(task)
            assert loaded_manifest["predictor"] == recipe.to_dict()
            assert loaded_manifest["parameter_count"] == sum(p.numel() for p in loaded.parameters())
            inputs = torch.ones(3, loaded_manifest["input_dim"])
            reference = regression_module.build_predictor(recipe, loaded_manifest["input_dim"]).eval()
            reference.load_state_dict(torch.load(tmp_path / "mixed/tasks" / task / "regression_head.pt", weights_only=False)["model"])
            assert torch.equal(loaded(inputs), reference(inputs))
        # Historical format1 has no predictor metadata and reconstructs a Linear.
        legacy_root = tmp_path / "legacy_head"
        legacy_root.mkdir()
        payload = torch.load(tmp_path / "regression/tasks/HOMO_eV/regression_head.pt", weights_only=False)
        for key in ("predictor", "parameter_count", "initialization"):
            payload.pop(key)
        payload["format_version"] = 1
        atomic_torch_save(legacy_root / "regression_head.pt", payload)
        atomic_json(legacy_root / "regression_head.json", {**{k: v for k, v in payload.items() if k != "model"},
                    "artifact_sha256": sha256_file(legacy_root / "regression_head.pt")})
        legacy_head, legacy_manifest = load_regression_head(legacy_root, output / "last.pt")
        assert legacy_manifest["format_version"] == 1
        assert all(torch.equal(value, payload["model"][name]) for name, value in legacy_head.state_dict().items())
        bad_root = tmp_path / "mixed/tasks/HOMO_eV"
        payload = torch.load(bad_root / "regression_head.pt", weights_only=False)
        payload["predictor"]["hidden_dims"] = [32, 8]
        atomic_torch_save(bad_root / "regression_head.pt", payload)
        atomic_json(bad_root / "regression_head.json", {**{k: v for k, v in payload.items() if k != "model"},
                    "artifact_sha256": sha256_file(bad_root / "regression_head.pt")})
        with pytest.raises(ValueError, match="predictor differs"):
            load_regression_head(bad_root, output / "last.pt")
        assert (output / "last.pt").read_bytes() == source_bytes
        assert (output / "stage1_encoder.pt").read_bytes() == final_encoder
    with pytest.raises(FileExistsError, match="new output"):
        run_regression(regression_config, output / "last.pt", tmp_path / "regression", selected)
    wrong = tmp_path / "wrong_anchor.pt"
    wrong.write_bytes(b"different source")
    with pytest.raises(ValueError, match="different Stage1 checkpoint"):
        load_regression_head(tmp_path / "regression/tasks/HOMO_eV", wrong)
    manifest_path = tmp_path / "regression/tasks/HOMO_eV/regression_head.json"
    corrupt = json.loads(manifest_path.read_text())
    corrupt["state_hash"] = "bad hash"
    manifest_path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="manifest/payload"):
        load_regression_head(manifest_path.parent, output / "last.pt")
    from stage1.dual_view import DualViewEncoder
    from stage1.identity import build_stage1_encoder_identity
    encoder_identity = build_stage1_encoder_identity(model=encoder.model, config=config,
                                                     feature_identity=metadata["semantic"]["identities"]["feature"])
    assert encoder_identity["payload"]["model"].get("graph_message_mode", "shared") == graph_message_mode
    with pytest.raises(RuntimeError, match="state_dict"):
        DualViewEncoder(alternate, encoder.vocabulary).load_state_dict(exported["model"], strict=True)
    shard = config.auxiliary.teacher_cache / manifest["shards"][0]["path"]
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity mismatch"):
        resolve_stage1_training_identity(config)
    with pytest.raises(ValueError, match="integrity mismatch"):
        prepare_teacher_cache(config, teacher=teacher)

def test_formal_stage1_has_one_large_capacity_base_profile() -> None:
    active = load_config(ROOT / "configs/v2/stage1/base.yaml")
    assert active.data.descriptor_dim == 217
    assert active.descriptor.mode == "full"
    assert active.descriptor.token_count == 1
    assert active.model.d_model == 512
    assert "fingerprint" not in active.to_dict()
    assert sorted(path.name for path in (ROOT / "configs/v1/stage1").glob("*.yaml")) == [
        "base.yaml"
    ]
    base = load_config(ROOT / "configs/v1/stage1/base.yaml")
    assert base.loss.role_weights == (2.0, 2.0, 1.0)
    assert "preparation" in base.to_dict()
    assert "preparation" not in base.experiment_dict()

def test_run_directory_allows_execution_change_on_resume(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(outputs_module, "REPOSITORY_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  compile: true\n", encoding="utf-8")
    original = {"training": {"epochs": 2, "compile": True}}
    identity = semantic_identity("test.train", {"epochs": 2})
    run = open_run_directory(
        stage="stage1",
        operation="train",
        config_path="config.yaml",
        config_payload=original,
        semantic_identity=identity,
        output="outputs/train",
        seed=42,
    )
    run.fail()
    resumed = open_run_directory(
        stage="stage1",
        operation="train",
        config_path="config.yaml",
        config_payload={"training": {"epochs": 2, "compile": False}},
        semantic_identity=identity,
        output="outputs/train",
        seed=42,
        resume="outputs/train/last.pt",
    )
    assert resumed.metadata["attempt_id"] != run.metadata["attempt_id"]
    assert resumed.metadata["locator"]["resume"] == "outputs/train/last.pt"

def test_data_identity_records_relative_hash_size_and_rows(tmp_path) -> None:
    source = tmp_path / "data" / "stage1" / "cation.csv"
    source.parent.mkdir(parents=True)
    source.write_text("SMILES\n[Na+]\nC[NH3+]\n", encoding="utf-8")
    identity = write_data_identity(tmp_path, "stage1", [source])
    logical_id = next(iter(identity["locator"]["files"]))
    assert identity["locator"]["files"][logical_id] == "data/stage1/cation.csv"
    source_payload = identity["semantic"]["identities"]["source"]["payload"]
    assert source_payload["sources"][logical_id]["rows"] == 2
    record = identity["integrity"]["files"][logical_id]
    assert record["size"] == source.stat().st_size
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert identity["provenance"]["source_repository_commit"] is None

def _write_smiles(path, values) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES"])
        writer.writeheader()
        writer.writerows({"SMILES": value} for value in values)


def _write_charge_resources(root, molecules):
    """Tiny verified heavy-atom MOL2 resources, including an untouched valid source."""
    resources = root / "resources"
    resources.mkdir()
    entries, train, valid = [], [], []
    for index, smiles in enumerate([*molecules, "CCC", "CCCC"]):
        molecule = Chem.MolFromSmiles(smiles)
        mol_id = f"mol{index}"
        atom_rows = [f"{i + 1} {atom.GetSymbol()}{i + 1} 0 0 0 {'Du' if atom.GetSymbol() == 'Na' else atom.GetSymbol()} 1 MOL {index / 10 + i / 100}"
                     for i, atom in enumerate(molecule.GetAtoms())]
        bond_rows = [f"{i + 1} {bond.GetBeginAtomIdx() + 1} {bond.GetEndAtomIdx() + 1} {int(bond.GetBondTypeAsDouble())}"
                     for i, bond in enumerate(molecule.GetBonds())]
        text = f"@<TRIPOS>MOLECULE\n{mol_id}\n{len(atom_rows)} {len(bond_rows)} 0 0 0\nSMALL\nUSER_CHARGES\n@<TRIPOS>ATOM\n" + "\n".join(atom_rows) + "\n@<TRIPOS>BOND\n" + "\n".join(bond_rows) + "\n"
        path = resources / f"{mol_id}.mol2"
        path.write_text(text)
        entries.append({"mol_id": mol_id, "relative_path": path.name, "format": "mol2",
                        "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "referenced_by_charge": "true"})
        (valid if smiles == "CCCC" else train).append({"SMILES": smiles, "mol_id": mol_id})
    manifest = resources / "structure_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    task = root / "simulation" / "partial_atomic_charge"
    task.mkdir()
    with (task / "train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SMILES", "mol_id"])
        writer.writeheader()
        writer.writerows(train)
    (task / "valid.csv").write_text("must not be read during pretraining")
    (task / "test.csv").write_text("must never be read")
    return manifest, valid


def test_global_rdkit_v2_corpus_uses_format3_without_fingerprints(
    tmp_path, capsys
) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "C[NH3+]"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "CC"])
    artifacts = tmp_path / "prepared"
    base = PretrainConfig()
    config = replace(
        base,
        architecture=ArchitectureConfig(kind="global_rdkit_v2"),
        data=replace(
            base.data,
            stage1_dir=source,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            max_smiles_tokens=64,
            shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="full", token_count=1),
        model=replace(
            base.model,
            d_model=16,
            n_heads=4,
            smiles_layers=1,
            graph_depth=2,
            descriptor_hidden_dim=32,
            descriptor_blocks=2,
            fusion_layers=1,
            feedforward_dim=32,
            dropout=0.0,
        ),
    )

    prepare_corpus(config)
    capsys.readouterr()
    metadata = json.loads((artifacts / "metadata.json").read_text())
    dataset = PreparedCorpusDataset(artifacts, "train")

    assert metadata["format_version"] == GLOBAL_RDKIT_CORPUS_FORMAT_VERSION
    assert metadata["descriptor_dim"] == 217
    assert metadata["descriptor_token_count"] == 1
    assert "fingerprint_kind" not in metadata
    assert "fingerprint_contract" not in metadata
    assert dataset.format_version == GLOBAL_RDKIT_CORPUS_FORMAT_VERSION
    assert "fingerprints" not in dataset[0]

def _ddp_training_worker(
    rank: int,
    world_size: int,
    init_path: str,
    config: PretrainConfig,
    output_dir: str,
    resume_from: str | None,
    stop_after_first_epoch: bool,
    audit_failure: bool = False,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        parameter = torch.tensor(1.0, requires_grad=True)
        local_numerator = parameter * (2.0 if rank == 0 else 9.0)
        statistics = LossStatistics(
            numerators=local_numerator.reshape(1),
            denominators=torch.tensor([2.0 if rank == 0 else 3.0]),
            role_numerators=torch.zeros((1, 3)),
            role_denominators=torch.zeros((1, 3)),
        )
        reduced_loss, _ = _global_training_losses(
            PretrainOutput(
                loss=local_numerator,
                losses={"smiles": local_numerator},
                loss_statistics={"smiles": statistics},
                logits={},
                fused_cls=torch.empty(0),
            ),
            _DistributedContext(rank, world_size, rank),
            config,
        )
        reduced_loss.backward()
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= world_size
        assert parameter.grad.item() == pytest.approx(11.0 / 5.0)

        if audit_failure:
            def fail_probe(*args, **kwargs):
                raise ValueError("intentional probe failure")
            train_module.prepare_gradient_probe = fail_probe
            with pytest.raises(RuntimeError, match="intentional probe failure"):
                run_training(config, output_dir=output_dir)
            dist.barrier()  # Both ranks received the ordinary error.
        elif stop_after_first_epoch:
            real_save = train_module._save_checkpoint

            class StopAfterCheckpoint(RuntimeError):
                pass

            def save_then_stop(paths, **kwargs):
                real_save(paths, **kwargs)
                if kwargs["completed_epoch"] == 1:
                    raise StopAfterCheckpoint

            train_module._save_checkpoint = save_then_stop
            try:
                run_training(config, output_dir=output_dir)
            except StopAfterCheckpoint:
                pass
        else:
            run_training(
                config,
                output_dir=output_dir,
                resume_from=resume_from,
            )
    finally:
        dist.destroy_process_group()

def test_two_rank_gloo_checkpoint_ownership_and_cross_world_resume(tmp_path, capsys) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "[K+]", "C[NH3+]", "C[NH2+]C"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "[Br-]", "[I-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "N", "CC", "CCO"])
    artifacts = tmp_path / "prepared"
    output = tmp_path / "ddp_train"
    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=source,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            max_smiles_tokens=64,
            shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="clean", token_count=1),
        fingerprint=FingerprintConfig(kind="maccs"),
        model=ModelConfig(
            d_model=8,
            n_heads=2,
            smiles_layers=1,
            graph_depth=1,
            descriptor_hidden_dim=16,
            descriptor_blocks=1,
            fusion_layers=1,
            feedforward_dim=16,
            dropout=0.0,
        ),
        training=TrainingConfig(
            batch_size=2,
            epochs=1,
            learning_rate=1.0e-3,
            num_workers=0,
            device="cpu",
            amp_dtype="none",
            compile=False,
            validation_interval_steps=100,
            quick_validation_samples_per_role=1,
        ),
    )
    prepare_corpus(config)
    capsys.readouterr()

    world_size = 2
    mp.spawn(
        _ddp_training_worker,
        args=(
            world_size,
            str(tmp_path / "first_init"),
            config,
            str(output),
            None,
            True,
        ),
        nprocs=world_size,
        join=True,
    )
    mid = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert mid["world_size_at_save"] == 2
    assert mid["global_step"] == 3
    assert mid["completed_epoch"] == 1
    assert "rank_rng" not in mid
    assert "epoch_cursor" not in mid

    mp.spawn(
        _ddp_training_worker,
        args=(
            world_size,
            str(tmp_path / "resume_init"),
            config,
            str(output),
            str(output / "last.pt"),
            False,
        ),
        nprocs=world_size,
        join=True,
    )
    completed = torch.load(
        output / "last.pt", map_location="cpu", weights_only=False
    )
    assert completed["completed_epoch"] == 1
    assert sorted(path.name for path in output.glob("*.pt")) == [
        "checkpoint_epoch_00001.pt",
        "last.pt",
    ]
    metric_steps = [
        json.loads(line)["global_step"]
        for line in (output / "metrics.jsonl").read_text().splitlines()
        if json.loads(line).get("event") != "attempt_start"
    ]
    assert metric_steps == [3]

    assert run_training(
        config, output_dir=output, resume_from=output / "last.pt",
        attempt_id="single-rank-resume",
    ) == []

def test_stage1_epoch_checkpoint_resume_and_attempt_log_preservation(
    tmp_path, capsys, monkeypatch
) -> None:
    source = tmp_path / "stage1"
    source.mkdir()
    _write_smiles(source / "cation.csv", ["[Na+]", "[K+]", "C[NH3+]", "C[NH2+]C"])
    _write_smiles(source / "anion.csv", ["[Cl-]", "[Br-]", "[I-]", "C(=O)[O-]"])
    _write_smiles(source / "molecule.csv", ["O", "N", "CC", "CCO"])
    artifacts = tmp_path / "prepared"
    baseline_output = tmp_path / "baseline"
    output = tmp_path / "train"
    config = PretrainConfig(
        data=DataConfig(
            stage1_dir=source, artifacts_dir=artifacts, valid_fraction=0.5,
            max_smiles_tokens=64, shard_size=2,
        ),
        descriptor=DescriptorConfig(mode="clean", token_count=8),
        fingerprint=FingerprintConfig(kind="both"),
        model=ModelConfig(
            d_model=16, n_heads=4, smiles_layers=1, graph_depth=2,
            descriptor_hidden_dim=32, descriptor_blocks=1, fusion_layers=1,
            feedforward_dim=32, dropout=0.1,
        ),
        training=TrainingConfig(
            batch_size=2, epochs=2,
            learning_rate=1.0e-3, num_workers=0, device="cpu",
            amp_dtype="none", compile=False,
            validation_interval_steps=2, quick_validation_samples_per_role=1,
        ),
    )
    prepare_corpus(config)
    capsys.readouterr()
    validation_calls: list[bool] = []
    real_validate = train_module._validate

    def record_validation(*args, **kwargs):
        validation_calls.append(bool(kwargs["quick"]))
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(train_module, "_validate", record_validation)
    baseline = run_training(config, output_dir=baseline_output, attempt_id="baseline")
    monkeypatch.setattr(train_module, "_validate", real_validate)
    assert [row["global_step"] for row in baseline] == [2, 3, 4, 6]
    assert validation_calls == [True, False, True, False]

    real_save = train_module._save_checkpoint

    class Interrupted(RuntimeError):
        pass

    def interrupt_after_epoch(paths, **kwargs):
        real_save(paths, **kwargs)
        if kwargs["completed_epoch"] == 1:
            raise Interrupted

    monkeypatch.setattr(train_module, "_save_checkpoint", interrupt_after_epoch)
    try:
        run_training(config, output_dir=output, attempt_id="attempt-1")
    except Interrupted:
        pass
    monkeypatch.setattr(train_module, "_save_checkpoint", real_save)

    mid = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert mid["kind"] == STAGE1_CHECKPOINT_KIND
    assert mid["format_version"] == STAGE1_CHECKPOINT_VERSION
    assert mid["completed_epoch"] == 1
    assert mid["global_step"] == 3
    assert set(mid).isdisjoint({"epoch_index", "epoch_cursor", "rank_rng", "micro_step"})
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"global_step": 999, "loss": 0}\n')

    rows = run_training(
        config, output_dir=output, resume_from=output / "last.pt",
        attempt_id="attempt-2",
    )
    assert [row["global_step"] for row in rows] == [4, 6]
    assert [row["loss"] for row in rows] == pytest.approx(
        [row["loss"] for row in baseline[2:]]
    )
    assert (output / "checkpoint_epoch_00001.pt").is_file()
    assert (output / "checkpoint_epoch_00002.pt").is_file()
    assert (output / "last.pt").is_file()
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert last["completed_epoch"] == 2
    metric_rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    attempt_rows = [row for row in metric_rows if row.get("event") == "attempt_start"]
    assert attempt_rows == [
        {
            "event": "attempt_start",
            "attempt_id": "attempt-2",
            "resumed_from_attempt_id": "attempt-1",
            "completed_epoch": 1,
            "global_step": 3,
            "world_size": 1,
            "compile": False,
        }
    ]
    training_rows = [row for row in metric_rows if row.get("event") != "attempt_start"]
    assert [row["global_step"] for row in training_rows] == [2, 3, 999, 4, 6]
    assert [row.get("attempt_id") for row in training_rows] == [
        "attempt-1", "attempt-1", None, "attempt-2", "attempt-2"
    ]

    assert run_training(
        config, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_preparation = replace(
        config, preparation=PreparationConfig(workers=4)
    )
    assert run_training(
        changed_preparation, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_compile = replace(config, training=replace(config.training, compile=True))
    assert run_training(
        changed_compile, output_dir=output, resume_from=output / "last.pt"
    ) == []
    changed_tokenizer = replace(
        config, tokenizer=replace(config.tokenizer, min_frequency=2)
    )
    assert run_training(
        changed_tokenizer, output_dir=output, resume_from=output / "last.pt"
    ) == []

# --- Data preparation and artifact contracts ---

HEADER = [
    "SMILES",
    "formal_charge",
    "origin_list",
    "seed_smiles_list",
    "rule_list",
    "pubchem_cid_list",
    "mol_id_list",
]

def _write_role_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row in rows:
            smiles, charge, *seed = row
            writer.writerow(
                [smiles, charge, "test", seed[0] if seed else "", "", "", ""]
            )

def test_prepare_uses_new_original_sources_and_sharded_artifacts(tmp_path):
    stage1 = tmp_path / "stage1"
    artifacts = tmp_path / "artifacts"
    stage1.mkdir()
    _write_role_csv(stage1 / "cation.csv", [("[Na+]", 1), ("C[NH3+]", 1)])
    _write_role_csv(stage1 / "anion.csv", [("[Cl-]", -1), ("C(=O)[O-]", -1)])
    _write_role_csv(stage1 / "molecule.csv", [("CCO", 0), ("O", 0)])
    for ignored in ("simulation_mol.csv", "solute.csv", "solvent.csv"):
        (stage1 / ignored).write_text("not,a,valid,csv\n", encoding="utf-8")
    (stage1 / "IL.csv").write_text("cation,anion\n[K+],[Br-]\n", encoding="utf-8")

    summary = prepare_corpus(PretrainConfig(
        fingerprint=FingerprintConfig(kind="both"),
        data=DataConfig(
            stage1_dir=stage1,
            artifacts_dir=artifacts,
            valid_fraction=0.5,
            seed=3,
            shard_size=2,
        ),
    ))
    assert summary["total"] == 6
    assert summary["train"] == summary["valid"] == 3
    assert summary["cation"] == summary["anion"] == summary["neutral"] == 2
    assert summary["augmented"] == 0
    assert summary["excluded_entities"] == 0
    assert not (artifacts / "corpus_index.json").exists()
    assert (artifacts / "train_index.npy").is_file()
    assert (artifacts / "valid_index.npy").is_file()
    assert (artifacts / "shard_manifest.json").is_file()
    assert (artifacts / "descriptor_schema.json").is_file()
    assert (artifacts / "excluded_entities.csv").is_file()
    assert not (artifacts / ".prepare.sqlite").exists()
    assert not (artifacts / ".raw_descriptors.npy").exists()
    assert not (artifacts / "preparation_state.json").exists()
    assert len(list((artifacts / "shards").glob("*.pt"))) == 4
    metadata_path = artifacts / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format_version"] == CORPUS_FORMAT_VERSION == 2
    assert metadata["kind"] == CORPUS_KIND
    assert set(metadata["source_hashes"]) == {
        "cation.csv",
        "anion.csv",
        "molecule.csv",
    }
    assert "excluded_entities.csv" in metadata["artifact_hashes"]
    assert "augmentation_audit.json" in metadata["artifact_hashes"]
    dataset = PreparedCorpusDataset(artifacts, "train")
    assert len(dataset) == 3
    sample = dataset[0]
    with (artifacts / "manifest.csv").open() as handle:
        source = next(row for row in csv.DictReader(handle) if row["sample_id"] == sample["sample_id"])
    expected = calculate_fingerprints(
        Chem.MolFromSmiles(source["canonical_smiles"]), FingerprintConfig(kind="both")
    )
    for family, values in expected.items():
        assert torch.equal(sample["fingerprints"][family].float(), torch.from_numpy(values))
    assert set(sample) == {
        "sample_id",
        "role_id",
        "token_ids",
        "atom_categorical",
        "atom_continuous",
        "bond_categorical",
        "bond_index",
        "descriptors",
        "descriptor_valid",
        "fingerprints",
    }
    assert all(value.dtype == torch.uint8 for value in sample["fingerprints"].values())
    vocabulary = SmilesTokenizer.load(artifacts / "tokenizer.json")
    packed = MultimodalPacker(vocabulary)([sample])
    assert all(
        value.dtype == torch.float32 for value in packed.fingerprints.values.values()
    )
    for family, stored in sample["fingerprints"].items():
        assert torch.equal(packed.fingerprints.values[family][0], stored.float())
    batched_dataset = PreparedCorpusDataset(artifacts, "train")
    expected_ids = [batched_dataset[index]["sample_id"] for index in (0, 0, 1)]
    batched = batched_dataset.__getitems__([0, 0, 1])
    assert [item["sample_id"] for item in batched] == expected_ids
    if torch.cuda.is_available():
        pinned = packed.pin_memory()
        assert pinned.token_ids.is_pinned()
        assert pinned.graphs.atom_categorical.is_pinned()
        assert pinned.fusion_layout.smiles_lengths.is_pinned()
        assert all(value.is_pinned() for value in pinned.fingerprints.values.values())

def test_prepare_excludes_qc_failures_before_descriptor_calculation(
    tmp_path, monkeypatch
):
    stage1 = tmp_path / "stage1"
    augmentation = stage1 / "augmentation"
    artifacts = tmp_path / "artifacts"
    augmentation.mkdir(parents=True)
    originals = {
        "cation": [("[Na+]", 1), ("[K+]", 1), ("C[NH3+]", 1), ("C[NH2+]C", 1)],
        "anion": [("[Cl-]", -1), ("[Br-]", -1), ("[I-]", -1), ("C(=O)[O-]", -1)],
        "molecule": [("CCO", 0), ("O", 0), ("N", 0), ("CC", 0)],
    }
    for name, rows in originals.items():
        _write_role_csv(stage1 / f"{name}.csv", rows)
    _write_role_csv(
        augmentation / "cation.csv",
        [],
    )
    _write_role_csv(
        augmentation / "anion.csv",
        [("CC(C)[CH2][AlH-](<-[CH2](C)C)[S](C)(=O)=O", -1)],
    )
    _write_role_csv(
        augmentation / "molecule.csv",
        [("CCCCCCCC", 0), ("P", 0)],
    )

    real_ipc = features_module._calculate_ipc

    def fake_ipc(mol):
        if Chem.MolToSmiles(mol, canonical=True) == "P":
            return IPC_SQUARE_OVERFLOW_LIMIT * 2
        return real_ipc(mol)

    descriptor_smiles = []

    def fake_descriptors(mol, names):
        descriptor_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        return np.arange(len(names), dtype=np.float64)

    monkeypatch.setattr(features_module, "_calculate_ipc", fake_ipc)
    monkeypatch.setattr(prepare_module, "calculate_descriptors", fake_descriptors)
    summary = prepare_corpus(
        PretrainConfig(
            data=DataConfig(
                stage1_dir=stage1,
                artifacts_dir=artifacts,
                valid_fraction=0.25,
                seed=7,
                max_smiles_tokens=8,
                include_augmentation=True,
                shard_size=4,
            )
        )
    )

    assert summary["total"] == 12
    assert summary["excluded_entities"] == 3
    assert "P" not in descriptor_smiles
    assert "CCCCCCCC" not in descriptor_smiles
    assert not any("AlH" in smiles for smiles in descriptor_smiles)
    with (artifacts / "excluded_entities.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        excluded = list(csv.DictReader(handle))
    reasons = {
        row["canonical_smiles"]: set(row["exclusion_reasons"].split(";"))
        for row in excluded
    }
    assert reasons["P"] == {"ipc_square_overflow"}
    assert reasons["CCCCCCCC"] == {"smiles_overlength"}
    dative_reasons = next(
        value for smiles, value in reasons.items() if "AlH" in smiles
    )
    assert "unsupported_bcut_bond_type" in dative_reasons
    metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["quality_control"]["excluded"]["total"] == 3
    assert metadata["augmentation_audit"]["anion"]["retained"] == 0
    assert metadata["augmentation_audit"]["neutral"]["retained"] == 0

def test_augmentation_leakage_and_worker_artifact_semantics(tmp_path):
    stage1 = tmp_path / "stage1"
    augmentation = stage1 / "augmentation"
    augmentation.mkdir(parents=True)
    originals = {
        "cation": [("[Na+]", 1), ("C[NH3+]", 1), ("C[NH2+]C", 1), ("[K+]", 1)],
        "anion": [("[Cl-]", -1), ("[Br-]", -1), ("C(=O)[O-]", -1), ("[I-]", -1)],
        "molecule": [("CCO", 0), ("O", 0), ("N", 0), ("CC", 0)],
    }
    for name, rows in originals.items():
        _write_role_csv(stage1 / f"{name}.csv", rows)
        charge = rows[0][1]
        all_seeds = ";".join(row[0] for row in rows)
        candidate = (
            "C1CC1"
            if name == "molecule"
            else ("C[NH+](C)C" if name == "cation" else "C[S-]")
        )
        _write_role_csv(
            augmentation / f"{name}.csv",
            [
                (candidate, charge, all_seeds),
                (rows[0][0], charge, "unrelated"),
                (candidate, charge, "unrelated"),
                (candidate, charge, "unrelated"),
                ("CCC" if name == "molecule" else ("CC[NH2+]C" if name == "cation" else "CC[S-]"), charge, "unrelated"),
                ("CCCC" if name == "molecule" else ("CCC[NH2+]C" if name == "cation" else "CCC[S-]"), charge, "unrelated"),
            ],
        )

    artifact_dirs = []
    for workers in (1, 4):
        artifacts = tmp_path / f"artifacts_{workers}"
        config = PretrainConfig(
            data=DataConfig(
                stage1_dir=stage1, artifacts_dir=artifacts,
                valid_fraction=0.25, seed=7, include_augmentation=True, shard_size=3,
            ),
            preparation=PreparationConfig(
                workers=workers, catalog_batch_size=2, qc_batch_size=2,
                tokenizer_batch_size=2, descriptor_batch_size=2,
            ),
        )
        summary = prepare_corpus(config)
        artifact_dirs.append(artifacts)
        assert summary["augmented"] == 9
        metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
        for role in ("cation", "anion", "neutral"):
            assert metadata["augmentation_audit"][role] == {
                "included": True,
                "source_rows": 6,
                "excluded_valid_seed": 1,
                "excluded_overlap": 1,
                "excluded_duplicate": 1,
                "eligible": 3,
                "excluded_qc": 0,
                "retained": 3,
            }
        audit = json.loads(
            (artifacts / "augmentation_audit.json").read_text(encoding="utf-8")
        )
        assert audit["roles"] == metadata["augmentation_audit"]
        with (artifacts / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        valid_smiles = {
            row["canonical_smiles"] for row in rows if row["split"] == "valid"
        }
        augmented_seeds = {
            seed
            for row in rows
            if row["is_augmented"] == "1"
            for seed in row["seed_smiles"].split(";")
            if seed
        }
        assert valid_smiles.isdisjoint(augmented_seeds)

    left, right = artifact_dirs
    for filename in (
        "tokenizer.json",
        "descriptor_schema.json",
        "descriptor_scaler.json",
        "manifest.csv",
        "excluded_entities.csv",
        "augmentation_audit.json",
    ):
        assert (left / filename).read_bytes() == (right / filename).read_bytes()
    np.testing.assert_array_equal(
        np.load(left / "train_index.npy"), np.load(right / "train_index.npy")
    )
    np.testing.assert_array_equal(
        np.load(left / "valid_index.npy"), np.load(right / "valid_index.npy")
    )
    for split in ("train", "valid"):
        left_dataset = PreparedCorpusDataset(left, split)
        right_dataset = PreparedCorpusDataset(right, split)
        assert len(left_dataset) == len(right_dataset)
        for index in range(len(left_dataset)):
            left_sample = left_dataset[index]
            right_sample = right_dataset[index]
            assert left_sample.keys() == right_sample.keys()
            for key in left_sample:
                if isinstance(left_sample[key], dict):
                    assert left_sample[key].keys() == right_sample[key].keys()
                    for name in left_sample[key]:
                        assert np.array_equal(
                            left_sample[key][name].numpy(),
                            right_sample[key][name].numpy(),
                        )
                elif hasattr(left_sample[key], "numpy"):
                    assert np.array_equal(
                        left_sample[key].numpy(), right_sample[key].numpy()
                    )
                else:
                    assert left_sample[key] == right_sample[key]



def test_ais_round_trip_and_vocabulary_save_load(tmp_path):
    import atomInSmiles

    smiles = "NCC(=O)O"
    encoded = " ".join(ais_tokenize(smiles))
    decoded = atomInSmiles.decode(encoded)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(decoded)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles)
    )

    vocabulary = SmilesTokenizer.fit([smiles, "CCO"], backend="ais")
    path = tmp_path / "tokenizer.json"
    vocabulary.save(path)
    loaded = SmilesTokenizer.load(path)
    assert loaded.tokens == vocabulary.tokens
    assert loaded.encode(smiles, max_length=32) == vocabulary.encode(
        smiles, max_length=32
    )

def test_smiles_masking_uses_bert_replacement_distribution():
    vocabulary = SmilesTokenizer.fit(["CCO", "NCC"], backend="ais")
    token_ids = np.full(10_002, vocabulary.token_to_id[ais_tokenize("CCO")[0]])
    token_ids[0] = vocabulary.cls_id
    token_ids[-1] = vocabulary.sep_id
    import torch

    token_ids_tensor = torch.from_numpy(token_ids).long()
    positions = torch.arange(1, 10_001)
    corrupted, labels = mask_smiles_tokens(
        token_ids_tensor,
        positions,
        ratio=1.0,
        vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(11),
        drop_entire_modality=False,
    )
    assert (labels != -100).sum().item() == 10_000
    assert labels[0].item() == labels[-1].item() == -100
    mask_fraction = (corrupted[positions] == vocabulary.mask_id).float().mean().item()
    assert 0.77 < mask_fraction < 0.83

def test_descriptor_schema_clean_pruned_groups_and_save_load(tmp_path):
    values = np.asarray(
        [
            [1.0, 2.0, 1.0, np.nan, 4.0, 1.01],
            [2.0, 3.0, 2.0, np.nan, 4.0, 2.01],
            [3.0, 4.0, 3.0, np.nan, 4.0, 3.01],
            [4.0, 5.0, 4.0, np.nan, 4.0, 4.01],
        ]
    )
    names = ("MolWt", "Chi0", "duplicate", "missing", "constant", "Chi1")
    clean = DescriptorSchema.fit(values, names, "clean", 8)
    assert clean.selected_names == ("MolWt", "Chi0", "Chi1")
    assert clean.removal_reasons["missing"] == "all_non_finite"
    assert clean.removal_reasons["constant"] == "zero_variance"
    assert clean.removal_reasons["duplicate"] == "duplicate_of:MolWt"
    assert len(clean.group_indices) == 8
    assert clean.semantic_mapping_version == "rdkit-217-v1"
    assert len(clean.raw_semantic_groups) == len(names)

    pruned = DescriptorSchema.fit(values, names, "pruned", 12, 0.98)
    assert pruned.selected_dim == 1
    assert len(pruned.correlation_clusters) == 1
    assert pruned.cluster_representatives == ("MolWt",)
    path = tmp_path / "schema.json"
    pruned.save(path)
    loaded = DescriptorSchema.load(path, expected_raw_names=names)
    assert loaded == pruned

# --- Scientific model behavior ---

def test_all_five_modalities_use_element_role_weights_and_component_means(
    tiny_config,
    tiny_samples,
) -> None:
    vocabulary, samples = tiny_samples
    batch = MultimodalCollator(
        vocabulary, tiny_config.masking, seed=tiny_config.data.seed
    )(samples)
    model = MultimodalPretrainModel(tiny_config, vocabulary)
    output = model(batch)
    role_weights = torch.tensor(tiny_config.loss.role_weights)

    expected: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    smiles_mask = batch.masks.smiles_labels != -100
    token_roles = batch.roles[:, None].expand_as(smiles_mask)
    expected["smiles"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["smiles"][smiles_mask],
                batch.masks.smiles_labels[smiles_mask],
                reduction="none",
            ),
            token_roles[smiles_mask],
            role_weights,
        )
    ]

    atom_mask = batch.masks.atom_mask
    atom_roles = batch.roles[batch.graphs.atom_batch][atom_mask]
    expected["atom"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["atom"][name][atom_mask],
                batch.graphs.atom_categorical[atom_mask, column],
                reduction="none",
            ),
            atom_roles,
            role_weights,
        )
        for column, name in enumerate(ATOM_FEATURE_NAMES)
    ]

    bond_mask = batch.masks.bond_mask
    bond_roles = batch.roles[batch.graphs.bond_batch][bond_mask]
    expected["bond"] = [
        _weighted_component(
            F.cross_entropy(
                output.logits["bond"][name][bond_mask],
                batch.graphs.bond_categorical[bond_mask, column],
                reduction="none",
            ),
            bond_roles,
            role_weights,
        )
        for column, name in enumerate(BOND_FEATURE_NAMES)
    ]

    descriptor_mask = batch.masks.descriptor_loss_mask
    descriptor_roles = batch.roles[:, None].expand_as(descriptor_mask)
    expected["descriptor"] = [
        _weighted_component(
            F.smooth_l1_loss(
                output.logits["descriptor"][descriptor_mask],
                batch.descriptors[descriptor_mask],
                reduction="none",
            ),
            descriptor_roles[descriptor_mask],
            role_weights,
        )
    ]

    expected["fingerprint"] = []
    for family, logits in output.logits["fingerprint"].items():
        loss_mask = batch.masks.fingerprint_loss_mask[family]
        fingerprint_roles = batch.roles[:, None].expand_as(loss_mask)
        expected["fingerprint"].append(
            _weighted_component(
                F.binary_cross_entropy_with_logits(
                    logits[loss_mask],
                    batch.fingerprints.values[family][loss_mask],
                    reduction="none",
                ),
                fingerprint_roles[loss_mask],
                role_weights,
            )
        )

    for modality, components in expected.items():
        statistics = output.loss_statistics[modality]
        assert torch.allclose(
            statistics.numerators,
            torch.stack([component[0] for component in components]),
        )
        assert torch.equal(
            statistics.denominators,
            torch.stack([component[1] for component in components]),
        )
        assert torch.allclose(output.losses[modality], statistics.mean())


    assert set(output.losses) == {
        "smiles",
        "descriptor",
        "atom",
        "bond",
        "fingerprint",
    }
    assert all(torch.isfinite(loss) for loss in output.losses.values())
    assert output.logits["smiles"].shape[:2] == batch.token_ids.shape
    assert output.logits["descriptor"].shape == (3, 217)
    assert output.logits["fingerprint"]["morgan"].shape == (3, 2048)
    assert output.logits["fingerprint"]["maccs"].shape == (3, 167)
    assert output.fused_cls.shape == (3, tiny_config.model.d_model)

    output.loss.backward()
    required_parameters = [
        model.smiles_encoder.token_embedding.weight,
        model.graph_encoder.atom_mask_feature,
        model.graph_encoder.bond_mask_feature,
        model.descriptor_encoder.group_encoders[4].input_projection[0].weight,
        model.fingerprint_encoder.chunk_encoder[0].weight,
        model.fusion.modality_embedding.weight,
        model.fusion.role_embedding.weight,
        model.smiles_head.bias,
        model.atom_heads["atomic_number"].weight,
        model.bond_heads["bond_type"].weight,
        model.descriptor_heads[4].weight,
        model.fingerprint_heads["morgan"].weight,
    ]
    assert all(parameter.grad is not None for parameter in required_parameters)
def test_regression_overlength_filter_keeps_observations_and_audits():
    from stage1.regression import filter_regression_inputs

    class Vocabulary:
        def token_count(self, key):
            if key == "invalid":
                raise ValueError("tokenization failed")
            return {"short": 255, "boundary": 256, "long": 259}[key]

    data = {"HOMO_eV": {"train": [{"canonical_smiles": "short"},
                                    {"canonical_smiles": "long"}],
                         "valid": [{"canonical_smiles": "boundary"}]},
            "partial_atomic_charge": {"train": [{"canonical_smiles": "long", "mol_id": "a"},
                                                  {"canonical_smiles": "long", "mol_id": "b"}],
                                      "valid": []}}
    filtered, audit = filter_regression_inputs(data, Vocabulary(), 256)
    assert filtered["HOMO_eV"]["train"] == [{"canonical_smiles": "short"}]
    assert filtered["HOMO_eV"]["valid"] == [{"canonical_smiles": "boundary"}]
    assert filtered["partial_atomic_charge"]["train"] == []
    assert audit["tasks"]["HOMO_eV"]["train"] == {"before": 2, "retained": 1, "excluded": 1}
    assert audit["tasks"]["partial_atomic_charge"]["train"] == {"before": 2, "retained": 0, "excluded": 2}
    assert [row.get("mol_id") for row in audit["excluded"]] == [None, "a", "b"]
    assert all(row["token_count"] == 259 and row["reason"] == "smiles_overlength"
               for row in audit["excluded"])
    with pytest.raises(ValueError, match="tokenization failed"):
        filter_regression_inputs({"HOMO_eV": {"train": [{"canonical_smiles": "invalid"}]}}, Vocabulary(), 256)
