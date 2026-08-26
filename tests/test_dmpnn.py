from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


pytest.importorskip("chemprop")

from benchmarks.common.config import load_benchmark_config
from benchmarks.common.data import BenchmarkTask, RawDataset
from benchmarks.common.engine import TargetStats
from benchmarks.common.environment import validate_dmpnn_environment
from benchmarks.dmpnn.adapter import (
    ConditionStats,
    DMPNNTrainingBundle,
    _partial_dataset,
    _predict,
    _prepare_scalar,
    _scalar_dataset,
    build_dmpnn_model,
    train_dmpnn_bundle,
)
from common.identity import semantic_identity


def _task(component_count: int, *, atom: bool = False) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark="stage2_physics",
        task_id="simulation/partial_atomic_charge" if atom else "simulation/tiny",
        slots=tuple(f"component_{index}" for index in range(component_count)),
        condition_columns=() if atom else ("temperature_K", "pressure_kPa"),
        target_columns=("partial_charge",) if atom else ("value",),
        audit_columns=(),
        train_paths=(Path("train.csv"),),
        valid_paths=(Path("valid.csv"),),
        test_path=Path("test.csv"),
        fold=None,
        meta_group=None,
        registry_payload={"test": True},
    )


def _scalar_bundle(component_count: int) -> DMPNNTrainingBundle:
    smiles = ("CC", "O", "[Na+]")[:component_count]
    rows = tuple(tuple(smiles) for _ in range(4))
    targets = np.asarray([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    conditions = np.asarray(
        [[290.0, 100.0], [300.0, 110.0], [310.0, 120.0], [320.0, 130.0]],
        dtype=np.float64,
    )
    raw = RawDataset(
        components=rows,
        component_count=component_count,
        conditions=conditions,
        targets=targets,
        source_rows=tuple(f"tiny:{index}" for index in range(2, 6)),
        audit_rows=({}, {}, {}, {}),
    )
    target_stats = TargetStats.fit(targets)
    condition_stats = ConditionStats.fit(conditions)
    dataset = _scalar_dataset(raw, target_stats, condition_stats)
    return DMPNNTrainingBundle(
        task=_task(component_count),
        train_dataset=dataset,
        valid_dataset=dataset,
        target_stats=target_stats,
        condition_stats=condition_stats,
        source_hashes={},
        training_identity=semantic_identity(
            "benchmark.training.v1", {"synthetic_components": component_count}
        ),
        target_level="molecule",
        component_count=component_count,
    )


@pytest.mark.parametrize("component_count", [1, 2, 3])
def test_dmpnn_component_blocks_are_independent_and_conditions_enter_predictor(
    component_count: int,
) -> None:
    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    bundle = _scalar_bundle(component_count)
    model = build_dmpnn_model(config, bundle)
    assert model.predictor.input_dim == component_count * 300 + 2
    if component_count == 1:
        assert model.message_passing.output_dim == 300
    else:
        blocks = list(model.message_passing.blocks)
        assert len(blocks) == component_count
        assert len({id(block) for block in blocks}) == component_count
        assert model.message_passing.shared is False
    first = bundle.train_dataset.datasets[0] if component_count > 1 else bundle.train_dataset
    assert first.d_xd == 2
    if component_count > 1:
        assert all(dataset.d_xd == 0 for dataset in bundle.train_dataset.datasets[1:])


def test_condition_stats_are_train_only_and_zero_variance_safe() -> None:
    stats = ConditionStats.fit(np.asarray([[10.0, 5.0], [20.0, 5.0]]))
    assert stats.mean == pytest.approx((15.0, 5.0))
    assert stats.scale == pytest.approx((5.0, 1.0))
    transformed = stats.normalize(np.asarray([[25.0, 5.0], [5.0, 7.0]]))
    np.testing.assert_allclose(transformed, [[2.0, 0.0], [-2.0, 2.0]])


def test_environment_lock_mismatch_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    monkeypatch.setattr(
        "benchmarks.common.environment._locked_versions",
        lambda _path: {"chemprop": "0.0.0"},
    )
    with pytest.raises(RuntimeError, match="does not match its lock"):
        validate_dmpnn_environment(config)


def test_partial_dataset_uses_prepared_rows_offsets_and_canonical_atom_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.csv"
    source.write_text(
        "mol_id,SMILES,role,formal_charge,source_list\n"
        "mol_1,OC,neutral,0,synthetic\n",
        encoding="utf-8",
    )
    audit = tmp_path / "partial_charge_mapping_audit.csv"
    audit.write_text(
        "mol_id,canonical_smiles,model_atom_count,status\n"
        "mol_1,CO,2,mapped\n",
        encoding="utf-8",
    )
    prepared = SimpleNamespace(
        source_rows=torch.tensor([2]),
        mol_ids=("mol_1",),
        atom_target_offsets=torch.tensor([0, 2]),
        atom_target_values=torch.tensor([-0.25, 0.25]),
        atom_target_mask=torch.tensor([True, True]),
    )
    dataset = _partial_dataset(prepared, source, audit)
    assert dataset.data[0].name == "mol_1"
    assert dataset.smiles == ["CO"]
    np.testing.assert_allclose(dataset.data[0].atom_y[:, 0], [-0.25, 0.25])


def test_orbital_bundle_fits_one_scaler_over_pooled_cation_and_anion_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    train_path.write_text("train\n", encoding="utf-8")
    valid_path.write_text("valid\n", encoding="utf-8")
    task = BenchmarkTask(
        benchmark="stage2_physics",
        task_id="simulation/homo",
        slots=("SMILES",),
        condition_columns=(),
        target_columns=("HOMO_eV",),
        audit_columns=(
            "ion_role",
            "provenance_source_file",
            "provenance_source_row",
        ),
        train_paths=(train_path,),
        valid_paths=(valid_path,),
        test_path=tmp_path / "test.csv",
        fold=None,
        meta_group=None,
        registry_payload={"pooled": True},
    )
    train = RawDataset(
        components=(("[Na+]",), ("[Cl-]",)),
        component_count=1,
        conditions=np.empty((2, 0)),
        targets=np.asarray([[-2.0], [-4.0]]),
        source_rows=("train:2", "train:3"),
        audit_rows=(
            {"ion_role": "cation"},
            {"ion_role": "anion"},
        ),
    )
    valid = RawDataset(
        components=(("[K+]",), ("[Br-]",)),
        component_count=1,
        conditions=np.empty((2, 0)),
        targets=np.asarray([[-1.0], [-5.0]]),
        source_rows=("valid:2", "valid:3"),
        audit_rows=(
            {"ion_role": "cation"},
            {"ion_role": "anion"},
        ),
    )
    monkeypatch.setattr("benchmarks.dmpnn.adapter.resolve_task", lambda *_: task)
    rows = iter((train, valid))
    monkeypatch.setattr("benchmarks.dmpnn.adapter.load_split", lambda *_: next(rows))
    bundle = _prepare_scalar(
        load_benchmark_config("configs/benchmarks/dmpnn.yaml"),
        "stage2_physics",
        "simulation/homo",
        None,
    )
    assert bundle.target_stats.mean == pytest.approx((-3.0,))
    assert bundle.target_stats.scale == pytest.approx((1.0,))
    assert bundle.component_count == 1


@pytest.mark.parametrize("component_count", [1, 2, 3])
def test_one_epoch_scalar_and_multicomponent_save_reload_smoke(
    tmp_path: Path, component_count: int,
) -> None:
    from chemprop.models.utils import load_model

    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    config = replace(
        config,
        training={
            **config.training,
            "batch_size": 2,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "warmup_epochs": 0,
        },
    )
    output = tmp_path / f"components-{component_count}"
    summary = train_dmpnn_bundle(config, _scalar_bundle(component_count), output)
    assert summary["epochs_ran"] == 1
    assert (output / "model.pt").is_file()
    first = load_model(
        output / "model.pt", multicomponent=component_count > 1
    )
    second = load_model(
        output / "model.pt", multicomponent=component_count > 1
    )
    dataset = _scalar_bundle(component_count).valid_dataset
    np.testing.assert_allclose(
        _predict(first, dataset, atom=False),
        _predict(second, dataset, atom=False),
        rtol=0,
        atol=0,
    )
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["best_valid_raw_mae"] == pytest.approx(
        checkpoint["best_valid_normalized_mae"]
        * checkpoint["target_statistics"]["scale"][0]
    )


def test_one_epoch_atom_property_save_reload_smoke(tmp_path: Path) -> None:
    from chemprop.data import MolAtomBondDatapoint, MolAtomBondDataset
    from chemprop.models.utils import load_model

    config = load_benchmark_config("configs/benchmarks/dmpnn.yaml")
    config = replace(
        config,
        training={
            **config.training,
            "batch_size": 2,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "warmup_epochs": 0,
        },
    )
    data = [
        MolAtomBondDatapoint.from_smi(
            smiles,
            atom_y=np.asarray(values, dtype=np.float32).reshape(-1, 1),
            reorder_atoms=False,
        )
        for smiles, values in (
            ("CC", (-1.0, 1.0)),
            ("CO", (-0.5, 0.5)),
            ("CN", (-0.25, 0.25)),
            ("CF", (-0.75, 0.75)),
        )
    ]
    dataset = MolAtomBondDataset(data)
    bundle = DMPNNTrainingBundle(
        task=_task(1, atom=True),
        train_dataset=dataset,
        valid_dataset=dataset,
        target_stats=TargetStats((0.0,), (1.0,)),
        condition_stats=ConditionStats((), ()),
        source_hashes={},
        training_identity=semantic_identity(
            "benchmark.training.v1", {"synthetic_atom": True}
        ),
        target_level="atom",
        component_count=1,
    )
    output = tmp_path / "atom"
    summary = train_dmpnn_bundle(config, bundle, output)
    assert summary["epochs_ran"] == 1
    model = load_model(output / "model.pt", mol_atom_bond=True)
    assert model.atom_predictor.input_dim == 300
    assert model.atom_constrainer is None
    assert model.bond_predictor is None
