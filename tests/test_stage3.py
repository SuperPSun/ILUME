from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from stage1.config import (
    STAGE1_CHECKPOINT_KIND,
    STAGE1_CHECKPOINT_VERSION,
    DataConfig,
    DescriptorConfig,
    FingerprintConfig,
    ModelConfig,
    PretrainConfig,
)
from stage1.data import PreparedCorpusDataset
from stage1.prepare import prepare_corpus
from stage1.model import MultimodalPretrainModel
from common.progress import ProgressReporter
from stage2.config import (
    Stage2Config,
    Stage2DataConfig,
    Stage2InitializationConfig,
)
from common.io import sha256_file
from stage2.model import Stage2AlignmentModel
from stage3.config import (
    AUX6_TASKS,
    IL21_TASKS,
    STAGE3_TASKS,
    Stage3Config,
    Stage3DataConfig,
    Stage3DomainTrainingConfig,
    Stage3InitializationConfig,
    Stage3ModelConfig,
    Stage3TrainingConfig,
    load_stage3_config,
)
from stage3.data import (
    TASK_REGISTRY,
    Stage3TaskDataset,
    SystemCursor,
    fit_fold_scalers,
    sanitize_task,
)
from stage3.evaluate import evaluate_checkpoints
from stage3.model import (
    Expert,
    Stage3MultiDomainModel,
)
from stage3.prepare import (
    load_frozen_embeddings,
    prepare_stage3,
)
from stage3.train import (
    STAGE3_CHECKPOINT_VERSION,
    STAGE3_DOMAIN_MODEL_KIND,
    STAGE3_MODEL_KIND,
    STAGE3_TRAINING_KIND,
    _build_runtime,
    _load_training_checkpoint,
    _train_domain_block,
    run_stage3_training,
)
from stage1.tokenizer import SmilesTokenizer


def _write_csv(
    path: Path, fields: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stage2_config_hash(raw: dict[str, object]) -> str:
    payload = json.loads(json.dumps(raw))
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _task_row(task: str, fold: int) -> dict[str, object]:
    spec = TASK_REGISTRY[task]
    cations = ("[Na+]", "[K+]", "[Li+]", "[Rb+]", "[Cs+]")
    anions = (
        "[Cl-]",
        "[Br-]",
        "[F-]",
        "[I-]",
        "O=[N+]([O-])[O-]",
    )
    solutes = ("C", "CC", "CCC", "CCCC", "CCCCC")
    solvents = ("O", "CO", "CCO", "CCCO", "CCCCO")
    row: dict[str, object] = {
        "temperature_K": 290 + fold,
        "pressure_kPa": 100 + fold,
        "frequency_MHz": 10 + fold,
        "wavelength_nm": 500 + fold,
        spec.target: fold + len(task) / 100,
        "source_list": "test",
    }
    for column, role in zip(
        spec.entity_columns, spec.entity_roles, strict=True
    ):
        if column == "cation":
            row[column] = cations[fold - 1]
        elif column == "anion":
            row[column] = anions[fold - 1]
        elif column in {"solute", "SMILES"}:
            row[column] = solutes[fold - 1]
        elif column == "solvent":
            row[column] = solvents[fold - 1]
        elif role == "cation":
            row[column] = cations[fold - 1]
        elif role == "anion":
            row[column] = anions[fold - 1]
        else:
            row[column] = solutes[fold - 1]
    return row


@pytest.fixture(scope="module")
def tiny_stage3(tmp_path_factory: pytest.TempPathFactory) -> Stage3Config:
    root = tmp_path_factory.mktemp("stage3")
    stage1 = root / "stage1"
    _write_csv(
        stage1 / "cation.csv",
        ["SMILES"],
        [
            {"SMILES": value}
            for value in ("[Na+]", "[K+]", "[Li+]", "[Rb+]", "[Cs+]")
        ],
    )
    _write_csv(
        stage1 / "anion.csv",
        ["SMILES"],
        [
            {"SMILES": value}
            for value in (
                "[Cl-]",
                "[Br-]",
                "[F-]",
                "[I-]",
                "O=[N+]([O-])[O-]",
            )
        ],
    )
    neutral = (
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
        "O",
        "CO",
        "CCO",
        "CCCO",
        "CCCCO",
    )
    _write_csv(
        stage1 / "molecule.csv",
        ["SMILES"],
        [{"SMILES": value} for value in neutral],
    )
    pretrain_dir = root / "pretrain"
    pretrain = PretrainConfig(
        data=DataConfig(
            stage1_dir=stage1,
            artifacts_dir=pretrain_dir,
            valid_fraction=0.2,
            max_smiles_tokens=64,
            shard_size=8,
        ),
        descriptor=DescriptorConfig(mode="full", token_count=8),
        fingerprint=FingerprintConfig(kind="both"),
        model=ModelConfig(
            d_model=16,
            n_heads=4,
            smiles_layers=1,
            graph_depth=2,
            descriptor_hidden_dim=32,
            descriptor_blocks=1,
            fusion_layers=1,
            feedforward_dim=32,
            dropout=0.0,
        ),
    )
    prepare_corpus(pretrain)
    vocabulary = SmilesTokenizer.load(pretrain_dir / "tokenizer.json")
    dataset = PreparedCorpusDataset(pretrain_dir, "train")
    backbone = MultimodalPretrainModel(
        pretrain, vocabulary, dataset.descriptor_schema
    )
    stage1_checkpoint = root / "stage1.pt"
    torch.save(
        {
            "kind": STAGE1_CHECKPOINT_KIND,
            "format_version": STAGE1_CHECKPOINT_VERSION,
            "model": backbone.state_dict(),
            "config": pretrain.to_dict(),
            "artifact_hash": sha256_file(pretrain_dir / "metadata.json"),
        },
        stage1_checkpoint,
    )
    stage2_data = root / "stage2_data"
    stage2_data.mkdir()
    (stage2_data / "metadata.json").write_text(
        '{"format_version": 2}\n', encoding="utf-8"
    )
    stage2_config = Stage2Config(
        data=Stage2DataConfig(
            stage2_dir=root / "unused",
            pretrain_artifacts_dir=pretrain_dir,
            artifacts_dir=stage2_data,
        ),
        initialization=Stage2InitializationConfig(
            checkpoint=stage1_checkpoint
        ),
    )
    stage2_model = Stage2AlignmentModel(
        backbone, head_dropout=stage2_config.model.head_dropout
    )
    raw_stage2 = stage2_config.to_dict()
    stage2_checkpoint = root / "stage2.pt"
    torch.save(
        {
            "format_version": 2,
            "kind": "ilume_stage2_alignment",
            "model": stage2_model.state_dict(),
            "config": raw_stage2,
            "config_hash": _stage2_config_hash(raw_stage2),
            "data_metadata_hash": sha256_file(
                stage2_data / "metadata.json"
            ),
            "pretrain_checkpoint_hash": sha256_file(stage1_checkpoint),
        },
        stage2_checkpoint,
    )
    stage3_dir = root / "stage3_data"
    for task, spec in TASK_REGISTRY.items():
        fields = list(spec.entity_columns) + [
            "temperature_K",
            "pressure_kPa",
            "frequency_MHz",
            "wavelength_nm",
            spec.target,
            "source_list",
        ]
        for fold in range(1, 6):
            _write_csv(
                stage3_dir
                / task
                / spec.fold_strategy
                / f"fold{fold}.csv",
                fields,
                [_task_row(task, fold)],
            )
        _write_csv(
            stage3_dir / task / "test.csv",
            fields,
            [_task_row(task, 1)],
        )
    domain_training = Stage3DomainTrainingConfig(
        batch_size=2,
        max_blocks=3,
        learning_rate=3.0e-4,
        weight_decay=1.0e-4,
        warmup_fraction=0.1,
        max_grad_norm=1.0,
        amp_dtype="none",
        validation_interval_blocks=1,
        early_stopping_patience=5,
        early_stopping_min_delta=1.0e-4,
        backward_mode="domain",
    )
    aux_training = replace(
        domain_training,
        early_stopping_patience=1,
        early_stopping_min_delta=1.0e9,
    )
    config = Stage3Config(
        data=Stage3DataConfig(
            stage3_dir=stage3_dir,
            artifacts_dir=root / "artifacts",
            entity_batch_size=4,
            seed=7,
        ),
        initialization=Stage3InitializationConfig(
            stage2_checkpoint=stage2_checkpoint
        ),
        model=Stage3ModelConfig(dropout=0.2),
        training=Stage3TrainingConfig(
            il21=domain_training,
            aux6=aux_training,
            device="cpu",
            save_every_n_cycles=1,
        ),
    )
    prepare_stage3(config, reporter=ProgressReporter(interactive=False))
    return config


@pytest.fixture(scope="module")
def trained_stage3(
    tiny_stage3: Stage3Config,
) -> tuple[Stage3Config, list[dict[str, Any]]]:
    rows = run_stage3_training(
        tiny_stage3, 1,
        output_dir=tiny_stage3.data.artifacts_dir.parent / "combined_train",
        reporter=ProgressReporter(interactive=False),
    )
    return tiny_stage3, rows


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_stage3_registry_and_formal_configs_are_explicit() -> None:
    assert len(IL21_TASKS) == 21
    assert len(AUX6_TASKS) == 6
    assert len(STAGE3_TASKS) == 27
    assert set(STAGE3_TASKS) == set(TASK_REGISTRY)
    home = load_stage3_config("configs/formal/stage3/reference.yaml")
    assert home.active_domains == ("il21", "aux6")
    assert home.data.artifacts_dir == Path(
        "outputs/formal_v1/stage3/reference/prepare/artifacts"
    )
    assert home.initialization.stage2_checkpoint == Path(
        "outputs/formal_v1/stage2/base/train/best.pt"
    )
    assert home.model.global_experts == home.model.group_experts == 2
    assert home.model.private_experts == 1
    assert home.training.cpu_threads == 4
    assert home.training.cpu_interop_threads == 1
    assert home.training.resident_data
    assert home.training.il21.batch_size == 128
    assert home.training.il21.max_blocks == 5000
    assert home.training.il21.validation_interval_blocks == 50
    assert home.training.aux6.batch_size == 256
    assert home.training.aux6.max_blocks == 2500
    assert home.training.aux6.validation_interval_blocks == 25
    for domain in home.active_domains:
        training = home.domain_training(domain)
        assert training.backward_mode == "domain"
        assert training.batch_size * training.max_blocks == 640000
        assert training.batch_size * training.validation_interval_blocks == 6400
    assert home.training.save_every_n_cycles == 25


def test_stage3_scripts_configure_runtime_before_loading_operation() -> None:
    root = Path(__file__).resolve().parents[1]
    operations = {
        "prepare.py": "from stage3.prepare import prepare_stage3",
        "train.py": "from stage3.train import run_stage3_training",
        "evaluate.py": "from stage3.evaluate import evaluate_checkpoints",
    }
    for filename, operation_import in operations.items():
        source = (root / "scripts" / "stage3" / filename).read_text(
            encoding="utf-8"
        )
        assert source.index("configure_process_runtime(config)") < source.index(
            operation_import
        )


def test_late_solute_is_invisible_to_first_layer_and_global_experts() -> None:
    config = Stage3Config(model=Stage3ModelConfig(dropout=0.0))
    model = Stage3MultiDomainModel(config, 16, seed=3).il21.eval()
    base = torch.randn(3, 16)
    conditions = torch.randn(3, 8)
    phase = torch.zeros(3, dtype=torch.long)
    solute_a = torch.randn(3, 16, requires_grad=True)
    solute_b = torch.randn(3, 16)
    first = model(
        "experiment/solvation",
        base,
        conditions,
        phase,
        solute_cls=solute_a,
    )
    second = model(
        "experiment/solvation",
        base,
        conditions,
        phase,
        solute_cls=solute_b,
    )
    for key in (
        "first_layer_shared",
        "first_layer_group",
        "second_layer_global",
    ):
        assert torch.equal(first.diagnostics[key], second.diagnostics[key])
    assert not torch.equal(
        first.diagnostics["second_layer_local"],
        second.diagnostics["second_layer_local"],
    )
    assert not torch.equal(
        first.diagnostics["task_gate"], second.diagnostics["task_gate"]
    )
    first.predictions.sum().backward()
    assert float(solute_a.grad.abs().sum()) > 0.0
    with pytest.raises(ValueError, match="must not receive solute"):
        model(
            "experiment/pec50",
            base,
            conditions,
            phase,
            solute_cls=solute_b,
        )


def test_aux_heads_and_backward_are_fully_isolated() -> None:
    config = Stage3Config(model=Stage3ModelConfig(dropout=0.0))
    model = Stage3MultiDomainModel(config, 16, seed=5).train()
    parameter_sets = [
        {id(parameter) for parameter in model.aux6.heads[sanitize_task(task)].parameters()}
        for task in AUX6_TASKS
    ]
    for index, first in enumerate(parameter_sets):
        for second in parameter_sets[index + 1 :]:
            assert first.isdisjoint(second)
    base = torch.randn(3, 16)
    conditions = torch.randn(3, 8)
    phase = torch.zeros(3, dtype=torch.long)
    model.il21(
        "experiment/density", base, conditions, phase
    ).predictions.sum().backward()
    il_state = {
        name: value.detach().clone()
        for name, value in model.il21.state_dict().items()
    }
    il_gradients = {
        name: (
            None if parameter.grad is None else parameter.grad.detach().clone()
        )
        for name, parameter in model.il21.named_parameters()
    }
    model.aux6(
        "simulation/charge", base, conditions, phase
    ).predictions.sum().backward()
    for name, value in model.il21.state_dict().items():
        assert torch.equal(value, il_state[name])
    for name, parameter in model.il21.named_parameters():
        expected = il_gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, expected)


def test_expert_and_system_cursor_contracts() -> None:
    expert = Expert(8)
    assert isinstance(expert.layers[0], torch.nn.Linear)
    assert expert.layers[0].in_features == 8
    assert expert.layers[0].out_features == 16
    assert isinstance(expert.layers[3], torch.nn.BatchNorm1d)
    cursor = SystemCursor(
        torch.tensor([0, 3, 5]), torch.arange(5), seed=11
    )
    cursor.next_indices(7)
    state = cursor.state_dict()
    expected = cursor.next_indices(9)
    restored = SystemCursor(
        torch.tensor([0, 3, 5]), torch.arange(5), seed=11
    )
    restored.load_state_dict(state)
    assert torch.equal(restored.next_indices(9), expected)


def test_domain_mode_calls_backward_once(
    tiny_stage3: Stage3Config,
) -> None:
    config = replace(tiny_stage3, active_domains=("il21",))
    device = torch.device("cpu")
    model = Stage3MultiDomainModel(config, 16, seed=13)
    runtime = _build_runtime(config, "il21", 1, model, device)
    original = torch.autograd.backward
    with patch("torch.autograd.backward", wraps=original) as backward:
        _train_domain_block(
            config=config,
            model=model,
            runtime=runtime,
            device=device,
        )
    assert backward.call_count == 1


def test_stage3_v2_prepare_is_domain_separated_and_reproducible(
    tiny_stage3: Stage3Config,
) -> None:
    il_scalers = fit_fold_scalers(tiny_stage3, 1, IL21_TASKS)
    aux_scalers = fit_fold_scalers(tiny_stage3, 1, AUX6_TASKS)
    assert il_scalers["experiment/density"]["target"]["count"] == 4
    assert aux_scalers["simulation/charge"]["target"]["count"] == 4
    metadata = prepare_stage3(
        tiny_stage3, reporter=ProgressReporter(interactive=False)
    )
    assert set(metadata) == {"il21", "aux6"}
    for domain, tasks in (("il21", IL21_TASKS), ("aux6", AUX6_TASKS)):
        root = tiny_stage3.data.artifacts_dir / domain
        assert metadata[domain]["format_version"] == 2
        assert metadata[domain]["domain"] == domain
        assert tuple(metadata[domain]["tasks"]) == tasks
        assert (root / "frozen_embeddings.pt").is_file()
        assert (root / "scalers.json").is_file()
    repeated = prepare_stage3(
        tiny_stage3, reporter=ProgressReporter(interactive=False)
    )
    assert repeated == metadata
    train = Stage3TaskDataset(
        tiny_stage3.data.artifacts_dir,
        "il21",
        1,
        "experiment/solvation",
        "train",
    )
    valid = Stage3TaskDataset(
        tiny_stage3.data.artifacts_dir,
        "il21",
        1,
        "experiment/solvation",
        "valid",
    )
    train_pairs = {
        tuple(map(int, row[:2])) for row in train.entity_ids.tolist()
    }
    valid_pairs = {
        tuple(map(int, row[:2])) for row in valid.entity_ids.tolist()
    }
    assert train_pairs.isdisjoint(valid_pairs)


def test_combined_aux_activity_cannot_change_il_training_or_resume(
    trained_stage3: tuple[Stage3Config, list[dict[str, Any]]],
) -> None:
    tiny_stage3, combined_rows = trained_stage3
    reporter = ProgressReporter(interactive=False)
    assert [row["domain"] for row in combined_rows] == [
        "il21",
        "aux6",
        "il21",
        "aux6",
        "il21",
    ]
    assert combined_rows[-1]["block"] == 3
    assert combined_rows[3]["stopped"] == 1
    il_only = replace(
        tiny_stage3,
        active_domains=("il21",),
    )
    combined_output = tiny_stage3.data.artifacts_dir.parent / "combined_train"
    il_output = tiny_stage3.data.artifacts_dir.parent / "il_only"
    il_rows = run_stage3_training(
        il_only, 1, output_dir=il_output, reporter=reporter
    )
    assert not (il_output / "best.pt").exists()
    combined_il_rows = [
        row for row in combined_rows if row["domain"] == "il21"
    ]
    _assert_nested_equal(combined_il_rows, il_rows)

    combined_final_path = (
        combined_output / "checkpoint_cycle_00000003.pt"
    )
    il_final_path = (
        il_output / "checkpoint_cycle_00000003.pt"
    )
    combined_final = torch.load(
        combined_final_path, map_location="cpu", weights_only=False
    )
    il_final = torch.load(
        il_final_path, map_location="cpu", weights_only=False
    )
    assert combined_final["kind"] == STAGE3_TRAINING_KIND
    assert combined_final["format_version"] == STAGE3_CHECKPOINT_VERSION
    combined_il_model = {
        key: value
        for key, value in combined_final["model"].items()
        if key.startswith("il21.")
    }
    _assert_nested_equal(combined_il_model, il_final["model"])
    _assert_nested_equal(
        combined_final["domains"]["il21"],
        il_final["domains"]["il21"],
    )

    original_final = combined_final
    resume_checkpoint = (
        combined_output / "checkpoint_cycle_00000001.pt"
    )
    replay_rows = run_stage3_training(
        tiny_stage3, 1, output_dir=combined_output,
        resume_from=resume_checkpoint, reporter=reporter,
    )
    expected_replay = [
        row
        for row in combined_rows
        if row["block"] >= 2
    ]
    _assert_nested_equal(replay_rows, expected_replay)
    replay_final = torch.load(
        combined_final_path, map_location="cpu", weights_only=False
    )
    _assert_nested_equal(replay_final, original_final)
    metric_rows = [
        json.loads(line)
        for line in (
            combined_output / "metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [
        row["block"] for row in metric_rows if row["domain"] == "il21"
    ] == [1, 2, 3]
    assert [
        row["block"] for row in metric_rows if row["domain"] == "aux6"
    ] == [1, 2]


def test_domain_bests_are_assembled_and_evaluation_runs(
    trained_stage3: tuple[Stage3Config, list[dict[str, Any]]],
) -> None:
    tiny_stage3, _ = trained_stage3
    output = tiny_stage3.data.artifacts_dir.parent / "combined_train"
    combined = torch.load(
        output / "best.pt", map_location="cpu", weights_only=False
    )
    assert combined["kind"] == STAGE3_MODEL_KIND
    assert combined["active_domains"] == ["il21", "aux6"]
    for domain in ("il21", "aux6"):
        domain_best = torch.load(
            output / f"best_{domain}.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert domain_best["kind"] == STAGE3_DOMAIN_MODEL_KIND
        for key, value in domain_best["model"].items():
            assert torch.equal(combined["model"][f"{domain}.{key}"], value)
    checkpoint_root = output.parent / "evaluation_folds"
    for fold in range(1, 6):
        fold_dir = checkpoint_root / f"fold{fold}"
        fold_dir.mkdir(parents=True)
        payload = dict(combined)
        payload["fold"] = fold
        torch.save(payload, fold_dir / "best.pt")
    result = evaluate_checkpoints(
        tiny_stage3,
        checkpoint_root,
        split="test",
        ensemble_folds=True,
    )
    assert result["ensemble_folds"] == 5
    assert set(result["tasks"]) == set(STAGE3_TASKS)
    il_checkpoint_root = output.parent / "il_evaluation_folds"
    il_best = torch.load(
        output / "best_il21.pt", map_location="cpu", weights_only=False
    )
    il_config = replace(tiny_stage3, active_domains=("il21",))
    for fold in range(1, 6):
        fold_dir = il_checkpoint_root / f"fold{fold}"
        fold_dir.mkdir(parents=True)
        payload = dict(il_best)
        payload["fold"] = fold
        torch.save(payload, fold_dir / "best_il21.pt")
    il_result = evaluate_checkpoints(
        il_config,
        il_checkpoint_root,
        split="test",
        ensemble_folds=True,
    )
    assert set(il_result["tasks"]) == set(IL21_TASKS)


def test_stage3_v1_and_evaluation_checkpoints_are_not_resumable(
    tmp_path: Path,
) -> None:
    v1 = tmp_path / "v1.pt"
    torch.save(
        {"format_version": 1, "kind": STAGE3_TRAINING_KIND}, v1
    )
    with pytest.raises(ValueError, match="v2 training checkpoint"):
        _load_training_checkpoint(v1)
    evaluation = tmp_path / "best.pt"
    torch.save(
        {"format_version": 2, "kind": STAGE3_MODEL_KIND}, evaluation
    )
    with pytest.raises(ValueError, match="v2 training checkpoint"):
        _load_training_checkpoint(evaluation)


def test_stage3_frozen_cache_hash_mismatch_is_rejected(
    tiny_stage3: Stage3Config, tmp_path: Path
) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(tiny_stage3.data.artifacts_dir, copied)
    config = replace(
        tiny_stage3,
        data=replace(tiny_stage3.data, artifacts_dir=copied),
    )
    path = copied / "aux6" / "frozen_embeddings.pt"
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="embedding hash mismatch"):
        load_frozen_embeddings(config, "aux6")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_aux_cuda_amp_overflow_cannot_change_il_runtime(
    tiny_stage3: Stage3Config,
) -> None:
    fp16 = replace(
        tiny_stage3,
        training=replace(
            tiny_stage3.training,
            il21=replace(tiny_stage3.training.il21, amp_dtype="fp16"),
            aux6=replace(tiny_stage3.training.aux6, amp_dtype="fp16"),
            device="cuda",
        ),
    )
    device = torch.device("cuda")
    model = Stage3MultiDomainModel(fp16, 16, seed=9).to(device)
    il_runtime = _build_runtime(fp16, "il21", 1, model, device)
    aux_runtime = _build_runtime(fp16, "aux6", 1, model, device)
    assert il_runtime.store.device.type == "cuda"
    assert aux_runtime.store.device.type == "cuda"
    assert il_runtime.store.entity.device.type == "cuda"
    view = il_runtime.store.view_for(
        il_runtime.train["experiment/solvation"]
    )
    assert view.base_indices.device.type == "cuda"
    assert view.conditions.device.type == "cuda"
    assert view.targets.device.type == "cuda"
    assert view.solute_indices is not None
    assert view.solute_indices.device.type == "cuda"
    assert il_runtime.cursors["experiment/solvation"].offsets.device.type == "cpu"
    il_snapshot = {
        "model": {
            name: value.detach().clone()
            for name, value in model.il21.state_dict().items()
        },
        "optimizer": il_runtime.optimizer.state_dict(),
        "scheduler": il_runtime.scheduler.state_dict(),
        "scaler": il_runtime.scaler.state_dict(),
        "cursors": {
            task: cursor.state_dict()
            for task, cursor in il_runtime.cursors.items()
        },
        "task_order_rng": il_runtime.task_order_rng.getstate(),
        "torch_rng": il_runtime.torch_rng.state_dict(),
        "block": il_runtime.block,
        "micro_step": il_runtime.micro_step,
    }
    initial_aux_scale = aux_runtime.scaler.get_scale()
    aux_runtime.store.entity.fill_(1.0e30)
    aux_runtime.store.neutral.fill_(1.0e30)
    _train_domain_block(
        config=fp16,
        model=model,
        runtime=aux_runtime,
        device=device,
    )
    assert aux_runtime.scaler.get_scale() < initial_aux_scale
    current_il = {
        "model": model.il21.state_dict(),
        "optimizer": il_runtime.optimizer.state_dict(),
        "scheduler": il_runtime.scheduler.state_dict(),
        "scaler": il_runtime.scaler.state_dict(),
        "cursors": {
            task: cursor.state_dict()
            for task, cursor in il_runtime.cursors.items()
        },
        "task_order_rng": il_runtime.task_order_rng.getstate(),
        "torch_rng": il_runtime.torch_rng.state_dict(),
        "block": il_runtime.block,
        "micro_step": il_runtime.micro_step,
    }
    _assert_nested_equal(current_il, il_snapshot)
