from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import random
from types import SimpleNamespace

import pytest
import torch

from common.identity import semantic_identity
from common.training import canonical_json_sha256
from generation.chemistry import ChemicalLibrary, Constraints, FragmentEnvironment, State, canonical, charge, extract_resources
from generation.model import DedicatedGFN
from generation.reward import ApplicabilityDomain, RewardOracle
from generation.runtime import prepare_protocol
from generation.sampling import hypervolume, sample_candidates
from generation.tasks import DesignRequest, Objective
from generation.training import load_checkpoint, rollout, train, trajectory_balance
from stage3.inference import FrozenPropertyPredictor, PredictionQuery, PropertyCapability


TASK = "experiment/viscosity"
OTHER = "experiment/density"


@pytest.fixture
def generation_library():
    return ChemicalLibrary({"format_version": 1, "status": "approved",
        "review": {"chemistry": "temporary test fixture", "synthesis": "temporary test fixture"},
        "anions": [{"id": "a", "smiles": "[Cl-]", "family": "halide", "sources": ["fixture"]},
            {"id": "b", "smiles": "O=S(=O)([O-])[O-]", "family": "sulfate", "sources": ["fixture"]}],
        "cores": [{"id": "c", "smiles": "[1*][N+](C)(C)C", "family": "ammonium", "sources": ["fixture"]},
            {"id": "d", "smiles": "[1*][N+](C)(C)CC[N+](C)(C)C", "family": "dication", "sources": ["fixture"]}],
        "fragments": [{"id": "m", "smiles": "[1*:1]C", "sources": ["fixture"]},
            {"id": "e", "smiles": "[1*:1]CC", "sources": ["fixture"]},
            {"id": "l", "smiles": "[1*:1]C[1*]", "sources": ["fixture"]}]})


class TinyPredictor:
    def __init__(self):
        self.identity = semantic_identity("generation.test.predictor", {"version": 1})
        self.capabilities = {
            TASK: PropertyCapability(TASK, "viscosity_mPa*s_log10", "mPa*s", "log10",
                ("temperature_K",), False, 1.0, 1.0, {"temperature_K": {"mean": 300., "scale": 10.}}),
            OTHER: PropertyCapability(OTHER, "density_g/cm^3", "g/cm^3", "identity",
                (), False, 1., 1., {})}
        self.calls = []

    def capability_table(self):
        return [asdict(self.capabilities[t]) for t in sorted(self.capabilities)]

    def reference_rows(self, task):
        return iter([{"cation": "C[N+](C)(C)C", "anion": "[Cl-]", "temperature_K": "300", "solute": "O=C=O"}])

    def predict(self, queries):
        self.calls.extend(queries)
        results = []
        for query in queries:
            result = {}
            for task, conditions in query.conditions.items():
                cap = self.capabilities[task]
                value = len(query.cation) / 100 + conditions.get("temperature_K", 300) / 300
                folds = [value + delta / 100 for delta in range(-2, 3)]
                result[task] = {"native": value, "value": cap.to_physical(value), "std_native": math.sqrt(.0002),
                    "folds_native": folds, "unit": cap.unit, "transform": cap.transform, "source": "experiment"}
            results.append(result)
        return results


def request(predictor, *, both=False, fixed=""):
    objectives = [Objective(TASK, "minimize", "mPa*s", 100., 1.)]
    conditions = {TASK: {"temperature_K": 300.}}
    if both:
        objectives.append(Objective(OTHER, "maximize", "g/cm^3", 1., 1.))
        conditions[OTHER] = {}
    return DesignRequest.from_dict({"objectives": [asdict(o) for o in objectives], "conditions": conditions,
        "constraints": {"max_atoms": 8}, "fixed_anion": fixed}, predictor.capabilities)


def oracle_for(predictor, cache_path=None):
    domain = ApplicabilityDomain({task: list(predictor.reference_rows(task)) for task in predictor.capabilities}, predictor.capabilities)
    return RewardOracle(predictor, domain, cache_path=cache_path)


def enumerate_dag(environment, constraints, fixed=""):
    children, todo = {}, [State()]
    while todo:
        state = todo.pop()
        if state in children:
            continue
        children[state] = environment.children(state, constraints, fixed)
        todo.extend(children[state])
    return children


def test_reversible_canonical_dag_and_exact_reward_distribution(generation_library):
    environment = FragmentEnvironment(generation_library, 8)
    constraints = Constraints(max_atoms=8)
    for fixed in ("", "a", "b"):
        graph = enumerate_dag(environment, constraints, fixed)
        parents = {s: environment.parents(s, constraints, fixed) for s in graph}
        for state, children in graph.items():
            assert len(children) == len(set(children))
            for child in children:
                assert state in parents[child]
            assert all(state in graph[p] for p in parents[state])
        terminals = [s for s in graph if s.terminal]
        rewards = {s: float(len(s.cation) + (3 if s.anion == "b" else 1)) for s in terminals}
        flows = dict(rewards)
        def flow(state):
            if state not in flows:
                flows[state] = sum(flow(child) / len(parents[child]) for child in graph[state])
            return flows[state]
        z = flow(State())
        assert z == pytest.approx(sum(rewards.values()))
        def visit(state, probability):
            if state.terminal:
                return {state: probability}
            values = {}
            for child in graph[state]:
                conditional = flows[child] / len(parents[child]) / flows[state] if flows[state] else 0.
                for terminal, mass in visit(child, probability * conditional).items():
                    values[terminal] = values.get(terminal, 0.) + mass
            return values
        distribution = visit(State(), 1.)
        assert sum(distribution.values()) == pytest.approx(1.)
        for terminal, reward in rewards.items():
            assert distribution[terminal] == pytest.approx(reward / z)
        class ExactPolicy:
            log_z = torch.tensor(math.log(z), dtype=torch.float64)
            def log_probabilities(self, state, children):
                return torch.tensor([math.log(flows[child] / len(parents[child]) / flows[state])
                    if flows[child] else -math.inf for child in children], dtype=torch.float64)
        def complete_paths(state, path):
            if state.terminal:
                yield path
            else:
                for child in graph[state]:
                    yield from complete_paths(child, path + [child])
        condition = request(TinyPredictor(), fixed=fixed)
        for path in complete_paths(State(), [State()]):
            assert float(trajectory_balance(ExactPolicy(), environment, condition, path,
                math.log(rewards[path[-1]]))) < 1e-24
        assert any(len(p) > 1 for p in parents.values())
        if fixed:
            assert {s.anion for s in terminals} == {fixed}
    assert charge("C[N+](C)(C)CC[N+](C)(C)C") == 2
    assert charge(generation_library.anions["b"]["smiles"]) == -2


def test_resource_review_invalid_chemistry_and_constraints(generation_library, tmp_path):
    payload = deepcopy(generation_library.payload)
    payload["status"] = "draft"
    with pytest.raises(ValueError, match="reviewed"):
        ChemicalLibrary(payload)
    with pytest.raises(ValueError, match="connected"):
        canonical("[Cl-].[Cl-]")
    payload = deepcopy(generation_library.payload)
    payload["fragments"][0]["smiles"] = "[1*:1][NH3+]"
    with pytest.raises(ValueError, match="preserve net charge"):
        ChemicalLibrary(payload)
    payload = deepcopy(generation_library.payload)
    payload["anions"].append({**payload["anions"][0], "id": "duplicate"})
    with pytest.raises(ValueError, match="distinct canonical"):
        ChemicalLibrary(payload)
    with pytest.raises(ValueError, match="neutral and single"):
        canonical("[1*+][N+](C)(C)C", ports=True)
    environment = FragmentEnvironment(generation_library, 8)
    with pytest.raises(ValueError, match="outside"):
        environment.children(State(), Constraints(max_atoms=8), "unknown")
    forbidden = Constraints(max_atoms=8, forbidden_smarts=("[N+]",))
    assert not environment.children(State("a"), forbidden)
    impossible = Constraints(max_atoms=4)
    state = State("a", generation_library.cores["c"]["smiles"])
    assert not environment.children(state, impossible)
    for kind, smiles in (("cation", "C[N+](C)(C)C"), ("anion", "[Cl-]")):
        (tmp_path / f"{kind}.csv").write_text("SMILES\n" + smiles + "\n")
    extracted = extract_resources(tmp_path / "cation.csv", tmp_path / "anion.csv")
    assert extracted["status"] == "draft"
    assert extracted["cores"][0]["pretraining_seen"]


def test_fixed_request_units_and_conditions():
    predictor = TinyPredictor()
    condition = request(predictor, both=True)
    raw = condition.to_dict()
    raw["objectives"][0]["unit"] = "unknown"
    with pytest.raises(ValueError, match="unit"):
        DesignRequest.from_dict(raw, predictor.capabilities)
    raw = request(predictor).to_dict()
    raw["conditions"][TASK] = {}
    with pytest.raises(ValueError, match="conditions"):
        DesignRequest.from_dict(raw, predictor.capabilities)
    assert predictor.capabilities[TASK].to_native(100., "mPa*s") == 2.
    with pytest.raises(ValueError, match="positive"):
        predictor.capabilities[TASK].to_native(0., "mPa*s")


def test_reward_cache_reuses_fixed_request_and_isolates_physical_context(tmp_path):
    predictor = TinyPredictor()
    path = tmp_path / "cache.jsonl"
    oracle = oracle_for(predictor, path)
    condition = request(predictor, both=True)
    first = oracle.evaluate("C[N+](C)(C)C", "[Cl-]", condition)
    assert len(predictor.calls) == 1
    assert oracle.new_queries == 2
    second = oracle.evaluate("C[N+](C)(C)C", "[Cl-]", condition)
    assert oracle.new_queries == 2 and first == second
    other_request = deepcopy(condition)
    other_request.conditions[TASK]["temperature_K"] = 310.
    oracle.evaluate("C[N+](C)(C)C", "[Cl-]", other_request)
    assert oracle.new_queries == 3
    oracle_for(predictor, path)
    predictor.execution_identity = semantic_identity("generation.test.execution", {"version": 2})
    with pytest.raises(ValueError, match="cache identity"):
        oracle_for(predictor, path)
    del predictor.execution_identity
    lines = path.read_text().splitlines()
    corrupt = json.loads(lines[0]); corrupt["prediction"]["native"] += 1
    path.write_text(json.dumps(corrupt) + "\n")
    with pytest.raises(ValueError, match="cache identity"):
        oracle_for(predictor, path)


def test_neural_tb_checkpoint_resume_and_candidate_shortfall(generation_library, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(10)
    predictor = TinyPredictor()
    oracle = oracle_for(predictor)
    environment = FragmentEnvironment(generation_library, 8)
    model = DedicatedGFN(generation_library, width=8, depth=1)
    condition = request(predictor, fixed="a")
    condition.objectives = (replace(condition.objectives[0], threshold=1e-10),)
    states = rollout(model, environment, condition, random.Random(10))
    assert states and states[-1].terminal
    loss = trajectory_balance(model, environment, condition, states, -2.)
    loss.backward()
    assert model.log_z.ndim == 0 and model.log_z.grad is not None
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters())
    settings = {"learning_rate": .001, "steps": 4, "batch_size": 2, "replay_batch_size": 2,
        "replay_capacity": 16, "max_grad_norm": 1., "checkpoint_interval": 2}
    identity = semantic_identity("generation.dedicated-gfn.v1", {"resources": generation_library.identity,
        "request": condition.to_dict()})
    complete = tmp_path / "complete"; complete.mkdir()
    train(model, environment, oracle, condition, random.Random(8), settings, complete, identity)
    resumed = DedicatedGFN(generation_library, width=8, depth=1)
    retry = tmp_path / "retry"; retry.mkdir()
    train(resumed, environment, oracle_for(predictor), condition, random.Random(8), settings, retry, identity,
        resume=complete / "checkpoint_step_0000002.pt")
    for key, value in model.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    assert all(query.conditions == condition.conditions for query in predictor.calls)
    selected, report = sample_candidates(model, environment, oracle, condition, random.Random(4),
        attempts=5, count=3, raw_path=tmp_path / "raw.jsonl")
    assert selected == [] and report["shortfall"] == 3
    with pytest.raises(ValueError, match="source identity"):
        load_checkpoint(complete / "gfn_final.pt", model, semantic_identity("generation.test.other", {}))
    legacy = tmp_path / "legacy.pt"
    legacy.write_bytes((complete / "gfn_final.pt").read_bytes())
    manifest = json.loads((complete / "gfn_final.json").read_text())
    manifest["kind"] = "ilume_universal_gfn_v1"
    legacy.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="universal or historical"):
        load_checkpoint(legacy, model, identity)
    assert hypervolume([[1, 2], [2, 1]]) == 3.


def test_protocol_freezes_request_and_only_selected_property_references():
    predictor = TinyPredictor()
    condition = request(predictor)
    protocol = prepare_protocol(predictor, condition)
    assert protocol["identity"] == canonical_json_sha256(protocol["payload"])
    assert protocol["payload"]["test_labels_used"] is False
    assert protocol["payload"]["format_version"] == 2
    assert protocol["payload"]["request"] == condition.to_dict()
    assert set(protocol["payload"]["references"]) == {TASK}
    assert [cap["task_id"] for cap in protocol["payload"]["capabilities"]] == [TASK]
    assert all("target" not in row for rows in protocol["payload"]["references"].values() for row in rows)


def test_generation_entrypoint_lifecycle_and_query_budget(monkeypatch, generation_library, tmp_path):
    import yaml
    import generation.runtime as runtime
    import common.outputs as outputs
    predictor = TinyPredictor()
    monkeypatch.setattr(outputs, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "FrozenPropertyPredictor", lambda *a, **k: predictor)
    (tmp_path / "resources.json").write_text(json.dumps(generation_library.payload))
    config = {"seed": 42, "predictor": {"config_path": "unused.yaml", "checkpoint_dir": "unused"},
        "resources": {"library_path": "resources.json", "protocol_path": "protocol/protocol.json"},
        "request": request(predictor, both=True).to_dict(),
        "model": {"width": 8, "depth": 1, "anion_conditioned": True},
        "reward": {"beta": 4., "uncertainty_penalty": 1., "domain_penalty": 1., "epsilon": 1e-8},
        "training": {"steps": 2, "batch_size": 1, "replay_batch_size": 1, "replay_capacity": 4,
            "learning_rate": .001, "max_grad_norm": 1., "checkpoint_interval": 1},
        "sampling": {"attempts": 3, "count": 2, "max_property_queries": 1}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    runtime.main("prepare", ["--config", "config.yaml", "--kind", "protocol", "--output", "protocol"])
    runtime.main("train", ["--config", "config.yaml", "--output", "train"])
    assert json.loads((tmp_path / "train/metadata.json").read_text())["status"] == "completed"
    arguments = ["--config", "config.yaml", "--checkpoint", "train/gfn_final.pt", "--output", "design"]
    runtime.main("sample", arguments)
    summary = json.loads((tmp_path / "design/summary.json").read_text())
    assert summary["query_budget_exhausted"]
    assert summary["new_property_queries"] == 0
    assert summary["returned"] == 0
    with pytest.raises(FileExistsError):
        runtime.main("sample", arguments)
    evaluation = ["--config", "config.yaml", "--checkpoint", "train/gfn_final.pt", "--output", "eval"]
    runtime.main("evaluate", evaluation)
    assert json.loads((tmp_path / "eval/summary.json").read_text())["request"] == json.loads(
        (tmp_path / "train/request.json").read_text())
    changed = deepcopy(config)
    changed["request"]["objectives"][0]["threshold"] += 1
    (tmp_path / "changed.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="Fixed design request"):
        runtime.main("sample", ["--config", "changed.yaml", "--checkpoint", "train/gfn_final.pt", "--output", "changed_design"])
    assert not (tmp_path / "changed_design").exists()
    changed["resources"]["protocol_path"] = "changed_protocol/protocol.json"
    (tmp_path / "changed.yaml").write_text(yaml.safe_dump(changed))
    runtime.main("prepare", ["--config", "changed.yaml", "--kind", "protocol", "--output", "changed_protocol"])
    for operation, selector in (("sample", "--checkpoint"), ("train", "--resume")):
        with pytest.raises(ValueError, match="source identity"):
            runtime.main(operation, ["--config", "changed.yaml", selector, "train/gfn_final.pt", "--output", f"changed_{operation}"])
        assert json.loads((tmp_path / f"changed_{operation}/metadata.json").read_text())["status"] == "failed"
    for operation, old_flag in (("train", ["--request", "request.yaml"]),
            ("evaluate", ["--partition", "test"])):
        with pytest.raises(SystemExit):
            runtime.main(operation, ["--config", "config.yaml", "--output", "unused", *old_flag])


@pytest.mark.parametrize("field", ("properties", "direction", "threshold", "weight", "temperature", "solute", "constraints", "fixed_anion"))
def test_fixed_request_identity_rejects_all_scientific_changes(monkeypatch, generation_library, tmp_path, field):
    import generation.runtime as runtime
    import common.outputs as outputs
    from generation.training import save_checkpoint
    predictor = TinyPredictor()
    predictor.capabilities[OTHER] = replace(predictor.capabilities[OTHER], requires_solute=True)
    raw = request(TinyPredictor(), both=True).to_dict()
    raw["solutes"] = {OTHER: "O=C=O"}
    condition = DesignRequest.from_dict(raw, predictor.capabilities)
    monkeypatch.setattr(outputs, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / "library.json").write_text(json.dumps(generation_library.payload))
    (tmp_path / "protocol.json").write_text(json.dumps(prepare_protocol(predictor, condition)))
    config = {"resources": {"library_path": "library.json", "protocol_path": "protocol.json"},
        "request": raw, "model": {"width": 8, "depth": 1}, "reward": {}, "training": {}, "seed": 42}
    *_, model, identity = runtime.build_components(config, device="cpu", predictor=predictor)
    save_checkpoint(tmp_path / "gfn.pt", model, torch.optim.Adam(model.parameters()), identity, 0, [],
        random.Random(4), [], oracle_for(predictor))
    changed = deepcopy(raw)
    if field == "properties":
        changed["objectives"] = [o for o in changed["objectives"] if o["task"] == TASK]
        del changed["conditions"][OTHER]
        changed["solutes"] = {}
    elif field in {"direction", "threshold", "weight"}:
        changed["objectives"][0][field] = {"direction": "minimize", "threshold": 3., "weight": 2.}[field]
    elif field == "temperature":
        changed["conditions"][TASK]["temperature_K"] += 10
    elif field == "solute":
        changed["solutes"][OTHER] = "O"
    elif field == "constraints":
        changed["constraints"]["max_atoms"] += 1
    else:
        changed["fixed_anion"] = "b"
    config["request"] = changed
    with pytest.raises(ValueError, match="Fixed design request"):
        runtime.build_components(config, device="cpu", predictor=predictor)
    new_request = DesignRequest.from_dict(changed, predictor.capabilities)
    (tmp_path / "protocol_new.json").write_text(json.dumps(prepare_protocol(predictor, new_request)))
    config["resources"]["protocol_path"] = "protocol_new.json"
    *_, new_model, new_identity = runtime.build_components(config, device="cpu", predictor=predictor)
    assert identity["hash"] != new_identity["hash"]
    with pytest.raises(ValueError, match="source identity"):
        load_checkpoint(tmp_path / "gfn.pt", new_model, new_identity)
    with pytest.raises(ValueError, match="Training request"):
        train(model, FragmentEnvironment(generation_library, 8), oracle_for(predictor), new_request,
            random.Random(4), {}, tmp_path, identity)


def test_frozen_public_inference_matches_native_ensemble_and_checks_inputs(monkeypatch, tmp_path):
    import stage3.inference as inference
    calls, models = [], []
    class Encoder:
        encoder_identity = semantic_identity("generation.test.encoder", {})
        def encode_slots(self, objects):
            return torch.ones(len(objects), 1, 1241), torch.zeros(len(objects), 1, dtype=torch.long)
    class Model(torch.nn.Module):
        def __init__(self, fold):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(float(fold)))
            self.task_specs = {TASK: SimpleNamespace(enabled=True, primary_slots=("cation", "anion"),
                partner_slots=(), condition_columns=("temperature_K",), target_column="viscosity_mPa*s_log10")}
        def forward(self, task, primary, conditions, **kwargs):
            assert not self.training and not self.weight.requires_grad and not torch.is_grad_enabled()
            return SimpleNamespace(predictions=primary.values[:, 0, 0] * self.weight + conditions[:, 0])
    def loader(config, prepared, path, fold, epoch, device, **kwargs):
        assert kwargs == {"three_phase_final": True}
        calls.append(fold)
        model = Model(fold); models.append(model)
        return model, {"normalization": {TASK: {"target": {"mean": 0., "scale": .1},
            "conditions": {"temperature_K": {"mean": 300., "scale": 10.}}}},
            "training_identity": {"hash": str(fold)}}, None
    for fold in range(1, 6):
        directory = tmp_path / f"fold{fold}"; directory.mkdir()
        (directory / "three_phase_final.pt").write_text(str(fold))
    config = SimpleNamespace(is_entity_home=True, runtime_backend=None,
        initialization=SimpleNamespace(stage1_encoder="unused", stage1_artifacts_dir="unused", simulation_artifacts_dir=None),
        training=SimpleNamespace(amp_dtype="fp32"))
    monkeypatch.setattr(inference, "load_prepared_stage3", lambda c: {"metadata": {"registry_hash": "r"}})
    monkeypatch.setattr(inference, "load_frozen_stage1_entities", lambda *a, **k: Encoder())
    monkeypatch.setattr(inference, "_load_model", loader)
    monkeypatch.setattr(inference, "_validate_three_phase_manifest", lambda *a: "verified")
    predictor = FrozenPropertyPredictor(config, tmp_path)
    query = PredictionQuery("C[N+](C)(C)C", "[Cl-]", {TASK: {"temperature_K": 300.}})
    result = predictor.predict([query])[0][TASK]
    assert calls == [1, 2, 3, 4, 5]
    assert result["native"] == pytest.approx(.3)
    assert result["value"] == pytest.approx(10 ** .3)
    assert result["value"] != pytest.approx(sum(10 ** p for p in result["folds_native"]) / 5)
    assert all(p.grad is None and not p.requires_grad for m in models for p in m.parameters())
    with pytest.raises(ValueError, match="conditions"):
        predictor.predict([replace(query, conditions={TASK: {}})])
    with pytest.raises(ValueError, match="positive cation"):
        predictor.predict([replace(query, cation="CC")])
    def rejected(*args, **kwargs):
        raise ValueError("Stage 3 checkpoint prepared-data identity mismatch")
    monkeypatch.setattr(inference, "_load_model", rejected)
    with pytest.raises(ValueError, match="identity mismatch"):
        FrozenPropertyPredictor(config, tmp_path)
