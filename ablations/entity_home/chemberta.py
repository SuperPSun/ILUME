"""Frozen ChemBERTa-2 sources; no ILUME Stage1 or descriptor computation."""
import importlib.metadata
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from common.entity_roles import molecular_role
from common.frozen_encoder import register_frozen_encoder
from common.identity import semantic_identity, tensor_state_hash, require_compatible_identity
from common.io import sha256_file


SOURCES = {
    "chemberta2_mtr": ("DeepChem/ChemBERTa-77M-MTR", "66b895cab8adebea0cb59a8effa66b2020f204ca"),
    "chemberta2_mlm": ("DeepChem/ChemBERTa-77M-MLM", "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"),
}
FILES = {
    "chemberta2_mtr": {
        "config.json": "6c15d3fd119919afb92b762c4e55cd46a595251ab799435168baa75a6ebd07e6",
        "pytorch_model.bin": "7882205587d3985c07e425632d0e57ba0fc2fc778e3b47aab02aef9b7961407f",
        "tokenizer.json": "1197005224bc8917f46f8d6fc360aec71c6626858966e167b2ff8af62214f814",
    },
    "chemberta2_mlm": {
        "config.json": "b9ec477f76a173461741cb0f339b89661c9d9249af2f6a84df7593c3baede9b0",
        "pytorch_model.bin": "95571af26b61ba8d56b9d61ff108b5c9afe96fd69ea3cbd1901d9421e3d2dd6d",
        "tokenizer.json": "1197005224bc8917f46f8d6fc360aec71c6626858966e167b2ff8af62214f814",
    },
}
VERSIONS = {"transformers": "5.3.0", "tokenizers": "0.22.2", "safetensors": "0.6.2"}
EXTRACTION = {"pooling": "last_hidden_cls", "native_dim": 384, "learned_padding": 640,
              "descriptor_padding": 217, "include_rdkit": False, "max_tokens": 512,
              "dtype": "float32", "fp32_matmul_precision": "ieee",
              "attention": "eager", "tokenizer": "serialized_json"}


def validate_spec(spec):
    if (spec.kind != "chemberta2" or spec.variant not in SOURCES
            or (spec.repo_id, spec.revision) != SOURCES[spec.variant]
            or spec.pooling != "last_hidden_cls" or spec.include_rdkit is not False):
        raise ValueError("ChemBERTa source must match the registered frozen C12/C13 contract")


def check_dependencies():
    for package, expected in VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Missing {package}; install the documented chemberta optional dependencies") from error
        if actual != expected:
            raise RuntimeError(f"ChemBERTa requires {package}=={expected}; found {actual}")


def source_features(spec, bundle=None):
    validate_spec(spec)
    if bundle is None:
        root = Path(spec.snapshot_dir)
        for filename, digest in FILES[spec.variant].items():
            if sha256_file(root / filename) != digest:
                raise ValueError(f"ChemBERTa snapshot SHA mismatch: {filename}")
        features = {name: json.loads((root / name).read_text()) for name in ("config.json", "tokenizer.json")}
    else:
        import hashlib
        features = bundle["features"]
        for filename in ("config.json", "tokenizer.json"):
            raw = bundle["feature_bytes"][filename]
            if (hashlib.sha256(raw).hexdigest() != FILES[spec.variant][filename]
                    or json.loads(raw) != features[filename]):
                raise ValueError(f"Embedded ChemBERTa feature SHA mismatch: {filename}")
        if bundle["source_files"] != FILES[spec.variant] or bundle["extraction"] != EXTRACTION:
            raise ValueError("Embedded ChemBERTa source or extraction mismatch")
        if bundle["spec"] != spec.to_dict():
            raise ValueError("Embedded ChemBERTa version mismatch")
    return features


def strict_encoder_state(raw, encoder, variant):
    encoder_state = {name.removeprefix("roberta."): value for name, value in raw.items() if name.startswith("roberta.")}
    ignored = ({"norm_mean", "norm_std", "regression.dense.weight", "regression.dense.bias",
                "regression.out_proj.weight", "regression.out_proj.bias"}
               if variant == "chemberta2_mtr" else
               {"lm_head.bias", "lm_head.dense.weight", "lm_head.dense.bias", "lm_head.layer_norm.weight",
                "lm_head.layer_norm.bias", "lm_head.decoder.weight", "lm_head.decoder.bias"})
    unexpected = set(raw) - {"roberta." + name for name in encoder_state} - ignored
    if unexpected:
        raise ValueError(f"Unexpected ChemBERTa checkpoint parameters: {sorted(unexpected)}")
    for name in ("embeddings.position_ids", "embeddings.token_type_ids"):
        if name in encoder_state and name not in encoder.state_dict():
            value = encoder_state.pop(name)
            expected = torch.arange(value.shape[-1])[None, :] if name.endswith("position_ids") else torch.zeros_like(value)
            if value.ndim != 2 or value.shape[0] != 1 or not torch.equal(value, expected):
                raise ValueError("Invalid legacy ChemBERTa position/type buffer")
    encoder.load_state_dict(encoder_state, strict=True)


class ChemBERTaEncoder(nn.Module):
    def __init__(self, spec, *, bundle=None):
        super().__init__()
        validate_spec(spec)
        check_dependencies()
        from tokenizers import Tokenizer
        from transformers import RobertaConfig, RobertaModel

        self.config = spec
        self.features = source_features(spec, bundle)
        self.feature_bytes = ({name: (Path(spec.snapshot_dir) / name).read_bytes() for name in self.features}
                              if bundle is None else bundle["feature_bytes"])
        self.tokenizer = Tokenizer.from_str(self.feature_bytes["tokenizer.json"].decode())
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        if self.pad_id is None or self.cls_id is None:
            raise ValueError("ChemBERTa serialized tokenizer lacks PAD/CLS")
        config = RobertaConfig.from_dict(deepcopy(self.features["config.json"]))
        if config.hidden_size != 384:
            raise ValueError("ChemBERTa must produce native 384D states")
        config._attn_implementation = "eager"
        self.encoder = RobertaModel(config, add_pooling_layer=False).float()
        if bundle is None:
            raw = torch.load(Path(spec.snapshot_dir) / "pytorch_model.bin", map_location="cpu", weights_only=True)
            strict_encoder_state(raw, self.encoder, spec.variant)
        else:
            self.load_state_dict(bundle["state"], strict=True)
        self.feature_identity = semantic_identity("ablation.chemberta2.features.v1", {
            "repo_id": spec.repo_id, "revision": spec.revision,
            "source_files": FILES[spec.variant], "extraction": EXTRACTION})
        state_hash = tensor_state_hash("stage1.encoding-state", self.state_dict())
        self.encoder_identity = semantic_identity("ablation.chemberta2.frozen-entities.v1", {
            "repo_id": spec.repo_id, "revision": spec.revision, "variant": spec.variant,
            "stage1_state_hash": state_hash, "feature_identity": self.feature_identity["hash"],
            "source_files": FILES[spec.variant], "extraction": EXTRACTION,
            "slot_contract": "ordered_entity_slots_v1", "role_policy": "formal_charge_v1"})
        if bundle is not None:
            require_compatible_identity(self.encoder_identity, bundle["encoder_identity"], context="Embedded ChemBERTa encoder")
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def prepare_stage2(self, config):
        from .chemberta_data import prepare_stage2
        return prepare_stage2(self, config)

    def validate_stage3_reference(self, config, registry, objects, normalization):
        from .chemberta_data import validate_stage3_reference
        validate_stage3_reference(config, registry, objects, normalization)

    def export_bundle(self):
        return {"spec": self.config.to_dict(), "features": self.features, "feature_bytes": self.feature_bytes,
                "source_files": FILES[self.config.variant], "extraction": EXTRACTION,
                "encoder_identity": self.encoder_identity,
                "state": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    @torch.inference_mode()
    def encode_smiles(self, smiles):
        tokens = self.tokenizer.encode_batch(list(smiles))
        if any(not t.ids or t.ids[0] != self.cls_id or len(t.ids) > 512 for t in tokens):
            raise ValueError("ChemBERTa input lacks leading CLS or exceeds 512 tokens; truncation is forbidden")
        device = next(self.parameters()).device
        ids = torch.full((len(tokens), max(len(t.ids) for t in tokens)), self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, token in enumerate(tokens):
            ids[i, :len(token.ids)] = torch.tensor(token.ids, device=device)
            mask[i, :len(token.ids)] = 1
        precision = torch.backends.cuda.matmul.fp32_precision if device.type == "cuda" else None
        try:
            if precision is not None:
                torch.backends.cuda.matmul.fp32_precision = "ieee"
            with torch.autocast(device_type=device.type, enabled=False):
                values = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0].float()
        finally:
            if precision is not None:
                torch.backends.cuda.matmul.fp32_precision = precision
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite ChemBERTa entity states")
        return torch.nn.functional.pad(values, (0, 857))

    @torch.inference_mode()
    def encode_slots(self, objects):
        from common.entity_inputs import EntityInputs
        role_ids = {"cation": 0, "anion": 1, "neutral": 2}
        counts = {len(o.slots) for o in objects}
        if len(counts) != 1 or not counts <= {1, 2}:
            raise ValueError("ChemBERTa slot batch must share topology")
        count = counts.pop()
        slots = [slot for o in objects for slot in o.slots]
        if any(molecular_role(smiles) != role for role, smiles in slots):
            raise ValueError("ChemBERTa role/formal-charge mismatch")
        if any(o.topology != ("molecule" if count == 1 else "il") for o in objects):
            raise ValueError("ChemBERTa topology/slot mismatch")
        values = self.encode_smiles([s for _, s in slots]).reshape(len(objects), count, 1241).cpu()
        roles = torch.tensor([role_ids[r] for r, _ in slots], dtype=torch.long).reshape(len(objects), count)
        EntityInputs.from_slots(values, roles)
        return values, roles

    def encode_entity(self, batch):
        values = batch.frozen_entity_embedding
        return SimpleNamespace(entity_embedding=values, atom_states=values.new_empty((0, 0)),
                               atom_batch=torch.empty(0, dtype=torch.long, device=values.device))


register_frozen_encoder("chemberta2", ChemBERTaEncoder)
