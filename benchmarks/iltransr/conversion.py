from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


UPSTREAM_REVISION = "ff5e55cfb8162b0706fc2d88ca5cd384705686b1"
GENERIC_CHECKPOINT_SHA256 = "36a3401175dd372725bd2f5e4e041642a754eeaf3f46845b6ff949065871735a"
PARITY_TOKENS = np.asarray(
    [
        [5, 23, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [12, 11, 37, 16, 13, 23, 12, 5, 24, 15, 13, 3],
        [5, 5, 6, 10, 8, 7, 8, 3, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)
PARITY_VALID_LENGTHS = np.asarray([4, 12, 8], dtype=np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_name(source: str) -> str | None:
    if source == "src_embed.0.weight":
        return "embedding.weight"
    if source == "encoder.position_weight":
        return "position_weight"
    if source == "encoder.layer_norm.gamma":
        return "input_norm.weight"
    if source == "encoder.layer_norm.beta":
        return "input_norm.bias"
    prefix = "encoder.transformer_cells."
    if not source.startswith(prefix):
        return None
    rest = source[len(prefix):]
    layer, suffix = rest.split(".", 1)
    translations = {
        "attention_cell.proj_query.weight": "query.weight",
        "attention_cell.proj_key.weight": "key.weight",
        "attention_cell.proj_value.weight": "value.weight",
        "proj.weight": "projection.weight",
        "layer_norm.gamma": "attention_norm.weight",
        "layer_norm.beta": "attention_norm.bias",
        "ffn.ffn_1.weight": "ffn_1.weight",
        "ffn.ffn_1.bias": "ffn_1.bias",
        "ffn.ffn_2.weight": "ffn_2.weight",
        "ffn.ffn_2.bias": "ffn_2.bias",
        "ffn.layer_norm.gamma": "ffn_norm.weight",
        "ffn.layer_norm.beta": "ffn_norm.bias",
    }
    translated = translations.get(suffix)
    return None if translated is None else f"layers.{layer}.{translated}"


def convert_checkpoint(
    checkpoint: str | Path,
    source_vocab: str | Path,
    target_vocab: str | Path,
    output: str | Path,
    parity_reference: str | Path,
    manifest: str | Path,
) -> dict[str, Any]:
    """Convert the pinned MXNet NMT checkpoint to the encoder-only safetensors artifact."""
    import importlib.metadata
    import gluonnlp as nlp
    import mxnet as mx
    from gluonnlp.model.transformer import get_transformer_encoder_decoder
    from safetensors.numpy import save_file

    source = Path(checkpoint)
    target = Path(output)
    reference_path = Path(parity_reference)
    manifest_path = Path(manifest)
    if _sha256(source) != GENERIC_CHECKPOINT_SHA256:
        raise ValueError("ILTransR generic MXNet checkpoint SHA256 mismatch")
    raw = mx.nd.load(str(source))
    converted: dict[str, np.ndarray] = {}
    ignored: list[str] = []
    for name, value in raw.items():
        target_name = _target_name(name)
        if target_name is None:
            ignored.append(name)
        else:
            converted[target_name] = np.asarray(value.asnumpy(), dtype=np.float32)
    expected_count = 4 + 3 * 12
    if len(converted) != expected_count:
        raise ValueError(
            f"ILTransR converted tensor count differs: expected {expected_count}, got {len(converted)}"
        )
    if not ignored or any(
        not name.startswith(("decoder.", "one_step_ahead_decoder.", "tgt_embed.", "tgt_proj."))
        for name in ignored
    ):
        raise ValueError("ILTransR checkpoint contains an unclassified non-downstream tensor")
    target.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(target), metadata={"upstream_revision": UPSTREAM_REVISION})

    def load_vocab(path: str | Path) -> Any:
        return nlp.Vocab.from_json(Path(path).read_text(encoding="utf-8"))

    src_vocab = load_vocab(source_vocab)
    tgt_vocab = load_vocab(target_vocab)
    encoder, decoder, one_step_decoder = get_transformer_encoder_decoder(
        units=128,
        hidden_size=1024,
        dropout=0,
        num_layers=3,
        num_heads=4,
        max_src_length=100,
        max_tgt_length=100,
        scaled=True,
    )
    reference_model = nlp.model.translation.NMTModel(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        encoder=encoder,
        decoder=decoder,
        one_step_ahead_decoder=one_step_decoder,
        embed_size=128,
        embed_initializer=None,
        prefix="transformer_",
    )
    reference_model.load_parameters(str(source), ctx=mx.cpu())
    reference_output, _ = reference_model.encode(
        mx.nd.array(PARITY_TOKENS, ctx=mx.cpu()),
        valid_length=mx.nd.array(PARITY_VALID_LENGTHS, ctx=mx.cpu()),
    )
    save_file(
        {
            "token_ids": PARITY_TOKENS,
            "valid_lengths": PARITY_VALID_LENGTHS,
            "mxnet_output": np.asarray(reference_output.asnumpy(), dtype=np.float32),
        },
        str(reference_path),
        metadata={"upstream_revision": UPSTREAM_REVISION},
    )
    tensor_hash = hashlib.sha256()
    for name in sorted(converted):
        value = converted[name]
        tensor_hash.update(name.encode())
        tensor_hash.update(str(value.dtype).encode())
        tensor_hash.update(json.dumps(value.shape).encode())
        tensor_hash.update(value.tobytes(order="C"))
    payload = {
        "format_version": 1,
        "upstream_revision": UPSTREAM_REVISION,
        "source_file": source.name,
        "source_sha256": GENERIC_CHECKPOINT_SHA256,
        "converted_file": target.name,
        "converted_sha256": _sha256(target),
        "parity_reference_file": reference_path.name,
        "parity_reference_sha256": _sha256(reference_path),
        "parity_thresholds": {"max_abs_error": 1.0e-5, "mean_abs_error": 1.0e-6},
        "tensor_state_sha256": tensor_hash.hexdigest(),
        "converted_tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(converted.items())
        },
        "ignored_pretraining_tensors": sorted(ignored),
        "conversion_environment": {
            "python": __import__("platform").python_version(),
            "mxnet": mx.__version__,
            "gluonnlp": importlib.metadata.version("gluonnlp"),
            "numpy": np.__version__,
            "safetensors": importlib.metadata.version("safetensors"),
            "device": "cpu",
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the pinned ILTransR MXNet checkpoint and export its parity reference."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-vocab", required=True)
    parser.add_argument("--target-vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parity-reference", required=True)
    parser.add_argument("--manifest", required=True)
    arguments = parser.parse_args()
    payload = convert_checkpoint(
        arguments.checkpoint,
        arguments.source_vocab,
        arguments.target_vocab,
        arguments.output,
        arguments.parity_reference,
        arguments.manifest,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


__all__ = [
    "GENERIC_CHECKPOINT_SHA256",
    "PARITY_TOKENS",
    "PARITY_VALID_LENGTHS",
    "UPSTREAM_REVISION",
    "convert_checkpoint",
]


if __name__ == "__main__":
    main()
