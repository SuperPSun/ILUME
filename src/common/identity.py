from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .io import sha256_file


IDENTITY_CONTRACT_VERSION = 1


def _update_hash(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, Path):
        raise TypeError("Semantic identity payloads must not contain Path values")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Semantic identity mapping keys must be strings")
        digest.update(b"mapping\0")
        for key in sorted(value):
            _update_hash(digest, key)
            _update_hash(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _update_hash(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0true\0" if value else b"bool\0false\0")
        return
    if isinstance(value, int):
        digest.update(f"int\0{value}\0".encode())
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Semantic identity floats must be finite")
        digest.update(f"float\0{value!r}\0".encode())
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"string\0{len(encoded)}\0".encode())
        digest.update(encoded)
        digest.update(b"\0")
        return
    raise TypeError(
        f"Unsupported semantic identity value: {type(value).__name__}"
    )


def semantic_hash(identity_type: str, payload: Any) -> str:
    if not identity_type:
        raise ValueError("Semantic identity type must not be empty")
    digest = hashlib.sha256()
    _update_hash(
        digest,
        {
            "identity_contract_version": IDENTITY_CONTRACT_VERSION,
            "identity_type": identity_type,
            "payload": payload,
        },
    )
    return digest.hexdigest()


def semantic_identity(identity_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    return {
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "type": identity_type,
        "payload": copied,
        "hash": semantic_hash(identity_type, copied),
    }


def validate_semantic_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("identity_contract_version") != IDENTITY_CONTRACT_VERSION:
        raise ValueError("Unsupported or missing identity contract version")
    identity_type = identity.get("type")
    payload = identity.get("payload")
    if not isinstance(identity_type, str) or not isinstance(payload, Mapping):
        raise ValueError("Malformed semantic identity")
    if identity.get("hash") != semantic_hash(identity_type, payload):
        raise ValueError(f"Semantic identity self-hash mismatch: {identity_type}")


def _identity_differences(expected: Any, actual: Any, path: str) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        result: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in expected:
                result.append(f"{child}: unexpected")
            elif key not in actual:
                result.append(f"{child}: missing")
            else:
                result.extend(_identity_differences(expected[key], actual[key], child))
        return result
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return [f"{path}: length {len(actual)} != {len(expected)}"]
        result = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            result.extend(_identity_differences(left, right, f"{path}[{index}]"))
        return result
    return [] if expected == actual else [f"{path}: {actual!r} != {expected!r}"]


def compare_semantic_identity(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> tuple[str, ...]:
    validate_semantic_identity(expected)
    validate_semantic_identity(actual)
    if expected["type"] != actual["type"]:
        return (f"type: {actual['type']!r} != {expected['type']!r}",)
    if expected["hash"] == actual["hash"]:
        return ()
    return tuple(
        _identity_differences(expected["payload"], actual["payload"], "payload")
    ) or ("hash: semantic payload differs",)


def require_compatible_identity(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, context: str
) -> None:
    differences = compare_semantic_identity(expected, actual)
    if differences:
        raise ValueError(
            f"{context} semantic identity mismatch ({expected.get('type')}): "
            + "; ".join(differences[:8])
        )


def tensor_state_hash(identity_type: str, state: Mapping[str, Any]) -> str:
    return semantic_hash(identity_type, state)


def verify_integrity(
    root: str | Path,
    locator: Mapping[str, str],
    manifest: Mapping[str, Mapping[str, Any]],
) -> None:
    root_path = Path(root).resolve()
    if set(locator) != set(manifest):
        raise ValueError("Integrity locator and manifest logical IDs differ")
    for logical_id in sorted(locator):
        relative = Path(locator[logical_id])
        if relative.is_absolute():
            raise ValueError(f"Integrity locator must be relative: {logical_id}")
        path = (root_path / relative).resolve()
        try:
            path.relative_to(root_path)
        except ValueError as error:
            raise ValueError(f"Integrity locator escapes root: {logical_id}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Missing integrity file: {logical_id}")
        expected = manifest[logical_id]
        if int(expected.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"Integrity size mismatch: {logical_id}")
        if expected.get("sha256") != sha256_file(path):
            raise ValueError(f"Integrity SHA256 mismatch: {logical_id}")


__all__ = [
    "IDENTITY_CONTRACT_VERSION",
    "compare_semantic_identity",
    "require_compatible_identity",
    "semantic_hash",
    "semantic_identity",
    "tensor_state_hash",
    "validate_semantic_identity",
    "verify_integrity",
]
