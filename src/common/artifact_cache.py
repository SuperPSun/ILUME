"""Local immutable artifacts with one writer and verified atomic publication."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import tempfile

from .io import atomic_json, sha256_file


@contextmanager
def artifact_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_artifact_cache(root, identity):
    root = Path(root)
    identity = json.loads(json.dumps(identity))
    manifest = json.loads((root / "complete.json").read_text())
    if manifest.get("identity") != identity or not manifest.get("files"):
        raise ValueError(f"Cache identity or completion mismatch: {root}")
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Cache artifact hash mismatch: {path}")
    return manifest


def artifact_cache(root, identity, build, *, dry_run=False):
    root = Path(root)
    def status():
        if not root.exists():
            return "CREATE"
        try:
            validate_artifact_cache(root, identity)
            return "REUSE"
        except (OSError, ValueError, KeyError):
            return "INVALID"
    if dry_run:
        return status()
    with artifact_lock(root.with_name(root.name + ".lock")):
        decision = status()
        print(f"{decision} cache {root}", flush=True)
        if decision == "INVALID":
            raise ValueError(f"Invalid cache retained; use a new cache location: {root}")
        if decision == "REUSE":
            return decision
        staging = Path(tempfile.mkdtemp(prefix=root.name + ".pending-", dir=root.parent))
        # Failures deliberately retain their staging directory and logs.
        build(staging)
        files = {p.relative_to(staging).as_posix(): sha256_file(p)
                 for p in sorted(staging.rglob("*")) if p.is_file()}
        if not files:
            raise ValueError("Cache builder produced no artifacts")
        atomic_json(staging / "complete.json", {"identity": identity, "files": files})
        staging.rename(root)
        validate_artifact_cache(root, identity)
        return decision
