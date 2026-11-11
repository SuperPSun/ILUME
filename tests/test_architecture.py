from __future__ import annotations

import ast
import io
from pathlib import Path
import subprocess
import sys
import tokenize




ROOT = Path(__file__).resolve().parents[1]
STAGES = {"stage1", "stage2", "stage3"}
EXTERNAL_PACKAGES = {"ablations", "benchmarks"}


def test_source_annotations_and_cli_messages_are_english() -> None:
    failures = []
    for directory in ('scripts', 'src', 'benchmarks', 'ablations', 'tests'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            source = path.read_text(encoding='utf-8')
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT and any('\u3400' <= char <= '\u9fff' for char in token.string):
                    failures.append(f'{path.relative_to(ROOT)}:{token.start[0]}: {token.string}')
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(node, clean=False)
                    if docstring and any('\u3400' <= char <= '\u9fff' for char in docstring):
                        failures.append(f'{path.relative_to(ROOT)}:{node.body[0].lineno}: {docstring}')
                message = node.exc if isinstance(node, ast.Raise) else node.msg if isinstance(node, ast.Assert) else None
                if isinstance(node, ast.Call):
                    name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                    if name in {'error', 'exception', 'exit', '_equal', '_stats', 'canonicalize_smiles',
                                'ArgumentParser', 'add_argument', 'print'}:
                        message = node
                if message is None:
                    continue
                for value in ast.walk(message):
                    if (isinstance(value, ast.Constant) and isinstance(value.value, str)
                            and any('\u3400' <= char <= '\u9fff' for char in value.value)):
                        failures.append(f'{path.relative_to(ROOT)}:{node.lineno}: {value.value}')
    assert not failures, '\n'.join(failures)


def test_stage2_registry_does_not_import_stage1_graph() -> None:
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
sys.path.insert(0, "src")
sys.modules["stage1.graph"] = None
from stage2.registry import load_stage2_registry, orbital_audit_columns
assert callable(load_stage2_registry)
assert orbital_audit_columns("simulation/homo") == (
    "ion_role", "provenance_source_file", "provenance_source_row",
)
assert "stage2.frozen" not in sys.modules
"""],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def _cross_stage_private_imports(
    source: str,
    current_stage: str,
) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            target_stage = node.module.split(".", 1)[0]
            if target_stage in STAGES and target_stage != current_stage:
                for alias in node.names:
                    if alias.name == "*" or alias.name.startswith("_"):
                        violations.append((node.lineno, f"{node.module}.{alias.name}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    parts[0] in STAGES
                    and parts[0] != current_stage
                    and any(part.startswith("_") for part in parts[1:])
                ):
                    violations.append((node.lineno, alias.name))
    return violations


def test_stage_packages_only_use_cross_stage_public_contracts() -> None:
    violations: list[str] = []
    for stage in sorted(STAGES):
        for path in sorted((ROOT / "src" / stage).rglob("*.py")):
            for line, imported in _cross_stage_private_imports(
                path.read_text(encoding="utf-8"), stage
            ):
                violations.append(f"{path.relative_to(ROOT)}:{line}: {imported}")
    assert violations == []


def test_stage_packages_do_not_import_benchmarks_or_ablations() -> None:
    violations = []
    for stage in sorted(STAGES):
        for path in sorted((ROOT / "src" / stage).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split(".", 1)[0] in EXTERNAL_PACKAGES
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                if isinstance(node, ast.Import) and any(
                    alias.name.split(".", 1)[0] in EXTERNAL_PACKAGES
                    for alias in node.names
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []
