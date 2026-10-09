"""Keep reusable research code independent of CLI modules and stage names."""
import ast
from pathlib import Path


def test_package_and_scripts_never_import_another_cli():
    root = Path(__file__).parents[2]
    for directory in ("src", "scripts"):
        for path in (root / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert not module.startswith(("scripts.", "train_", "evaluate_", "precompute_")), path
                elif isinstance(node, ast.Import):
                    assert not any(alias.name.startswith("scripts.") for alias in node.names), path


def test_phase_names_do_not_define_source_paths():
    root = Path(__file__).parents[2]
    for directory in ("src", "scripts", "configs"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                assert not any(term in str(path.relative_to(root)) for term in ("phase0", "phase1", "phase2", "phase3")), path
