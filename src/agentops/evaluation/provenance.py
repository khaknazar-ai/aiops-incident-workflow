"""Stable provenance for reports when no Git commit is available."""

from __future__ import annotations

import hashlib
from pathlib import Path

_SOURCE_ROOTS = ("src", "migrations", "tests", "scripts", ".github")
_SOURCE_FILES = (
    "pyproject.toml",
    "alembic.ini",
    "Dockerfile",
    "compose.yaml",
    "requirements-runtime.lock",
    "requirements-dev.lock",
    ".env.example",
)
_EXCLUDED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".git"})
SOURCE_REVISION_SCOPE = "agentops-source-dependencies-v2"


class SourceProvenanceError(ValueError):
    """A source-layout tree is required for honest working-tree provenance."""


def working_tree_revision(root: Path | None = None) -> str:
    """Return a content hash for the reproducible source tree.

    The result is deliberately labeled as a working-tree hash. It is not a Git
    commit and does not include generated runtime state, caches, or installer
    metadata directories (*.egg-info and *.dist-info).
    """

    module = Path(__file__).resolve()
    tree_root = root or module.parents[3]
    if root is None and (
        not (tree_root / "pyproject.toml").is_file()
        or tree_root / "src/agentops/evaluation/provenance.py" != module
    ):
        raise SourceProvenanceError(
            "evaluation working-tree provenance requires a source-layout tree; "
            "installed wheels do not contain the required source snapshot"
        )
    digest = hashlib.sha256()
    candidates = [tree_root / name for name in _SOURCE_FILES]
    for name in _SOURCE_ROOTS:
        directory = tree_root / name
        if directory.is_dir() and not directory.is_symlink():
            candidates.extend(directory.rglob("*"))
    files = (
        path
        for path in candidates
        if path.is_file()
        and not path.is_symlink()
        and not any(parent.is_symlink() for parent in path.parents if parent != tree_root)
        and not any(
            part in _EXCLUDED_PARTS or part.endswith((".egg-info", ".dist-info"))
            for part in path.relative_to(tree_root).parts
        )
        and path.suffix not in {".pyc", ".pyo"}
        and (path.name == ".env.example" or not path.name.startswith(".env"))
    )
    for path in sorted(files, key=lambda item: item.relative_to(tree_root).as_posix()):
        relative = path.relative_to(tree_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"wt-sha256:{digest.hexdigest()}"


__all__ = ["SOURCE_REVISION_SCOPE", "SourceProvenanceError", "working_tree_revision"]
