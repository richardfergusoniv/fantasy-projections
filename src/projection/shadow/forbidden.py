"""Forbidden-module and input-path contracts for leakage-safe shadow tracks.

Sleeper comparison artifacts and release-mutation modules must never appear on
the shadow dependency path. Production promotion/publishing stays unreachable.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import sys
from pathlib import Path
from typing import Iterable

from src.projection.contracts import REPO_ROOT

# Modules whose import (static or dynamic) fails any shadow / repair guard.
FORBIDDEN_MODULES: frozenset[str] = frozenset({
    "src.comparison.sleeper_compare",
    "src.comparison.spot_check",
    "src.projection.promote_release",
    "src.projection.release_bundle_publish",
})

FORBIDDEN_MODULE_PREFIXES: tuple[str, ...] = (
    "src.comparison.sleeper",
)

# Path substrings that mark Sleeper-derived or comparison artifacts.
FORBIDDEN_INPUT_PATH_MARKERS: tuple[str, ...] = (
    "sleeper_comparison",
    "sleeper_snapshots",
    "sleeper_spot",
    "spot_check",
    "/comparison/sleeper",
    "\\comparison\\sleeper",
)


class ForbiddenDependencyError(RuntimeError):
    """Shadow code reached a forbidden module or input path."""


class ForbiddenImportGuard:
    """Raise immediately if a forbidden module is imported dynamically."""

    def __init__(self, forbidden: Iterable[str] | None = None):
        self.forbidden = frozenset(forbidden or FORBIDDEN_MODULES)
        self._orig = None

    def __enter__(self):
        self._orig = builtins.__import__
        builtins.__import__ = self._guarded_import  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        builtins.__import__ = self._orig  # type: ignore[assignment]
        return False

    def _guarded_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        self._check_name(name)
        if fromlist:
            for item in fromlist:
                if name:
                    self._check_name(f"{name}.{item}")
        return self._orig(name, globals, locals, fromlist, level)

    def _check_name(self, name: str) -> None:
        if not name:
            return
        if _is_forbidden(name, self.forbidden):
            raise ForbiddenDependencyError(
                f"Forbidden module imported on shadow path: {name}"
            )


def _is_forbidden(name: str, forbidden: frozenset[str] = FORBIDDEN_MODULES) -> bool:
    if name in forbidden or any(name == mod or name.startswith(mod + ".") for mod in forbidden):
        return True
    return any(name == prefix or name.startswith(prefix) for prefix in FORBIDDEN_MODULE_PREFIXES)


def normalize_repo_path(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    root = str(Path(REPO_ROOT)).replace("\\", "/")
    if text.lower().startswith(root.lower()):
        text = text[len(root):].lstrip("/")
    return text


def assert_input_path_allowed(path: str | Path) -> str:
    """Reject Sleeper comparison / snapshot / derived diagnostic inputs."""
    normalized = normalize_repo_path(path).lower()
    for marker in FORBIDDEN_INPUT_PATH_MARKERS:
        if marker.lower().replace("\\", "/") in normalized:
            raise ForbiddenDependencyError(
                f"Sleeper-derived or comparison input path rejected: {path}"
            )
    return normalized


def local_import_graph(entrypoint_modules: Iterable[str]) -> set[str]:
    """Walk local ``src.*`` imports from entrypoints via AST (no execution)."""
    root = Path(REPO_ROOT)
    src_root = root / "src"
    seen: set[str] = set()
    queue = list(entrypoint_modules)

    def module_to_path(module: str) -> Path | None:
        if not module.startswith("src."):
            return None
        parts = module.split(".")[1:]
        py_file = src_root.joinpath(*parts).with_suffix(".py")
        pkg_init = src_root.joinpath(*parts) / "__init__.py"
        if py_file.is_file():
            return py_file
        if pkg_init.is_file():
            return pkg_init
        return None

    while queue:
        module = queue.pop()
        # Normalize attribute imports (src.foo.bar.Class) back to modules.
        while module and module_to_path(module) is None and "." in module:
            module = module.rsplit(".", 1)[0]
        if not module or module in seen:
            continue
        seen.add(module)
        path = module_to_path(module)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src."):
                        queue.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("src."):
                    queue.append(node.module)
                    for alias in node.names:
                        if alias.name != "*":
                            queue.append(f"{node.module}.{alias.name}")
    return seen


def assert_no_forbidden_imports(entrypoint_modules: Iterable[str]) -> set[str]:
    graph = local_import_graph(entrypoint_modules)
    offenders = sorted(mod for mod in graph if _is_forbidden(mod))
    if offenders:
        raise ForbiddenDependencyError(
            "Forbidden modules reachable from shadow entrypoints: "
            + ", ".join(offenders)
        )
    return graph


def ensure_not_already_imported(forbidden: Iterable[str] | None = None) -> None:
    """Fail if a forbidden module is already present in ``sys.modules``."""
    banned = frozenset(forbidden or FORBIDDEN_MODULES)
    hits = [
        name for name in sys.modules
        if _is_forbidden(name, banned)
    ]
    if hits:
        raise ForbiddenDependencyError(
            "Forbidden modules already imported: " + ", ".join(sorted(hits))
        )


def import_module_guarded(name: str):
    ensure_not_already_imported()
    if _is_forbidden(name):
        raise ForbiddenDependencyError(f"Forbidden module: {name}")
    return importlib.import_module(name)
