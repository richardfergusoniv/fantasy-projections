"""Import smoke tests for primary pipeline entrypoints."""
from __future__ import annotations

import importlib
import inspect


ENTRYPOINTS = (
    "src.projection.data_prep",
    "src.projection.train",
    "src.projection.inference.simulate",
    "src.projection.release_bundle_publish",
    "src.projection.evaluation.release_bundle_validation",
    "src.projection.evaluation.promotion_invariants",
    "src.projection.promote_release",
)


def test_entrypoint_modules_import():
    for module_name in ENTRYPOINTS:
        importlib.import_module(module_name)


def test_promote_release_has_no_git_bypass():
    from src.projection import promote_release as module

    assert "skip_git" not in inspect.signature(module.promote_release).parameters
    assert "skip_git" not in inspect.signature(module.rollback_release).parameters


def test_validate_promotion_invariants_has_no_git_bypass():
    from src.projection.evaluation import promotion_invariants as module

    assert "skip_git" not in inspect.signature(module.validate_promotion_invariants).parameters
