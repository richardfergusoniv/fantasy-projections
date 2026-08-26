"""Smoke tests for rankings comparison export."""

from __future__ import annotations

from src.draft_assistant.compare_prepare import _join_keys, _norm_name


def test_norm_name_strips_suffixes():
    assert _norm_name("Ja'Marr Chase") == "jamarr chase"
    assert _norm_name("Kenneth Walker III") == "kenneth walker"


def test_join_keys_team_alias():
    n, p, t = _join_keys("A Player", "WR", "JAC")
    assert n == "a player"
    assert p == "WR"
    assert t == "JAX"
