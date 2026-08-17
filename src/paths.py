"""Filesystem locations shared by ingestion and projection code.

Large, reproducible data can live outside the repository by setting
``FANTASY_PROJECTIONS_DATA_DIR``.  The database and raw cache can also be
overridden independently for one-off jobs.  Defaults preserve the original
``<repo>/data`` layout.
"""
from __future__ import annotations

import os
from collections.abc import Mapping


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def configured_data_dir(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return os.path.abspath(env.get(
        "FANTASY_PROJECTIONS_DATA_DIR",
        os.path.join(REPO_ROOT, "data"),
    ))


def configured_db_path(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return os.path.abspath(env.get(
        "FANTASY_PROJECTIONS_DB_PATH",
        os.path.join(configured_data_dir(env), "projections.db"),
    ))


def configured_raw_dir(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return os.path.abspath(env.get(
        "FANTASY_PROJECTIONS_RAW_DIR",
        os.path.join(configured_data_dir(env), "raw"),
    ))


DATA_DIR = configured_data_dir()
DB_PATH = configured_db_path()
RAW_DIR = configured_raw_dir()
