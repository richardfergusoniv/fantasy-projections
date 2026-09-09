"""Production-safe QB evaluation/allocation safeguards.

Extracted and adapted from PR #8 infrastructure repairs without promoting
H3/H4 model candidates. These helpers never change sealed release defaults.
"""

from src.projection.qb_eval_safeguards.archetype import classify_qb_archetype_safe
from src.projection.qb_eval_safeguards.composition_contract import (
    assert_availability_applied_once,
    compose_season_opportunity,
    detect_double_availability,
)
from src.projection.qb_eval_safeguards.conservation import (
    assert_team_qb_room_conserved,
    check_team_volume_conservation,
)
from src.projection.qb_eval_safeguards.portable_contract import (
    ReconciliationSkipped,
    assert_no_label_leakage,
    resolve_reconciliation_source,
    validate_portable_fixture,
)
from src.projection.qb_eval_safeguards.projections_db import (
    ProjectionsDbUnusable,
    projections_db_status,
    require_usable_projections_db,
)
from src.projection.qb_eval_safeguards.role_allocation import (
    allocate_league_expected_starts,
    allocate_team_expected_starts,
    assert_backups_do_not_inherit_starter_volume,
    role_from_preseason,
)

__all__ = [
    "ProjectionsDbUnusable",
    "ReconciliationSkipped",
    "allocate_league_expected_starts",
    "allocate_team_expected_starts",
    "assert_availability_applied_once",
    "assert_backups_do_not_inherit_starter_volume",
    "assert_no_label_leakage",
    "assert_team_qb_room_conserved",
    "check_team_volume_conservation",
    "classify_qb_archetype_safe",
    "compose_season_opportunity",
    "detect_double_availability",
    "projections_db_status",
    "require_usable_projections_db",
    "resolve_reconciliation_source",
    "role_from_preseason",
    "validate_portable_fixture",
]
