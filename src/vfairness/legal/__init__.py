"""Legal-admissibility layer.

Universal at runtime: every Pulse run classifies its columns against a
(use_case x jurisdiction) rule pack and emits a `legalAdmissibility`
block. Where no pack exists, the block reports `uncovered` honestly
instead of fabricating a verdict.

Source of truth: hand-curated JSON rule packs under ``legal/data/``.
LLMs are NEVER consulted at runtime; they may only draft new packs
that a human reviews and merges. See data/README.md.
"""

from .admissibility import (
    LEGAL_DATA_REVISION,
    classify_columns,
    load_rules,
    map_domain_to_use_case,
)

__all__ = [
    "classify_columns",
    "map_domain_to_use_case",
    "load_rules",
    "LEGAL_DATA_REVISION",
]
