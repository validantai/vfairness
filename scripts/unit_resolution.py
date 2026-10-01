"""Does the code unit a grade is ABOUT actually exist?

EXTRACTED 2026-09-30 so that exactly ONE definition of "this unit exists" serves
both the check that reports a phantom grade and the applier that would otherwise
write one. It lived only in tests/test_graded_evidence_resolves.py, which meant
the applier had no way to ask the question and guarded the SHAPE of a unit key
instead: ``_DOTTED`` accepts any dotted identifier, so
``training_analysis_report_to_svg.base_rate_disparity`` (a field of a rendering,
not a code unit) and ``engine._get_jinja_env._to_float`` (a function nested
inside another function) both passed and were published as graded rows about
nothing. Two definitions of the same question is the defect this repository is
audited for, so there is now one, here, imported by both.

The resolver itself is unchanged, comments included, because those comments
record the two incidents that shaped it.
"""

from __future__ import annotations


def resolve(qualname: str) -> tuple[bool, str]:
    """(exists, how). ``how`` is "absent" only when the parent itself resolved."""
    import importlib
    import importlib.util

    parts = qualname.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        for depth, attr in enumerate(parts[i:]):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
                continue
            # A dataclass field whose default is None is not reachable by
            # hasattr on the class in every Python, and it is still declared.
            declared = {
                **getattr(obj, "__annotations__", {}),
                **getattr(obj, "__dataclass_fields__", {}),
            }
            if attr in declared:
                obj = None
                continue
            # THIRD STATE, added 2026-09-28, because this returned "absent" for a
            # unit that is REALLY THERE. vfairness.mcp.server is a module on disk,
            # but importing it raises ImportError on purpose when the optional
            # 'mcp' dependency is not installed, and this loop swallows that with
            # `except Exception: continue`. It then fell back to vfairness.mcp,
            # found no 'server' attribute (a package does not expose a submodule
            # it never imports) and reported the grade as being about nothing. An
            # over-accusation is the same class of error as a fabricated pass: it
            # would have had someone delete a valid grade.
            parent = ".".join(parts[: i + depth])
            spec = None
            try:
                spec = importlib.util.find_spec(f"{parent}.{attr}")
            except Exception:
                spec = None
            if spec is not None:
                return False, (
                    f"unimportable: {parent}.{attr} exists on disk ({spec.origin}) but "
                    "importing it raised here, most often a missing optional "
                    "dependency, so this method cannot say whether the graded unit "
                    "inside it is present"
                )
            return False, f"absent: {parent} has no {attr!r}"
        return True, "resolved"
    return False, "unresolved: no importable prefix"
