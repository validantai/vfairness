#!/usr/bin/env python3
"""Execute the explainability surface, which no probe could reach before.

WHY THIS EXISTS. Measured 2026-09-25, the XAI surface was the library's largest
blind spot: 79 of 81 code units carried no grade at all, and the reason was never
that anybody judged them risky. It was that `scripts/surface_probe.py` builds its
nine worlds out of labels, scores, groups and a frame, and every explainer needs
something none of those is: a FITTED MODEL, a matrix to explain, and a background
sample. The generic binder handed them None, the healthy call raised, and clause 1
of `grade_from_probe.py` correctly refused to learn anything from that.

So the fixture is the whole job, and it is a program's job, not an agent's.

WHAT THE WORLDS ARE. One healthy world carrying a model with a real, findable
dependence on one feature, and eight where the thing an explanation claims to
measure does not exist:

  single_feature      every feature but one is constant, so attribution has one
                      place to go and the rest is noise
  all_constant        every feature constant: no attribution is defined at all
  one_row             a single instance, so no distribution to explain against
  empty               zero rows
  all_nan             every feature missing
  no_background       the background sample the method needs is absent
  constant_model      the model returns one class whatever it is shown, so no
                      feature can move the prediction
  single_class_fit    trained on one label only

THE RULE, unchanged from grade_from_probe.py because the defect is the same one:
the HEALTHY world must produce a value (otherwise we called it wrong and learned
nothing), at least four degenerate worlds must have run, and every degenerate
world that ran must have raised or refused. An explainer that hands back an
attribution vector when nothing is attributable is the fabricated verdict in its
explainability shape, and no script may grade that; it goes to judgement.

Usage:
    python scripts/xai_probe.py --selftest
    python scripts/xai_probe.py --out docs/xai-probe.json
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import library_kpis  # noqa: E402
import surface_probe as SP  # noqa: E402

# WHICH WORLDS MAKE AN ATTRIBUTION UNDEFINED, and this distinction is the whole
# rule. The first version of it treated every non-healthy world as one where
# nothing is attributable, and its own selftest caught it: an HONEST explainer was
# graded NEEDS JUDGEMENT for answering on four worlds where an attribution is
# perfectly well defined. GRADING.md step 4 says it exactly: "only the cases
# where its own claim becomes undefined", and these are those cases:
#
#   all_constant   no feature varies, so no attribution is defined
#   empty          no rows to attribute over
#   all_nan        no values at all
#
# The rest are DEFINED and deliberately kept in the record as over-refusal
# controls. `constant_model` is the sharpest of them: the model ignores every
# feature, so the true attribution is exactly zero, and a vector of zeros there is
# the CORRECT answer rather than a fabricated one. Grading it as fabrication would
# push an explainer towards refusing a case it can answer, which is the same
# defect pointing the other way.
UNDEFINED_WORLDS = ("all_constant", "empty", "all_nan")
MIN_UNDEFINED = 2

SEMI_PROVEN = "SEMI-PROVEN"
NEEDS_JUDGEMENT = "NEEDS JUDGEMENT"
NOT_EXAMINED = "NOT EXAMINED"

N = 120
RNG = np.random.default_rng(20260925)
FEATURES = ["income", "age", "tenure", "score"]

WORLDS = (
    "healthy",
    "single_feature",
    "all_constant",
    "one_row",
    "empty",
    "all_nan",
    "no_background",
    "constant_model",
    "single_class_fit",
)


class _ConstantModel:
    """Predicts one class whatever it is shown. Nothing is attributable."""

    def fit(self, X, y):  # noqa: N803
        return self

    def predict(self, X):  # noqa: N803
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):  # noqa: N803
        out = np.zeros((len(X), 2))
        out[:, 0] = 1.0
        return out


def _fit(X, y):  # noqa: N803
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(max_iter=200).fit(X, y)


def _world(name: str) -> dict:
    X = RNG.normal(0, 1, size=(N, len(FEATURES)))
    # A real, findable dependence: the prediction follows feature 0 and nothing else.
    y = (X[:, 0] > 0).astype(int)

    if name == "single_feature":
        X[:, 1:] = 1.0
    elif name == "all_constant":
        X[:, :] = 1.0
    elif name == "one_row":
        X, y = X[:1], y[:1]
    elif name == "empty":
        X, y = X[:0], y[:0]
    elif name == "all_nan":
        X = np.full_like(X, np.nan)
    elif name == "single_class_fit":
        y = np.zeros(N, dtype=int)

    if name == "constant_model":
        model = _ConstantModel()
    elif name in ("empty", "all_nan", "all_constant", "single_class_fit", "one_row"):
        # The model is fitted on the HEALTHY data in these worlds: the degeneracy
        # under test is the data being explained, not the estimator. Fitting on a
        # constant or empty frame would raise in sklearn and the probe would be
        # measuring sklearn's input validation instead of the explainer.
        good_X = RNG.normal(0, 1, size=(N, len(FEATURES)))
        model = _fit(good_X, (good_X[:, 0] > 0).astype(int))
    else:
        model = _fit(X, y)

    background = None if name == "no_background" else X[: min(30, len(X))]
    attributions = (
        np.zeros(len(FEATURES)) if name == "all_constant" else RNG.normal(0, 1, len(FEATURES))
    )
    return {
        "model": model,
        "x": X,
        "X": X,
        "y": y,
        "background": background,
        "feature_names": list(FEATURES),
        "attributions": attributions,
        "instance_id": "probe-instance",
        "subject_id": "probe-subject",
        "model_hash": "0" * 64,
        "data_hash": "1" * 64,
        "model_type": "linear",
        "n_features": len(FEATURES),
        "seeds": [1, 2, 3],
        "predict_fn": getattr(model, "predict_proba", getattr(model, "predict", None)),
    }


# Parameter name -> world key. Names taken from the real signatures on the XAI
# surface, never guessed: base.Explainer.explain_local defines model / x /
# background / instance_id / subject_id / model_hash / data_hash, and the
# diagnostics take predict_fn, attributions and seeds.
_BY_NAME = {
    "model": "model",
    "estimator": "model",
    "clf": "model",
    "predict_fn": "predict_fn",
    "predict": "predict_fn",
    "predict_proba": "predict_fn",
    "x": "x",
    "X": "X",
    "instance": "x",
    "instances": "x",
    "data": "X",
    "features": "X",
    "X_train": "X",
    "X_test": "X",
    "y": "y",
    "labels": "y",
    "background": "background",
    "background_data": "background",
    "baseline": "background",
    "feature_names": "feature_names",
    "columns": "feature_names",
    "attributions": "attributions",
    "attribution": "attributions",
    "values": "attributions",
    "instance_id": "instance_id",
    "subject_id": "subject_id",
    "model_hash": "model_hash",
    "data_hash": "data_hash",
    "model_type": "model_type",
    "method": "model_type",
    "seeds": "seeds",
    "n_features": "n_features",
}


def _bind(fn: Any, world: dict) -> tuple[list | None, dict | None, str | None, set]:
    """Fill a signature from the XAI world, falling back to the generic binder."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError) as exc:
        return None, None, f"no signature: {exc}", set()
    args: list = []
    kwargs: dict = {}
    unfilled: list[str] = []
    used: set = set()
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        key = _BY_NAME.get(name)
        if key is not None:
            value = world[key]
            used.add(key)
        elif p.default is not inspect._empty:
            continue
        else:
            # Let the generic binder try its own vocabulary before giving up.
            ok, value = SP._from_annotation(p.annotation, SP._world("healthy"))
            if not ok:
                unfilled.append(name)
                continue
        if p.kind == p.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
    if unfilled:
        return None, None, f"no fixture for parameter(s): {', '.join(unfilled)}", set()
    return args, kwargs, None, used


def _describe(value: Any) -> str:
    """What came back, in the same vocabulary surface_probe uses."""
    kind, _shown = SP._describe(value)
    return kind


# The fixture keys that carry the data an explanation is ABOUT. A unit that binds
# none of them cannot be judged by these worlds at all, and the first version of
# this rule judged them anyway: `supports(model, model_type)` and
# `available_methods()` were reported as suspected fabrications on the strength of
# answering over a constant matrix they never received. A false accusation costs
# the same judgement tokens as a real one and spends them on nothing.
_DATA_KEYS = {"x", "X", "attributions"}


def probe_one(obj: Any, owner: Any | None) -> dict:
    runs: dict[str, dict] = {}
    consumed: set = set()
    for name in WORLDS:
        world = _world(name)
        target, bound_self = obj, None
        if owner is not None:
            try:
                bound_self = owner()
            except Exception as exc:  # noqa: BLE001
                runs[name] = {
                    "outcome": "NOT_REACHED",
                    "detail": f"owner {owner.__name__} would not construct: {type(exc).__name__}",
                }
                continue
        args, kwargs, why, used = _bind(target, world)
        if why:
            runs[name] = {"outcome": "NOT_REACHED", "detail": why}
            continue
        consumed |= used
        call_args = [bound_self, *args] if bound_self is not None else args
        old = SP._deadline(SP.CALL_TIMEOUT)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with open(os.devnull, "w") as null:
                    with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                        value = target(*call_args, **kwargs)
            runs[name] = {"outcome": _describe(value), "n_warnings": len(caught)}
        except SP._TimeoutError:
            runs[name] = {"outcome": "hung", "detail": "never returned"}
        except SystemExit as exc:
            runs[name] = {"outcome": "raised", "detail": f"SystemExit({exc.code})"}
        except Exception as exc:  # noqa: BLE001 - an exception IS a refusal
            runs[name] = {
                "outcome": "raised",
                "detail": f"{type(exc).__name__}: {SP._public_diagnostic(exc, 120)}",
            }
        finally:
            SP._clear_deadline(old)
    return {"runs": runs, "consumed": sorted(consumed), **grade(runs, consumed)}


def grade(runs: dict, consumed: set | None = None) -> dict:
    if consumed is not None and not (_DATA_KEYS & set(consumed)):
        return {
            "grade": NOT_EXAMINED,
            "reason": (
                "it takes none of the data these worlds vary (no x, X or attributions "
                "parameter), so nothing they do says anything about it"
            ),
        }
    healthy = runs.get("healthy") or {}
    if healthy.get("outcome") in (None, "NOT_REACHED", "raised", "hung"):
        return {
            "grade": NOT_EXAMINED,
            "reason": f"the healthy world did not produce a value ({healthy.get('detail', healthy.get('outcome'))})",
        }
    undefined = [(k, runs.get(k) or {}) for k in UNDEFINED_WORLDS]
    ran = [(k, v) for k, v in undefined if v.get("outcome") not in (None, "NOT_REACHED")]
    if len(ran) < MIN_UNDEFINED:
        return {
            "grade": NOT_EXAMINED,
            "reason": (
                f"only {len(ran)} of the {len(UNDEFINED_WORLDS)} worlds where an attribution is "
                "undefined could be reached, which is not enough to conclude anything"
            ),
        }
    answered = [
        k for k, v in ran if v.get("outcome") in ("neutral", "measured") and not v.get("n_warnings")
    ]
    # Recorded, never graded on: a unit that refuses every DEFINED world too may be
    # throwing evidence away, and that is a finding for a human, not a pass or a
    # fail a script may hand out.
    defined_refusals = [
        k
        for k, v in runs.items()
        if k not in UNDEFINED_WORLDS and k != "healthy" and (v or {}).get("outcome") == "raised"
    ]
    if answered:
        return {
            "grade": NEEDS_JUDGEMENT,
            "reason": (
                f"answered without a warning on {len(answered)} world(s) where no attribution is "
                f"defined ({', '.join(sorted(answered))})"
            ),
            "refused_defined_worlds": sorted(defined_refusals),
        }
    return {
        "grade": SEMI_PROVEN,
        "reason": (
            f"executed on the healthy world, and refused or warned on all {len(ran)} world(s) "
            "where no attribution is defined; no test pins that"
        ),
        "refused_defined_worlds": sorted(defined_refusals),
    }


def selftest() -> int:
    """An explainer that fabricates an attribution must not be gradeable."""

    class Honest:
        def explain_local(
            self, model, x, background=None, *, instance_id, subject_id, model_hash, data_hash
        ):
            if len(x) == 0 or np.isnan(x).all() or np.ptp(x, axis=0).max() == 0:
                raise ValueError("nothing is attributable on this input")
            return {"attributions": np.abs(x).mean(axis=0).tolist()}

    class Fabricates:
        def explain_local(
            self, model, x, background=None, *, instance_id, subject_id, model_hash, data_hash
        ):
            if len(x) == 0 or np.isnan(x).all() or np.ptp(x, axis=0).max() == 0:
                return {"attributions": [0.0, 0.0, 0.0, 0.0]}  # a clean-looking nothing
            return {"attributions": np.abs(x).mean(axis=0).tolist()}

    class RefusesEverything:
        """Answers the healthy world and refuses every other, including the
        answerable ones. It is SEMI-PROVEN on the fabrication axis and its
        over-refusal must still be visible in the record."""

        def explain_local(
            self, model, x, background=None, *, instance_id, subject_id, model_hash, data_hash
        ):
            # Refuses on ANY imperfection: a missing background, a constant
            # column, a single row. The undefined worlds are correctly refused
            # (so the fabrication axis is clean) and three answerable ones are
            # refused with them, which is what must show up in the record.
            if background is None or len(x) == 0 or not np.isfinite(x).all():
                raise ValueError("refusing")
            if np.ptp(x, axis=0).min() == 0:
                raise ValueError("refusing")
            return {"attributions": np.abs(x).mean(axis=0).tolist()}

    ok = True
    for cls, want in (
        (Honest, SEMI_PROVEN),
        (Fabricates, NEEDS_JUDGEMENT),
        (RefusesEverything, SEMI_PROVEN),
    ):
        got = probe_one(cls.explain_local, cls)
        flag = "ok " if got["grade"] == want else "FAIL"
        extra = got.get("refused_defined_worlds") or []
        print(
            f"  {flag} {cls.__name__:<18} -> {got['grade']:<16} {got['reason'][:60]}"
            + (f" | over-refused: {len(extra)}" if extra else "")
        )
        ok = ok and got["grade"] == want
        if cls is RefusesEverything and not extra:
            print("  FAIL over-refusal was not recorded")
            ok = False
    print("  selftest passed" if ok else "  SELFTEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]
    with contextlib.suppress(Exception):
        os.dup2(open(os.devnull).fileno(), 0)
    if args.selftest:
        return selftest()
    if not args.out:
        ap.error("--out is required unless --selftest")
    out_path = Path(args.out).resolve()

    surface = library_kpis._walk_public_surface()
    wanted = [
        q
        for q in sorted(set(surface["functions"]["ungraded"]) | set(surface["methods"]["ungraded"]))
        if ".xai." in q or q.endswith(".explainer") or ".explainer." in q
    ]
    os.chdir(tempfile.mkdtemp(prefix="xai-probe-"))

    out: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for qual in wanted:
        entry = library_kpis._OBJECTS.get(qual)
        if entry is None:
            out[qual] = {"grade": NOT_EXAMINED, "reason": "object not resolvable", "runs": {}}
        else:
            obj = entry[1]
            owner = None
            if "." in getattr(obj, "__qualname__", ""):
                mod = sys.modules.get(obj.__module__)
                owner = getattr(mod, obj.__qualname__.split(".")[0], None) if mod else None
            out[qual] = probe_one(obj, owner)
        counts[out[qual]["grade"]] = counts.get(out[qual]["grade"], 0) + 1
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(wanted)} XAI code units probed: {counts}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
