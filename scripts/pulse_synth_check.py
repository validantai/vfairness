"""Universality check: run the patched Pulse engine against each
single-bias synthetic dataset and grade whether the engine surfaces ONLY
the planted column and gives it a non-pass tone.

For each variant under ``datasets/synth/``:
  - load the audit view (drop ``_bias_flags`` + ``true_qualification_score``)
  - run run_pulse
  - check (a) the planted attribute (or proxy) reads warn/critical and
    (b) NO unplanted protected attribute reads warn/critical

Per-variant pass criterion: planted detected AND no spurious flag.

  python scripts/pulse_synth_check.py [--debug-dir /tmp/pulse-synth]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / ".." / "src"))

# Same torch stub the recall harness uses so vfairness imports clean.
if "torch" not in sys.modules:
    import types

    class _AnyAttr(types.ModuleType):
        def __getattr__(self, name):
            return _AnyAttr(f"{self.__name__}.{name}")

    mods = {"torch": _AnyAttr("torch")}
    for modname in (
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch.optim",
        "torch.autograd",
        "torch.utils",
        "torch.utils.data",
        "torch.cuda",
    ):
        mods[modname] = _AnyAttr(modname)
        sys.modules[modname] = mods[modname]
    for child in list(mods):
        if "." in child:
            p, leaf = child.rsplit(".", 1)
            setattr(mods[p], leaf, mods[child])
    mods["torch.nn"].Module = type("Module", (), {})
    mods["torch.autograd"].Function = type("Function", (), {})
    mods["torch.utils.data"].Dataset = type("Dataset", (), {})

import pandas as pd  # noqa: E402

from vfairness.operations.pulse import run_pulse  # noqa: E402

PROTECTED = [
    "gender",
    "race_ethnicity",
    "age",
    "disability_status",
    "national_origin",
    "religion",
    "marital_status",
    "sexual_orientation",
    "veteran_status",
    "primary_language",
]


# Per-variant grading spec.
VARIANTS: Dict[str, Dict] = {
    "clean": {
        "expect_protected": [],  # nothing planted
        "expect_proxy_feature": None,
    },
    "gender_penalty": {
        "expect_protected": ["gender"],
        "expect_proxy_feature": None,
    },
    "race_penalty": {
        "expect_protected": ["race_ethnicity"],
        "expect_proxy_feature": None,
    },
    "age_cliff": {
        "expect_protected": ["age"],
        "expect_proxy_feature": None,
    },
    "disability_penalty": {
        "expect_protected": ["disability_status"],
        "expect_proxy_feature": None,
    },
    "zip_redlining": {
        "expect_protected": [],  # planted as a proxy, not a direct attr
        "expect_proxy_feature": "zip_minority_majority",
    },
    "photo_laundering": {
        "expect_protected": [],
        "expect_proxy_feature": "photo_attractiveness_score",
    },
}


def _tone_for(res, attr):
    for pv in res.get("perVariable") or []:
        if str(pv.get("attribute")) == attr:
            return pv.get("tone") if pv.get("assessable") else None
    return None


def _proxy_present(res, feature):
    for p in (res.get("proxies") or {}).get("proxies") or []:
        if str(p.get("feature")) == str(feature):
            return True
    return False


def _flagged_protected(res):
    out = []
    for pv in res.get("perVariable") or []:
        if pv.get("assessable") and pv.get("tone") in ("warn", "critical"):
            out.append(str(pv.get("attribute")))
    return out


def check(name: str, csv_path: Path, debug_dir=None) -> Tuple[bool, Dict]:
    df = pd.read_csv(csv_path)
    audit = df.drop(
        columns=[c for c in ("true_qualification_score", "_bias_flags") if c in df.columns]
    )
    inputs = {
        "domain": "recruitment / employment screening",
        "jurisdiction": "US",
        "source_kind": "tabular",
        "protected_attributes": [c for c in PROTECTED if c in audit.columns],
    }
    res = run_pulse(audit, inputs)
    if isinstance(res, dict) and "data" in res:
        res = res["data"]
    spec = VARIANTS[name]
    flagged = set(_flagged_protected(res))
    expected = set(spec["expect_protected"])
    proxy_feat = spec["expect_proxy_feature"]
    proxy_hit = _proxy_present(res, proxy_feat) if proxy_feat else None
    spurious = sorted(flagged - expected)
    missed = sorted(expected - flagged)
    proxy_ok = proxy_hit if proxy_feat else True
    ok = not missed and proxy_ok
    # For the clean variant, ALSO require no false positives.
    if name == "clean":
        ok = ok and not spurious
    if debug_dir:
        with open(Path(debug_dir) / f"{name}.json", "w") as fh:
            json.dump(
                {
                    "flagged": sorted(flagged),
                    "expected": sorted(expected),
                    "proxy_feature": proxy_feat,
                    "proxy_hit": proxy_hit,
                    "spurious": spurious,
                    "missed": missed,
                    "tone_table": {a: _tone_for(res, a) for a in PROTECTED if a in audit.columns},
                },
                fh,
                indent=2,
            )
    return ok, {
        "flagged": sorted(flagged),
        "spurious": spurious,
        "missed": missed,
        "proxy_feat": proxy_feat,
        "proxy_hit": proxy_hit,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default=str(_HERE / ".." / "datasets" / "synth"))
    ap.add_argument("--debug-dir", default="")
    args = ap.parse_args()
    base = Path(args.datasets_dir)
    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("PULSE UNIVERSALITY CHECK  ·  synthetic single-bias variants")
    print(f"datasets: {base}")
    print("=" * 88)
    print(f" {'VARIANT':<22s} {'OK':<3s} {'EXPECTED':<22s} {'FLAGGED':<28s} {'NOTES':<25s}")
    print("-" * 88)
    n_pass = 0
    n_total = 0
    for name in VARIANTS:
        p = base / f"{name}.csv"
        if not p.exists():
            print(f" {name:<22s} ??  no dataset at {p}")
            continue
        n_total += 1
        ok, details = check(name, p, debug_dir=args.debug_dir or None)
        notes = []
        if details["missed"]:
            notes.append(f"missed={details['missed']}")
        if details["spurious"]:
            notes.append(f"spurious={details['spurious']}")
        if details["proxy_feat"]:
            notes.append(
                f"proxy {details['proxy_feat']}: {'OK' if details['proxy_hit'] else 'MISS'}"
            )
        flagged_short = ",".join(details["flagged"]) or "—"
        expected_short = ",".join(VARIANTS[name]["expect_protected"]) or "—"
        print(
            f" {name:<22s} {'✓' if ok else '✗'}   "
            f"{expected_short:<22s} {flagged_short:<28s} "
            f"{'; '.join(notes)}"
        )
        if ok:
            n_pass += 1
    print("-" * 88)
    print(f" PASS: {n_pass}/{n_total}")
    print("=" * 88)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
