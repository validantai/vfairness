"""Run every downloadable test object through the library and record what executed.

    python scripts/verify_test_objects.py

Writes docs/site/test-objects/verification.json, which the test-objects page
quotes. A file earns its place on that page by being RUN here, not by being
described: each entry records the inputs used, whether the run succeeded, the
route the library took, and what it found or could not assess.

The synthetic pack is graded both ways. Each planted file must surface its
planted mechanism, and the clean control must surface nothing; a detector that
only ever says "bias" would pass the first half and fail the second.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from vfairness.operations.pulse import run_pulse  # noqa: E402

DATA = HERE.parent / "docs" / "site" / "test-objects" / "data"
OUT = DATA.parent / "verification.json"

SYNTH_PROTECTED = [
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
# What each synthetic file must show. "attr": a protected column that must read
# warn or critical. "proxy": a feature the proxy battery must name.
SYNTH_EXPECT = {
    "clean": {},
    "gender_penalty": {"attr": "gender"},
    "race_penalty": {"attr": "race_ethnicity"},
    "age_cliff": {"attr": "age"},
    "disability_penalty": {"attr": "disability_status"},
    # A planted proxy that tracks race produces a real race gap downstream, so
    # race_ethnicity is EXPECTED there, not a false flag.
    "zip_redlining": {"proxy": "zip_minority_majority", "also": ["race_ethnicity"]},
    "surname_proxy": {},  # race is not in the file: undetectable by construction
    "photo_laundering": {"proxy": "photo_attractiveness_score", "also": ["race_ethnicity"]},
}

RUNS = [
    # (relative path, inputs, reader)
    (
        "predictive/adult_test_with_predictions.csv",
        {
            "source_kind": "predictive",
            "domain": "Income",
            "jurisdiction": "US",
            "protected_attributes": ["sex", "race"],
        },
        "csv",
    ),
    # A1 with no labels: the decisions and the groups, nothing else.
    (
        "predictive/adult_test_with_predictions.csv",
        {
            "source_kind": "predictive",
            "domain": "Income",
            "jurisdiction": "US",
            "protected_attributes": ["sex", "race"],
            "_label": "A1 without labels",
        },
        "adult_a1",
    ),
    (
        "predictive/lending_fairness.csv",
        {
            "source_kind": "predictive",
            "domain": "Consumer lending",
            "jurisdiction": "EU",
            "protected_attributes": ["gender", "race"],
        },
        "csv",
    ),
    (
        "predictive/hiring_fairness.csv",
        {
            "source_kind": "predictive",
            "domain": "Employment",
            "jurisdiction": "EU",
            "protected_attributes": ["gender", "ethnicity"],
        },
        "csv",
    ),
    (
        "predictive/healthcare_fairness.csv",
        {
            "source_kind": "predictive",
            "domain": "Healthcare",
            "jurisdiction": "US",
            "protected_attributes": ["sex", "race"],
        },
        "csv",
    ),
    (
        "predictive/recruitment_fairness_dataset.csv",
        {
            "source_kind": "predictive",
            "domain": "recruitment / employment screening",
            "jurisdiction": "US",
            "protected_attributes": ["gender", "race_ethnicity", "age", "disability_status"],
        },
        "recruit",
    ),
    (
        "llm/generative_support_fairness.csv",
        {
            "source_kind": "generative",
            "domain": "Customer support",
            "jurisdiction": "EU",
            "protected_attributes": ["gender", "ethnicity"],
        },
        "csv",
    ),
    (
        "llm/discrim_eval_qwen25_7b.csv",
        {
            "source_kind": "predictive",
            "domain": "Consequential decisions",
            "jurisdiction": "US",
            "protected_attributes": ["gender", "race"],
        },
        "discrim",
    ),
    (
        "agent/agent_traces_support.csv",
        {
            "source_kind": "agent",
            "domain": "Public benefits",
            "jurisdiction": "EU",
            "protected_attributes": ["gender"],
        },
        "csv",
    ),
    (
        "agent/agent_traces_otel.jsonl",
        {"source_kind": "agent", "domain": "Public benefits", "jurisdiction": "EU"},
        "jsonl",
    ),
    (
        "agent/agent_traces_langfuse.jsonl",
        {"source_kind": "agent", "domain": "Public benefits", "jurisdiction": "EU"},
        "jsonl",
    ),
    (
        "agent/loan_agent_persona_runs.csv",
        {
            "source_kind": "agent",
            "domain": "Consumer lending",
            "jurisdiction": "CH",
            "protected_attributes": ["group"],
        },
        "agent_runs",
    ),
    (
        "agent/loan_agent_otel.jsonl",
        {"source_kind": "agent", "domain": "Consumer lending", "jurisdiction": "CH"},
        "jsonl",
    ),
]


def _read(rel: str, how: str) -> pd.DataFrame:
    p = DATA / rel
    if how == "jsonl":
        return pd.read_json(p, lines=True)
    df = pd.read_csv(p, low_memory=False)
    if how == "adult_a1":
        return df.drop(columns=["y_true", "y_prob"])
    if how == "recruit":
        return df.drop(columns=["true_qualification_score", "_bias_flags"])
    if how == "discrim":
        # An LLM's yes/no decisions ARE a predictive object: the answer is the
        # decision and p_yes, from the logprobs, is its score. A row whose
        # p_yes could not be measured is dropped, not filled.
        df = df[pd.to_numeric(df["p_yes"], errors="coerce").notna()].copy()
        return pd.DataFrame(
            {
                "gender": df["gender"],
                "race": df["race"],
                "age": df["age"],
                "y_pred": (df["p_yes"] >= 0.5).astype(int),
                "y_prob": df["p_yes"],
            }
        )
    if how == "agent_runs":
        return df[df["group"] != "control"]
    return df


def _summary(res: dict) -> dict:
    if not res.get("success"):
        return {"success": False, "error": str(res.get("error"))[:300]}
    d = res["data"]
    out = {
        "success": True,
        "sourceKind": d.get("sourceKind"),
        "scope": {k: (d.get("scope") or {}).get(k) for k in ("level", "pathway")},
        "verdict": {k: (d.get("verdict") or {}).get(k) for k in ("tone", "headline")},
        "assurance": (d.get("assurance") or {}).get("overall"),
        "perVariable": [
            {
                "attribute": p.get("attribute"),
                "assessable": p.get("assessable"),
                "tone": p.get("tone"),
                "significant": p.get("significant"),
                "pValueAdjusted": p.get("pValueAdjusted"),
                "gap": p.get("gap"),
                "fourFifthsRatio": p.get("fourFifthsRatio"),
            }
            for p in (d.get("perVariable") or [])
        ],
        "calibration": bool((d.get("calibration") or {}).get("available")),
        "proxies": [str(p.get("feature")) for p in ((d.get("proxies") or {}).get("proxies") or [])],
    }
    for k in ("agent", "generative", "llm"):
        if isinstance(d.get(k), dict):
            blk = d[k]
            out[k] = {
                kk: blk.get(kk)
                for kk in ("tone", "headline", "ingestion", "available", "summary")
                if kk in blk
            }
    return out


def _flagged(summary: dict) -> list:
    """Attributes Pulse CONFIRMED: significant after its own correction.

    Tone alone is not detection. In US employment the four-fifths screen sets
    the tone from the ratio by design, so a 27-person group can read critical
    on data with no planted bias (measured on clean.csv, 2026-10-01). Grading
    on tone would call that a detection and grade every file the same."""
    return [
        p["attribute"]
        for p in summary.get("perVariable", [])
        if p.get("assessable") and p.get("significant") is True
    ]


def _toned(summary: dict) -> list:
    return [
        p["attribute"]
        for p in summary.get("perVariable", [])
        if p.get("assessable") and p.get("tone") in ("warn", "critical")
    ]


def main() -> None:
    only_synth = "--synthetic-only" in sys.argv
    record = {"run_on": date.today().isoformat(), "objects": [], "synthetic_grading": []}
    if only_synth and OUT.exists():
        record["objects"] = json.loads(OUT.read_text())["objects"]
    for rel, inputs, how in [] if only_synth else RUNS:
        if not (DATA / rel).exists():
            record["objects"].append({"path": rel, "built": False})
            continue
        df = _read(rel, how)
        s = _summary(run_pulse(df, inputs))
        record["objects"].append(
            {"path": rel, "built": True, "rows_used": int(len(df)), "inputs": inputs, **s}
        )
        print(f"{rel}: success={s['success']} verdict={s.get('verdict')}", flush=True)

    for name, expect in SYNTH_EXPECT.items():
        df = pd.read_csv(DATA / "synthetic" / f"{name}.csv")
        df = df.drop(columns=["true_qualification_score", "_bias_flags"])
        inputs = {
            "source_kind": "predictive",
            "domain": "recruitment / employment screening",
            "jurisdiction": "US",
            "protected_attributes": [c for c in SYNTH_PROTECTED if c in df.columns],
        }
        s = _summary(run_pulse(df, inputs))
        flagged = _flagged(s)
        planted_attr, planted_proxy = expect.get("attr"), expect.get("proxy")
        found = (
            (planted_attr in flagged)
            if planted_attr
            else (planted_proxy in s.get("proxies", []))
            if planted_proxy
            else None
        )
        spurious = [a for a in flagged if a != planted_attr and a not in expect.get("also", [])]
        record["synthetic_grading"].append(
            {
                "file": f"synthetic/{name}.csv",
                "planted": expect or "nothing",
                "planted_found": found,
                "flagged": flagged,
                "spurious": spurious,
                "proxies_named": s.get("proxies", []),
                "verdict": s.get("verdict"),
                "toned_warn_or_critical": _toned(s),
            }
        )
        print(f"synthetic/{name}: found={found} flagged={flagged} spurious={spurious}", flush=True)

    OUT.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
