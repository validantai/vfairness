"""Build the downloadable test-object pack served at /test-objects/ on the docs site.

Every file a reader can download from docs/site/test-objects/data/ is either
generated here or recorded here, and manifest.json is written from this file so
that every download carries its source, licence, sha256, shape, the access tier
it reaches, and (for synthetic data) the bias that was planted in it.

    python scripts/build_test_objects.py synth      # planted-bias variants
    python scripts/build_test_objects.py adult      # UCI Adult + a pinned model
    python scripts/build_test_objects.py manifest   # sha256 + shape for every file
    python scripts/build_test_objects.py page       # fill the page's generated tables

The LLM, agent and multi-agent packs need a running model and are built by
scripts/build_live_test_objects.py; this script only records them.

WHAT IS DELIBERATELY NOT HOSTED. COMPAS: the ProPublica repository carries no
licence file, so the pack ships get_compas.py, which downloads the origin file
and derives the prediction columns locally, instead of a copy of the data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "docs" / "site" / "test-objects" / "data"

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
# sha256 of the two members as downloaded on 2026-10-01. A changed origin file
# must be noticed, not silently retrained on.
ADULT_SHA256 = {
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
}
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
# The model never sees sex or race. It is the "fairness through unawareness"
# baseline on purpose: the page uses it to show that removing the attribute
# does not remove the disparity (relationship = Husband / Wife carries sex).
ADULT_FEATURES_NUM = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
ADULT_FEATURES_CAT = ["workclass", "marital_status", "occupation", "relationship"]

CC_BY_4 = "CC BY 4.0"
# The lending, hiring and healthcare files are copies of the Validant platform's
# Navigator samples (lending is its lending_full_pipeline file: the plain lending
# sample does not contain the gaps the platform describes) with three columns renamed (the label to y_true,
# model_prediction to y_pred, model_probability to y_prob) so that run_pulse and
# FairnessAnalyzer find them with no column mapping. Values are unchanged.
OURS = "CC BY 4.0 (Validant)"

# Everything a reader can download. Paths are relative to DATA.
CATALOGUE = {
    # predictive
    "predictive/lending_fairness.csv": dict(
        pathway="predictive",
        tier="A2",
        licence=OURS,
        origin="Validant (synthetic)",
        title="Synthetic credit decisions with model scores",
        label="y_true",
        prediction="y_pred",
        score="y_prob",
        protected=["gender", "race", "age"],
        planted="Measured: Black applicants approved 6.0 points less often than White (42.0 vs "
        "48.0 percent), Hispanic 4.5 points; women 9.2 points less than men (41.0 vs "
        "50.2).",
    ),
    "predictive/hiring_fairness.csv": dict(
        pathway="predictive",
        tier="A2",
        licence=OURS,
        origin="Validant (synthetic)",
        title="Synthetic hiring decisions with model scores, includes non-binary",
        label="y_true",
        prediction="y_pred",
        score="y_prob",
        protected=["gender", "ethnicity", "age"],
        planted="Measured: small gaps only. At equal interview and technical scores women are "
        "selected 2.0 points less often (not significant); non-binary n = 50. Useful to "
        "check that a tool does not over-flag.",
    ),
    "predictive/healthcare_fairness.csv": dict(
        pathway="predictive",
        tier="A2",
        licence=OURS,
        origin="Validant (synthetic)",
        title="Synthetic clinical risk scores",
        label="y_true",
        prediction="y_pred",
        score="y_prob",
        protected=["sex", "race", "age"],
        planted="Measured: at similar clinical profiles Black patients get risk scores 4.2 points "
        "lower. Probabilities understate risk for every group by 9 to 17 points, so the "
        "miscalibration is global, not group-specific.",
    ),
    "predictive/recruitment_fairness_dataset.csv": dict(
        pathway="predictive",
        tier="A1",
        licence=OURS,
        origin="Validant (synthetic)",
        title="Synthetic recruitment funnel, 9 planted biases, ground truth included",
        label="true_qualification_score",
        prediction="invite_decision",
        score="model_score",
        protected=[
            "gender",
            "race_ethnicity",
            "age",
            "religion",
            "disability_status",
            "sexual_orientation",
            "national_origin",
            "marital_status",
            "veteran_status",
        ],
        planted="Nine mechanisms, flagged per row in _bias_flags (B1 to B9). All personal "
        "details are invented.",
    ),
    "predictive/adult_test_with_predictions.csv": dict(
        pathway="predictive",
        tier="A2",
        licence=CC_BY_4,
        origin="UCI Adult, Becker and Kohavi (1996), doi:10.24432/C5XW20; predictions by Validant",
        origin_url="https://archive.ics.uci.edu/dataset/2/adult",
        title="UCI Adult test split with a pinned logistic regression's predictions",
        label="y_true",
        prediction="y_pred",
        score="y_prob",
        protected=["sex", "race", "age"],
        planted="Nothing planted: a real 1994 census extract. The model never sees sex or race.",
    ),
    "predictive/adult_logreg_model.json": dict(
        pathway="predictive",
        tier="A3",
        licence=CC_BY_4,
        origin="Validant, trained on UCI Adult",
        origin_url="https://archive.ics.uci.edu/dataset/2/adult",
        title="The same model, every weight: rebuild predict() in ten lines",
        protected=[],
        planted="n/a",
    ),
    "scripts/get_compas.py": dict(
        pathway="predictive",
        tier="A2",
        licence="MIT (script); data is not re-hosted",
        origin="ProPublica, compas-analysis",
        origin_url="https://github.com/propublica/compas-analysis",
        title="Download COMPAS from ProPublica and add prediction columns",
        protected=["race", "sex", "age"],
        planted="Nothing planted: the real Northpointe decile scores.",
    ),
    # synthetic
    **{
        f"synthetic/{name}.csv": dict(
            pathway="synthetic",
            tier="A1",
            licence=OURS,
            origin="Validant (generated)",
            origin_url="",
            title=title,
            label="true_qualification_score",
            prediction="invite_decision",
            score="model_score",
            protected=["gender", "race_ethnicity", "age", "disability_status"],
            planted=planted,
        )
        for name, title, planted in [
            ("clean", "Control: no bias planted", "None. A correct detector reports nothing here."),
            (
                "gender_penalty",
                "Gender penalty in tech roles",
                "Half of female invites in tech roles removed (B1).",
            ),
            ("race_penalty", "Race penalty", "Half of Black and Hispanic invites removed (B2)."),
            ("age_cliff", "Age cliff", "Applicants under 25 never invited (B3)."),
            (
                "disability_penalty",
                "Disability penalty",
                "80 percent of invites removed for any reported disability (B7).",
            ),
            (
                "zip_redlining",
                "ZIP code redlining (a proxy)",
                "Minority-majority ZIP invites halved; race itself untouched (B4).",
            ),
            (
                "surname_proxy",
                "Surname proxy, race column removed",
                "Surnames typical of Black and Hispanic applicants lose half their invites; "
                "race_ethnicity is not in the file (B5).",
            ),
            (
                "photo_laundering",
                "Photo score laundering",
                "A photo score correlated with race drives half the decision (B9).",
            ),
        ]
    },
    # llm
    "llm/generative_support_fairness.csv": dict(
        pathway="llm",
        tier="A1",
        licence=OURS,
        origin="Validant (synthetic)",
        title="Support-bot prompts and replies by customer gender and ethnicity",
        protected=["gender", "ethnicity"],
        planted="Tone and refusal differ by group in the replies.",
    ),
    "llm/discrim_eval_qwen25_7b.csv": dict(
        pathway="llm",
        tier="A2",
        licence=CC_BY_4,
        origin="Prompts: Anthropic discrim-eval (Tamkin et al. 2023); answers and logprobs: "
        "Validant run of qwen2.5:7b",
        origin_url="https://huggingface.co/datasets/Anthropic/discrim-eval",
        title="Real yes/no decisions with log probabilities from a pinned open model",
        protected=["age", "gender", "race"],
        planted="Nothing planted: measured behaviour of the model.",
    ),
    # agent
    "agent/agent_traces_support.csv": dict(
        pathway="agent",
        tier="A1",
        licence=OURS,
        origin="Validant (synthetic)",
        title="Benefits-triage agent, one row per episode",
        protected=["gender"],
        planted="Tool choice and routing differ by gender.",
    ),
    "agent/agent_traces_otel.jsonl": dict(
        pathway="agent",
        tier="A1",
        licence=OURS,
        origin="Validant (synthetic)",
        title="The same agent as raw OpenTelemetry GenAI spans",
        protected=["gender"],
        planted="As above.",
    ),
    "agent/agent_traces_langfuse.jsonl": dict(
        pathway="agent",
        tier="A1",
        licence=OURS,
        origin="Validant (synthetic)",
        title="The same agent as a Langfuse export",
        protected=["gender"],
        planted="As above.",
    ),
    "agent/loan_agent_persona_runs.csv": dict(
        pathway="agent",
        tier="A1",
        licence=OURS,
        origin="Validant run of qwen2.5:7b",
        title="A real tool-calling loan agent, counterfactual personas, one row per episode",
        protected=["persona_gender", "persona_origin"],
        planted="Nothing planted: measured behaviour of the agent.",
    ),
    "agent/loan_agent_otel.jsonl": dict(
        pathway="agent",
        tier="A1",
        licence=OURS,
        origin="Validant run of qwen2.5:7b",
        title="The same runs as OpenTelemetry GenAI spans",
        protected=["persona_gender", "persona_origin"],
        planted="Nothing planted.",
    ),
    "scripts/loan_agent_persona_harness.py": dict(
        pathway="agent",
        tier="A1",
        licence="MIT",
        origin="Validant",
        title="The harness that produced the runs: point it at your own agent",
        protected=[],
        planted="n/a",
    ),
    # multi-agent
    "multi_agent/loan_committee_runs.csv": dict(
        pathway="multi_agent",
        tier="A1",
        licence=OURS,
        origin="Validant run of qwen2.5:7b",
        title="Two-agent loan committee: screener, then reviewer, per-agent outputs",
        protected=["persona_gender", "persona_origin"],
        planted="Nothing planted: measured behaviour of the system.",
    ),
    "multi_agent/loan_committee_harness_trace.json": dict(
        pathway="multi_agent",
        tier="A1",
        licence=OURS,
        origin="Validant run of qwen2.5:7b",
        title="The same run as a MultiAgentRunHarness trace",
        protected=["persona_gender", "persona_origin"],
        planted="Nothing planted.",
    ),
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_synth() -> None:
    out = DATA / "synthetic"
    subprocess.run(
        [sys.executable, str(HERE / "synth_single_bias_datasets.py"), "--out", str(out)],
        check=True,
    )


def _load_adult():
    raw = urllib.request.urlopen(ADULT_URL, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    frames = {}
    for member, expected in ADULT_SHA256.items():
        blob = z.read(member)
        got = hashlib.sha256(blob).hexdigest()
        if got != expected:
            raise SystemExit(f"{member}: origin changed (sha256 {got}, pinned {expected})")
        df = pd.read_csv(
            io.BytesIO(blob),
            header=None,
            names=ADULT_COLUMNS,
            skipinitialspace=True,
            na_values="?",
            skiprows=1 if member == "adult.test" else 0,
        )
        df["income"] = df["income"].str.rstrip(".")
        frames[member] = df.dropna().reset_index(drop=True)
    return frames["adult.data"], frames["adult.test"]


def build_adult() -> None:
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    train, test = _load_adult()
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ADULT_FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ADULT_FEATURES_CAT),
        ]
    )
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    y_train = (train["income"] == ">50K").astype(int)
    model.fit(train[ADULT_FEATURES_NUM + ADULT_FEATURES_CAT], y_train)

    X_test = test[ADULT_FEATURES_NUM + ADULT_FEATURES_CAT]
    prob = model.predict_proba(X_test)[:, 1]
    out = test.drop(columns=["income"]).copy()
    out["y_true"] = (test["income"] == ">50K").astype(int)
    out["y_prob"] = np.round(prob, 6)
    out["y_pred"] = (prob >= 0.5).astype(int)
    (DATA / "predictive").mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA / "predictive" / "adult_test_with_predictions.csv", index=False)

    scaler = model.named_steps["pre"].named_transformers_["num"]
    enc = model.named_steps["pre"].named_transformers_["cat"]
    clf = model.named_steps["clf"]
    names = list(model.named_steps["pre"].get_feature_names_out())
    card = {
        "model": "logistic regression, sklearn " + sklearn.__version__,
        "trained_on": "UCI Adult adult.data, rows with no missing value",
        "features_numeric": ADULT_FEATURES_NUM,
        "features_categorical": ADULT_FEATURES_CAT,
        "not_used": ["sex", "race", "native_country", "fnlwgt", "education"],
        "scaler_mean": dict(zip(ADULT_FEATURES_NUM, map(float, scaler.mean_))),
        "scaler_scale": dict(zip(ADULT_FEATURES_NUM, map(float, scaler.scale_))),
        "categories": {c: list(map(str, v)) for c, v in zip(ADULT_FEATURES_CAT, enc.categories_)},
        "intercept": float(clf.intercept_[0]),
        "coefficients": dict(zip(names, map(float, clf.coef_[0]))),
        "threshold": 0.5,
        "how_to_predict": "z = intercept + sum(coef * feature), numeric features standardised "
        "with scaler_mean / scaler_scale, categorical one-hot; "
        "y_prob = 1 / (1 + exp(-z)); y_pred = y_prob >= threshold",
    }
    (DATA / "predictive" / "adult_logreg_model.json").write_text(json.dumps(card, indent=2))
    acc = float(((prob >= 0.5).astype(int) == out["y_true"]).mean())
    print(f"adult: train {len(train)}, test {len(test)}, accuracy {acc:.4f}")


def build_manifest() -> None:
    files = []
    missing = []
    for rel, meta in CATALOGUE.items():
        p = DATA / rel
        if not p.exists():
            missing.append(rel)
            continue
        entry = {"path": rel, "bytes": p.stat().st_size, "sha256": _sha256(p), **meta}
        if p.suffix == ".csv":
            df = pd.read_csv(p, low_memory=False)
            entry["rows"], entry["columns"] = int(len(df)), list(map(str, df.columns))
        elif p.suffix == ".jsonl":
            entry["rows"] = sum(1 for line in p.open() if line.strip())
        files.append(entry)
    stray = sorted(
        str(p.relative_to(DATA))
        for p in DATA.rglob("*")
        if p.is_file() and p.name != "manifest.json" and str(p.relative_to(DATA)) not in CATALOGUE
    )
    if stray:
        raise SystemExit(f"files in the pack that the catalogue does not describe: {stray}")
    (DATA / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(f"manifest: {len(files)} files")
    if missing:
        print(f"NOT YET BUILT ({len(missing)}): {missing}")


PAGE = DATA.parent / "index.html"
PATHWAY_LABEL = {
    "predictive": "Predictive",
    "synthetic": "Synthetic",
    "llm": "Generative (LLM)",
    "agent": "Agent",
    "multi_agent": "Multi-agent",
}


def _esc(x) -> str:
    import html

    return html.escape(str(x), quote=True)


def _fill(page: str, name: str, body: str) -> str:
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    i, j = page.index(begin) + len(begin), page.index(end)
    return page[:i] + "\n" + body + "\n" + page[j:]


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(1, round(n / 1e3))} KB"


def build_page() -> None:
    """Fill the generated blocks of the page from manifest.json and verification.json,
    so the page cannot describe a file differently from the file's own record."""
    manifest = json.loads((DATA / "manifest.json").read_text())["files"]
    ver_path = DATA.parent / "verification.json"
    ver = (
        json.loads(ver_path.read_text())
        if ver_path.exists()
        else {"objects": [], "synthetic_grading": []}
    )
    page = PAGE.read_text()

    rows = []
    for f in manifest:
        shape = f"{f['rows']:,} rows" if "rows" in f else ""
        origin = _esc(f.get("origin", ""))
        if f.get("origin_url"):
            origin = (
                f'<a href="{_esc(f["origin_url"])}" target="_blank" rel="noopener">{origin}</a>'
            )
        rows.append(
            f'      <tr><td><a class="to-dl" href="data/{_esc(f["path"])}">{_esc(f["path"])}</a>'
            f'<br><span class="to-note">{_esc(f["title"])}</span></td>'
            f"<td>{_esc(PATHWAY_LABEL[f['pathway']])}</td>"
            f'<td><span class="to-tier to-tier--{_esc(f["tier"])}">{_esc(f["tier"])}</span></td>'
            f"<td>{_esc(f.get('planted', ''))}</td>"
            f'<td>{origin}<br><span class="to-note">{_esc(f["licence"])}</span></td>'
            f'<td>{shape}<br><span class="to-note">{_size(f["bytes"])}</span><br>'
            f'<code title="sha256">{f["sha256"][:12]}</code></td></tr>'
        )
    catalogue = (
        '  <div class="to-scroll"><table class="to-cat">\n    <thead><tr><th>File</th><th>Pathway</th><th>Tier</th>'
        "<th>What is in it</th><th>Source and licence</th><th>Size and sha256</th></tr></thead>\n"
        "    <tbody>\n" + "\n".join(rows) + "\n    </tbody>\n  </table></div>"
    )
    page = _fill(page, "catalogue", catalogue)

    vrows = []
    for o in ver["objects"]:
        if not o.get("built"):
            vrows.append(
                f'      <tr><td>{_esc(o["path"])}</td><td colspan="3" class="to-na">'
                "Not built yet, so not run.</td></tr>"
            )
            continue
        label = o.get("inputs", {}).get("_label")
        name = _esc(o["path"]) + (
            f'<br><span class="to-note">{_esc(label)}</span>' if label else ""
        )
        if not o.get("success"):
            vrows.append(
                f'      <tr><td>{name}</td><td colspan="3" class="to-no">Run failed: '
                f"{_esc(o.get('error'))}</td></tr>"
            )
            continue
        v = o.get("verdict") or {}
        flagged = (
            ", ".join(
                f"{_esc(p['attribute'])} ({_esc(p['tone'])})"
                for p in o.get("perVariable", [])
                if p.get("assessable")
            )
            or '<span class="to-na">none assessable</span>'
        )
        verdict = _esc(v.get("headline", ""))
        probe = (o.get("agent") or {}).get("summary")
        if probe:
            # The agent route's own measured finding, shown beside the top-level
            # verdict because the two can disagree (see the note under the table).
            verdict += f'<br><span class="to-note">Agent probe: {_esc(probe)}</span>'
        vrows.append(
            f"      <tr><td>{name}</td><td>{o['rows_used']:,}</td>"
            f"<td>{verdict}</td><td>{flagged}</td></tr>"
        )
    verification = (
        f'  <p class="to-note">Run on {_esc(ver.get("run_on", "not yet run"))}.</p>\n'
        '  <div class="to-scroll"><table>\n    <thead><tr><th>File</th><th>Rows used</th>'
        "<th>Verdict returned</th><th>Per attribute</th></tr></thead>\n    <tbody>\n"
        + "\n".join(vrows)
        + "\n    </tbody>\n  </table></div>"
    )
    page = _fill(page, "verification", verification)

    grows = []
    for g in ver["synthetic_grading"]:
        planted = g["planted"]
        if planted == "nothing" or not planted:
            what = "nothing" if "clean" in g["file"] else "not checkable (race is not in the file)"
        elif "attr" in planted:
            what = f"a gap on {planted['attr']}"
        else:
            what = f"a proxy, {planted['proxy']}"
        if isinstance(planted, dict) and planted.get("also"):
            what += f" (a {', '.join(planted['also'])} gap follows from it)"
        if g["planted_found"] is True:
            found = '<span class="to-yes">found</span>'
        elif g["planted_found"] is False:
            found = '<span class="to-no">missed</span>'
        else:
            found = '<span class="to-na">nothing to find</span>'
        spur = ", ".join(map(_esc, g["spurious"])) or "none"
        spur_cls = "to-no" if g["spurious"] else "to-yes"
        tone = (g.get("verdict") or {}).get("tone", "")
        grows.append(
            f"      <tr><td>{_esc(g['file'])}</td><td>{_esc(what)}</td><td>{found}</td>"
            f'<td><span class="{spur_cls}">{spur}</span></td><td>{_esc(tone)}</td></tr>'
        )
    grading = (
        '  <div class="to-scroll"><table>\n    <thead><tr><th>File</th><th>Planted</th>'
        "<th>Planted mechanism</th><th>Confirmed but not planted</th><th>Top-level verdict</th>"
        "</tr></thead>\n    <tbody>\n" + "\n".join(grows) + "\n    </tbody>\n  </table></div>"
    )
    page = _fill(page, "synthetic-grading", grading)
    PAGE.write_text(page)
    print(f"page: {len(manifest)} files, {len(ver['objects'])} runs, {len(grows)} graded")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["synth", "adult", "manifest", "page"])
    step = ap.parse_args().step
    {"synth": build_synth, "adult": build_adult, "manifest": build_manifest, "page": build_page}[
        step
    ]()


if __name__ == "__main__":
    main()
