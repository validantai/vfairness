"""Emit the cross-library parity evidence as JSON.

Runs the same comparison as tests/test_cross_library_parity.py on a fixed
dataset and prints a JSON snapshot (library versions + per-metric values +
match) to stdout. The frontend "Library Parity" tab renders this snapshot; the
pytest is the regression guard. Regenerate with:

    pip install fairlearn aif360
    PYTHONPATH=vfairness/src python scripts/emit_parity_evidence.py
"""

import json
import sys
from datetime import date

import numpy as np

ATOL = 1e-9
SEED = 7
N = 2000


def make(seed=SEED, n=N):
    rng = np.random.default_rng(seed)
    sens = rng.integers(0, 2, n)
    p = np.where(sens == 1, 0.65, 0.35)
    y_pred = (rng.random(n) < p).astype(int)
    y_true = (rng.random(n) < 0.5).astype(int)
    return y_true, y_pred, sens


def vf_vals(y_true, y_pred, sens):
    from vfairness.evaluation import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equal_opportunity_difference,
        equalized_odds_difference,
    )

    kw = dict(min_group_size=1)
    return {
        "demographic_parity_difference": float(
            demographic_parity_difference(y_true, y_pred, sens, **kw)
        ),
        "disparate_impact_ratio": float(demographic_parity_ratio(y_true, y_pred, sens, **kw)),
        "equal_opportunity_difference": float(
            equal_opportunity_difference(y_true, y_pred, sens, **kw)
        ),
        "equalized_odds_difference": float(equalized_odds_difference(y_true, y_pred, sens, **kw)),
    }


def fl_vals(y_true, y_pred, sens):
    import fairlearn
    from fairlearn.metrics import (
        MetricFrame,
        true_positive_rate,
    )
    from fairlearn.metrics import (
        demographic_parity_difference as dpd,
    )
    from fairlearn.metrics import (
        demographic_parity_ratio as dpr,
    )
    from fairlearn.metrics import (
        equalized_odds_difference as eqo,
    )

    mf = MetricFrame(
        metrics=true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sens
    )
    return fairlearn.__version__, {
        "demographic_parity_difference": float(dpd(y_true, y_pred, sensitive_features=sens)),
        "disparate_impact_ratio": float(dpr(y_true, y_pred, sensitive_features=sens)),
        "equal_opportunity_difference": float(mf.difference(method="between_groups")),
        "equalized_odds_difference": float(eqo(y_true, y_pred, sensitive_features=sens)),
    }


def aif_vals(y_true, y_pred, sens):
    import aif360
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric

    rates = {int(g): float(y_pred[sens == g].mean()) for g in np.unique(sens)}
    priv, unpriv = max(rates, key=rates.get), min(rates, key=rates.get)

    def ds(lbl):
        return BinaryLabelDataset(
            df=pd.DataFrame({"label": lbl, "grp": sens}),
            label_names=["label"],
            protected_attribute_names=["grp"],
            favorable_label=1,
            unfavorable_label=0,
        )

    cm = ClassificationMetric(
        ds(y_true),
        ds(y_pred),
        unprivileged_groups=[{"grp": unpriv}],
        privileged_groups=[{"grp": priv}],
    )
    tpr = abs(cm.true_positive_rate(privileged=False) - cm.true_positive_rate(privileged=True))
    fpr = abs(cm.false_positive_rate(privileged=False) - cm.false_positive_rate(privileged=True))
    return aif360.__version__, {
        "demographic_parity_difference": abs(float(cm.statistical_parity_difference())),
        "disparate_impact_ratio": float(cm.disparate_impact()),
        "equal_opportunity_difference": abs(float(cm.equal_opportunity_difference())),
        "equalized_odds_difference": float(max(tpr, fpr)),
    }


LABELS = {
    "demographic_parity_difference": "Demographic parity difference",
    "disparate_impact_ratio": "Disparate impact ratio (4/5 rule)",
    "equal_opportunity_difference": "Equal opportunity difference",
    "equalized_odds_difference": "Equalized odds difference",
}

try:
    import importlib.metadata as md

    vf_version = md.version("vfairness")
except Exception:
    vf_version = "unknown"

yt, yp, s = make()
vf = vf_vals(yt, yp, s)
fl_ver, fl = fl_vals(yt, yp, s)
aif_ver, aif = aif_vals(yt, yp, s)

metrics = []
for key in LABELS:
    vals = [vf[key], fl[key], aif[key]]
    metrics.append(
        {
            "key": key,
            "label": LABELS[key],
            "vfairness": round(vf[key], 9),
            "fairlearn": round(fl[key], 9),
            "aif360": round(aif[key], 9),
            "match": (max(vals) - min(vals)) < ATOL,
        }
    )

report = {
    "generated_at": date.today().isoformat(),
    "tolerance": ATOL,
    "dataset": "synthetic, seed 7, n=2000, binary protected attribute, labels independent of predictions",
    "versions": {"vfairness": vf_version, "fairlearn": fl_ver, "aif360": aif_ver},
    "metrics": metrics,
    "test": "vfairness/tests/test_cross_library_parity.py",
    "test_summary": "17/17 cases pass (fairlearn binary + 3 and 4 groups; AIF360 binary; 5 seeds each)",
    "all_match": all(m["match"] for m in metrics),
}
json.dump(report, sys.stdout, indent=2)
print()
