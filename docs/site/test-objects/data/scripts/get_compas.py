"""Download COMPAS from its origin and add prediction columns. Needs only pandas.

    python get_compas.py                # writes compas_with_predictions.csv

Why a script and not a file: ProPublica's repository carries no licence, and the
origin file names real defendants. So nothing is re-hosted. This downloads the
original from ProPublica, checks it is the file this script was written against,
applies ProPublica's own filters, drops every name and case number, and adds:

    y_true  two_year_recid: re-arrested within two years (the label)
    score   decile_score, 1 to 10: the REAL score of the deployed COMPAS tool
    y_pred  1 when score_text is Medium or High (decile 5 or more), the split
            ProPublica used in "Machine Bias" (2016)

score is a rank, not a probability. Calibration checks need a probability, so
treat this file as A1 (decisions) with an ordinal score, and say so.

Source: https://github.com/propublica/compas-analysis
Method: https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
"""

import hashlib
import io
import sys
import urllib.request

import pandas as pd

URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)
SHA256 = "c451db85908b2f7fef1d83203bedf6b71ecda0d5af468d82ae62178f91d0cc7d"
KEEP = ["id", "sex", "race", "age", "age_cat", "juv_fel_count", "juv_misd_count",
        "priors_count", "c_charge_degree", "decile_score", "score_text", "two_year_recid"]


def main(out="compas_with_predictions.csv"):
    raw = urllib.request.urlopen(URL, timeout=60).read()
    got = hashlib.sha256(raw).hexdigest()
    if got != SHA256:
        sys.exit(f"The origin file changed (sha256 {got}). Check it before trusting this script.")
    # The origin repeats two column names; keep the first of each.
    df = pd.read_csv(io.BytesIO(raw))
    df = df.loc[:, ~df.columns.duplicated()]
    # ProPublica's filters, from their analysis notebook.
    df = df[
        df["days_b_screening_arrest"].between(-30, 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ]
    df = df[KEEP].rename(columns={"two_year_recid": "y_true", "decile_score": "score"})
    df["y_pred"] = df["score_text"].isin(["Medium", "High"]).astype(int)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows (ProPublica reports 6,172)")


if __name__ == "__main__":
    main(*sys.argv[1:])
