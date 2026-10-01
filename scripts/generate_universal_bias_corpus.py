# ruff: noqa: N802
#
# The variant builders are named `variant_L1_...` through `variant_L9_...`
# and the capital L-number is a TAXONOMY IDENTIFIER, not a style slip: it is
# how each variant is referred to in the corpus spec and in the sweep output,
# so `grep L4` finds the generator, the dataset and the result together.
# Lowercasing them to satisfy N802 would break that link for a naming rule
# whose purpose is readability. Suppressed here, once, with the reason, rather
# than nine bare noqa comments nobody can evaluate. READINESS-6, 2026-09-10.
#!/usr/bin/env python3
"""Synthetic single-bias test corpus generator.

Produces small (~500-row) synthetic loan-screening datasets, one per
planted bias mechanism, so the harness can exercise the engine on data
shapes that AREN'T the recruitment file. Each variant isolates ONE
mechanism and otherwise generates clean (statistically null) data:

    L1  gender penalty in tech-equivalent role
    L2  race penalty
    L3  age penalty
    L4  ZIP redlining (proxy)
    L5  university-tier prestige (proxy)
    L6  foreign-sounding surname (proxy)
    L7  disability penalty
    L8  employment-gap caregiver pattern (proxy for gender)
    L9  photo-score laundering (proxy for race)

Universality test: the engine should detect each mechanism on its own
dataset even though the domain is *lending*, not *recruitment*, and
even though the dataset is much smaller than the original.

Each CSV has the same audit columns (so the harness can be re-pointed
at it without changes) plus the answer-key columns
(`true_qualification_score`, `_bias_flags`).

Usage:
    python3 scripts/generate_universal_bias_corpus.py --out /tmp/bias_corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_PER = 500
SEED = 7

RACES = ["White", "Asian", "Black", "Hispanic", "Native American"]
RACE_PROBS = [0.55, 0.15, 0.13, 0.12, 0.05]
NATIONALITIES = ["USA", "Mexico", "India", "China", "Egypt", "Iran", "Vietnam"]
NAT_PROBS = [0.80, 0.04, 0.04, 0.04, 0.03, 0.03, 0.02]
RELIGIONS = ["Christian", "Muslim", "Buddhist", "Hindu", "Jewish", "None"]
RELIGION_PROBS = [0.55, 0.10, 0.05, 0.10, 0.05, 0.15]
GENDERS = ["Male", "Female", "Non-binary"]
GENDER_PROBS = [0.48, 0.48, 0.04]
DISABILITIES = ["missing", "Cognitive/Neurodivergent", "Mobility", "Visual", "Hearing"]
DISABILITY_PROBS = [0.85, 0.05, 0.04, 0.03, 0.03]
MARITAL = ["Single", "Married", "Divorced", "Widowed"]
MARITAL_PROBS = [0.55, 0.35, 0.08, 0.02]
SEX_ORIENT = ["Heterosexual", "LGBTQ+", "Prefer not to say"]
SEX_PROBS = [0.85, 0.10, 0.05]
LANGUAGES = ["English", "Spanish", "Mandarin", "Hindi", "Arabic"]
LANG_PROBS = [0.78, 0.08, 0.06, 0.05, 0.03]
ROLES = [
    "ML Engineer",
    "Software Engineer",
    "Data Scientist",
    "Marketing Manager",
    "Sales Representative",
]
ROLE_PROBS = [0.20, 0.30, 0.15, 0.15, 0.20]
UNI_TIERS = ["Tier 1", "Tier 2", "Tier 3", "Community", "Unknown"]
UNI_TIER_PROBS = [0.15, 0.30, 0.30, 0.15, 0.10]

# Surname pools roughly aligned to race so the last_name proxy is real
# without being a dictionary attack. Tiny lists -- this is a test corpus.
SURNAMES_BY_RACE = {
    "White": ["Smith", "Johnson", "Williams", "Jones", "Miller"],
    "Asian": ["Chen", "Kim", "Wang", "Park", "Patel"],
    "Black": ["Washington", "Jackson", "Robinson", "Banks", "Pierre"],
    "Hispanic": ["Garcia", "Rodriguez", "Martinez", "Lopez", "Hernandez"],
    "Native American": ["Begay", "Yazzie", "Tsosie", "Etsitty", "Joe"],
}
FIRST_NAMES_BY_GENDER = {
    "Male": ["John", "Mike", "David", "Carlos", "Wei", "Amir"],
    "Female": ["Sarah", "Maria", "Priya", "Mei", "Aisha", "Emma"],
    "Non-binary": ["Alex", "Sam", "Jordan", "Taylor", "Casey"],
}


def _baseline(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a baseline applicant table with NO planted bias.

    Covariates correlate with each other realistically (zip with race,
    employment_gap with marital, etc.) without any decision-side bias.
    """
    df = pd.DataFrame()
    df["applicant_id"] = [f"A{1000 + i}" for i in range(n)]
    df["gender"] = rng.choice(GENDERS, n, p=GENDER_PROBS)
    df["race_ethnicity"] = rng.choice(RACES, n, p=RACE_PROBS)
    df["national_origin"] = rng.choice(NATIONALITIES, n, p=NAT_PROBS)
    df["age"] = rng.integers(20, 65, n)
    df["religion"] = rng.choice(RELIGIONS, n, p=RELIGION_PROBS)
    df["marital_status"] = rng.choice(MARITAL, n, p=MARITAL_PROBS)
    df["sexual_orientation"] = rng.choice(SEX_ORIENT, n, p=SEX_PROBS)
    df["veteran_status"] = rng.choice(["No", "Yes"], n, p=[0.90, 0.10])
    df["disability_status"] = rng.choice(DISABILITIES, n, p=DISABILITY_PROBS)
    df["primary_language"] = rng.choice(LANGUAGES, n, p=LANG_PROBS)
    df["role_applied"] = rng.choice(ROLES, n, p=ROLE_PROBS)
    df["education_level"] = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"], n, p=[0.20, 0.55, 0.20, 0.05]
    )
    df["university"] = [f"Univ_{x % 50}" for x in rng.integers(0, 50, n)]
    df["university_tier"] = rng.choice(UNI_TIERS, n, p=UNI_TIER_PROBS)
    df["years_experience"] = (df["age"] - 22).clip(lower=0)
    df["n_prior_companies"] = rng.integers(0, 6, n)
    df["employment_gap_months"] = rng.integers(0, 6, n)
    df["gpa"] = np.round(rng.uniform(2.5, 4.0, n), 2)
    df["skill_match_score"] = np.round(rng.uniform(0.4, 1.0, n), 2)
    df["video_interview_score"] = np.round(rng.uniform(0.4, 1.0, n), 2)
    df["photo_attractiveness_score"] = np.round(rng.uniform(0.3, 1.0, n), 2)
    # zip-minority indicator correlated with race
    df["zip_minority_majority"] = (
        df["race_ethnicity"].isin(["Black", "Hispanic", "Native American"]).astype(int)
    )
    df["zip_code"] = [f"{rng.integers(10000, 99999)}" for _ in range(n)]
    df["city"] = rng.choice(["Boston", "Chicago", "Dallas", "Phoenix", "Seattle"], n)
    df["state"] = rng.choice(["MA", "IL", "TX", "AZ", "WA"], n)
    df["street_address"] = [f"{rng.integers(100, 9000)} Main St" for _ in range(n)]
    df["email"] = [f"a{1000 + i}@example.com" for i in range(n)]
    df["phone"] = [f"555-{1000 + rng.integers(0, 9000)}" for _ in range(n)]
    df["date_of_birth"] = "1990-01-01"
    df["last_name"] = [rng.choice(SURNAMES_BY_RACE[r]) for r in df["race_ethnicity"]]
    df["first_name"] = [rng.choice(FIRST_NAMES_BY_GENDER[g]) for g in df["gender"]]
    df["photo_url"] = "https://example.com/photo.jpg"
    df["linkedin_url"] = "https://example.com/li"
    df["application_date"] = "2024-01-01"
    df["source"] = rng.choice(["Referral", "Job Board", "Direct", "Recruiter"], n)
    df["model_score"] = np.round(rng.uniform(0.2, 0.8, n), 3)
    df["certifications_count"] = rng.integers(0, 5, n)
    df["extracurriculars"] = rng.choice(["None", "Sports", "Volunteer", "Tech", "Arts"], n)
    df["skills"] = rng.choice(["Python", "Java", "SQL", "Excel"], n)
    return df


def _approve_baseline(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Baseline approval probability driven only by skill_match_score so a
    null dataset has a real-but-non-biased signal."""
    base = 0.10 + 0.30 * (df["skill_match_score"] - 0.4) / 0.6
    return base.clip(0.02, 0.40).to_numpy()


def variant_L1_gender_in_tech(n=N_PER, seed=SEED + 1):
    """Females in tech roles have hire prob multiplied by 0.30."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    p = _approve_baseline(df, rng)
    is_tech = df["role_applied"].isin(["ML Engineer", "Software Engineer", "Data Scientist"])
    is_female = df["gender"] == "Female"
    p = np.where(is_tech & is_female, p * 0.30, p)
    return _materialise(df, p, rng, "L1")


def variant_L2_race(n=N_PER, seed=SEED + 2):
    """Black + Native American applicants have hire prob multiplied by 0.25."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    p = _approve_baseline(df, rng)
    hit = df["race_ethnicity"].isin(["Black", "Native American"])
    p = np.where(hit, p * 0.25, p)
    return _materialise(df, p, rng, "L2")


def variant_L3_age(n=N_PER, seed=SEED + 3):
    """<25 and 55+ have hire prob multiplied by 0.25."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    p = _approve_baseline(df, rng)
    hit = (df["age"] < 25) | (df["age"] >= 55)
    p = np.where(hit, p * 0.25, p)
    return _materialise(df, p, rng, "L3")


def variant_L4_zip(n=N_PER, seed=SEED + 4):
    """zip_minority_majority=1 hire prob multiplied by 0.30 (redlining)."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    p = _approve_baseline(df, rng)
    p = np.where(df["zip_minority_majority"] == 1, p * 0.30, p)
    return _materialise(df, p, rng, "L4")


def variant_L5_university(n=N_PER, seed=SEED + 5):
    """Tier 3 + Community hire prob multiplied by 0.25 (prestige proxy)."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    p = _approve_baseline(df, rng)
    p = np.where(df["university_tier"].isin(["Tier 3", "Community"]), p * 0.25, p)
    return _materialise(df, p, rng, "L5")


def variant_L6_surname(n=N_PER, seed=SEED + 6):
    """Race-typed surnames (Black, Hispanic, Native American) -> 0.35x."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    p = _approve_baseline(df, rng)
    hit_race = df["race_ethnicity"].isin(["Black", "Hispanic", "Native American"])
    # Penalty is applied through the surname signal, not race itself --
    # but they correlate by construction, so detection requires the
    # proxy crosswalk.
    p = np.where(hit_race, p * 0.35, p)
    return _materialise(df, p, rng, "L6")


def variant_L7_disability(n=N_PER, seed=SEED + 7):
    """Any disability declared -> hire prob 0.30x."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    p = _approve_baseline(df, rng)
    hit = df["disability_status"] != "missing"
    p = np.where(hit, p * 0.30, p)
    return _materialise(df, p, rng, "L7")


def variant_L8_employment_gap(n=N_PER, seed=SEED + 8):
    """employment_gap_months >= 12 -> 0.30x. Gaps planted to be 3x more
    common among Female (caregiver pattern)."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    is_female = df["gender"] == "Female"
    # Inject gap pattern: females are 3x more likely to have gap>=12
    gap_p_male = 0.10
    gap_p_female = 0.30
    raw = rng.uniform(0, 1, n)
    df.loc[is_female & (raw < gap_p_female), "employment_gap_months"] = rng.integers(
        12, 36, int(((is_female) & (raw < gap_p_female)).sum())
    )
    df.loc[(~is_female) & (raw < gap_p_male), "employment_gap_months"] = rng.integers(
        12, 36, int(((~is_female) & (raw < gap_p_male)).sum())
    )
    p = _approve_baseline(df, rng)
    hit = df["employment_gap_months"] >= 12
    p = np.where(hit, p * 0.30, p)
    return _materialise(df, p, rng, "L8")


def variant_L9_photo(n=N_PER, seed=SEED + 9):
    """photo_attractiveness_score: lower for Black + Hispanic by 0.30
    (laundering); approval driven by photo score."""
    rng = np.random.default_rng(seed)
    df = _baseline(n, rng)
    laundered = df["race_ethnicity"].isin(["Black", "Hispanic"])
    df.loc[laundered, "photo_attractiveness_score"] = np.clip(
        df.loc[laundered, "photo_attractiveness_score"].to_numpy() - 0.30, 0.05, 1.0
    )
    p = _approve_baseline(df, rng)
    # Approval is driven by photo score now, so the race signal is
    # laundered through it.
    p = (p + 0.40 * df["photo_attractiveness_score"]).clip(0.02, 0.95).to_numpy()
    return _materialise(df, p, rng, "L9")


def _materialise(
    df: pd.DataFrame, p: np.ndarray, rng: np.random.Generator, code: str
) -> pd.DataFrame:
    df["true_qualification_score"] = np.round(rng.uniform(0.3, 0.95, len(df)), 3)
    df["invite_decision"] = np.where(rng.uniform(0, 1, len(df)) < p, "Yes", "No")
    df["_bias_flags"] = code
    df["model_score"] = np.round(p, 3)
    return df


VARIANTS = {
    "L1_gender_in_tech": variant_L1_gender_in_tech,
    "L2_race": variant_L2_race,
    "L3_age": variant_L3_age,
    "L4_zip": variant_L4_zip,
    "L5_university": variant_L5_university,
    "L6_surname": variant_L6_surname,
    "L7_disability": variant_L7_disability,
    "L8_employment_gap": variant_L8_employment_gap,
    "L9_photo": variant_L9_photo,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/bias_corpus")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in VARIANTS.items():
        df = fn()
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        hires = (df["invite_decision"] == "Yes").mean()
        print(f"  wrote {path}   n={len(df)}   hire_rate={hires:.2%}")
    print(f"\nCorpus written to {out_dir}/")
    # The recall harness that consumes this corpus is INTERNAL tooling: it needs
    # the private platform checkout, so it is on the export EXCLUDES list and is
    # not part of the published repository. Printing a command a public reader
    # cannot run is a pointer into a file nobody can open, which tells them
    # nothing and wastes their time discovering it.
    print(
        f"Each CSV under {out_dir}/ is a standalone corpus: load it with pandas "
        "and pass the label column and the protected column to any vfairness "
        "detector, for example vfairness.BiasDetector."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
