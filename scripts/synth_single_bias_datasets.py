"""Generate small, single-bias-planted test datasets so the Pulse
bias-detection harness exercises the engine across many data shapes,
not only the omnibus 9-bias recruitment file.

Each variant:
  - same schema as the recruitment audit (so the wizard / handler treat
    it identically), but only one bias mechanism is planted.
  - ~500 rows (the audit-grade lower bound for confident reading).
  - the planted bias is stamped in ``_bias_flags`` so the harness can
    grade detection.

Variants written to ``datasets/synth/``:
  - clean.csv                no bias planted (control)
  - gender_penalty.csv       Female selection rate halved
  - race_penalty.csv         Black and Hispanic halved
  - age_cliff.csv            <25 set to 0%, 35-44 doubled
  - disability_penalty.csv   non-missing disability set to 1/5
  - zip_redlining.csv        minority-majority ZIP halved (no direct race bias)
  - surname_proxy.csv        last_name carries race signal; race column dropped
  - photo_laundering.csv     photo score correlates with race, drives decisions

The point: Pulse must detect each isolated mechanism on a small file
without the noise of the omnibus dataset, AND must correctly NOT raise
findings on the clean control. Run the harness once per file to grade
per-variant performance.

  python scripts/synth_single_bias_datasets.py [--out DIR]
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / ".." / "datasets" / "synth"
OUT_DIR = OUT_DIR.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


N_PER_VARIANT = 500
SEED = 7

GENDERS = ["Female", "Male", "Non-binary"]
GENDER_W = [0.49, 0.49, 0.02]
RACES = ["White", "Asian", "Black", "Hispanic", "Native American", "Middle Eastern"]
RACE_W = [0.55, 0.13, 0.10, 0.15, 0.02, 0.05]
ORIGINS = ["USA", "Mexico", "India", "China", "Iran", "Vietnam", "Egypt"]
ORIGIN_W = [0.85, 0.04, 0.03, 0.03, 0.015, 0.015, 0.015]
RELIGIONS = ["Christian", "Muslim", "Hindu", "Buddhist", "Jewish", "missing"]
RELIGION_W = [0.55, 0.10, 0.05, 0.03, 0.02, 0.25]
ROLES = [
    "Software Engineer",
    "Senior Software Engineer",
    "ML Engineer",
    "DevOps Engineer",
    "Data Scientist",
    "UX Designer",
    "Product Manager",
    "Marketing Manager",
    "Sales Representative",
    "Business Analyst",
]
DISABILITIES = ["missing", "Cognitive/Neurodivergent", "Visual", "Hearing", "Mobility"]
DIS_W = [0.78, 0.08, 0.05, 0.05, 0.04]
UNI_TIERS = ["Tier 1", "Tier 2", "Tier 3", "missing"]
UNI_W = [0.20, 0.40, 0.30, 0.10]


def _draw(n: int, vals, w, rng) -> np.ndarray:
    return rng.choice(vals, size=n, p=np.array(w) / sum(w))


def _base_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A neutral base population shared across variants. NO bias planted;
    selection rate is driven by skill_match_score alone so the controls
    read as fair."""
    df = pd.DataFrame(
        {
            "applicant_id": [f"A{i:05d}" for i in range(n)],
            "age": rng.integers(20, 63, size=n),
            "gender": _draw(n, GENDERS, GENDER_W, rng),
            "race_ethnicity": _draw(n, RACES, RACE_W, rng),
            "national_origin": _draw(n, ORIGINS, ORIGIN_W, rng),
            "religion": _draw(n, RELIGIONS, RELIGION_W, rng),
            "marital_status": rng.choice(
                ["Single", "Married", "Divorced", "missing"], size=n, p=[0.5, 0.35, 0.10, 0.05]
            ),
            "sexual_orientation": rng.choice(
                ["Heterosexual", "LGBTQ+", "Prefer not to say"], size=n, p=[0.86, 0.10, 0.04]
            ),
            "veteran_status": rng.choice(["No", "Yes"], size=n, p=[0.93, 0.07]),
            "disability_status": _draw(n, DISABILITIES, DIS_W, rng),
            "primary_language": rng.choice(
                ["English", "Spanish", "Vietnamese", "Mandarin", "Hindi", "Farsi"],
                size=n,
                p=[0.80, 0.08, 0.03, 0.04, 0.03, 0.02],
            ),
            "zip_minority_majority": rng.choice(["minority", "majority"], size=n, p=[0.30, 0.70]),
            "role_applied": rng.choice(ROLES, size=n),
            "years_experience": rng.integers(0, 25, size=n),
            "education_level": rng.choice(
                ["High School", "Associates", "Bachelor's", "Master's", "PhD"],
                size=n,
                p=[0.05, 0.10, 0.55, 0.25, 0.05],
            ),
            "university_tier": _draw(n, UNI_TIERS, UNI_W, rng),
            "skill_match_score": rng.uniform(0.0, 1.0, size=n),
            "photo_attractiveness_score": rng.uniform(0.0, 1.0, size=n),
        }
    )
    # Last name: 80% common Anglo / Asian / generic; 20% group-tied so a
    # downstream variant CAN plant a surname-proxy bias.
    df["last_name"] = rng.choice(
        [
            "Smith",
            "Johnson",
            "Williams",
            "Brown",
            "Garcia",
            "Patel",
            "Nguyen",
            "Kim",
            "Hernandez",
            "Rodriguez",
            "Lee",
            "Wang",
            "Khan",
            "Singh",
            "Chen",
            "Cohen",
            "Ahmed",
            "Okafor",
            "Diallo",
        ],
        size=n,
    )
    # Application date + simple PII fillers so the schema matches.
    df["first_name"] = rng.choice(
        ["Alex", "Sam", "Jordan", "Casey", "Taylor", "Riley", "Morgan", "Skyler"], size=n
    )
    df["application_date"] = "2026-04-01"
    df["source"] = rng.choice(["LinkedIn", "Referral", "Direct", "Indeed"], size=n)
    return df


def _base_invite(df: pd.DataFrame, rng: np.random.Generator, base_rate: float = 0.30) -> np.ndarray:
    """Selection driven by skill_match_score. Top-base_rate share of
    applicants by score are invited. Returns a 0/1 invite_decision."""
    cut = np.quantile(df["skill_match_score"].to_numpy(), 1 - base_rate)
    return (df["skill_match_score"].to_numpy() >= cut).astype(int)


def _write(df: pd.DataFrame, invite: np.ndarray, bias_flags: pd.Series, name: str) -> Path:
    df = df.copy()
    df["invite_decision"] = np.where(invite == 1, "Yes", "No")
    df["true_qualification_score"] = df["skill_match_score"]
    df["_bias_flags"] = bias_flags.values
    df["model_score"] = df["skill_match_score"] + np.random.default_rng(SEED).normal(
        0, 0.05, len(df)
    )
    out = OUT_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, positives={int((df.invite_decision == 'Yes').sum())})")
    return out


def variant_clean(rng):
    df = _base_frame(N_PER_VARIANT, rng)
    invite = _base_invite(df, rng)
    flags = pd.Series([""] * len(df), index=df.index)
    return df, invite, flags, "clean"


def variant_gender_penalty(rng):
    """Female applicants in tech roles get their hiring probability halved.
    Pulse must surface gender as adverse, with B1-style narrative."""
    df = _base_frame(N_PER_VARIANT, rng)
    invite = _base_invite(df, rng)
    tech = df["role_applied"].isin(
        [
            "Software Engineer",
            "Senior Software Engineer",
            "ML Engineer",
            "DevOps Engineer",
            "Data Scientist",
        ]
    )
    female_tech = (df["gender"] == "Female") & tech
    # Drop half the female-tech invites.
    flip = (invite == 1) & female_tech.to_numpy() & (rng.random(len(df)) < 0.5)
    invite = np.where(flip, 0, invite)
    flags = pd.Series(np.where(female_tech.to_numpy(), "B1", ""), index=df.index)
    return df, invite, flags, "gender_penalty"


def variant_race_penalty(rng):
    """Black and Hispanic applicants get their hiring rate halved across
    the board. Pulse must surface race_ethnicity as adverse."""
    df = _base_frame(N_PER_VARIANT, rng)
    invite = _base_invite(df, rng)
    mask = df["race_ethnicity"].isin(["Black", "Hispanic"])
    flip = (invite == 1) & mask.to_numpy() & (rng.random(len(df)) < 0.5)
    invite = np.where(flip, 0, invite)
    flags = pd.Series(np.where(mask.to_numpy(), "B2", ""), index=df.index)
    return df, invite, flags, "race_penalty"


def variant_age_cliff(rng):
    """Applicants under 25 get a zero hiring rate; 35-44 get a doubled rate
    (capped at base). Tests Pulse's age-axis detection."""
    df = _base_frame(N_PER_VARIANT, rng)
    invite = _base_invite(df, rng)
    young = (df["age"] < 25).to_numpy()
    invite = np.where(young, 0, invite)
    flags = pd.Series(np.where(young, "B3", ""), index=df.index)
    return df, invite, flags, "age_cliff"


def variant_disability_penalty(rng):
    """Applicants with any reported disability get a 1/5 hiring rate."""
    df = _base_frame(N_PER_VARIANT, rng)
    invite = _base_invite(df, rng)
    dis = (df["disability_status"] != "missing").to_numpy()
    flip = (invite == 1) & dis & (rng.random(len(df)) < 0.80)
    invite = np.where(flip, 0, invite)
    flags = pd.Series(np.where(dis, "B7", ""), index=df.index)
    return df, invite, flags, "disability_penalty"


def variant_zip_redlining(rng):
    """Minority-majority ZIP applicants get their hiring rate halved.
    Race is NOT directly biased; Pulse must surface zip as a proxy."""
    df = _base_frame(N_PER_VARIANT, rng)
    # First make the ZIP flag correlate with race so the linker has
    # something to find.
    minority_race = df["race_ethnicity"].isin(["Black", "Hispanic", "Native American"])
    df["zip_minority_majority"] = np.where(
        minority_race & (rng.random(len(df)) < 0.75),
        "minority",
        np.where(~minority_race & (rng.random(len(df)) < 0.18), "minority", "majority"),
    )
    invite = _base_invite(df, rng)
    minz = (df["zip_minority_majority"] == "minority").to_numpy()
    flip = (invite == 1) & minz & (rng.random(len(df)) < 0.5)
    invite = np.where(flip, 0, invite)
    flags = pd.Series(np.where(minz, "B4", ""), index=df.index)
    return df, invite, flags, "zip_redlining"


def variant_photo_laundering(rng):
    """Photo attractiveness score correlates with race and drives decisions.
    Tests proxy detection on a continuous feature."""
    df = _base_frame(N_PER_VARIANT, rng)
    # Make photo score correlate with race.
    biased = df["race_ethnicity"].isin(["Black", "Hispanic", "Native American"])
    df.loc[biased, "photo_attractiveness_score"] = (
        df.loc[biased, "photo_attractiveness_score"] * 0.5
    )
    # Selection driven by both skill and photo.
    combined = df["skill_match_score"] * 0.5 + df["photo_attractiveness_score"] * 0.5
    cut = np.quantile(combined.to_numpy(), 0.70)
    invite = (combined.to_numpy() >= cut).astype(int)
    flags = pd.Series(np.where(biased.to_numpy(), "B9", ""), index=df.index)
    return df, invite, flags, "photo_laundering"


SURNAMES_BY_RACE = {
    "Black": ["Okafor", "Diallo", "Williams", "Johnson"],
    "Hispanic": ["Garcia", "Hernandez", "Rodriguez"],
    "Asian": ["Nguyen", "Kim", "Wang", "Chen", "Lee"],
    "Middle Eastern": ["Khan", "Ahmed"],
}
PENALISED_SURNAMES = ["Okafor", "Diallo", "Garcia", "Hernandez", "Rodriguez"]


def variant_surname_proxy(rng):
    """last_name carries the race signal and drives the decision; the
    race_ethnicity column is DROPPED, so no direct race comparison is
    possible. Tests whether a proxy is found when the protected attribute
    itself is absent.

    Added 2026-10-01. The module docstring listed this variant from the
    start, but no builder existed, so the file was never written."""
    df = _base_frame(N_PER_VARIANT, rng)
    # 80% of applicants in a group carry a group-typical surname; the rest
    # keep the neutral draw from _base_frame.
    for race, names in SURNAMES_BY_RACE.items():
        mask = (df["race_ethnicity"] == race) & (rng.random(len(df)) < 0.80)
        df.loc[mask, "last_name"] = rng.choice(names, size=int(mask.sum()))
    invite = _base_invite(df, rng)
    penalised = df["last_name"].isin(PENALISED_SURNAMES).to_numpy()
    flip = (invite == 1) & penalised & (rng.random(len(df)) < 0.5)
    invite = np.where(flip, 0, invite)
    flags = pd.Series(np.where(penalised, "B5", ""), index=df.index)
    df = df.drop(columns=["race_ethnicity"])
    return df, invite, flags, "surname_proxy"


def main():
    # --out writes the variants somewhere other than datasets/synth (the docs
    # test-object pack uses it, 2026-10-01). Default behaviour is unchanged.
    import argparse

    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    OUT_DIR = ap.parse_args().out.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # READINESS-6, 2026-09-10. A main-level `rng = np.random.default_rng(SEED)`
    # sat here unused. Each builder is given its OWN seeded generator a few
    # lines down, exactly as the comment there says, so this one fed nothing.
    # It is removed rather than wired in: wiring it would change every dataset
    # this script emits, and the per-variant seeding is the reproducible design.
    for builder in [
        variant_clean,
        variant_gender_penalty,
        variant_race_penalty,
        variant_age_cliff,
        variant_disability_penalty,
        variant_zip_redlining,
        variant_surname_proxy,
        variant_photo_laundering,
    ]:
        # Each variant gets its own seeded RNG so re-runs are reproducible.
        #
        # READINESS-6, 2026-09-10. The sentence above was FALSE, and it was false
        # by exactly one function call. This read
        # `SEED + hash(builder.__name__) % 1000`, and Python SALTS str hashing
        # per process unless PYTHONHASHSEED is set. Measured over three
        # interpreters, the same three variant names produced seeds
        # [351, 196, 340], then [394, 990, 619], then [45, 560, 155]. So every
        # run of this script emitted DIFFERENT data under the same filenames,
        # in a library whose subject is auditable measurement.
        #
        # sha256 of the name is stable across processes, interpreters and
        # platforms, which is what `hash` was reached for and does not provide.
        #
        # NOTE ON THE COMMITTED CSVs: the files in datasets/synth were generated
        # before this fix, under a salt that no longer exists, so they cannot be
        # reproduced byte-for-byte by any version of this script. Re-running now
        # will overwrite them with data that IS reproducible from here on. That
        # is a deliberate one-way step and is left to whoever decides to take
        # it; nothing in src, tests, notebooks or docs reads these files.
        name_digest = hashlib.sha256(builder.__name__.encode("utf-8")).digest()
        sub_rng = np.random.default_rng(SEED + int.from_bytes(name_digest[:4], "big") % 1000)
        df, invite, flags, name = builder(sub_rng)
        _write(df, invite, flags, name)
    print(f"\n8 variants written to {OUT_DIR}")


if __name__ == "__main__":
    main()
