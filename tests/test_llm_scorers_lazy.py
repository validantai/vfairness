"""VF-4 regression tests: the optional LLM scorer backends load lazily.

The finding: vfairness/llm/scorers.py imported transformers, flair, detoxify,
vaderSentiment and profanity_check at module scope inside try/except ImportError.
The first three pull torch; profanity_check deserialises an sklearn SVM. So any
environment that had the `llm` / `llm-transformers` extras installed paid the
whole cost on `import vfairness`, and an optional dependency became effectively
required at import time rather than at use time.

What is pinned here:

1. Importing the module locates those backends but does NOT execute them, so
   they stay out of sys.modules until a scorer that needs one actually runs.
2. The scoring path still works when the backends ARE present, and the ladder
   still selects the same production scorer it used to.
3. A backend that is installed but unimportable raises a clear, actionable
   ImportError naming the install command. It must never degrade quietly to a
   keyword placeholder.
4. CONTROLS: on a bare install (no optional backends at all) the ladder, every
   default scorer's numeric output, and the PlaceholderScorerWarning behaviour
   are unchanged.

The subprocess arms load scorers.py directly from its file rather than via
`import vfairness.llm.scorers`, because the vfairness.llm package imports torch
for unrelated reasons (embedding bias) and that would mask exactly the
sys.modules assertions these tests make.
"""

import json
import math
import os
import subprocess
import sys
import warnings

import pytest

import vfairness.llm.scorers as scorers_mod
from vfairness.llm.scorers import (
    KeywordRegardScorer,
    KeywordSentimentScorer,
    KeywordToxicityScorer,
    PlaceholderScorerWarning,
    UnscorableTextWarning,
)

REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
SCORERS_PATH = os.path.join(REPO_SRC, "vfairness", "llm", "scorers.py")

# Stub backends. Each stands in for a real optional dependency: same import
# path, same call shape, deterministic output. They are deliberately CHEAP, so
# these tests measure *whether* the module is executed, not how long it takes.
_STUBS = {
    "transformers/__init__.py": '''
"""Stub standing in for transformers."""
IMPORTED = True


class _FakePipe:
    def __call__(self, text):
        return [[
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.1},
            {"label": "neutral", "score": 0.1},
        ]]


def pipeline(*args, **kwargs):
    return _FakePipe()
''',
    "flair/__init__.py": "",
    "flair/data.py": """
class _Label:
    def __init__(self, value, score):
        self.value = value
        self.score = score


class Sentence:
    def __init__(self, text):
        self.text = text
        self.labels = [_Label("POSITIVE", 0.9)]
""",
    "flair/models.py": """
class TextClassifier:
    @staticmethod
    def load(model):
        return TextClassifier()

    def predict(self, sentence):
        return None
""",
    "detoxify/__init__.py": """
class Detoxify:
    def __init__(self, model_type="unbiased"):
        self.model_type = model_type

    def predict(self, text):
        if isinstance(text, list):
            return {"toxicity": [0.25] * len(text)}
        return {"toxicity": 0.25}
""",
    "vaderSentiment/__init__.py": "",
    "vaderSentiment/vaderSentiment.py": """
class SentimentIntensityAnalyzer:
    def polarity_scores(self, text):
        return {"compound": 0.5}
""",
    "profanity_check/__init__.py": """
def predict_prob(texts):
    return [0.125] * len(texts)
""",
}


def _write_stubs(root, names):
    """Materialise the named stub packages under `root`; return root as str."""
    for rel, body in _STUBS.items():
        pkg = rel.split("/")[0]
        if pkg not in names:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return str(root)


_RUNNER_PREAMBLE = f"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("_vf_scorers_probe", {SCORERS_PATH!r})
m = importlib.util.module_from_spec(spec)
sys.modules["_vf_scorers_probe"] = m
spec.loader.exec_module(m)


def loaded(name):
    return name in sys.modules


def emit(d):
    print("VF4_RESULT " + json.dumps(d))
"""


def _run(body, stub_dir=None, extra_args=()):
    """Execute `body` in a fresh interpreter after loading scorers.py.

    Returns the dict the body passed to emit().
    """
    env = dict(os.environ)
    env.pop("VFAIRNESS_SIDECAR_PYTHON", None)
    env.pop("VFAIRNESS_SIDECAR_SCRIPT", None)
    # Only the stub dir (if any) goes on the path. REPO_SRC is deliberately
    # absent: the module is loaded by file path, so nothing else is needed and
    # an installed vfairness cannot shadow the file under test.
    if stub_dir:
        env["PYTHONPATH"] = stub_dir
    else:
        env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, *extra_args, "-c", _RUNNER_PREAMBLE + body],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("VF4_RESULT ")]
    assert lines, f"probe produced no result:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(lines[-1][len("VF4_RESULT ") :])


# ===========================================================================
# 1. The pin: importing the module must not execute the optional backends.
# ===========================================================================


class TestImportDoesNotExecuteBackends:
    def test_backends_are_detected_but_not_imported(self, tmp_path):
        stub_dir = _write_stubs(
            tmp_path,
            {"transformers", "flair", "detoxify", "vaderSentiment", "profanity_check"},
        )
        r = _run(
            """
emit({
    "flags": {
        "transformers": m._HF_PIPELINE_AVAILABLE,
        "flair": m._FLAIR_AVAILABLE,
        "detoxify": m._DETOXIFY_AVAILABLE,
        "vader": m._VADER_AVAILABLE,
        "profanity": m._ALT_PROFANITY_AVAILABLE,
    },
    "loaded": {n: loaded(n) for n in
               ("transformers", "flair", "detoxify",
                "vaderSentiment", "profanity_check")},
})
""",
            stub_dir,
        )
        # Detection still works: every backend is reported available.
        assert r["flags"] == {
            "transformers": True,
            "flair": True,
            "detoxify": True,
            "vader": True,
            "profanity": True,
        }, "availability detection regressed"
        # But nothing was executed.
        assert r["loaded"] == {
            "transformers": False,
            "flair": False,
            "detoxify": False,
            "vaderSentiment": False,
            "profanity_check": False,
        }, "an optional backend was imported at module scope"

    def test_default_scorers_are_still_the_production_rungs(self, tmp_path):
        """Laziness must not have demoted the ladder to placeholders."""
        stub_dir = _write_stubs(
            tmp_path,
            {"transformers", "flair", "detoxify", "vaderSentiment", "profanity_check"},
        )
        r = _run(
            """
emit({
    "sentiment": type(m.DEFAULT_SENTIMENT_SCORER).__name__,
    "toxicity": type(m.DEFAULT_TOXICITY_SCORER).__name__,
    "regard": type(m.DEFAULT_REGARD_SCORER).__name__,
})
""",
            stub_dir,
        )
        assert r == {
            "sentiment": "TransformerSentimentScorer",
            "toxicity": "DetoxifyScorer",
            "regard": "TransformerRegardScorer",
        }


# ===========================================================================
# 2. The scoring path still works when the backends ARE present.
# ===========================================================================


class TestScoringPathWithBackendsPresent:
    def test_transformer_rungs_score_and_import_on_first_use(self, tmp_path):
        stub_dir = _write_stubs(tmp_path, {"transformers", "flair", "detoxify"})
        r = _run(
            """
before = {n: loaded(n) for n in ("transformers", "flair", "detoxify")}
sent = m.DEFAULT_SENTIMENT_SCORER.score("a perfectly ordinary sentence")
tox = m.DEFAULT_TOXICITY_SCORER.score("a perfectly ordinary sentence")
tox_batch = list(m.DEFAULT_TOXICITY_SCORER.score_batch(["one", "two"]))
reg = m.DEFAULT_REGARD_SCORER.score("they are respected leaders")
after = {n: loaded(n) for n in ("transformers", "flair", "detoxify")}
emit({"before": before, "after": after, "sent": sent, "tox": tox,
      "tox_batch": tox_batch, "reg": reg})
""",
            stub_dir,
        )
        assert r["before"] == {"transformers": False, "flair": False, "detoxify": False}
        assert r["after"] == {"transformers": True, "flair": True, "detoxify": True}
        # Flair stub always labels POSITIVE at 0.9 -> polarity +0.9.
        assert r["sent"] == pytest.approx(0.9)
        # Detoxify stub returns toxicity 0.25.
        assert r["tox"] == pytest.approx(0.25)
        assert r["tox_batch"] == pytest.approx([0.25, 0.25])
        # regardv3 stub: positive 0.8 minus negative 0.1.
        assert r["reg"] == pytest.approx(0.7)

    def test_vader_and_profanity_rungs_score_and_import_on_first_use(self, tmp_path):
        stub_dir = _write_stubs(tmp_path, {"vaderSentiment", "profanity_check"})
        r = _run(
            """
before = {n: loaded(n) for n in ("vaderSentiment", "profanity_check")}
kinds = (type(m.DEFAULT_SENTIMENT_SCORER).__name__,
         type(m.DEFAULT_TOXICITY_SCORER).__name__)
sent = m.DEFAULT_SENTIMENT_SCORER.score("whatever")
tox = m.DEFAULT_TOXICITY_SCORER.score("whatever")
tox_batch = list(m.DEFAULT_TOXICITY_SCORER.score_batch(["a", "b", "c"]))
after = {n: loaded(n) for n in ("vaderSentiment", "profanity_check")}
emit({"before": before, "after": after, "kinds": list(kinds),
      "sent": sent, "tox": tox, "tox_batch": tox_batch})
""",
            stub_dir,
        )
        assert r["before"] == {"vaderSentiment": False, "profanity_check": False}
        assert r["after"] == {"vaderSentiment": True, "profanity_check": True}
        assert r["kinds"] == ["VADERSentimentScorer", "AltProfanityCheckScorer"]
        assert r["sent"] == pytest.approx(0.5)
        assert r["tox"] == pytest.approx(0.125)
        assert r["tox_batch"] == pytest.approx([0.125, 0.125, 0.125])


# ===========================================================================
# 3. A broken backend must be reported, never quietly downgraded.
# ===========================================================================


class TestBrokenBackendFailsLoud:
    def test_unimportable_transformers_raises_actionable_error(self, tmp_path):
        """transformers present on the path but unimportable (e.g. no torch).

        Before VF-4 the module-scope try/except swallowed this, the ladder
        silently dropped to KeywordRegardScorer, and the caller got a
        keyword number where they had asked for sasha/regardv3.
        """
        (tmp_path / "transformers").mkdir(parents=True, exist_ok=True)
        (tmp_path / "transformers" / "__init__.py").write_text(
            "raise ImportError('No module named torch')\n"
        )
        r = _run(
            """
kind = type(m.DEFAULT_REGARD_SCORER).__name__
try:
    m.DEFAULT_REGARD_SCORER.score("they are respected leaders")
    outcome, msg = "returned", ""
except ImportError as exc:
    outcome, msg = "ImportError", str(exc)
emit({"kind": kind, "outcome": outcome, "msg": msg})
""",
            str(tmp_path),
        )
        assert r["kind"] == "TransformerRegardScorer"
        assert r["outcome"] == "ImportError", (
            "an unimportable backend degraded quietly instead of reporting"
        )
        # The message has to be actionable: what is missing and how to fix it.
        assert "transformers" in r["msg"]
        assert "pip install" in r["msg"]
        assert "llm-transformers" in r["msg"]


# ===========================================================================
# 4. CONTROLS. Ordinary healthy usage must be numerically unchanged.
# ===========================================================================

# Fixed corpus, fixed expected values. These are the numbers the default
# scorers produced at 7c5fff8, read off the pre-change file, before the
# laziness work. Any drift here means the fix altered results, not just
# import timing.
_CONTROL_CORPUS = [
    "She is a wonderful and talented engineer with an outstanding record.",
    "Those people are lazy criminals and they always cause trouble.",
    "I cannot help with that request, as an AI I must decline.",
    "You are a stupid worthless idiot and a disgusting loser.",
    "",
]

_CONTROL_EXPECTED = {
    # Beta Go-Live Stage 2, 2026-09-17. Three metrics moved, and every move is a
    # refusal replacing a number that was never measured. The control's SUBJECT
    # is unchanged: these numbers still may not drift silently, and every NaN
    # below has to be a refusal the scorer explains, which the assertions after
    # this table check per position.
    #
    # framing: index 0 moved 0.8 -> 1.0 and indexes 1..3 moved 0.575 -> NaN.
    # The score is a weighted blend of a hedging, a certainty and an evaluative
    # ratio. A text carrying NONE of those markers used to come back 0.575 for
    # ANY text whatsoever, which is why three different sentences shared it; the
    # scorer had read nothing in any of them. Index 0 does carry evaluative
    # markers, and the blend now renormalises over the dimensions it actually
    # read instead of averaging in a neutral 0.5 for the ones it did not, which
    # is what moves a wholly positive sentence from 0.8 to 1.0.
    #
    # framing index 4 moved 0.5 -> NaN on 2026-09-25, and it is the same move as
    # indexes 1..3 arriving one branch earlier. Index 4 is the EMPTY string, and
    # the scorer returned 0.5 for it, the exact midpoint of the [0, 1] scale, while
    # refusing a real sentence that carried no framing marker. The carve-out
    # justified itself "on the same ground as KeywordSentimentScorer's empty-text
    # carve-out", and that carve-out had been withdrawn on 2026-09-17 by the
    # argument that a midpoint is a measured value and not an absence: the
    # cross-reference outlived the thing it pointed at. Two groups of empty
    # generations reported 0.5 against 0.5, delta 0.0, assessed=True.
    "framing": [1.0, float("nan"), float("nan"), float("nan"), float("nan")],
    "helpfulness": [0.31, 0.31, 0.31, 0.385, 0.0],
    "information_quality": [0.45, 0.45, 0.25, 0.45, 0.0],
    "refusal": [0.0, 0.0, 1.0, 0.0, 0.0],
    # regard: indexes 2..4 moved 0.0 -> NaN. NEGATIVE_REGARD is a 30-word
    # social-standing lexicon (criminal, lazy, dropout, illiterate), not a
    # general insult list, so "I cannot help with that request" and even "You
    # are a stupid worthless idiot" carry no regard evidence at all: they are
    # abusive, which the toxicity scorer reads, but they say nothing about how a
    # group is portrayed. The old 0.0 asserted MEASURED NEUTRAL STANDING for
    # three texts the scorer never read, and a 0.0 on this scale is the exact
    # midpoint between the two poles. Index 4 is the empty string and refuses
    # for the same reason rather than keeping 0.0: a blank generation is not a
    # neutral one, and scoring it neutral gave a group that produced nothing a
    # significant regard ADVANTAGE over one that did.
    "regard": [1.0, -1.0, float("nan"), float("nan"), float("nan")],
    "representation": [0.367, 0.367, 0.0, 0.0, 0.0],
    # semantic_quality: indexes 0..3 moved to NaN. The scorer measures the
    # quality of ADVICE: its markers are recommendation verbs, specificity
    # connectives and institution-tier words. None of the four sentences here is
    # advice, so it read no evidence in any of them, and the number it returned
    # was WORD COUNT wearing a quality label. Measured on the old code,
    # score("zzzz qqqq") was 0.133, of which 0.125 (94%) was the length term.
    # Index 4 is the empty string and keeps a real 0.0: an empty answer is
    # genuinely of no quality, which is a measurement rather than an absence.
    "semantic_quality": [float("nan"), float("nan"), float("nan"), float("nan"), 0.0],
    # Index 2 MOVED, 0.0 -> NaN, on 2026-09-10, and not by the laziness work
    # this control was built to guard. "I cannot help with that request, as an
    # AI I must decline." contains none of KeywordSentimentScorer's 30 lexicon
    # words, so the scorer read no evidence in it and used to report the
    # neutral 0.0 anyway. It now refuses, loudly, per text. The refusal scorer
    # one row up reads the same sentence correctly as 1.0, which is what makes
    # the old 0.0 so easy to believe: every OTHER metric had an answer for it.
    # Reasoning in full on KeywordSentimentScorer. Index 4, the empty string,
    # MOVED 0.0 -> NaN on 2026-09-17: the carve-out that kept it neutral was
    # closed. On a signed scale 0.0 is measured NEUTRAL sentiment, the exact
    # midpoint, and it beats every negative score. Measured at the public entry
    # before the change, eight whitespace-only generations against eight
    # denigrating sentences came back delta=1.0, p=0.000138,
    # is_significant=True, assessed=True: a significant sentiment ADVANTAGE for
    # the group that generated nothing at all. The regard sibling had already
    # closed exactly this and named this scorer as the shape it copied.
    "sentiment": [1.0, -1.0, float("nan"), -1.0, float("nan")],
    "stereotype": [0.0, 1.0, 0.0, 0.0, 0.0],
    # Index 3 MOVED, 0.4 -> 0.5, on 2026-09-10, and it is a correction rather
    # than drift. "You are a stupid worthless idiot and a disgusting loser."
    # holds FIVE of the 25 toxic words, but "loser." ends the sentence, and the
    # whitespace-only tokeniser this scorer used could not see it while still
    # counting it in the denominator. 4/10 was the arithmetic of a numerator and
    # a denominator tokenised differently. See _lexicon_tokens.
    "toxicity": [0.0, 0.0, 0.0, 0.5, 0.0],
}


class TestBareInstallUnchanged:
    """The over-correction control: with no optional backend installed at all,
    nothing about the ladder or its numbers may move."""

    def test_ladder_selection_unchanged(self, tmp_path):
        r = _run(
            """
emit({
    "flags": [m._HF_PIPELINE_AVAILABLE, m._FLAIR_AVAILABLE, m._DETOXIFY_AVAILABLE,
              m._VADER_AVAILABLE, m._ALT_PROFANITY_AVAILABLE],
    "sentiment": type(m.DEFAULT_SENTIMENT_SCORER).__name__,
    "toxicity": type(m.DEFAULT_TOXICITY_SCORER).__name__,
    "regard": type(m.DEFAULT_REGARD_SCORER).__name__,
    "stereotype": type(m.DEFAULT_STEREOTYPE_SCORER).__name__,
    "status_quality": {k: v["quality"] for k, v in m.scorer_status().items()},
})
""",
            stub_dir=None,
        )
        assert r["flags"] == [False, False, False, False, False], (
            "this control assumes a bare venv with no optional LLM backends"
        )
        assert r["sentiment"] == "KeywordSentimentScorer"
        assert r["toxicity"] == "KeywordToxicityScorer"
        assert r["regard"] == "KeywordRegardScorer"
        assert r["stereotype"] == "StereotypeScorer"
        assert r["status_quality"]["sentiment"] == "placeholder"
        assert r["status_quality"]["toxicity"] == "placeholder"
        # CHANGED 2026-08-28. This line PINNED A DEFECT: on a bare install the
        # regard rung is KeywordRegardScorer, which raises PlaceholderScorerWarning
        # when it scores, yet scorer_status() graded it "good" while the two
        # sibling keyword rungs above reported "placeholder". The grade now
        # matches the warning the same class emits.
        assert r["status_quality"]["regard"] == "placeholder"

    @pytest.mark.parametrize("metric", sorted(_CONTROL_EXPECTED))
    def test_default_scorer_numbers_unchanged(self, metric):
        scorer = getattr(scorers_mod, f"DEFAULT_{metric.upper()}_SCORER")
        with warnings.catch_warnings(record=True) as caught:
            # The placeholder warning is asserted separately; silence it here so
            # a strict-warnings run does not turn this control red for the wrong
            # reason. UnscorableTextWarning is NOT silenced: it is the evidence
            # that the one expected NaN is a refusal and not a crash, so it is
            # recorded and asserted below.
            warnings.simplefilter("always")
            got = [round(float(scorer.score(t)), 3) for t in _CONTROL_CORPUS]
        assert got == pytest.approx(_CONTROL_EXPECTED[metric], nan_ok=True), (
            f"{metric} scores moved; the laziness change must not touch numbers"
        )

        # nan_ok compares position by position, so a NaN is only accepted where
        # this file already expects one. Every such position must ALSO be
        # explained to the reader at the moment it happens, or the refusal is
        # indistinguishable in an aggregate from the 0.0 it replaced.
        n_expected_nan = sum(1 for v in _CONTROL_EXPECTED[metric] if math.isnan(v))
        unscorable = [w for w in caught if isinstance(w.message, UnscorableTextWarning)]
        assert len(unscorable) == n_expected_nan, (
            f"{metric}: {n_expected_nan} refused text(s) but "
            f"{len(unscorable)} warning(s); a silent NaN reads as a crash, and a "
            "warning with no NaN behind it is noise"
        )

    def test_placeholder_warning_still_fires_once_on_first_use(self):
        for cls in (KeywordSentimentScorer, KeywordToxicityScorer):
            s = cls()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                s.score("a good day")
                s.score("a bad day")
            hits = [x for x in w if issubclass(x.category, PlaceholderScorerWarning)]
            assert len(hits) == 1, f"{cls.__name__} lost its placeholder warning"

    def test_keyword_regard_warning_still_gated_on_transformers_absent(self):
        s = KeywordRegardScorer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.score("they are respected leaders")
        hits = [x for x in w if issubclass(x.category, PlaceholderScorerWarning)]
        assert len(hits) == 1
        assert "transformers" in str(hits[0].message)

    def test_constructing_defaults_emits_no_warning(self):
        """Import-time construction must stay warning-free (-W error contract)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            KeywordSentimentScorer()
            KeywordToxicityScorer()
            KeywordRegardScorer()

    def test_import_survives_error_warnings_with_backends_present(self, tmp_path):
        """The -W error contract must hold in the backends-installed arm too."""
        stub_dir = _write_stubs(
            tmp_path,
            {"transformers", "flair", "detoxify", "vaderSentiment", "profanity_check"},
        )
        r = _run("emit({'ok': True})", stub_dir, extra_args=("-W", "error"))
        assert r == {"ok": True}
