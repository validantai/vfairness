"""Audit wave 4 regression tests for vfairness.llm.scorers ("LLM scorers honesty").

Pins the fixes for five adversarially confirmed findings:

1. LLMJudgeScorer no longer masks judge outages as neutral {5,5,5,5}; a
   failed judge call returns NaN plus a RuntimeWarning.
2. Sidecar sentiment/toxicity scorers warn once (SidecarUnavailableWarning)
   and expose .available instead of silently returning 0.0 forever.
3. SemanticQualityScorer depth score is continuous and monotonic at the
   20-word boundary (21 words no longer scores lower than 20).
4. The judge rubric prompt ships single braces; the old .format-style
   double-brace escaping leaked literal {{...}} to the judge model.
5. Placeholder-scorer warnings are lazy (first use) with a custom
   PlaceholderScorerWarning category, so `import vfairness` survives -W error.
"""

import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import vfairness.llm.scorers as scorers_mod
from vfairness.llm.scorers import (
    KeywordRegardScorer,
    KeywordSentimentScorer,
    KeywordToxicityScorer,
    LLMJudgeScorer,
    PlaceholderScorerWarning,
    SemanticQualityScorer,
    SidecarSentimentScorer,
    SidecarToxicityScorer,
    SidecarUnavailableWarning,
)

REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


# Finding ~1347: import-time placeholder warnings aborted -W error runs


class TestPlaceholderWarningsLazy:
    def test_import_survives_error_warnings(self):
        # A fresh interpreter with -W error must import the module cleanly.
        env = dict(os.environ, PYTHONPATH=REPO_SRC)
        # Ensure the sidecar path is not accidentally active in CI.
        env.pop("VFAIRNESS_SIDECAR_PYTHON", None)
        env.pop("VFAIRNESS_SIDECAR_SCRIPT", None)
        r = subprocess.run(
            [sys.executable, "-W", "error", "-c", "import vfairness.llm.scorers"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, f"import aborted under -W error:\n{r.stderr}"

    def test_constructor_emits_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            KeywordSentimentScorer()
            KeywordToxicityScorer()
            KeywordRegardScorer()

    def test_warning_fires_on_first_use_with_custom_category(self):
        s = KeywordSentimentScorer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.score("a good day")
            s.score("a bad day")
        hits = [x for x in w if issubclass(x.category, PlaceholderScorerWarning)]
        assert len(hits) == 1, "expected exactly one lazy warning on first use"
        # Category subclasses UserWarning so existing UserWarning filters
        # still apply.
        assert issubclass(PlaceholderScorerWarning, UserWarning)

    def test_toxicity_warning_lazy_once(self):
        s = KeywordToxicityScorer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.score_batch(["hate this", "fine text"])
        hits = [x for x in w if issubclass(x.category, PlaceholderScorerWarning)]
        assert len(hits) == 1


# Finding ~355: depth_score non-monotonic at the 20-word boundary


class TestSemanticQualityDepthMonotonic:
    """The depth curve must not go DOWN as the answer gets longer.

    Subject unchanged. The FIXTURE changed on 2026-09-17: it used to be filler
    only ("word word word ..."), and the scorer now refuses text in which it can
    read no quality evidence at all, so filler alone came back NaN and these
    three tests compared NaN with NaN and passed vacuously in one case and failed
    in the others. The refusal is correct: the number the old code returned for
    pure filler was WORD COUNT wearing a quality label, which is the defect the
    Beta Go-Live work removed.

    So the filler now carries ONE marker, "I recommend", which is the smallest
    thing that makes the text readable to this scorer. Every other component
    stays constant across the sweep, so the depth term is still the only thing
    moving and the property is still the property.
    """

    #: The smallest prefix that makes filler readable: one actionable marker.
    READABLE = "I recommend "

    def _filler(self, n: int) -> str:
        return self.READABLE + " ".join(["word"] * n)

    def test_exact_counterexample_20_vs_21_words(self):
        sq = SemanticQualityScorer()
        # Before the fix: score(20w)=0.205 > score(21w)=0.153. A longer answer
        # scored WORSE, which is the non-monotonicity this pins.
        t20, t21 = self._filler(20), self._filler(21)
        assert not math.isnan(sq.score(t20)), "fixture is unreadable; it pins nothing"
        assert sq.score(t21) >= sq.score(t20)

    def test_score_nondecreasing_in_pure_length(self):
        # Every component except depth is constant across these, so the total
        # must be non-decreasing with word count.
        sq = SemanticQualityScorer()
        prev = -1.0
        for n in range(1, 201):
            cur = sq.score(self._filler(n))
            assert not math.isnan(cur), f"fixture unreadable at {n} words"
            assert cur >= prev, f"score dropped at {n} words: {prev} -> {cur}"
            prev = cur

    def test_depth_caps_at_150_words(self):
        sq = SemanticQualityScorer()
        capped = sq.score(self._filler(150))
        assert not math.isnan(capped), "fixture is unreadable; it pins nothing"
        assert capped == sq.score(self._filler(400))


# Finding ~958: rubric prompt shipped literal double braces


class TestJudgeRubricBraces:
    def test_template_has_no_double_braces(self):
        assert "{" + "{" not in LLMJudgeScorer._RUBRIC_PROMPT
        assert "}" + "}" not in LLMJudgeScorer._RUBRIC_PROMPT

    def test_built_prompt_single_braces_and_placeholder_filled(self, monkeypatch):
        captured = {}

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"helpfulness": 8, "fairness": 9, "specificity": 7, "completeness": 8}'
                            }
                        }
                    ]
                }

        def fake_post(url, *a, json=None, headers=None, timeout=None, **k):
            captured["prompt"] = json["messages"][0]["content"]
            return _Resp()

        from vfairness.net import egress

        monkeypatch.setattr(egress, "guarded_post", fake_post)
        j = LLMJudgeScorer(endpoint_url="http://judge.invalid/v1/chat/completions")
        j.score("Sample response text.")
        prompt = captured["prompt"]
        assert "{" + "{" not in prompt
        assert "}" + "}" not in prompt
        assert "{text}" not in prompt
        assert "Sample response text." in prompt
        # The JSON shape instruction reaches the model with single braces.
        assert '{"helpfulness": N' in prompt


# VF-2 (2026-08-27): LLMJudgeScorer no longer calls requests.post. The judge
# request carries the API key to a caller-supplied URL, so it now goes through
# net.egress.guarded_post, which validates the target BEFORE the header is sent.
# These tests therefore patch guarded_post, not requests.post.
#
# Patching requests.post after that change did not fail loudly: the guard
# refused the unresolvable fixture host "judge.invalid" first, so the fake was
# never reached. Two tests in this class kept PASSING while no longer exercising
# the judge at all, because they only assert "a RuntimeWarning and NaN" and the
# guard's own refusal produces exactly that. Fixing only the three that went red
# would have left those two green and hollow.


# Finding ~1010: judge outage masked as neutral {5,5,5,5}


class TestJudgeFailureIsLoud:
    def _failing_judge(self, monkeypatch):
        """A judge that is REACHED and then fails, not one the guard refused.

        The distinction is the point of the test: a mid-flight ConnectionError
        is an outage, while a guard refusal means nothing was dialled. They warn
        differently and must not be conflated.
        """
        import requests

        from vfairness.net import egress

        def fake_post(*a, **k):
            raise requests.exceptions.ConnectionError("judge endpoint down")

        monkeypatch.setattr(egress, "guarded_post", fake_post)
        return LLMJudgeScorer(endpoint_url="http://judge.invalid/v1/chat/completions", timeout=1)

    def test_score_returns_nan_and_warns_on_outage(self, monkeypatch):
        j = self._failing_judge(monkeypatch)
        with pytest.warns(RuntimeWarning, match="judge call failed"):
            s = j.score("a substantive response")
        assert isinstance(s, float)
        assert math.isnan(s), "outage must NOT read as a neutral mid-scale score"

    def test_score_batch_propagates_nan(self, monkeypatch):
        j = self._failing_judge(monkeypatch)
        # match= is load-bearing, not decoration. A guard refusal also yields
        # NaN plus a RuntimeWarning, so a bare pytest.warns(RuntimeWarning)
        # passes whether or not the judge was ever reached. Requiring the
        # OUTAGE wording is what makes this test able to fail.
        with pytest.warns(RuntimeWarning, match="judge call failed"):
            arr = j.score_batch(["text one", "text two"])
        assert arr.shape == (2,)
        assert np.isnan(arr).all()
        # A group mean over failed judge calls is NaN, visibly broken,
        # rather than a fake 0.5 "no bias" signal.
        assert math.isnan(float(np.mean(arr)))

    def test_unparseable_reply_returns_nan(self, monkeypatch):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "not json at all"}}]}

        from vfairness.net import egress

        monkeypatch.setattr(egress, "guarded_post", lambda *a, **k: _Resp())
        j = LLMJudgeScorer(endpoint_url="http://judge.invalid/v1")
        # As above: require the wording that only the reached-and-unparseable
        # path produces, so a guard refusal cannot satisfy this test.
        with pytest.warns(RuntimeWarning, match="judge call failed"):
            assert math.isnan(j.score("hello there world"))

    def test_success_path_unchanged(self, monkeypatch):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"helpfulness": 10, "fairness": 10, "specificity": 10, "completeness": 10}'
                            }
                        }
                    ]
                }

        from vfairness.net import egress

        monkeypatch.setattr(egress, "guarded_post", lambda *a, **k: _Resp())
        j = LLMJudgeScorer(endpoint_url="http://judge.invalid/v1")
        assert j.score("great answer") == 1.0

    def test_empty_text_still_zero_not_nan(self):
        j = LLMJudgeScorer(endpoint_url="http://judge.invalid/v1")
        assert j.score("") == 0.0
        assert j.score("   ") == 0.0


# Finding ~161: sidecar scorers silently return 0.0 when sidecar is down


class TestSidecarDownIsLoud:
    @pytest.fixture()
    def dead_sidecar(self, tmp_path, monkeypatch):
        script = tmp_path / "dead_sidecar.py"
        script.write_text("import sys; sys.exit(1)\n")
        monkeypatch.setenv("VFAIRNESS_SIDECAR_PYTHON", sys.executable)
        monkeypatch.setenv("VFAIRNESS_SIDECAR_SCRIPT", str(script))
        # Reset module state so the once-per-process warning and the bridge
        # singleton do not leak between tests.
        monkeypatch.setattr(scorers_mod, "_sidecar_down_warned", False)
        monkeypatch.setattr(scorers_mod._SidecarBridge, "_instance", None)
        yield
        scorers_mod._SidecarBridge._instance = None

    def test_available_flag_false_when_spawn_fails(self, dead_sidecar):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert SidecarSentimentScorer().available is False
            assert SidecarToxicityScorer().available is False

    def test_warns_once_not_per_text(self, dead_sidecar):
        """The warning half of the wave-4 decision, unchanged: exactly once, and
        it says the scores are not real. The VALUE half changed on 2026-09-25,
        because once per process cannot describe ten thousand scores."""
        sent = SidecarSentimentScorer()
        tox = SidecarToxicityScorer()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert math.isnan(sent.score("I love this"))
            assert math.isnan(tox.score("you are awful"))
            assert all(math.isnan(v) for v in sent.score_batch(["a", "b"]))
        hits = [x for x in w if issubclass(x.category, SidecarUnavailableWarning)]
        assert len(hits) == 1, "sidecar-down warning must fire exactly once per process"
        assert "NOT real scores" in str(hits[0].message)

    def test_the_return_shape_is_unchanged_so_pipelines_still_run(self, dead_sidecar):
        """The half of the API-compatibility promise that survives, and it is the
        half that mattered: a caller still gets an ndarray of the right length, so
        nothing crashes. What it no longer gets is a column of clean zeros that
        reads as measured non-toxicity. Renamed from
        test_zeros_still_returned_api_compat on 2026-09-25."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = SidecarToxicityScorer().score_batch(["x", "y", "z"])
        assert arr.shape == (3,)
        assert np.isnan(arr).all(), "an outage produced scores that read as measured"
