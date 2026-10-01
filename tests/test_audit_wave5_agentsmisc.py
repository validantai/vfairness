"""Audit wave 5 pins: agents, llm, multi_agent, vision, legal, xai lows.

Each test pins one adversarially confirmed audit finding so it cannot
regress. Findings covered:
- rag_bias: alpha reused as disparity threshold; positional id fallback
  collision; trigram distance wrong for short strings
- temporal: CUSUM/EWMA returned array indices instead of turn numbers;
  cumulative_drift docstring said "from initial value" but code computes
  total variation
- action_bias: KeyError instead of documented ValueError on missing field
- llm/output_analysis: docstring default mismatch; dead keyword heuristics
- multi_agent/delegation: docstring claimed None for always-float fields
- vision: bias_amplification docstring formula
- legal/admissibility: infinite recursion on cyclic inherits
- xai/diagnostics/faithfulness: IndexError when n_steps > n_features
- pyproject: llm-transformers extra for advertised scorer backends
"""

import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

import vfairness.legal.admissibility as admissibility
from vfairness.agents.action_bias import ActionBiasAnalyzer
from vfairness.agents.rag_bias import RAGBiasAnalyzer
from vfairness.agents.temporal import TemporalTracker, TrajectoryResult
from vfairness.llm.output_analysis import OutputAnalyzer
from vfairness.multi_agent.delegation import DelegationResult, DelegationRoutingAuditor
from vfairness.vision import bias_amplification
from vfairness.xai.diagnostics.faithfulness import removal_curve_auc

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


# ================================================================ rag_bias


class TestRAGBiasThresholds:
    def _docs(self, ids):
        return [{"id": i} for i in ids]

    def test_alpha_no_longer_used_as_disparity_threshold(self):
        # 19 of 20 docs shared: Jaccard distance ~0.095, above the old
        # alpha default (0.05) but below the new disparity default (0.1).
        docs_a = self._docs(range(20))
        docs_b = self._docs(list(range(19)) + ["extra"])
        analyzer = RAGBiasAnalyzer(alpha=0.05)
        result = analyzer.full_analysis(
            [], [], docs_a, docs_b, ["same text here"], ["same text here"]
        )
        assert 0.05 < result.retrieval_disparity < 0.1
        assert result.is_retrieval_biased is False

    def test_explicit_disparity_threshold_respected(self):
        docs_a = self._docs(range(20))
        docs_b = self._docs(list(range(19)) + ["extra"])
        analyzer = RAGBiasAnalyzer(alpha=0.05, disparity_threshold=0.05)
        result = analyzer.full_analysis(
            [], [], docs_a, docs_b, ["same text here"], ["same text here"]
        )
        assert result.is_retrieval_biased is True

    def test_disparity_threshold_recorded_in_metadata(self):
        analyzer = RAGBiasAnalyzer(disparity_threshold=0.2)
        result = analyzer.full_analysis(
            [], [], self._docs([1]), self._docs([1]), ["abc def"], ["abc def"]
        )
        assert result.metadata.parameters["disparity_threshold"] == 0.2
        assert result.metadata.parameters["alpha"] == 0.05


class TestRAGBiasDocIdentity:
    def test_distinct_unlabeled_docs_do_not_collide(self):
        analyzer = RAGBiasAnalyzer()
        distance = analyzer.analyze_retrieval(
            [{"text": "completely different document A"}],
            [{"text": "unrelated document B"}],
        )
        assert distance == 1.0

    def test_identical_unlabeled_docs_still_match(self):
        analyzer = RAGBiasAnalyzer()
        distance = analyzer.analyze_retrieval(
            [{"text": "same content"}], [{"text": "same content"}]
        )
        assert distance == 0.0

    def test_explicit_ids_take_precedence(self):
        analyzer = RAGBiasAnalyzer()
        distance = analyzer.analyze_retrieval(
            [{"id": "d1", "text": "x"}], [{"id": "d1", "text": "y"}]
        )
        assert distance == 0.0

    def test_unhashable_content_falls_back_to_object_identity(self):
        # Dicts with unsortable keys must not crash and distinct objects
        # must stay distinct.
        analyzer = RAGBiasAnalyzer()
        doc_a = {1: "x", "k": "v"}
        doc_b = {2: "y", "k": "w"}
        distance = analyzer.analyze_retrieval([doc_a], [doc_b])
        assert distance == 1.0


class TestTrigramDistanceShortStrings:
    def test_distinct_short_strings_are_not_identical(self):
        assert RAGBiasAnalyzer._trigram_distance("ab", "cd") == 1.0

    def test_equal_short_strings_are_identical(self):
        assert RAGBiasAnalyzer._trigram_distance("ab", "ab") == 0.0

    def test_partial_char_overlap_short_strings(self):
        # chars {a, b} vs {b, c}: Jaccard 1/3, distance 2/3.
        dist = RAGBiasAnalyzer._trigram_distance("ab", "bc")
        assert dist == pytest.approx(2.0 / 3.0)

    def test_empty_string_behaviour_preserved(self):
        assert RAGBiasAnalyzer._trigram_distance("", "") == 0.0
        assert RAGBiasAnalyzer._trigram_distance("", "abcd") == 1.0

    def test_long_string_path_unchanged(self):
        assert RAGBiasAnalyzer._trigram_distance("abcdef", "abcdef") == 0.0
        assert RAGBiasAnalyzer._trigram_distance("abcdef", "uvwxyz") == 1.0


# ================================================================ temporal


class TestTemporalTurnIdentifiers:
    def _tracker_with_shift(self):
        tracker = TemporalTracker()
        # Non-contiguous turn numbers so indices and turns differ.
        turns = [10, 20, 30, 40, 50, 60]
        for i, turn in enumerate(turns):
            value = 0.0 if i < 3 else 5.0
            tracker.record_turn(turn, [value], [0.0])
        return tracker, turns

    def test_cusum_drift_point_is_turn_number(self):
        tracker, turns = self._tracker_with_shift()
        result = tracker.detect_drift_cusum(threshold=0.1, drift_limit=1.0)
        assert result["has_drift"] is True
        assert result["drift_point"] in turns

    def test_ewma_drift_points_are_turn_numbers(self):
        tracker, turns = self._tracker_with_shift()
        result = tracker.detect_drift_ewma(span=2, sigma_limit=0.5)
        assert result["has_drift"] is True
        assert len(result["drift_points"]) > 0
        assert set(result["drift_points"]).issubset(set(turns))

    def test_contiguous_zero_based_turns_unchanged(self):
        tracker = TemporalTracker()
        for turn in range(6):
            value = 0.0 if turn < 3 else 5.0
            tracker.record_turn(turn, [value], [0.0])
        result = tracker.detect_drift_cusum(threshold=0.1, drift_limit=1.0)
        assert result["drift_point"] in range(6)


class TestCumulativeDriftDocs:
    def test_behaviour_is_total_variation(self):
        # 0 -> 1 -> 0: net change from initial is 0, total variation is 2.
        tracker = TemporalTracker()
        tracker.record_turn(0, [0.0], [0.0])
        tracker.record_turn(1, [1.0], [0.0])
        tracker.record_turn(2, [0.0], [0.0])
        trajectory = tracker.compute_trajectory()
        assert trajectory[-1].cumulative_drift == pytest.approx(2.0)

    def test_docstring_no_longer_claims_from_initial_value(self):
        doc = TrajectoryResult.__doc__
        assert "from initial value" not in doc
        assert "turn-to-turn" in doc or "total variation" in doc.lower()


# ============================================================= action_bias


class TestActionBiasMissingField:
    def test_missing_outcome_field_raises_valueerror(self):
        analyzer = ActionBiasAnalyzer()
        actions = [{"action": "approve", "score": 1.0}] * 3
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="outcome_field"):
                analyzer.analyze_outcomes(actions, actions, outcome_field="missing")

    def test_present_outcome_field_still_works(self):
        analyzer = ActionBiasAnalyzer()
        rng = np.random.default_rng(42)
        actions_a = [{"action": "approve", "score": float(v)} for v in rng.normal(0.7, 0.05, 40)]
        actions_b = [{"action": "approve", "score": float(v)} for v in rng.normal(0.5, 0.05, 40)]
        result = analyzer.analyze_outcomes(actions_a, actions_b, outcome_field="score")
        assert result.disparity > 0


# ========================================================== output_analysis


class TestOutputAnalysisCleanup:
    def test_analyze_all_docstring_matches_signature_default(self):
        doc = OutputAnalyzer.analyze_all.__doc__
        assert "Default is 'benjamini_hochberg'" in doc
        assert "Default is 'bonferroni'" not in doc

    def test_dead_keyword_heuristics_removed(self):
        # Canonical keyword scorers live in vfairness/llm/scorers.py; the
        # duplicated static heuristics must stay deleted.
        for name in ("_sentiment_score", "_toxicity_score", "_refusal_score"):
            assert not hasattr(OutputAnalyzer, name)
        import vfairness.llm.output_analysis as oa

        for name in ("_POSITIVE_WORDS", "_NEGATIVE_WORDS", "_TOXIC_KEYWORDS", "_REFUSAL_KEYWORDS"):
            assert not hasattr(oa, name)

    def test_canonical_scorers_still_exist(self):
        from vfairness.llm.scorers import (
            DEFAULT_REFUSAL_SCORER,
            DEFAULT_SENTIMENT_SCORER,
            DEFAULT_TOXICITY_SCORER,
        )

        assert DEFAULT_SENTIMENT_SCORER is not None
        assert DEFAULT_TOXICITY_SCORER is not None
        assert DEFAULT_REFUSAL_SCORER is not None


# =============================================================== delegation


class TestDelegationDocstring:
    def test_fields_documented_as_floats_not_none(self):
        doc = DelegationResult.__doc__
        assert "None if Fisher's exact was used" not in doc
        assert "else None" not in doc
        assert "nan" in doc.lower()

    def test_fisher_path_returns_floats(self):
        result = DelegationRoutingAuditor().analyze(["r1", "r2", "r1", "r2"], ["A", "A", "B", "B"])
        assert isinstance(result.chi_square, float)
        assert isinstance(result.odds_ratio, float)
        assert result.test_used == "fisher"

    def test_chi2_path_returns_nan_odds_ratio(self):
        """The odds ratio is a 2x2 Fisher quantity, so every larger-table path
        reports nan rather than a number.

        Readiness 6 (2026-09-10) split that path in two, and this pin followed
        it. A table whose smallest EXPECTED cell count is below 5 now runs an
        exact-by-permutation chi-square instead of the asymptotic one, because
        the asymptotic p-value is not valid there: measured on 13 decisions,
        12 from one group over two routes and 1 from another on a third, the
        asymptotic test answered p=0.0015 and "significant" where the exact
        conditional answer is 0.0769. This fixture's 3x2 table has expected
        counts of 3.0, so it takes the exact route; the larger table below
        takes the asymptotic one. The subject of the pin is unchanged: neither
        reports an odds ratio.
        """
        routes = ["r1", "r2", "r3"] * 6
        groups = ["A", "B"] * 9
        result = DelegationRoutingAuditor().analyze(routes, groups)
        assert result.test_used == "permutation_chi2"
        assert isinstance(result.chi_square, float)
        assert np.isnan(result.odds_ratio)

        big = DelegationRoutingAuditor().analyze(["r1", "r2", "r3"] * 60, ["A", "B"] * 90)
        assert big.test_used == "chi2"
        assert isinstance(big.chi_square, float)
        assert np.isnan(big.odds_ratio)


# =================================================================== vision


class TestBiasAmplificationDocstring:
    def test_docstring_states_proportion_difference(self):
        doc = bias_amplification.__doc__
        assert "proportion difference" in doc
        assert "generated-set skew" not in doc

    def test_formula_is_proportion_difference(self):
        # gen: m=2/3, f=1/3; ref: 0.5 each. Delta m = +1/6, f = -1/6.
        out = bias_amplification(["m", "m", "f"], {"m": 0.5, "f": 0.5})
        assert out["perGroup"]["m"] == pytest.approx(2 / 3 - 0.5, abs=1e-4)
        assert out["perGroup"]["f"] == pytest.approx(1 / 3 - 0.5, abs=1e-4)


# ==================================================================== legal


class TestLegalInheritsCycle:
    @pytest.fixture()
    def rules_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(admissibility, "_RULES_DIR", tmp_path)
        admissibility.load_rules.cache_clear()
        yield tmp_path
        admissibility.load_rules.cache_clear()

    def test_two_pack_cycle_raises_valueerror(self, rules_dir):
        (rules_dir / "uc.a.json").write_text(
            json.dumps({"inherits": "uc.b", "rules": {"x": {"status": "allowed"}}})
        )
        (rules_dir / "uc.b.json").write_text(
            json.dumps({"inherits": "uc.a", "rules": {"y": {"status": "allowed"}}})
        )
        with pytest.raises(ValueError, match="Cyclic 'inherits'"):
            admissibility.load_rules("uc", "a")

    def test_self_inherit_raises_valueerror(self, rules_dir):
        (rules_dir / "uc.a.json").write_text(json.dumps({"inherits": "uc.a", "rules": {}}))
        with pytest.raises(ValueError, match="Cyclic 'inherits'"):
            admissibility.load_rules("uc", "a")

    def test_acyclic_inheritance_still_merges(self, rules_dir):
        (rules_dir / "uc.child.json").write_text(
            json.dumps({"inherits": "uc.parent", "rules": {"b": {"status": "restricted"}}})
        )
        (rules_dir / "uc.parent.json").write_text(
            json.dumps({"rules": {"a": {"status": "allowed"}, "b": {"status": "allowed"}}})
        )
        pack = admissibility.load_rules("uc", "child")
        assert pack["rules"]["a"]["status"] == "allowed"
        assert pack["rules"]["b"]["status"] == "restricted"

    def test_missing_pack_returns_none(self, rules_dir):
        assert admissibility.load_rules("uc", "nowhere") is None


# ============================================================= faithfulness


class TestRemovalCurveClamp:
    def _setup(self):
        predict_fn = lambda X: X.sum(axis=1)  # noqa: E731
        x = np.array([1.0, 2.0, 3.0])
        attributions = np.array([0.5, 0.2, 0.1])
        background = np.zeros(3)
        return predict_fn, x, attributions, background

    def test_n_steps_beyond_n_features_is_clamped(self):
        predict_fn, x, attr, bg = self._setup()
        auc_over = removal_curve_auc(
            predict_fn=predict_fn, x=x.copy(), attributions=attr, background_mean=bg, n_steps=10
        )
        auc_exact = removal_curve_auc(
            predict_fn=predict_fn, x=x.copy(), attributions=attr, background_mean=bg, n_steps=3
        )
        assert auc_over == pytest.approx(auc_exact)

    def test_partial_n_steps_still_works(self):
        predict_fn, x, attr, bg = self._setup()
        auc = removal_curve_auc(
            predict_fn=predict_fn, x=x.copy(), attributions=attr, background_mean=bg, n_steps=2
        )
        assert np.isfinite(auc)


# ================================================================ packaging


class TestLLMTransformersExtra:
    # This test used to pin the literal "torch>=1.9.0", and that literal froze a
    # floor NO resolver could ever reach: requires-python is >=3.11, and torch
    # published no cp311 wheel until 1.13.0 (checked against the PyPI file lists
    # on 2026-08-28; 1.9.0 -> no cp311, 2.0.0 -> cp311). A floor below the
    # minimum supported interpreter is not a support claim, it is a number
    # nobody can install, and pinning the literal in a test made it permanent:
    # raising the floor to a reachable one turned this test red, which is
    # exactly backwards.
    #
    # So assert the PROPERTY, not the spelling. The backends must be declared,
    # and the torch floor must be one a 3.11 user can actually resolve. Raising
    # the floor is now a one-line pyproject change; dropping it back below
    # cp311 availability fails here, which is the direction that should fail.
    _EARLIEST_TORCH_WITH_CP311 = (1, 13, 0)

    def test_extra_declares_advertised_backends(self):
        # Read the extra out of the PARSED table, not by slicing the file around
        # the first "llm-transformers" substring. That slice was how this test
        # used to work, and it broke the moment a comment mentioning the extra
        # was added above the table: it then measured prose instead of
        # dependencies, and still reported a confident pass/fail about neither.
        with _PYPROJECT.open("rb") as handle:
            extras = tomllib.load(handle)["project"]["optional-dependencies"]
        assert "llm-transformers" in extras, sorted(extras)
        section = " ".join(extras["llm-transformers"])
        assert "transformers>=4.30" in section
        assert "flair>=0.13" in section
        assert "detoxify>=0.5" in section

        match = re.search(r"torch>=(\d+)\.(\d+)\.(\d+)", section)
        assert match is not None, (
            f"the llm-transformers extra declares no torch floor: {section!r}. "
            "The transformers/flair/detoxify scorers all import torch, so the "
            "extra has to pin it or the backends are advertised without their "
            "runtime."
        )
        floor = tuple(int(g) for g in match.groups())
        assert floor >= self._EARLIEST_TORCH_WITH_CP311, (
            f"torch floor {'.'.join(map(str, floor))} predates the first release "
            f"with a cp311 wheel "
            f"({'.'.join(map(str, self._EARLIEST_TORCH_WITH_CP311))}), but "
            "requires-python is >=3.11. No resolver on a supported interpreter "
            "can install that floor, so the dependency-floor CI job cannot test "
            "it and the number means nothing."
        )
