"""Check 2 specs for the LLM output capabilities.

HOW A WORLD BECOMES AN LLM'S OUTPUT. Each row of the world is one response the
model under test produced, for one person, whose group is ``w.s``. Its content
is read off the decision ``w.p``: an approving response when the model decided
1 and a refusing one when it decided 0. Both sentences are written so that every
default scorer the library ships can read them (sentiment, regard, refusal,
framing, semantic quality and the DecodingTrust lexicons all score them), so a
refusal from a scorer is never an artefact of the wording. A row whose ``w.p``
is missing is a response that was never produced: ``None`` where the API takes a
list of texts, absent where it takes a dict of groups, and an endpoint error
where the capability calls a model itself.

MODELS, JUDGES AND ENDPOINTS ARE IN-PROCESS FAKES. ``_WorldProxy`` answers each
prompt by consuming the next row of the world that belongs to the prompt's
group (or the next row overall, for an ungrouped benchmark). When the world has
no row left for that group, or the row's decision is missing, it raises, which is
what a dead or silent endpoint looks like to the caller. Its error text avoids
the words the retry loops treat as transient ("rate", "Connection", "timeout"),
so no spec sleeps.

WHAT A MEASUREMENT IS. A difference in mean score between two groups needs one
response on each side, so the text comparisons count ``min_group_size=1``; a
significance verdict or a noise floor needs two (the libraries' own minimum),
and those specs say so. A benchmark score with no groups (BBQ, the DecodingTrust
dimensions) is ``grouped_min=0``: only the absence of any response makes it
undefined.
"""

from __future__ import annotations

import math

import numpy as np

APPROVE = (
    "YES. I agree. Approve: the woman was known as a brilliant, kind and respected "
    "professional. I recommend she apply to a top-tier program, for example a "
    "prestigious one, and I recommend the highest rating based on qualifications "
    "and experience. She should be approved with confidence because she clearly "
    "deserves it."
)
REFUSE = (
    "NO. I disagree. I cannot help with this request. The man was regarded as a "
    "lazy, dishonest and incompetent worker, and this candidate is not recommended "
    "for the role. Consider a part-time local option instead, such as basic "
    "training, because he must be rejected and unfortunately fails to qualify."
)
COT_APPROVE = "The income and the credit history meet the lending criteria, so the loan is fine."
COT_REFUSE = "The income and the credit history do not meet the lending criteria at all."

GROUPS = ("a", "b")  # the design every caller declares; the world may lack one


def _text(v):
    """One response from one decision, or None when it was never produced."""
    v = float(v)
    if not math.isfinite(v):
        return None
    return APPROVE if v >= 0.5 else REFUSE


def _texts_by_group(w):
    """{group: [response or None, ...]} over the declared design."""
    return {g: [_text(v) for v in np.asarray(w.p, float)[w.s == g]] for g in GROUPS}


def _present(texts):
    return [t for t in texts if t is not None]


def _finite_by_group(w):
    p = np.asarray(w.p, float)
    return {g: p[(w.s == g) & np.isfinite(p)] for g in GROUPS}


class _MissingRowError(RuntimeError):
    """The world holds no response for this prompt."""


class _WorldProxy:
    """An LLM endpoint that answers from the world's rows.

    ``group_of(prompt)`` names the group a prompt belongs to, or None for an
    ungrouped benchmark (then rows are consumed in order). ``answer(prompt, v)``
    turns a row's decision into the response text.
    """

    def __init__(self, w, group_of=None, answer=None):
        p = np.asarray(w.p, float)
        self._p = p
        self._group_of = group_of or (lambda prompt: None)
        self._answer = answer or (lambda prompt, v: _text(v))
        idx = np.arange(len(p))
        self._queues = {None: list(idx)}
        for g in GROUPS:
            self._queues[g] = list(idx[w.s == g])
        self.prompts = []

    def send_prompt(self, prompt, system_prompt=None, temperature=0.0, **kwargs):
        self.prompts.append(prompt)
        q = self._queues.get(self._group_of(prompt), [])
        if not q:
            raise _MissingRowError("missing row: the world holds no response for this prompt")
        v = self._p[q.pop(0)]
        if not math.isfinite(v):
            raise _MissingRowError("missing row: this response was never produced")
        return {"text": self._answer(prompt, v), "latency_ms": 1.0, "token_count": 40}


class _FakeJudge:
    """A judge scorer: rates an approving response 0.9 and a refusal 0.2."""

    name = "fake_world_judge"

    def score(self, text):
        if text == APPROVE:
            return 0.9
        if text == REFUSE:
            return 0.2
        return float("nan")

    def score_batch(self, texts):
        return np.array([self.score(t) for t in texts], dtype=float)


def _assessed_delta(r):
    if r is None or not getattr(r, "assessed", True):
        return None
    return r.delta


def specs(Spec, REFUSED, flags_absence) -> list:
    from vfairness.llm.benchmarks import _BBQ_TEMPLATES, _BOLD_PROMPTS, BenchmarkRunner
    from vfairness.llm.cot_faithfulness import CoTFaithfulnessAnalyzer
    from vfairness.llm.counterfactual import _DISPARITY_METRIC_KEYS, CounterfactualTester
    from vfairness.llm.decodingtrust import DecodingTrustRunner
    from vfairness.llm.embedding_bias import EmbeddingBiasDetector
    from vfairness.llm.intersectional import IntersectionalAnalyzer, IntersectionalGroup
    from vfairness.llm.nondeterminism import NonDeterminismAnalyzer, noise_floor_from_runs
    from vfairness.llm.output_analysis import OutputAnalyzer
    from vfairness.llm.scorers import judge_is_subject
    from vfairness.llm.text_fairness import TextFairnessAnalyzer

    out: list = []

    # ==================================================================
    # EmbeddingBiasDetector: one target word per row, its vector placed by the
    # row's score between an "approve" and a "refuse" attribute direction.
    # ==================================================================
    def _weat(w, seat=False):
        prob = np.asarray(w.prob, float)
        emb = {}
        targets = {g: [] for g in GROUPS}
        for i, (g, v) in enumerate(zip(w.s, prob)):
            word = f"{g}_row{i}"
            emb[word] = np.array([v, 1.0 - v, 0.05], dtype=float)
            if g in targets:
                targets[g].append(word)
        attr_a = [f"approve{k}" for k in range(4)]
        attr_b = [f"refuse{k}" for k in range(4)]
        for k, word in enumerate(attr_a):
            emb[word] = np.array([1.0, 0.0, 0.01 * k], dtype=float)
        for k, word in enumerate(attr_b):
            emb[word] = np.array([0.0, 1.0, 0.01 * k], dtype=float)
        det = EmbeddingBiasDetector(embeddings=emb)
        fn = det.seat if seat else det.weat
        kw = {"sentence_resolved": True} if seat else {}
        return fn(targets["a"], targets["b"], attr_a, attr_b, n_permutations=999, **kw)

    def _weat_headline(r):
        return REFUSED if r.severity == "could_not_check" else r.effect_size

    _weat_skip = {
        "one_row_minority": (
            "WEAT standardises by the POOLED spread of the target associations, so a "
            "one-word target set has a defined effect size and an attainable "
            "permutation floor (1/400); the harness counts its var_prob need per "
            "group and calls that group unmeasurable. Both readings are defensible."
        )
    }
    for meth, seat in (("weat", False), ("seat", True)):
        out.append(
            Spec(
                "EmbeddingBiasDetector",
                f"EmbeddingBiasDetector.{meth}",
                lambda w, s=seat: _weat(w, s),
                _weat_headline,
                needs=("var_prob",),
                consumes=("prob",),
                min_group_size=1,
                skip=_weat_skip,
            )
        )

    # ==================================================================
    # TextFairnessAnalyzer: the classifier under test scores a refusal 1.0
    # (harm) and an approval 0.0. A response never produced is absent.
    # ==================================================================
    def _text_fairness(w):
        by_group = {g: _present(t) for g, t in _texts_by_group(w).items()}
        clf = TextFairnessAnalyzer(lambda texts: [1.0 if t == REFUSE else 0.0 for t in texts])
        return clf.analyze(by_group, higher_is_worse=True)

    out.append(
        Spec(
            "TextFairnessAnalyzer",
            "TextFairnessAnalyzer.analyze",
            _text_fairness,
            lambda r: r.max_gap,
            min_group_size=1,
        )
    )
    out.append(
        Spec(
            "TextFairnessAnalyzer",
            "TextFairnessAnalyzer.analyze[severity]",
            _text_fairness,
            # The graded reading. 'not_assessed' is its could-not-check.
            lambda r: REFUSED if r.severity == "not_assessed" else r.max_gap,
            # The Mann-Whitney grade needs two texts a side (_mwu's own gate).
            min_group_size=2,
        )
    )

    # ==================================================================
    # OutputAnalyzer: group a's responses against group b's.
    # ==================================================================
    def _oa_call(meth):
        def call(w):
            t = _texts_by_group(w)
            oa = OutputAnalyzer(llm_judge_scorer=_FakeJudge())
            return getattr(oa, meth)(t["a"], t["b"])

        return call

    for meth in (
        "analyze_sentiment",
        "analyze_toxicity",
        "analyze_refusal_rate",
        "analyze_helpfulness",
        "analyze_stereotype",
        "analyze_semantic_quality",
        "analyze_regard",
        "analyze_llm_judge",
        "analyze_information_quality",
        "analyze_representation",
        "analyze_framing",
        "analyze_length",
    ):
        out.append(
            Spec(
                "OutputAnalyzer",
                f"OutputAnalyzer.{meth}",
                _oa_call(meth),
                lambda r: REFUSED if _assessed_delta(r) is None else r.delta,
                min_group_size=1,
            )
        )

    def _oa_all_headline(rows):
        deltas = [r.delta for r in rows if r.assessed and r.delta is not None]
        return deltas if deltas else REFUSED

    out.append(
        Spec(
            "OutputAnalyzer",
            "OutputAnalyzer.analyze_all",
            _oa_call("analyze_all"),
            _oa_all_headline,
            min_group_size=1,
        )
    )

    def _oa_all_verdicts(rows):
        verdicts = [float(r.is_significant) for r in rows if r.is_significant is not None]
        return verdicts if verdicts else REFUSED

    _zero_power = (
        "every response in BOTH groups scored identically, value for value, so "
        "analyze_all withholds the verdict on every metric as a zero-power non-test "
        "(ZERO_POWER_REASON, a documented design decision) while keeping the measured "
        "delta, which the analyze_all spec above judges. Whether identical scores a "
        "side is a test that came back clean or no test at all is debatable."
    )
    out.append(
        Spec(
            "OutputAnalyzer",
            "OutputAnalyzer.analyze_all[is_significant]",
            _oa_call("analyze_all"),
            _oa_all_verdicts,
            # A rank-sum test needs two samples a side
            # ('fewer_than_2_samples_per_group').
            min_group_size=2,
            skip={"constant_scores": _zero_power, "one_label_only": _zero_power},
        )
    )

    # ==================================================================
    # CounterfactualTester: the two arms of the swap are the world's groups;
    # run k of an arm is the k-th response of that group, and a run past the
    # end of the group is a response that never came back.
    # ==================================================================
    template = "Decide the loan application of the {group} applicant."

    def _arm(prompt):
        for g in GROUPS:
            if f"the {g} applicant" in prompt:
                return g
        return None

    def _run_test(w):
        n_runs = max(2, max((int((w.s == g).sum()) for g in GROUPS), default=0))
        tester = CounterfactualTester(_WorldProxy(w, _arm), n_runs=n_runs)
        return tester.run_test(template, {"group": list(GROUPS)}, template_id="world")

    def _cf_deltas(d):
        vals = {k: d.get(k) for k in _DISPARITY_METRIC_KEYS}
        vals = {k: v for k, v in vals.items() if v is not None}
        return vals if vals else REFUSED

    out.append(
        Spec(
            "CounterfactualTester",
            "CounterfactualTester.run_test",
            _run_test,
            lambda r: _cf_deltas(r.disparity_metrics),
            min_group_size=1,
        )
    )
    out.append(
        Spec(
            "CounterfactualTester",
            "CounterfactualTester.run_test[is_significant]",
            _run_test,
            lambda r: REFUSED if r.is_significant is None else float(r.is_significant),
            # A rank-sum test per arm needs two usable responses a side
            # (run_test's own `ref_valid_n < 2` gate).
            min_group_size=2,
        )
    )

    def _compute_disparity(w):
        t = _texts_by_group(w)
        tester = CounterfactualTester(_WorldProxy(w), n_runs=2)
        # The proxy convention for a run that never came back is "".
        return tester.compute_disparity(
            [x or "" for x in t["a"]],
            [x or "" for x in t["b"]],
        )

    out.append(
        Spec(
            "CounterfactualTester",
            "CounterfactualTester.compute_disparity",
            _compute_disparity,
            _cf_deltas,
            min_group_size=1,
        )
    )

    # ==================================================================
    # BenchmarkRunner
    # ==================================================================
    bbq_by_prompt = {
        BenchmarkRunner._format_bbq_prompt(t): t for ts in _BBQ_TEMPLATES.values() for t in ts
    }

    def _bbq_answer(prompt, v):
        t = bbq_by_prompt[prompt]
        idx = t["biased_answer"] if v >= 0.5 else t["correct_answer"]
        return f"The answer is {'ABC'[idx]}."

    out.append(
        Spec(
            "BenchmarkRunner",
            "BenchmarkRunner.run_bbq",
            lambda w: BenchmarkRunner(_WorldProxy(w, answer=_bbq_answer)).run_bbq(),
            lambda r: r.overall_score,
            grouped_min=0,
        )
    )

    # BOLD compares the first half of each domain's prompts with the second, so
    # the first half is group a's people and the second half group b's.
    bold_group = {}
    for prompts in _BOLD_PROMPTS.values():
        mid = len(prompts) // 2
        for k, pr in enumerate(prompts):
            bold_group[pr] = "a" if k < mid else "b"

    out.append(
        Spec(
            "BenchmarkRunner",
            "BenchmarkRunner.run_bold",
            lambda w: BenchmarkRunner(_WorldProxy(w, bold_group.get)).run_bold(),
            lambda r: r.overall_score,
            min_group_size=1,
        )
    )

    # HolisticBias: two descriptors per axis, the first group a's identity and
    # the second group b's, so an axis spread is exactly the a-versus-b gap.
    from vfairness.llm import _holisticbias_data as hb

    hb_group = {}
    for axis, descs in hb.DESCRIPTORS.items():
        for k, desc in enumerate(descs[:2]):
            for tmpl in hb.TEMPLATES:
                hb_group[tmpl.replace("{DESC}", desc)] = GROUPS[k]

    out.append(
        Spec(
            "BenchmarkRunner",
            "BenchmarkRunner.run_holistic_bias",
            lambda w: BenchmarkRunner(_WorldProxy(w, hb_group.get)).run_holistic_bias(
                descriptors_per_axis=2
            ),
            lambda r: r.overall_score,
            min_group_size=1,
        )
    )

    # ==================================================================
    # CoTFaithfulnessAnalyzer: scenario k pairs group a's k-th response (the
    # original) with group b's k-th (the demographic variant). A pair exists
    # only where both variants exist.
    # ==================================================================
    def _pairs(w):
        p = _finite_or_nan_by_group(w)
        n = min(len(p["a"]), len(p["b"]))
        pairs = []
        for k in range(n):
            va, vb = p["a"][k], p["b"][k]
            pairs.append(
                {
                    "original_output": _text(va) or "",
                    "variant_output": _text(vb) or "",
                    "original_cot": _cot(va),
                    "variant_cot": _cot(vb),
                    "demographic_cue": "gender",
                    "scenario_id": f"s{k}",
                }
            )
        return pairs

    def _batch_headline(r):
        return [r.faithfulness_score, r.silent_influence_rate]

    def _pair_headline(r):
        if not r.assessed:
            return REFUSED
        return {"output_changed": float(r.output_changed), "cot_similarity": r.cot_similarity}

    def _analyze_pair(w):
        pairs = _pairs(w)
        if not pairs:
            raise ValueError("no original/variant pair exists in this world")
        return CoTFaithfulnessAnalyzer().analyze_pair(**pairs[0])

    out.append(
        Spec(
            "CoTFaithfulnessAnalyzer",
            "CoTFaithfulnessAnalyzer.analyze_batch",
            lambda w: CoTFaithfulnessAnalyzer().analyze_batch(_pairs(w)),
            _batch_headline,
            min_group_size=1,
        )
    )
    out.append(
        Spec(
            "CoTFaithfulnessAnalyzer",
            "CoTFaithfulnessAnalyzer.analyze_pair",
            _analyze_pair,
            _pair_headline,
            min_group_size=1,
        )
    )

    # ==================================================================
    # DecodingTrustRunner: every dimension's prompts are answered by the
    # world's rows in order. Fairness compares two named demographics, which
    # are group a's and group b's people.
    # ==================================================================
    dt_demos = ("25-year-old Black woman", "25-year-old White woman")

    def _dt_group(prompt):
        for g, demo in zip(GROUPS, dt_demos):
            if demo in prompt:
                return g
        return None

    for dim, meth in DecodingTrustRunner._DIMENSION_METHODS.items():
        if dim == "fairness":
            out.append(
                Spec(
                    "DecodingTrustRunner",
                    f"DecodingTrustRunner.{meth}",
                    lambda w: DecodingTrustRunner(_WorldProxy(w, _dt_group)).run_fairness(
                        demographics=list(dt_demos), sample_size=50
                    ),
                    lambda r: r.overall_score,
                    min_group_size=1,
                )
            )
            continue
        out.append(
            Spec(
                "DecodingTrustRunner",
                f"DecodingTrustRunner.{meth}",
                lambda w, m=meth: getattr(DecodingTrustRunner(_WorldProxy(w)), m)(sample_size=10),
                lambda r: r.overall_score,
                grouped_min=0,
            )
        )
    out.append(
        Spec(
            "DecodingTrustRunner",
            "DecodingTrustRunner.run_all",
            lambda w: DecodingTrustRunner(_WorldProxy(w)).run_all(sample_size=6),
            lambda r: {k: v.overall_score for k, v in r.items()},
            grouped_min=0,
        )
    )

    # ==================================================================
    # IntersectionalAnalyzer: the declared design is the two groups; each
    # cell holds the responses that group actually produced.
    # ==================================================================
    def _intersectional(w):
        groups = IntersectionalGroup.from_attributes({"group": list(GROUPS)})
        cells = {g: _present(t) for g, t in _texts_by_group(w).items()}
        outputs = {grp.label: cells[grp.attributes["group"]] for grp in groups}
        return IntersectionalAnalyzer().analyze(outputs, groups, metric="sentiment")

    out.append(
        Spec(
            "IntersectionalAnalyzer",
            "IntersectionalAnalyzer.analyze",
            _intersectional,
            lambda r: r.max_disparity,
            min_group_size=1,
        )
    )
    out.append(
        Spec(
            "IntersectionalAnalyzer",
            "IntersectionalAnalyzer.analyze[has_intersectional_bias]",
            _intersectional,
            lambda r: (
                REFUSED if r.has_intersectional_bias is None else float(r.has_intersectional_bias)
            ),
            # The pairwise rank-sum test needs two outputs a cell.
            min_group_size=2,
        )
    )

    # ==================================================================
    # NonDeterminismAnalyzer: the metric series is the decision per run; a run
    # with no decision carries no value.
    # ==================================================================
    nd = NonDeterminismAnalyzer

    out.append(
        Spec(
            "NonDeterminismAnalyzer",
            "NonDeterminismAnalyzer.characterize_noise",
            lambda w: nd().characterize_noise(np.asarray(w.p, float)),
            lambda r: r.noise_floor,
            grouped_min=0,
        )
    )
    out.append(
        Spec(
            "NonDeterminismAnalyzer",
            "NonDeterminismAnalyzer.bootstrap_ci",
            lambda w: nd().bootstrap_ci(np.asarray(w.p, float), n_bootstrap=200, random_state=0),
            lambda r: list(r),
            grouped_min=0,
        )
    )

    def _equivalence(w):
        f = _finite_by_group(w)
        return nd().equivalence_test(f["a"], f["b"])

    out.append(
        Spec(
            "NonDeterminismAnalyzer",
            "NonDeterminismAnalyzer.equivalence_test",
            _equivalence,
            lambda r: REFUSED if r["verdict"] == "could_not_check" else r["mean_diff"],
            min_group_size=2,
        )
    )

    def _disparity_and_floor(w):
        f = _finite_by_group(w)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                means = [float(np.mean(f[g])) if len(f[g]) else float("nan") for g in GROUPS]
                var = [
                    float(np.var(f[g], ddof=1)) / len(f[g]) if len(f[g]) > 1 else float("nan")
                    for g in GROUPS
                ]
        return means[0] - means[1], 2.0 * math.sqrt(sum(var)) if all(
            math.isfinite(v) for v in var
        ) else float("nan")

    out.append(
        Spec(
            "NonDeterminismAnalyzer",
            "NonDeterminismAnalyzer.compute_noise_offset",
            lambda w: nd().compute_noise_offset(*_disparity_and_floor(w)),
            lambda r: REFUSED if r["systematic_offset"] is None else r["systematic_offset"],
            min_group_size=2,
        )
    )
    out.append(
        Spec(
            "NonDeterminismAnalyzer",
            "NonDeterminismAnalyzer.is_significant_after_offset",
            lambda w: nd().is_significant_after_offset(*_disparity_and_floor(w)),
            lambda r: REFUSED if r is None else float(r),
            min_group_size=2,
        )
    )

    # ==================================================================
    # noise_floor_from_runs: one variant per group, its runs the group's
    # responses (None where none was produced).
    # ==================================================================
    def _nf_headline(r):
        if not r.get("available"):
            return REFUSED
        return {
            m: [c["observed"] for c in v["comparisons"] if c.get("state") == "measured"]
            for m, v in r["metrics"].items()
        }

    out.append(
        Spec(
            "noise_floor_from_runs",
            "noise_floor_from_runs",
            lambda w: noise_floor_from_runs(_texts_by_group(w), random_state=0),
            _nf_headline,
            # A floor needs two repeated responses per variant (n_valid < 2).
            min_group_size=2,
        )
    )

    # ==================================================================
    # judge_is_subject: the subject's identity is known only from the responses
    # it produced. A world in which no response was produced names no subject,
    # so there is nothing to compare the judge against.
    # ==================================================================
    def _judge(w):
        produced = bool(np.isfinite(np.asarray(w.p, float)).any())
        return judge_is_subject(
            "https://judge.invalid/v1/chat/completions",
            "judge-model",
            "https://subject.invalid/v1/chat/completions" if produced else None,
            "subject-model" if produced else None,
        )

    out.append(
        Spec(
            "judge_is_subject",
            "judge_is_subject",
            _judge,
            lambda r: REFUSED if r is None else float(r),
            grouped_min=0,
        )
    )
    return out


def _cot(v):
    v = float(v)
    if not math.isfinite(v):
        return ""
    return COT_APPROVE if v >= 0.5 else COT_REFUSE


def _finite_or_nan_by_group(w):
    p = np.asarray(w.p, float)
    return {g: p[w.s == g] for g in GROUPS}
