# Can I use this capability in the beta?

One question, asked by every reader of this library, answered for every single
public function, class and method: **1,558 of them**, with no unit left silent.

This page is the contract for the status badge. The badge appears in the API
reference, on the status grid, in the docstring of every graded capability and in
the machine-readable ledger `docs/capability-status.json`. All of those are
generated from one measurement by `scripts/capability_status.py`, so they cannot
disagree with each other, and `--check` fails the build when they drift.

## The three states

| Badge | What it means | What you should do |
| --- | --- | --- |
| **CHECKED** | Graded and assessed. It was executed on healthy input and on input where the thing it measures does not exist. If it was found inventing a value, that was fixed. | Use it. Read the strength line: some rows are held by a test, some were only verified by running them once. |
| **FIX PENDING** | Graded and assessed, and a defect is still open. | Do not rely on it in the beta. The row says what kind of defect and what closing it takes. |
| **NOT CHECKED** | No evidence either way. Nobody has established that it refuses honestly, and nobody has established that it does not. We are blind here. | Treat its output as unverified. It may be perfectly correct; this state is the absence of a measurement, not a failed one. |

**NOT CHECKED is not a soft pass.** Two thirds of this surface is in that state
today, and saying so is the point of this page. A library that publishes 100%
beside a scope nobody can see is the exact defect this library exists to catch.

### The strength line inside CHECKED

CHECKED is not a correctness certificate for the number a function returns. It
says the *fabrication* class was examined. Each row carries which of these it is:

- **pinned**: refuses honestly when nothing is measurable, and a test fails if
  that regresses. (Internally `PROVEN`.)
- **verified, not pinned**: behaves honestly today, confirmed by running it, but
  no test holds it there, so a regression would be silent. (`SEMI-PROVEN`.)
- **no verdict to fabricate**: returns no fairness number and no verdict, so the
  defect class cannot apply. Assigned only after running it. (`NOT A MEASUREMENT`.)

### Why NOT CHECKED, per row

The reason travels with the badge, because the reasons are not equivalent:

- *never executed by the census or the probe*
- *could not be called without a hand-built fixture*, the probe could not reach it
- *called, but the call itself failed, so nothing was learned about it*, the
  largest group, and a statement about our fixture rather than about the code
- *executed on healthy and degenerate input; the evidence has not been judged yet*

## Where it stands, measured

<!-- STATUS:matrix:start -->
| Published status | Units | Share |
| --- | ---: | ---: |
| **CHECKED** | 1580 | 100.0% |
| **FIX PENDING** | 0 | 0.0% |
| **NOT CHECKED** | 0 | 0.0% |
| | **1580** | 100% |

| Scope | Units | Checked | Fix pending | Not checked | Not independently checked |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core, what the beta promises | 1398 | 1398 | 0 | 0 | 485 |
| Preview, documented, not certified | 182 | 182 | 0 | 0 | 123 |
| **Total** | **1580** | **1580** | **0** | **0** | **608** |

The release gate reports 618 not independently checked; this table reports 608. The gate counts graded rows and this table counts code units, and 10 of the gate's rows grade a re-export spelling of a unit already counted here.

*Not independently checked* counts grades one examiner reached and nobody has yet tried to refute. Of the grades that HAVE been independently checked, 291 of 872 (33.4%) were overturned, so treat those rows as claims with evidence behind them rather than settled results.

| Area | Tier | Checked | Fix pending | Not checked | Total | Not independently checked |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fairness metrics | core | 247 | 0 | 0 | 247 | 74 |
| Post-processing mitigation | core | 191 | 0 | 0 | 191 | 55 |
| In-processing mitigation | core | 175 | 0 | 0 | 175 | 44 |
| LLM fairness testing | core | 151 | 0 | 0 | 151 | 60 |
| Preprocessing mitigation | core | 103 | 0 | 0 | 103 | 36 |
| Explainability (XAI) | core | 82 | 0 | 0 | 82 | 54 |
| Reporting | core | 79 | 0 | 0 | 79 | 18 |
| CI/CD gates | core | 77 | 0 | 0 | 77 | 32 |
| Monitoring | core | 65 | 0 | 0 | 65 | 11 |
| Bias detection | core | 65 | 0 | 0 | 65 | 21 |
| Agent fairness testing | core | 61 | 0 | 0 | 61 | 21 |
| Report rendering | core | 57 | 0 | 0 | 57 | 39 |
| Top level | core | 45 | 0 | 0 | 45 | 20 |
| Operations (other) | preview | 130 | 0 | 0 | 130 | 84 |
| MCP tools | preview | 23 | 0 | 0 | 23 | 14 |
| Validity axis | preview | 11 | 0 | 0 | 11 | 11 |
| Networking | preview | 9 | 0 | 0 | 9 | 9 |
| Vision | preview | 6 | 0 | 0 | 6 | 2 |
| Legal | preview | 3 | 0 | 0 | 3 | 3 |

| Open defect, by what closing it takes | Units |
| --- | ---: |

| The public path (`vfairness.X`), 831 units | Units | Share |
| --- | ---: | ---: |
| CHECKED | 831 | 100.0% |
| FIX PENDING | 0 | 0.0% |
| NOT CHECKED | 0 | 0.0% |
<!-- STATUS:matrix:end -->

## What has to change before the beta, and what only has to be honest

Two different bars, deliberately. Publishing a beta does not require every unit
proved; it requires **nothing unexamined that a user is told to rely on**, and
**nothing open and quiet**.

| Bar | Scope | Requirement | State on 2026-09-25 |
| --- | --- | --- | --- |
| **100% accuracy** | Tier 1: the assessment path. Fairness metrics, bias detection, explainability. | Zero FIX PENDING. | **MET for FIX PENDING**: 0 of 390 Tier-1 units. Grading the Tier-1 blind pile is NOT complete: 167 units remain NOT CHECKED, each saying why. |
| **100% transparency** | All 1,558 units, every tier. | Every unit carries a badge wherever it is described, including the ones nobody has looked at. No unit absent from the grid. | **MET**: 1,558 of 1,558 rows, four drift guards, each sabotage-tested. |
| **Disclosed, not closed** | Tier 2 and 3 open defects. | Each appears in the grid as FIX PENDING with its kind. A known, disclosed, non-dangerous defect is allowed in a beta. An unknown one is not. | **PARTLY**: all 85 are published with their kind; none carries an assessed SEVERITY yet, which is why the gate still fails G4. |

### The three kinds of open defect, which cost different things

"95 open defects" read as 95 broken functions for a week. It was never that, and
the split is why the core path could be cleared in one pass:

- **fabrication open**: seen returning a value nobody measured, not fixed. These
  are the real ones and they block. **0 units**, down from 10. Every observed
  fabrication on the public surface is closed, each reproduced at the public entry
  point, pinned, and each cause sabotaged separately. Three of the last ten turned
  out not to reproduce at all: their records were stale, and they are pinned so
  that a stale record and a fixed defect stop looking alike.
- **fix not accepted**: the fabrication was fixed and an independent audit
  refused the claim, usually because the pin covered less than the fix did. Needs
  re-verification, often no code change. **25 units** (29 when first counted).
- **evidence insufficient**: no fabrication was ever seen; the pins do not cover
  enough of the input space to call it proven. Needs evidence, not a fix.
  **39 units** (56 when first counted).

Measured 2026-09-25, after thirty-five closures. Where the remaining 64 sit:
in-processing mitigation 18, preprocessing mitigation 15, LLM testing 14,
monitoring 12, post-processing mitigation 12, MCP tools 5, reporting 4,
operations 3, CI/CD gates 2. **The assessment path itself, fairness metrics,
bias detection and explainability, holds none.** Core is wider than that and does
carry open defects, which is stated where core is defined.

## The plan, and what it costs

Measured rates from this repository's own campaign: an agent grading a code unit
costs 69,000 tokens with audits batched, an audit alone 16,000, and **a program
running the code costs nothing**. Every phase below that can be a program is one.

| Phase | What happens | Cost | Closes | State |
| --- | --- | --- | --- | --- |
| **T-1** | One ledger over the whole surface, three states, fail closed. Badges generated into the API reference, the markdown reference, the status grid and the quality page. Drift tests so no surface can disagree with the ledger. | machine | transparency, all 1,558 | **DONE** |
| **T-2** | Raise probe reach: build library types from the annotation naming them; probe the 392 classes nothing had ever executed; probe explainability against a fitted model. | machine | part of NOT CHECKED | **DONE**, blind pile 1,020 to 660 |
| **T-3** | Generate honesty pins from probe evidence, then sabotage each generated pin mechanically and keep only the ones that turn red. | machine | *verified, not pinned* to *pinned* | NOT STARTED |
| **T-4** | Judge and fix the open fabrications, then the Tier 1 fix-pending rows. | agent, bounded | FIX PENDING in Tier 1 to 0 | **DONE for Tier 1**: 12 closed, each cause sabotaged |
| **T-5** | The explainability blind spot: 79 of 81 XAI units carried no grade at all. | machine + bounded agent | Tier 1 XAI | **PARTLY**: 36 of 81 checked, 0 open, 45 still blind |
| **T-6** | Cross-validate: every badge against the ledger, every ledger row against a rerun measurement, every doc surface against both. | machine | rot | **DONE**: 4 guards, each sabotage-tested |

### What T-2 measured, and the honest part of it

The blind pile fell from 1,020 to 660 for nothing in agent tokens, and the reason
each unit is still blind is published per row rather than as one number:

| Why a unit is still NOT CHECKED | Units |
| --- | ---: |
| called, but the call itself failed, so nothing was learned | 283 |
| executed on healthy and degenerate input, evidence not yet judged | 195 |
| could not be constructed without a hand-built fixture | 58 |
| could not be called without a hand-built fixture | 46 |
| the explainability probe could not reach it | 36 |
| executed, and it answered where nothing was measurable (suspected, unjudged) | 16 |

The largest group is a statement about OUR fixtures, not about the library: the
probe called those units with arguments they cannot accept, so their refusals say
nothing about whether they refuse honestly. Counting them as examined is what an
earlier version of the readiness gate did, and it reported the surface as 98.6%
examined. They stay in the blind pile and say which kind of blind they are.

### Both new probes ship a selftest that must catch a fabricator

`class_probe.py --selftest` and `xai_probe.py --selftest` build a fixture that
fabricates a neutral value in the shape this library actually ships it, and fail
unless the rule refuses to grade it. **Both selftests failed the first time they
ran, and both failures were in the rule rather than in the library**: the class
rule was blind to a fabrication one container deep (`self.metrics = {"gap": 0.0}`
read as a container storing its arguments), and the XAI rule called an honest
explainer suspect for answering three worlds where an attribution is perfectly
well defined. A rule that cannot fail looks exactly like a rule that passed.

**A program may never assign a grade it cannot support.** `scripts/grade_from_probe.py`
holds that line already: of 316 units whose evidence looked qualifying, 287 had
raised on the healthy world too, meaning the probe had called them wrong and
learned nothing. 29 qualified. The same rule governs every phase above.

## Which files own this

So a parallel contributor can avoid a collision:
`scripts/capability_status.py`, `scripts/class_probe.py`, `scripts/xai_probe.py`,
`scripts/stamp_status_badges.py`, `docs/CAPABILITY_STATUS.md`,
`docs/capability-status.json`, `docs/class-probe.json`, `docs/xai-probe.json`,
`docs/surface-grading-2026-09-25.json`, `docs/site/status/`, and the test files
named in the wave record.

The durable record of what changed and why is this page plus
`docs/surface-grading-2026-09-25.json`. Coordination that belongs to the
development process rather than to the library is kept out of this page on
purpose: it is published, and an internal register of who is editing what is
reconnaissance detail with no value to a reader of the library.

## The correction that changed the size of the problem

**The readiness gate was not counting the test suite, and that single omission made
the remaining work look about seven times larger than it is.**

B1 asks whether every public code unit "has been executed and its result recorded".
It counted three sources: the capability census, the grading waves, and the surface
probe. It never counted the 10,000-plus test suite that runs on every push.

| | before | corrected 2026-09-25 | after wave 4 |
| --- | ---: | ---: | ---: |
| Units executed and recorded | 920 | 1,474 | **1,564** |
| Never executed by anything | 644 | 94 | **0** |

The probe exists precisely because nobody had executed this code, and it was
credited for 459 units. The suite, which executes three times as many, was credited
for none, and neither was the class probe (which constructs every ungraded class on
nine worlds) nor the explainability probe. Only the FUNCTION probe counted, which
was an accident of the order the three were written in. 528 units were published as "never executed by the census or the probe"
while CI ran them every time it fired.

This is a measurement error in the UNFLATTERING direction, and it was expensive in
its own way: it put "close B1" at roughly 1,100 hand-built fixtures over 343 owners,
a seven-to-twelve-million-token programme, which is a bar nobody would ever reach.
The real figure was 94 units, and **grading wave 4 closed all 94 on 2026-09-25, so
B1 now passes.** Closing them was not bookkeeping: executing those units for the
first time found six live defects, including 10% ISO 42001 coverage reported for an
empty wizard, a cluster randomisation that returned a single-arm design in silence,
and a fabricated 0.0 in this library's own status API.

**A second correction sits underneath the first.** Of the units the coverage record
calls "untouched" (132 as measured on 2026-09-25, and the figure falls as the suite
grows), only 43 execute nowhere; the other **89 are class bodies**,
which run at import, before `pytest-cov` begins measuring. No amount of use can put
those lines in a coverage report, so 36 public enums read as "never executed" while
the suite used them throughout. The split falls exactly on the class/function
boundary, which is what confirms it: every unit it calls import-executed is a class,
and every unit it calls unexecuted is a function or a method. It is measured in a subprocess so coverage starts before the library is
imported, recorded under `untouched_split` in `docs/suite-coverage.json`, and
deliberately not counted towards B1, because executing a class body is not recording
a result. What closed those 36 was an assertion, not a coverage line.

The first version of that measurement ran in-process and reported **zero** units
executing at import, the exact opposite of the truth, because this script imports the
surface walker at module scope and that imports the library before coverage can
start. Coverage said so on stderr ("Module vfairness was previously imported, but not
measured") and the number was still almost published.

**What the suite evidence does and does not buy**, because the distinction is the
whole point of the three states:

- It supports **B1**: a unit whose body lines run under a green suite has been
  executed, and an assertion recorded the outcome.
- It does **not** support **CHECKED**. Executing a unit is not judging whether it
  refuses honestly when nothing is measurable; a unit called incidentally inside
  another test has run and nobody has asked it that question. Those units stay
  NOT CHECKED, with a true reason instead of a false one.

**It fails closed on the suite's own result.** `scripts/suite_coverage.py` records
the run's pass and fail counts, and refuses to contribute anything from a run that
was not green. The first record taken carried one failure and was rejected, which is
why B1 did not move until the suite was green. Lines executed on the way to a
failing assertion did not verify anything.

## The other correction: tests that had already graded the code

Fifteen renderers were published NOT CHECKED while an existing harness had been
grading them to the full bar all along. `tests/test_adapters_zero_and_empty.py`
drives each one through BOTH directions: an `EMPTY_CASES` entry lists the verdict
tokens that must not survive empty input, a `CONTROL_CASES` entry lists the tokens
that must still be drawn on healthy input, and around them it asserts the canvas
states NOT CHECKED, that every number is withheld, that the accessible description
leads with could-not-check, that the severity is never info or low, and that
`save_path` never writes a zero-byte file.

None of that was recorded in a grading wave, so none of it reached the ledger. The
lesson for the ledger, not for the library: **evidence that exists and is not
recorded reads exactly like evidence that does not exist.**

## Grading wave 4: what closing the last 94 units found

The 94 units were the whole of the B1 gap: 39 classes, 36 of them enums, and 55
functions and methods. They are recorded in `docs/surface-grading-2026-09-25b.json`,
which also carries the sabotage for every PROVEN row and states plainly, per row,
where a pin was not individually sabotaged. 17 rows are PROVEN, 42 SEMI-PROVEN and
36 NOT A MEASUREMENT.

Closing them was expected to be bookkeeping. It was not. **Six live defects, each one
a neutral value or a silent omission where a refusal belonged:**

| Where | What it published | Now |
| --- | --- | --- |
| `generate_iso42001_evidence_map` | **10.0% ISO 42001 coverage for an empty wizard.** Two controls were hardcoded `"partial"` with no evidence test of any kind, a third returned `"partial"` in the branch where no evidence existed, and every partial earned half credit. | The two the map cannot determine are `"manual"` and excluded from both sides of the fraction; absence of evidence is a gap. Empty wizard reports 0.0%. |
| `assign_clusters` | **A single-arm cluster randomisation, in silence.** `max(1, int(n * fraction))` had no upper clamp, so a stratum of one cluster always went to treatment: 400 rows, one cluster, `treatment_fraction=0.5`, result all-treatment with no control and no warning. Stratified, ten singleton strata at a requested 0.5 delivered 1.0. | Clamped so a control cluster survives where the count allows, with warnings for the strata that could not be randomised and for any single-arm result. A deliberate `fraction >= 1.0` rollout is honoured, and still warned about. |
| `_correlation_ratio_with_pvalue`, both copies | **`corr=0.0, pvalue=1.0` for a correlation that is 0/0.** eta is `ss_between / ss_total`; with zero variance there is nothing to divide. The 1.0 is a p-value for an ANOVA that cannot run. | `(nan, nan)` with a warning. Found through an inconsistency: the same input against a two-level attribute already returned nan, so one undefined quantity had two published answers depending on cardinality. This overturns a standing "legitimate" verdict, which is kept verbatim under `_retired` in the fabricated-verdict ledger. |
| `SurfaceStatus.share` | **A fabricated zero in this library's own status API.** `counts.get(state, 0) / total` over upper-case keys, while `__str__` prints them lower case, so `share("checked")` answered 0.0 about the 874 units it had just printed. | Case and separator insensitive, and an unrecognised state raises rather than answering. There is no honest number for the share of a state that does not exist. |
| `IntersectionalGateDecision.to_dict` | Six declared dataclass fields, five emitted: `summary_decision` was dropped, so a hierarchical gate decision written to JSON lost its summary arm with no key to show it had one. | Serialised, or explicitly `null`, which is a stated absence rather than a missing key. |
| `create_regularizer` | `RegularizerType.MUTUAL_INFORMATION` is a member of the public enum with no implementation, and the factory said `"Unknown regularizer type"`, which reads as "you mistyped it". | `NotImplementedError` saying which of the two it is, that nothing is being substituted, and naming HSIC as the kernel dependence penalty over the same relationship. The enum member stays so the gap stays visible. |

**What was found honest is pinned too**, because an unpinned honest behaviour is one
refactor from becoming a defect with every gate still green. The gate fails closed on a
metric it cannot compute and puts the reason in the PR comment verbatim; the report
card carries small-sample warnings to where the merge decision is made; the validator
says `disparity_not_measurable` in so many words; the test suite returns SKIPPED, not
PASSED, for a check it could not run; an empty store scores `None / not_assessed` with
the sentence "This is not a score of 100 and certifies nothing"; the correlation
heatmap leaves a blank stripe rather than a band of confident zeros; and the fairness
regulariser over one group keeps its penalty at 0.0 so the training loss is not
poisoned, while saying `measured=False`, `dependence_measure=nan` and
`not_assessed="fewer_than_two_groups"` everywhere else at once.

Seven of those behaviours were sabotaged one at a time to prove the new pins can fail.
One did not fire: sabotaging the empty-store health score left
`FairnessDashboard.get_explanation` green, because it derives its refusal from the
record count independently. Two independent paths is a good property, but it leaves
that pin's discrimination unproven, so the unit keeps SEMI-PROVEN rather than
borrowing its neighbours' evidence.

## Grading wave 5: checking the evidence rather than the code

Every graded row names a `test_file`, and that name is the evidence behind the row's
published badge. Nothing had ever checked whether the named test *executes the unit*.
`scripts/verify_grade_attribution.py` does: it runs each named file under coverage and
asks whether the unit's body lines were reached.

Across **415 graded rows** it confirmed 348 by line coverage, reported 40 as blind, and
found **26 rows whose named test does not execute them**. Each of those was a published
CHECKED badge with nothing behind it, which is the defect this library is audited for,
one level up and in the surface built to prevent it.

| What was wrong | How many | Why it happened |
| --- | ---: | --- |
| Lazy chart wrappers graded on the test of the function they wrap | 18 | Every chart is reachable by three names. `vfairness.plot_fairness_metrics` and the subpackage spelling are each their OWN two-line function (a deferred import so that importing `vfairness` does not pull in matplotlib, then a forward). Only the third, canonical name was ever executed, and the first is the one the documentation tells a user to call. |
| `score_batch` on the six keyword scorers, graded PROVEN with no sabotage | 6 | The named file never calls `score_batch`. The single-text `score` path was thoroughly pinned and the batch path beside it was not pinned at all, although every aggregate comparison in the LLM surface goes through it. |
| The drift explanation, resting on a transitive justification | 1 | Its row said "none: already pinned by [two files]". Those files pin the explainer; neither reaches this method. |
| Wave 4's own rows, from a silent per-module default | 11 | Five named a file with no test for them at all (two base-class methods whose tests exercised overriding subclasses, two dataclass serialisers inspected but never constructed, one worker entry point) and six named the wrong one of five files. |

**One defect came out of it.** `FramingScorer` returned **0.5**, the exact midpoint of
its [0, 1] scale, for empty and whitespace-only text, while a real text carrying no
framing marker two branches below correctly returned NaN. So "the doctor was competent
and kind" was refused and `""` was scored. The carve-out justified itself "on the same
ground as `KeywordSentimentScorer`'s empty-text carve-out", and that carve-out had been
withdrawn on 2026-09-17 by the argument that a midpoint is a measured value and not an
absence. The cross-reference outlived the thing it pointed at.

**What the check cannot see, stated because a check that hides its blind spots is worse
than no check.** 38 rows are class bodies, which execute once at import before coverage
starts, so no assertion can put their lines in a report; 2 are bodies excluded by
configuration, because `raise NotImplementedError` is in this project's
`exclude_lines`; and `# pragma: no cover` hides an entry point the same way. Each is
reported under its own heading and none is counted as a finding.

**Two of the corrections were to the checker, not the code.** It first tested
`token.endswith(".py")` on a raw reference that carries a `::` node-id suffix, so every
row naming a specific test rather than a bare file was dropped, and it examined 89 of
435 rows while still printing a verdict.
And it read the test files *mentioned inside* an honest "none needed for the body"
explanation as that row's claimed evidence, which made it accuse
`BaseFeatureTransformer.fit` of naming a test for an abstract `pass`. A verifier that
quietly checks a tenth of its input, or accuses the rows that were candid, is the thing
it exists to prevent.

**And one of the new pins was blind.** The drift pin searched the rendered report for
"COULD NOT CHECK", a phrase that occurs three times in it, so rewriting the summary to
"Drift NOT DETECTED: the data looks stable" left the test green. It now asserts on
`report.summary`, the line a reader sees first. Similarly, sabotaging all six
`score_batch` methods turned five red and left `ContextualStereotypeScorer` green,
because every text in the corpus scored 0.0 for it and a constant-zero batch matched
its honest output exactly; a scoring text was added and it then went red too.

## How this cannot rot

- `python scripts/capability_status.py --check` fails if the ledger disagrees with
  a fresh measurement of the surface, if a unit changed status, or if the row
  count drifts from the surface total. A unit cannot drop out of the denominator.
- `tests/test_capability_status_badges.py` fails if any capability documented on a
  published page carries no badge, or a badge the ledger does not support.
- The grid renders from the same JSON the badges are stamped from.
