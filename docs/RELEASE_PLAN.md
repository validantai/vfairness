# How this library gets published, and what it costs

This page says what is still wrong, what closing it requires, and roughly what
that costs. It is published because a readiness claim without a plan behind it
is an opinion, and this project has already shipped one of those.


**Why the denominator moved on 2026-09-18.** It reads 1,558 here and 1,546 in
earlier versions. The public surface did not grow. The measuring environment
changed: the optional `mcp` extra was installed, so the scan could import
`vfairness.mcp.server` and count the twelve public functions it ships. Before
that, the published surface silently omitted a shipped feature, which is a
number leaving out part of the whole. The extra is now in the CI install list
and `scripts/library_kpis.py` publishes which extras were importable beside
the count, because a surface figure with no record of its environment cannot
be reproduced.

**Correction, 2026-09-18: "examined" was overstated, and this is the third time.**
B1 read 1,217 of 1,558. It now reads 890. Nothing regressed; the definition was
wrong. A code unit counted as examined if the probe managed to CALL it, and 287
units raised on every input world including the healthy one, with errors like
`'numpy.ndarray' object has no attribute 'keys'` and `could not convert string
to float: 'a'`. The probe's generic argument binder had called them with
nonsense, so their refusals say nothing about whether they refuse honestly. A
call that produced only an argument error is not an examination, and those units
are back in the ungraded pile where they belong.

**What this cost the free grading pass.** Grading from probe evidence alone was
expected to close 316 code units at no cost. Measured against the corrected
rule, **29 qualify**. The other 287 are the same units described above. The rule
now requires the healthy run to have produced a value before any grade is
assigned, and that single clause is the difference between 316 and 29. Awarding
the 316 would have been this library's own defect, committed by the tool built
to prevent it: a grade asserted from evidence that does not support it, in the
flattering direction.

The 29 are graded SEMI-PROVEN and counted as unaudited, because no agent argued
with them. A script may never award PROVEN: that requires a test which fails
when the fix is removed, and no script can write one.

## Where it stands today

**Measured 2026-10-01: BETA READY.** All eight beta conditions are met: every one
of the 1,580 public code units executed and recorded, no open defect anywhere, the
capability census clear, the 34 capabilities with an outside reference giving the
same answer as it (B6), and no capability that measures inventing a number on
broken data (B7). The full 1.0 gate is NOT READY on one condition only, G2: about
600 examined code units still await a second, independent examiner. That moved out
of the beta bar on 2026-10-01 by the maintainer's decision, because B6 and B7
re-check every capability that returns a number by running it. What remains before
the 0.1.0 publish is release operations, listed in RELEASE_PIPELINE.md, not quality
work. Re-run `scripts/release_gate.py` rather than trusting the figures here.

The measurements below are the record of how it got there, kept as they were taken.

Two bars, both measured by `scripts/release_gate.py`, which exits non-zero while
anything is open. Measured 2026-09-25:

**Beta gate: NOT BETA READY.** Four of the six criteria now pass (B1, B1b, B3, B5).
**Measured again on 2026-09-27, after the BGL3 campaign. B2 PASSES for the first
time: 0 open defects, down from 64.** 342 units were examined by execution across 28
batches, 178 of them were fabricating, and all 178 are fixed with a sabotage-checked
pin. FIX PENDING in the capability ledger went from 63 to 0.

**Two criteria still fail, and both numbers moved in the unflattering direction as a
direct result of the work.** B4 is now 556 grades not independently checked, up from
207, because every one of the 342 new grades is recorded `audited: false`: they were
produced by one agent each and nobody has argued with them, and the measured overturn
rate here is roughly three in ten. Marking them audited would make the criterion pass
by assertion. B1 read 1,564 of 1,567 for part of that day, because the fixes added
three new public units and the suite-coverage record that would credit them predated
the campaign's 29 new test files; re-measuring coverage closed it and **B1 passes at
1,567 of 1,567**. B3 and B5 pass.

**Full release gate: NOT READY.** Four of five criteria fail: G1 (1,094 of 1,567
code units carry a grade, so several hundred remain ungraded, counted in the
generated block below), G2 (517
unaudited), G3 and G4 (both now report "cannot measure" rather than a count, because
the open-defect split has nothing left to classify). G5 passes, and it is the same
census as B5.

The figures from 2026-09-25 that this replaces: B1 passing at 1,564 of 1,564, B2 64,
B4 207, G1 817 of 1,564 with 747 ungraded, G2 207, G3 64, G4 64.

**What moved between 2026-09-18 and 2026-09-25, and in which direction.** The
figures above replace B1 890, B2 86, B3 9, B4 10, G1 671, G2 39, G3/G4 95.
Ten further Tier-1 defects closed the same day took G3 from 93 to 85 and B4 from
49 to 74: every closure adds an unaudited grade, because nobody has yet argued
with the wave that closed it.

* **examined 890 to 919, graded 671 to 684.** Machine work only: the probe's
  argument binder learned to build the library's own types from an annotation, so
  61 more units got a healthy call, and 10 more met the from-probe bar.
* **open defects 95 to 64, and observed fabrications 10 to ZERO.** Thirty-five were closed by execution and pinned, with each
  cause sabotaged separately: `fisher_exact_test` (every degenerate margin
  returned 1.0, which reads as "no association found") and
  `BiasAuditReport.to_svg` (an audit that assessed nothing rendered "OVERALL RISK
  0% MINIMAL"). Recorded in `docs/surface-grading-2026-09-25.json`.
* **unaudited 39 to 49, which is worse and is the honest direction.** Ten of the
  new grades came from probe evidence alone and nobody has argued with them. More
  evidence means more claims owing an independent check, and the gate says so
  rather than netting it off against the progress above.

**Two countings of the same evidence, found and fixed the same day.** The gate
printed 95 open under G3 and 86 under B2's split, because the split read the
2026-09-18 file directly while G3 read the merged view. Both now come from
`library_kpis.graded_items()`. 80 measuring-or-unclassified plus 5 non-measuring
is 85, and those add up on purpose.

**B1 was overstated and is now corrected in the script.** `release_gate.py` used to add the graded count to the probe's reached count and print 1,524 of 1,546, which is 98.6% examined. The two sets overlap by 307, so the sum double-counted them. It now takes the union, which read **890 of 1,558** or 78.1% when that correction landed on 2026-09-18. The overstatement was found by an independent documentation review, not by the gate's own tests, and a test now pins the union with an anti-vacuity clause that fails if the two sets ever stop overlapping.

**Corrected twice more on 2026-09-25, both times in the unflattering direction.** The union never counted the test suite, the class probe or the explainability probe, which took B1 from 920 to 1,474 with no work done. And the comparison itself was a COUNT (`len(union) >= total`), which could have read PASS with real units unexamined, because the union is keyed by qualified name and some of those names are re-export spellings the surface walk does not count as units: 1,569 names against 1,564 units. B1 now names the units missing from the union and passes only when there are none.

The 64 open defects are in the wider surface, not in the capability census. None of them is a unit anybody has seen fabricate. The
census has none. Reporting the census total as the library's position would be
this library's own defect committed in prose, and it was: for two days the status
given here was "publishable, the 215 capabilities have zero open defects", which
was true and left the 95 out.

The full gate asks for every public code unit graded and no open defect
anywhere. At the measured cost that is tens of millions of tokens of machine
work, so a release held to it is a release that never ships. A bar nobody can
reach stops working as a bar. The beta gate is the smallest set of criteria
under which publishing is still honest.

## What a beta is allowed to be

A beta may ship with known defects. It may not ship with unknown ones, and it
may not be quiet about the known ones.

So the beta bar is:

1. Nothing on the public surface is unexamined.
2. Nothing that can hand a user a false clean bill is left open.
3. Everything still open is disclosed and ranked by severity.
4. Every grade was checked by someone trying to disprove it.
5. The 215 capabilities are complete and clear.

Criterion 2 is the one that decides the shape of the bar. A function that
returns a fairness number nobody measured can put a false clean bill in front of
a user. A renderer that returns an empty chart cannot. Both are defects; only
one blocks a beta. A code unit nobody has classified counts as the dangerous
kind until someone says otherwise, because not having worked out what something
returns is not evidence that it is harmless.

## What it costs, from what we actually spent

Measured across 420 graded code units:

| Approach | Tokens per code unit |
| --- | ---: |
| One agent to grade, one dedicated agent to audit | 138,000 |
| Grading, with audits batched eleven to an agent | 69,000 |
| Audit alone, batched | 16,000 |

**Batching the audits cut their cost by 81%.** Eleven audits in one agent
produce the same verdicts as eleven agents, because each audit is a short,
self-contained check and the expensive part is the agent's start-up context.
That single change is the difference between this plan being affordable and not.

<!-- BGL:remaining:start -->
_Generated 2026-10-02 from `scripts/release_gate.py`._

**0** of the **1580** public code units carry no grade.
1580 do: 222 from the capability census and 1374 from the wider-surface waves.

Executed is a separate and weaker measurement, and the two are never added: criterion B1 reads `1580 of 1580 executed and recorded`. Running a unit is not judging whether it refuses honestly.

**618** of the grades that do exist have had no independent check, and 291 of 872 grades were overturned when one was made. That queue, not the ungraded remainder, is the nearest thing to a blocker.
<!-- BGL:remaining:end -->

That figure is generated because it was typed here and rotted four times: 887 on
2026-09-18, 842 and then 747 on 2026-09-25 after grading wave 4, and twice more
within a few hours of 2026-09-27. The history is deliberately written without
repeating the current number, because a figure quoted as scenery satisfies the pin
that is supposed to hold the live one, and a check that cannot disagree is the
thing this whole plan is about. The surface total moved from 1,564 to 1,567 because
three of the campaign's fixes added a public helper. It moved by one
when a grade was retired rather than counted, because the row graded
`FairnessMonitor.get_metric_summary`, which does not exist. It then moved by 39 when
the probe-derived grades were withdrawn: every one of them claimed "measured on
healthy input" and the probe had recorded a refusal on the healthy world for all 39,
so none of them was.

The per-unit rates quoted below were measured on earlier waves and are left as they
were. What BGL3 actually cost is a better figure than either: 342 units examined by 28
agents working in parallel on module-disjoint batches, which is roughly 8.5 million
tokens of agent work, or about 25,000 tokens a unit including the fixes, the pins and
the sabotages. That is an order of magnitude under the 121-million-token projection
for the remainder above, and the reason is batching rather than anything clever: the
expensive part is an agent's start-up context, not the unit.

Note that "ungraded" and "never executed" are different questions, and only the
first is still open: B1 passes at 1,567 of 1,567, so every public unit has been run
and its result recorded, while 473 of them carry no grade. Running a unit is not
judging whether it refuses honestly, which is why the two criteria exist separately
and why the weaker one passing is not a reason to relax about the stronger one.
Grading wave 4 closed the last 94
of the previous total on 2026-09-25 and found six live defects doing it; BGL3 then
examined 342 and found 178, all recorded in `docs/CAPABILITY_STATUS.md` and in
`docs/surface-grading-2026-09-27.json`.

## The strategy: stop paying agent prices for machine work

An agent is needed to **judge** code. It is not needed to **run** it. Almost all
of the measured cost was an agent building a fixture and calling a function nine
times, which a program does for nothing.

**Three levers, in order of size.**

**1. Execute with a program, not with agents.** `scripts/surface_probe.py` calls
every ungraded code unit against nine worlds: one healthy world carrying a real,
findable difference, and eight where the thing being measured does not exist. It
records what came back, every warning and any exception. Cost in agent capacity:
zero. This was planned to close criterion 1 on its own. It does not, and the
measured reason is below: 167 code units cannot be called without a hand-built
fixture, so the probe says nothing about them.

**2. Spend judgement only where the probe raises a question.** A code unit that
refused honestly on all eight degenerate worlds needs no agent. One that handed
back a neutral number with no warning does. That second group was estimated at
roughly a third of code units, from the rate at which the earlier waves found
fabrication. The probe has since measured it: see below.

**3. Batch every audit.** Eleven to an agent, never one each.

The probe assigns no grades, deliberately. Its output is an observation:
"returned 0.0 on a single-group frame and emitted no warning". Turning that into
"this fabricates" is judgement, because for some functions 0.0 on that input is
the correct answer and refusing would destroy evidence. A script that graded its
own observations would be this library's defect one level up.

## The plan

| Phase | What happens | Machine cost | Closes | State |
| --- | --- | ---: | --- | --- |
| 0 | Probe executes every remaining code unit on nine worlds | none | part of B1 | RUN |
| 1 | A program sorts the output into refused, suspect, unreachable | none | feeds 2 | RUN |
| 1b | Probe the 392 CLASSES, which no probe had ever executed | none | part of B1 | RUN 2026-09-25 |
| 1c | Probe the explainability surface against a fitted model | none | part of B1 | RUN 2026-09-25 |
| 2 | Reviewers judge the remaining suspects, batched | ~3.4M | B4 | moved to 1.0 on 2026-10-01: second-examiner confirmation is now G2, a 1.0 condition |
| 3 | Fix the open defects that can produce a false clean bill | ~4.0M | B2 | 35 of 95 done, plus 6 more found and fixed by wave 4; Tier 1 at ZERO and every OBSERVED fabrication closed |
| 4 | Rank the remaining open defects by severity | 0 | B3 | **CLOSED**: the five non-measuring open defects were each executed, found honest and pinned, so there is nothing left to rank |
| 5 | Hand-build fixtures for the units the probe still cannot call | 5.5M to 9.5M estimated; **under 0.5M actual** | rest of B1 | **CLOSED 2026-09-25**: grading wave 4 executed all 94 in five test files and found six live defects |

**Beta total: about 9.3 million tokens** for what is left, down from 14.8 to 18.8,
because phase 5 closed for a small fraction of its estimate. The figure quoted for it
is an upper bound on one working session that also found and fixed six defects and
reconciled this page, so the fixture work alone was less. The estimate assumed
one hand-built fixture per unit at 40,000 to 69,000 tokens each. What the 94 actually
needed was five fixtures shared across families, plus reading the code: the 36 enums
took one parametrised contract test between them, and the serialisers were driven
through the objects their own public entry points already build. The lesson for the
remaining estimates on this page is that per-unit costing overstates any set of units
that share a construction path.

### The three machine phases added on 2026-09-25, and what each one cost

Nothing, in agent tokens. All three are programs, and together they moved 361
units out of "nobody has looked" without a single judgement call.

* **`scripts/class_probe.py`** executes every ungraded CLASS on the same nine
  worlds. 392 classes had never been probed at all, because `surface_probe.py`
  covers functions and methods only, and 167 of them are plain dataclasses whose
  whole job is to hold what somebody else measured. It settles 216 as NOT A
  MEASUREMENT and refuses to grade a constructor that derives a number from the
  data: that one goes to judgement. Its discriminator is not the class's type but
  whether the state it holds CHANGES with the data, because a dataclass can have
  a computing `__post_init__` and a class called `...Result` can run an analysis.
* **`scripts/xai_probe.py`** builds the fixture the explainability surface needs
  and nobody had built: a fitted model, a matrix, a background sample. 79 of 81
  XAI units carried no grade, and the reason was our fixtures, not their honesty.
* **The binder improvement** in `surface_probe.py`, which builds a library type
  from the annotation that names it. Healthy reach 398 to 459.

Both new probes ship a `--selftest` that builds a FABRICATING fixture and fails
unless the rule refuses to grade it. Both selftests failed the first time they
were run, and both failures were in the rule rather than in the library: the
class rule was blind to a fabrication one container deep, and the XAI rule
called an honest explainer suspect for answering three worlds where an
attribution is perfectly well defined.

### What the first full probe changed, in both directions

The probe has run, over **1,049 code units** on nine input worlds each. It
reached **882** of them, could not call **167** without a hand-built fixture, and
no individual call hung on that run. 416 of the 1,049 already carried a surface
grade and 633 did not, so the probe is not a second count of the ungraded set and
must not be read as one. Its full output is committed as
`docs/surface-probe.json`. Two of the plan's estimates were wrong, and they were
wrong the opposite way from each other.

**Better than estimated.** The suspect count was guessed at roughly a third of
the surface, around 300, from the rate at which earlier waves found fabrication.
Measured on that run: **100**. Phase 2 falls from 11 million tokens to 3.7
million. A suspect is a code unit that, on input where the thing it measures does
not exist, handed back a neutral value and emitted no warning at all.

**A correction to this page, made on 2026-09-18.** The suspect count was first
published as 79. The probe's own records give **100** code units that returned a
neutral value on a degenerate world and emitted no warning. The 79 came from an
extra condition nobody had stated: that the healthy run also had to return a
value. Twenty-one code units returned a neutral silent value on degenerate input
while raising on the healthy world, and they are suspects too. The unstated
filter shrank the published number in the flattering direction, which is the
defect this library was audited for, applied to its own plan. Found by an
independent review of this document, not by the person who wrote it.

**Worse than estimated.** The first plan assumed the probe would close criterion
B1 on its own. It does not. **167 code units could not be called without a
hand-built fixture**, and the probe therefore says nothing about them. They are
recorded as not reached, with the reason, and they count against B1 rather than
being quietly dropped. That is phase 5, and it was missing from the first plan
entirely.

The two roughly cancel. The total moves from 17 million to a range of 15.6 to
20.4 million, and the range is honest: the per-unit cost of building fixtures
for unreachable code has not been measured yet.

**One defect in the probe itself, worth recording.** Its first full run hung for
75 minutes on five seconds of processor time, holding a pipe: something it
called was waiting to read input that was never coming. A second run wrote a
132 kilobyte file into the repository root, because a renderer took a column
name from the argument binder as an output path. Arbitrary code does arbitrary
things, so the probe now closes its own input, gives every call a fifteen second
deadline, saves as it goes, and runs from a scratch directory it cannot escape.
With those in place the same sweep finished in under a minute.

## What is uncertain, stated rather than hidden

The suspect rate in phase 2 was first estimated from the waves so far, where 138
of 420 graded code units were found to be fabricating. That estimate has been
replaced by the probe's own measurement, recorded above, rather than quietly
kept. The judgement it feeds has not been spent yet, so whether the suspects turn
out to fabricate at the rate the waves did is still unmeasured.

The per-unit cost for probe-fed grading is an estimate. Fixture construction and
execution were the bulk of the measured 53,000, and the probe removes them, but
the first probe-fed batch is what settles it. It will be measured and published.

The per-unit cost of phase 5, hand-building fixtures for the 167 code units the
probe could not call, has not been measured at all. That is why phase 5 is
published as a range and why the beta total is a range rather than a figure.

## What this plan does not promise

Passing either gate means one specific thing: this library does not report a
number it did not measure, and tests hold it to that. It does **not** mean the
statistics are correct, that a method suits your problem, or that any test
covers every input. Those are different claims and they need different evidence.
