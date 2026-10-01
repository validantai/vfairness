# What grading means

This page explains, without jargon, what was actually done to this library and
what the words on the quality page mean. It is written for a reader who is
deciding whether to trust the library, not for the people who built it.

## The one defect everything here is about

A fairness library exists to answer questions like "does this model treat these
two groups the same". Sometimes it cannot answer. The data has one group in it.
Every score is missing. There are four relevant rows and no arrangement of them
could ever reach statistical significance.

The defect is what happens next. Instead of saying "I could not check this", the
code returns a neutral-looking number, and that number then gets graded,
counted, compared, ranked, plotted and put in a report as though it had been
measured.

A `0.0` that means "no association found" when the statistic was undefined.
A `False` that means "no bias detected" when nothing was detected at all.
An empty list that means "no proxies found" when the scan never ran.

Each of those reads as a clean bill of health. None of them is one.

**The fix is always three states, never two: measured, failed, or could not
check.** A caller has to be able to tell the third from the first without
reading the source code.

The reverse counts too, and it is worse. A function that refuses to answer when
it genuinely could have measured something throws away evidence. A detector that
refuses everything passes every test we could write about undefined input and
finds nothing real. Both directions are checked.

## What "grading" is, step by step

Grading one piece of code means doing all of this to it:

1. **Read how it is really called.** Its callers and the existing tests are the
   source for a realistic fixture. A test written against an unrealistic call
   proves nothing about the library anyone actually uses.

2. **Check the thresholds in its signature first.** Three times in this campaign
   a function looked honest when the test data was simply below a minimum group
   size it declares in its own parameters. The refusal came from the fixture,
   not from the code being examined.

3. **Run it on healthy data carrying a real, findable difference**, and confirm
   the measurement is exact. The expected value is recomputed independently in
   the test rather than copied from what the code returned, because copying the
   output only proves the code agrees with itself.

4. **Run it on data where the thing it claims to measure does not exist.** One
   group only. One outcome only. Every score missing. Zero rows. Every score
   identical. Two rows. One group with a single row. Text in an alphabet the
   code cannot read. Only the cases where its own claim becomes undefined.

5. **Run it with warnings switched on.** A refusal carried only in a warning is
   invisible to anyone who suppresses warnings, and most callers do.

6. **If it fabricates, fix it**, and then trace every caller of what changed.
   This is where the defect escapes: a caller comparing a not-a-number against a
   threshold silently gets "False", and the fabrication reappears one layer up
   wearing a different shape.

7. **Write a test that pins the behaviour**, with a control asserting that real
   data is still measured exactly.

8. **Put the bug back on purpose and confirm the test fails.** A test that
   cannot fail looks exactly like a test that passed. This step has caught more
   false confidence than every other step combined.

9. **Hand the whole claim to a second, independent agent whose instruction is to
   prove it wrong**, with access to run anything.

Step 9 is not a formality. **291 of 872 grades were overturned at that step**,
roughly one in three. Every one was a claim made by someone who had already run the
code, written a test and sabotaged it, and it still did not survive an independent
check.

That figure is a union over every wave file, including rows a later wave replaced,
and it has to be. Computed from the MERGED view it read 35, because re-grading a
row deletes the record that its first grade was wrong, so the number reporting the
method's fallibility fell by a factor of three as the campaign got more thorough.
The live merged figure is deliberately NOT quoted here: a reader who met it beside
the union would take it for a second opinion on the same question, and
tests/test_grading_explanation_is_current.py refuses this page if it appears.
The live counts are generated in
[QUALITY_AND_HARDENING.md](QUALITY_AND_HARDENING.md); the ones in this paragraph
are stated here because a reader needs them beside the step they judge.

## The five states

Every graded piece of code gets exactly one.

| State | What it means |
| --- | --- |
| **Proven** | It refuses honestly when nothing is measurable, it measures correctly when something is, and a test holds it there that fails when the fix is removed. |
| **Semi-proven** | It behaves honestly today, confirmed by running it, but no test fails if that behaviour regresses tomorrow. |
| **Unproven** | No fixture could reach it. This is a positive statement that nothing is known, not a clean bill. |
| **Defect open** | It was proved, by running it, to report a value nobody measured. Not yet fixed. |
| **Not a measurement** | It produces no fairness number and no verdict, so the defect cannot apply to it. A renderer, a converter, an accessor. |

**"Not a measurement" is the dangerous one**, because it is the cheapest to
assign and it closes the question. So it carries a rule: it may only be given
after running the code and stating what it returns instead. Never from the name,
never from the type, never from a guess.

That rule exists because the guess was wrong repeatedly. Across two waves,
**26 of 119 such dismissals were overturned**, and most of them were `to_dict`
methods. It read 27 of 120 until 2026-09-27, when one of the 27 was retired: the
row graded a callable that does not exist, `FairnessMonitor.get_metric_summary`,
and the auditor who overturned it said so in its own note and recorded a grade
anyway. A grade about nothing is not a dismissal that was rescued. A `to_dict` looks like plumbing. It is the boundary where a careful
three-state value gets flattened for a caller, and where "could not check"
quietly becomes a number.

## Two different measurements, never added together

**The capability census** graded the 215 **capabilities**: the things this
library does that a user would name, like measuring calibration disparity or
detecting proxy variables. It used four grades and a four-pass method, and as of
2026-09-17 it is complete, with no open defects and nothing unproven. That is a
statement about 222 of the 1,558 code units on the public surface, and about
nothing else.

**Surface grading** grades **code units**: public functions, classes and methods
in the package, one at a time. It uses the five states above, because the surface
contains something capabilities never do: code that produces no verdict at all.

The counts are not typed here. They moved four times in nine days and the typed
ones went stale each time, in the document that defines the grades. They are
generated from the evidence in
[QUALITY_AND_HARDENING.md](QUALITY_AND_HARDENING.md), under "What has happened to
that scope since", with a row per wave.

What a reader should take from those counts: the open-defect column is where this
library's open defects are, not the capability census, which has none. A sentence
that reports the census total as the library's position is the defect described at
the top of this page, written in prose instead of in code.

A capability is not a fourth kind of code unit. It is a name that points at one
or more of them: the 215 capabilities resolve to 222 code units.

Adding the census total to the surface-grading total would produce a number that
neither method measured. The two are reported separately everywhere, on purpose.

## Why a grade is not a guarantee

Being graded **proven** means one specific thing: this code does not report a
number it did not measure, and a test holds it to that.

It does **not** mean the statistics are correct, that the method is the right
one for your problem, or that the test covers every input. Those are different
claims needing different evidence, and this library does not make them on the
strength of a grade.


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

## What is still ungraded, and why that is stated rather than hidden

671 of the 1,558 code units carried a grade and 887 did not, measured 2026-09-18.
Both halves of that have moved since, and the live figures are in the generated
block in [QUALITY_AND_HARDENING.md](QUALITY_AND_HARDENING.md) rather than restated
here. Grading one code unit by agent costs about 138,000 tokens, measured over 242
of them. The whole remaining surface at that rate is around 125 million tokens, which
is why a cheaper route is planned rather than promised in
[RELEASE_PLAN.md](RELEASE_PLAN.md).

So the honest position is that grading is incomplete, the count of what is not
yet graded is published beside the count of what is, and the readiness verdict
is produced by `scripts/release_gate.py`, which fails while anything is open.
Run on 2026-10-01 it reports **BETA READY**, all eight beta conditions met, and,
on the stricter 1.0 bar, **NOT READY**: part of the graded code still waits for a
second, independent examiner. An ungraded
code unit is not a clean one. It is the absence of a measurement, and it is
exactly where every defect this campaign found was living.
