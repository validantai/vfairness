# vfairness Beta Programme

vfairness is heading for its first public beta on PyPI, versioned `0.1.0` with
the `Development Status :: 4 - Beta` classifier. A plain SemVer 0.x release
installs with a normal `pip install`, signals beta maturity through the
classifier, and is honest about stability: the API is not frozen until `1.0.0`.
Beta means: ready for real users to rely on, while we keep improving it. It is
free and open source under Apache-2.0, and it stays that way.

**The beta gate passes; nothing is published yet.** `scripts/release_gate.py`
is what decides it, and on 2026-10-01 it reports **BETA READY**: all eight beta
conditions are met and no known defect is open. The stricter 1.0 gate is not met:
a second, independent examiner still has to confirm part of the graded code.
Nothing has been published: there is no `v0.1.0` tag and no artifact on PyPI or
TestPyPI.

## What "done enough for beta" means (exit criteria)

The beta is a checklist, not a date, and the honesty half of that checklist is now
a program rather than prose: `python scripts/release_gate.py` computes it and
exits non-zero while anything is open. It supersedes the insufficient-evidence
criterion this page used to state in words, which is kept further down as the
record of what was asked for before there was a gate.

### The gate

The verdict is generated on every docs build rather than typed, because this table
was typed once, dated 2026-09-18, and every figure in it was wrong within nine days:
B1 passed, B2 went to zero and back up on an audit, B4 passed 500 and came back
down. Reproduce it with `python scripts/release_gate.py`, which is the authority.

<!-- BGL:gate:start -->
_Generated 2026-10-01 by `scripts/release_gate.py`._

**Beta bar: READY.** 8 of 8 criteria pass.

| | Criterion | Measured | Verdict |
| --- | --- | --- | --- |
| B1 | Every public code unit has been EXECUTED and its result recorded | 1580 of 1580 executed and recorded | passes |
| B1b | Every public code unit's examination state is PUBLISHED where it is described | 1580 of 1580 carry a published state | passes |
| B2 | No open defect in code that returns a fairness number or verdict | 0 open (0 measuring, 0 unclassified) | passes |
| B2b | No open defect recorded by the second-round audit | 0 open of 59 recorded claims | passes |
| B3 | Every remaining open defect carries an assessed severity | 0 non-measuring open, severity unassessed | passes |
| B5 | Capability census complete and clear | see G5 | passes |
| B6 | Same answer as an established library, where one exists | 34 of 34 agree; 76 have no outside reference | passes |
| B7 | No invented number on broken data | 109 of 109 measuring capabilities pass; 0 fabricate, 0 over-refuse, 0 not covered | passes |

**Full release bar: NOT READY.** 4 of 5 criteria pass.

| | Criterion | Measured | Verdict |
| --- | --- | --- | --- |
| G1 | Every public code unit carries a grade | 1580 of 1580 | passes |
| G2 | Every grade was confirmed by an independent check | 618 unaudited | **fails** |
| G3 | No open defect anywhere in the public surface | 0 open | passes |
| G4 | Every open defect has an assessed severity | 0 unassessed | passes |
| G5 | Capability census complete and clear | 0 open, 0 unproven | passes |

Reproduce with `python scripts/release_gate.py`. Every figure above is read from the recorded evidence, so a criterion cannot pass by assertion.
<!-- BGL:gate:end -->

The snapshot that used to sit here, kept as the record of what was asked for before
the block existed: measured 2026-09-18, B1 890 of 1,558 examined FAIL, B2 86 open
(49 measuring, 37 unclassified) FAIL, B3 9 non-measuring open with severity
unassessed FAIL, B4 39 unaudited FAIL, B5 0 open and 0 unproven PASS.

**B1's figure is not a count of distinct code units, and it flatters the
library.** `scripts/release_gate.py` counts a code unit as examined when a
grading wave graded it or when the probe reached it, then **adds** those two
counts instead of taking their union. They overlap: measured 2026-09-18, 307 of
the 420 wave-graded items are also in the probe's reached list, so the sum counts
them twice. The union of the 222 census-graded objects, the 420 wave-graded items
and the 882 the probe reached is **890 distinct code units of 1,558**. The
script used to print 1,524, the sum rather than the union, which double-counted
the 307 in both sets and reported the surface as better examined than it is. It
was corrected on 2026-09-18 and a test pins the union. B1 fails on either
figure.

B5 is the criterion that passes, and on its own it was never sufficient to call
the library ready. It covers the 215 registered capabilities, which resolve to 222
of the code units on the public surface. **The open defects are in the rest**, and
the live count is in the block above rather than repeated here: it has moved to zero
and back up twice since this paragraph was written, once because the defects were
fixed and once because an independent audit proved a batch of grades wrong. The
split matters more than the total: a defect in a unit that returns a fairness number
or a result can hand a reader a false clean bill, one that returns neither cannot,
and the gate counts an UNCLASSIFIED return with the first group rather than the
second, because not having looked is not evidence of harmlessness.

For two days the status given for this library was "publishable, the 215
capabilities have zero open defects". That sentence was true and it left the whole
wider surface out. Scoping a claim to the flattering subset and presenting it as the whole is
the defect this library is audited for, committed in prose instead of in code.

The two measurements are never added together. The registered capabilities and the
surface-graded code units count different things under different methods, and a
total of them is a number neither produced. [GRADING.md](GRADING.md) explains
both.

### The criteria the gate does not compute

These are conditions on the release rather than on the honesty of the code, so the
gate says nothing about them. They are still required before a publish:

- all P0 (blocker) issues on the roadmap are closed;
- correctness is proven: agreement with reference libraries documented, known-
  answer (synthetic-oracle) checks, impossibility-relationship checks, and a
  mutation-testing baseline on the metric core;
- the supply-chain gates are green: trusted publishing, provenance, dependency
  and code scanning, SBOM;
- the code-quality gates are green: `ruff` lint and format, `mypy src` (with a
  `mypy --strict` island on the foundational core), and branch coverage with a
  new-code diff-cover gate;
- the built wheel installs into a clean environment and its public API runs (a
  clean-room smoke test) before any publish;
- the methodology is versioned and stamped into every report;
- the documented examples are executed in the test suite.

Most of these are met. The [Quality and Hardening](QUALITY_AND_HARDENING.md) page
is the full inventory, and its "What is still open" section states the ones that
are not.

### The criterion this page used to state in prose

> no code path silently passes or drops a small group (an explicit
> insufficient-evidence verdict is returned instead).

That sentence is now measured by B2 and B3 above, and its history is why the gate
exists at all. The Beta Go-Live census of 2026-09-11 (commit `8559dda`) executed
all 215 registered capabilities on input where nothing they claim to measure
exists, and **124 of them returned a confident value anyway**, each reproduced at
the public API by a third independent pass. The census reproduced **157 defects**
at the public API across those capabilities, and all 157 are now fixed: as of
2026-09-17 it carries **0 open defects and 0 unproven capabilities**, which is
what B5 reports. Every capability carries its own grade
in its own docstring, so no reader has to take this on trust:
[BETA_GO_LIVE_PLAN.md](BETA_GO_LIVE_PLAN.md).

Closing the census did not close the criterion. The same question was then put to
the wider surface, under a five-state method described in [GRADING.md](GRADING.md),
and that is where the wider surface's open defects are.

The remaining work is staged and costed in [RELEASE_PLAN.md](RELEASE_PLAN.md):
five phases, 15.6 to 20.4 million tokens of machine work to reach the beta bar.
The figure this page used to carry, about 100 engineer-hours, was the census
remediation estimate made on 2026-09-11; that work is done, and the estimate no
longer describes what is left.

This page said "the remaining gate is the first publish itself" until 2026-09-11.
That was written when the could-not-check campaign had fixed everything it had
found, and it was wrong in the specific way this library exists to catch: the
campaign had never measured most of the surface, and "nothing found" was being
read as "nothing there".

## How to join

1. Install the beta (it is not published yet; there is nothing on PyPI to
   install until the gate above passes and a tag is cut):

   ```bash
   pip install vfairness
   ```

2. Try it on your own data or one of the standard datasets, and tell us what
   worked and what did not (see "Giving feedback" below).

We are looking for a handful (about five) of external testers across different
use cases (classification, regression, ranking, LLM or agent audits).

## Giving feedback

Open a "Beta feedback" issue on GitHub (there is a template). Useful reports
include: what you tried, what you expected, what happened, your Python and
vfairness versions, and a minimal snippet if something broke. Bugs, confusing
errors, unclear docs, and missing-but-expected features are all welcome.

## What beta does NOT promise

A beta may ship with known defects. It may not ship with unknown ones, and it may
not be quiet about the known ones, which is why the count above is published at
full weight rather than summarised. Passing the gate would mean one specific
thing: this library does not report a number it did not measure, and tests hold it
to that. It would not mean the statistics are correct, that a method suits your
problem, or that any test covers every input. Those are different claims and they
need different evidence.

vfairness is a community project with no service-level agreement: no guaranteed
response time, no uptime commitment, and no legal or contractual assurance. APIs
may still change before the stable `1.0.0`. If you need managed, independent,
service-backed assurance, that lives on the validant.ai platform, not in this
library. If the library is useful to you, please consider sponsoring its
development (this funds maintenance; it does not buy private features).

## After beta

`1.0.0` is the first stable release with a formal backwards-compatibility
contract. Before then, the **frozen public surface** listed in
`docs/API_STABILITY.md` is already treated as stable: it changes only after a
deprecation warning, and removals are reserved for `1.0.0`. Experimental
(non-frozen) surface may still change on `0.x` minor bumps. Either way, every
change is announced in the changelog.
