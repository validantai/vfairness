"""Executable documentation: run the examples in docs/examples.md (VB-DOC-3).

Keeps the documented snippets honest. Every ``>>>`` block in docs/examples.md
is executed by the standard doctest runner as part of the normal test suite
(no separate CI workflow), so a snippet that stops matching the real API fails
here.
"""

import doctest
import os

import vfairness


def _examples_path():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(vfairness.__file__))))
    return os.path.join(repo, "docs", "examples.md")


def test_examples_md_doctests_pass():
    path = _examples_path()
    assert os.path.exists(path), f"examples doc not found: {path}"
    result = doctest.testfile(path, module_relative=False, verbose=False)
    assert result.failed == 0, f"{result.failed} of {result.attempted} doctest examples failed"
    assert result.attempted > 0, "no doctest examples were found to run"
