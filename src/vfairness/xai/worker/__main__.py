"""Make ``python -m vfairness.xai.worker`` work, as runner.py has always said.

``runner.py`` has documented that invocation since it was written, and it never
worked: without this file Python answers

    No module named vfairness.xai.worker.__main__; 'vfairness.xai.worker' is a
    package and cannot be directly executed

The alternative was to edit the docstring down to
``python -m vfairness.xai.worker.runner``. Making the documented form real is the
better of the two: it is the invocation an operator would guess, it is the one
already written down, and a deployment script copied from that docstring keeps
working rather than needing a change on the host.

Deliberately nothing but the delegation. Any behaviour added here would be
invisible to ``python -m vfairness.xai.worker.runner``, which still resolves, and
two entry points that do different things is how they drift.
"""

from .runner import main

if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
