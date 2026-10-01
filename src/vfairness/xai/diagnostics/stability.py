"""
Stability diagnostics. Reference: Alvarez-Melis & Jaakkola (2018) ICML
WHI Workshop; Yeh et al. (2019) NeurIPS. Smaller sigma is better.
"""

from __future__ import annotations

import warnings

import numpy as np


def attribution_stability(reruns: list[np.ndarray]) -> float:
    """Mean per-feature standard deviation across repeated explainer runs.

    Each entry in ``reruns`` is a ``(n_features,)`` attribution vector
    from one run (different seed). Returns the average sigma; useful as
    a unit-free fidelity check on perturbation-based methods.
    """
    if len(reruns) < 2:
        # 0.0 sigma is PERFECTLY stable, the best score on this scale, and it
        # was what a caller got for supplying nothing to compare. Stability is
        # variation ACROSS runs; one run has none to measure.
        warnings.warn(
            f"attribution_stability: {len(reruns)} rerun(s) supplied, and stability is "
            f"variation ACROSS runs, so nothing was measured. Returning nan, not 0.0, "
            f"which would read as perfectly stable.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    stack = np.vstack(reruns)
    return float(stack.std(axis=0).mean())
