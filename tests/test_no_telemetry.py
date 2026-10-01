"""No-telemetry / no-network governance test (VB-GOV-2).

Core fairness metric computation must be a pure, offline operation: it must
not phone home, fetch remote resources, or open any socket. This test
neutralizes the socket layer so that ANY attempt to open a connection or
resolve a hostname raises, then runs a representative FairnessAnalyzer plus
the core metrics and asserts they complete without touching the network.

Note on import order: ``vfairness`` transitively imports ``requests`` (via
the optional LLM API proxy), and importing ``requests`` builds an
``ssl.SSLSocket`` subclass at import time, which needs the real
``socket.socket``. We therefore import ``vfairness`` FIRST and only then
monkeypatch the socket layer, so the block covers metric COMPUTATION (the
thing under test) rather than the one-time import.
"""

import socket

import numpy as np
import pytest

# Import the library BEFORE any socket patching (see module docstring).
import vfairness as v


class _NetworkAccessAttemptedError(AssertionError):
    """Raised if computation tries to open a socket or resolve a host."""


@pytest.fixture
def no_network(monkeypatch):
    """Make every socket / DNS operation fail loudly for the test body."""

    def _blocked_socket(*args, **kwargs):
        raise _NetworkAccessAttemptedError("socket.socket() called during metric computation")

    def _blocked_conn(*args, **kwargs):
        raise _NetworkAccessAttemptedError("socket.create_connection() called")

    def _blocked_dns(*args, **kwargs):
        raise _NetworkAccessAttemptedError("socket.getaddrinfo() (DNS) called")

    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_conn)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked_dns)
    return None


def _dataset(seed=0):
    rng = np.random.default_rng(seed)
    n = 120
    groups = np.array(["A"] * n + ["B"] * n)
    y_true = np.concatenate([(rng.random(n) < 0.5).astype(int), (rng.random(n) < 0.4).astype(int)])
    y_pred = np.concatenate([(rng.random(n) < 0.6).astype(int), (rng.random(n) < 0.3).astype(int)])
    y_prob = np.where(y_pred == 1, 0.8, 0.2)
    return y_true, y_pred, y_prob, groups


def test_sanity_socket_is_blocked(no_network):
    """Confirm the fixture actually neutralizes the socket layer."""
    with pytest.raises(_NetworkAccessAttemptedError):
        socket.socket()
    with pytest.raises(_NetworkAccessAttemptedError):
        socket.getaddrinfo("example.com", 80)


def test_core_metrics_make_no_network_calls(no_network):
    y_true, y_pred, y_prob, groups = _dataset(seed=1)

    dp = v.demographic_parity_difference(y_true, y_pred, groups)
    di = v.demographic_parity_ratio(y_true, y_pred, groups)
    eod = v.equalized_odds_difference(y_true, y_pred, groups)
    ece = v.expected_calibration_error(y_true, y_prob).overall_value

    assert not np.isnan(dp)
    assert not np.isnan(di)
    assert not np.isnan(eod)
    assert not np.isnan(ece)


def test_fairness_analyzer_makes_no_network_calls(no_network):
    y_true, y_pred, y_prob, groups = _dataset(seed=2)

    analyzer = v.FairnessAnalyzer(y_true, y_pred, groups, y_prob=y_prob)
    results = analyzer.compute_all_metrics(include_ci=False)

    # Computation completed offline and produced a non-empty result.
    assert results
    assert isinstance(results, dict)


def test_seeded_randomized_metrics_make_no_network_calls(no_network):
    """Bootstrap CI and permutation tests use RNG, not the network."""
    y_true, y_pred, y_prob, groups = _dataset(seed=3)

    ci = v.demographic_parity_difference_with_ci(
        y_true, y_pred, groups, n_bootstrap=200, random_state=17
    )
    perm = v.permutation_test_demographic_parity(
        y_pred, groups, n_permutations=200, random_state=17
    )

    assert not np.isnan(ci.point_estimate)
    assert 0.0 <= perm.p_value <= 1.0
