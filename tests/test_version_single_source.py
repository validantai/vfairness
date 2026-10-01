"""Guard: there is exactly one source of version truth.

The build (hatchling dynamic version) reads the version from
src/vfairness/__init__.py::__version__. This test asserts the runtime
attribute and the installed distribution metadata agree, so the two can
never silently drift.
"""

import importlib.metadata

import pytest

import vfairness


def test_version_single_source():
    try:
        dist_version = importlib.metadata.version("vfairness")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            "vfairness is not installed as a distribution (running from a source "
            "tree). Single-source versioning is exercised by the build and by the "
            "installed-wheel CI job; there is no distribution metadata to compare "
            "against here."
        )
    assert vfairness.__version__ == dist_version
