"""BGL7 PIN, batch NO-BATCH-b-w2 (2026-09-29): bias_detection/geographic_data.py.

``fetch_holc_data`` was graded "Every path refuses with None plus a warning that
names WHICH refusal it is, or raises. No path fabricates data or an availability
claim." Both halves were false: one path neither refused nor warned, and another
fabricated exactly an availability claim.

EVERY TEST HERE REPLACES ``requests.get`` IN PROCESS. No network call is made and
nothing is written.

ATTACK 3, an unvalidated 200 body returned as HOLC data. MEASURED BEFORE, each with
a 200 whose ``raise_for_status`` passes::

    {'message': 'Not Found'} -> returned {'message': 'Not Found'}, warnings []
    None                     -> returned None, warnings []   (collapsing into the
                                'not available' state, which is a different fact)
    {}                       -> returned {}, warnings []
    [1, 2, 3]                -> returned [1, 2, 3], warnings []  (violating the
                                declared Optional[Dict] return)
    'not json data'          -> returned 'not json data', warnings []

This function's own documented reader, ``len(data.get('areas', []))``, gives 0 for
the first three, so the example in its docstring prints "Found 0 HOLC areas" for a
city whose data was never read. A measured zero redlined areas is the most harmful
answer this module can give.

ATTACK 4, over-correction. MEASURED BEFORE::

    'Detroit_MI' in table: True | 'detroit_mi' in table: False
    fetch_holc_data('Detroit', 'MI') -> dict, warnings []
    fetch_holc_data('detroit', 'mi') -> None, "HOLC data not available for detroit, mi."

That warning asserts a fact about the world when the truth is a case-sensitive key
miss, and for a redlining library that direction is the harmful one.
"""

from __future__ import annotations

import warnings

import pytest

from vfairness.preprocessing.bias_detection import geographic_data as gd

_GOOD_BODY = {"type": "FeatureCollection", "features": [{"id": 1, "properties": {"grade": "D"}}]}


class _Response:
    """A 200 whose body is whatever the test hands it."""

    def __init__(self, body, raise_on_json: Exception | None = None):
        self._body = body
        self._raise_on_json = raise_on_json

    def raise_for_status(self) -> None:
        return None

    def json(self):
        if self._raise_on_json is not None:
            raise self._raise_on_json
        return self._body


@pytest.fixture
def served(monkeypatch):
    """Serve a body from ``requests.get`` without touching the network."""

    def _serve(body, raise_on_json: Exception | None = None):
        monkeypatch.setattr(
            gd.requests,
            "get",
            lambda *args, **kwargs: _Response(body, raise_on_json),
            raising=True,
        )

    return _serve


def _fetch(city: str = "Detroit", state: str = "MI"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = gd.fetch_holc_data(city, state)
    return out, [str(w.message) for w in caught]


# ===========================================================================
# ATTACK 3. A 200 is not a document.
# ===========================================================================


@pytest.mark.parametrize(
    "body",
    [
        {"message": "Not Found"},
        None,
        {},
        [1, 2, 3],
        "not json data",
        {"features": "not a list"},
        {"areas": None},
    ],
    ids=[
        "error-object",
        "null",
        "empty-object",
        "list",
        "bare-string",
        "features-not-a-list",
        "areas-null",
    ],
)
def test_a_200_that_is_not_a_holc_document_is_refused_and_named(body, served):
    """BEFORE: returned verbatim with ZERO warnings, so a failed fetch answering
    200 became HOLC data and read as 0 areas through the docstring's own example."""
    served(body)
    out, warned = _fetch()

    assert out is None, f"an unvalidated 200 body was returned as HOLC data: {out!r}"
    assert len(warned) == 1, warned
    assert "fetch_holc_data" in warned[0]
    assert "could-not-check" in warned[0], warned[0]
    # It must NOT read as a finding about the city.
    assert "NOT a finding that the city has no HOLC" in warned[0] or (
        "NOT a finding of zero HOLC areas" in warned[0]
    ), warned[0]


def test_the_refusal_names_what_actually_arrived(served):
    """A refusal a reader cannot act on is half a refusal. The type is named for a
    non-object body and the keys for an object that is not a HOLC document."""
    served([1, 2, 3])
    _out, warned = _fetch()
    assert "with a list, not a JSON object" in warned[0], warned[0]

    served({"message": "Not Found", "status": 404})
    _out, warned = _fetch()
    assert "Keys received: ['message', 'status']" in warned[0], warned[0]


def test_a_body_that_is_not_json_at_all_is_refused(served):
    """The SIBLING of the shape check: ``response.json()`` raising. requests wraps
    it in JSONDecodeError, which is both a RequestException and a ValueError, but a
    non-requests transport raises the plain one and it was not caught."""
    served(None, raise_on_json=ValueError("Expecting value: line 1 column 1"))
    out, warned = _fetch()

    assert out is None
    assert len(warned) == 1, warned
    assert "failed to fetch HOLC data" in warned[0], warned[0]
    assert "could-not-check" in warned[0]


def test_a_well_formed_document_with_no_areas_is_returned_and_warned(served):
    """THREE STATES, and this is the third. The document WAS read, so it is not a
    failed fetch and it is not refused; but a HOLC city with zero graded areas is
    the shape an error body imitates, so the zero is announced rather than left for
    a reader to infer from a silent empty list."""
    served({"type": "FeatureCollection", "features": []})
    out, warned = _fetch()

    assert out == {"type": "FeatureCollection", "features": []}
    assert len(warned) == 1, warned
    assert "ZERO areas" in warned[0], warned[0]
    assert "The document WAS read" in warned[0]


def test_control_a_real_holc_document_is_returned_verbatim_and_silently(served):
    """OVER-CORRECTION CONTROL. A validator that refused real HOLC data would pass
    every assertion above and make the module useless. The GeoJSON the endpoint
    actually serves comes back byte-for-byte with no warning, and the docstring's
    own reader gives its real count."""
    served(_GOOD_BODY)
    out, warned = _fetch()

    assert out == _GOOD_BODY
    assert out is not None and len(out.get("features", [])) == 1
    assert warned == [], warned


def test_control_the_other_documented_payload_key_is_accepted(served):
    """The docstring's example reads ``data.get('areas', [])`` while the endpoint
    serves ``features``. Both are HOLC documents; refusing either would break one of
    the two readers this module already has."""
    served({"areas": [{"grade": "D"}, {"grade": "C"}]})
    out, warned = _fetch()

    assert out is not None and len(out["areas"]) == 2
    assert warned == [], warned


# ===========================================================================
# ATTACK 4. The table is not the world.
# ===========================================================================


@pytest.mark.parametrize(
    ("city", "state"),
    [
        ("Detroit", "MI"),
        ("detroit", "mi"),
        ("DETROIT", "Mi"),
        (" Detroit ", " MI "),
        ("dEtRoIt", "mI"),
    ],
    ids=["canonical", "lower", "upper-mixed", "padded", "mixed"],
)
def test_a_case_or_spacing_mismatch_no_longer_denies_the_city(city, state, served):
    """BEFORE: ('detroit', 'mi') got "HOLC data not available for detroit, mi",
    a statement about a city with a 1939 HOLC map, from a case-sensitive dict miss.
    All five spellings must resolve to the same document, silently."""
    served(_GOOD_BODY)
    out, warned = _fetch(city, state)

    assert out == _GOOD_BODY, (city, state, out)
    assert warned == [], warned


def test_a_city_genuinely_absent_from_the_table_no_longer_asserts_a_fact(served):
    """The refusal is kept, and it now says what it IS. "HOLC data not available for
    X" is a claim about the world; this table is a curated subset of the Mapping
    Inequality project, so absence from it is a lookup result, not a finding."""
    served(_GOOD_BODY)
    out, warned = _fetch("Nowhere", "ZZ")

    assert out is None
    assert len(warned) == 1, warned
    assert "is not one of the" in warned[0], warned[0]
    assert "NOT a finding that the city has no HOLC record" in warned[0], warned[0]
    assert "curated subset" in warned[0]
    # And the old absolute claim is gone.
    assert "HOLC data not available for" not in warned[0], warned[0]


def test_multiword_and_punctuated_city_names_still_resolve(served):
    """The normalisation discards case and surrounding/repeated whitespace only.
    Three canonical keys depend on what it keeps: an internal space, a second
    internal space, and a full stop."""
    served(_GOOD_BODY)
    for city, state in (
        ("kansas city", "mo"),
        ("NEW YORK", "ny"),
        ("st. louis", "MO"),
        ("los  angeles", "ca"),
    ):
        out, warned = _fetch(city, state)
        assert out == _GOOD_BODY, (city, state, warned)
        assert warned == [], (city, state, warned)


def test_control_the_normalised_index_has_no_collisions():
    """A normalisation that mapped two canonical cities onto one key would silently
    fetch the wrong city's redlining map, which is worse than the defect it fixes."""
    assert len(gd._HOLC_CITY_INDEX) == len(gd.HOLC_AVAILABLE_CITIES)
    assert gd._HOLC_CITY_INDEX["kansas city_mo"] == "Kansas City_MO"
    assert gd._HOLC_CITY_INDEX["st. louis_mo"] == "St. Louis_MO"
    assert gd._HOLC_CITY_INDEX["new york_ny"] == "New York_NY"


def test_control_get_available_holc_cities_still_reports_canonical_spellings():
    """The public list is what a caller pastes back in, so it must keep the
    canonical spellings the table holds, not the normalised keys."""
    cities = gd.get_available_holc_cities()
    assert "Detroit_MI" in cities
    assert "detroit_mi" not in cities
    assert cities == sorted(gd.HOLC_AVAILABLE_CITIES)
