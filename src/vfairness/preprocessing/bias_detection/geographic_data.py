"""
Geographic Discrimination Data Integration

This module provides integration with official databases for geographic
discrimination analysis, including historical redlining maps and contemporary
area-level disparity data.

Data Sources:
    1. HOLC Redlining Maps (1930s-1940s)
       - Source: University of Richmond's Mapping Inequality Project
       - URL: https://dsl.richmond.edu/panorama/redlining/
       - API: https://dsl.richmond.edu/panorama/redlining/data/
       - License: CC BY-NC-SA 4.0

    2. CDC Social Vulnerability Index (SVI)
       - Source: CDC/ATSDR
       - URL: https://www.atsdr.cdc.gov/placeandhealth/svi/
       - Data: Census tract-level vulnerability scores
       - Updated: Annually

    3. FFIEC HMDA Data
       - Source: Federal Financial Institutions Examination Council
       - URL: https://ffiec.cfpb.gov/
       - Data: Home mortgage lending patterns
       - Updated: Annually

    4. HUD Affirmatively Furthering Fair Housing (AFFH) Data
       - Source: Department of Housing and Urban Development
       - URL: https://hudgis-hud.opendata.arcgis.com/
       - Data: Segregation indices, opportunity mapping

Academic References:
    - Rothstein, R. (2017). The Color of Law: A Forgotten History of How Our
      Government Segregated America. Liveright Publishing.

    - Aaronson, D., Hartley, D., & Mazumder, B. (2021). The Effects of the 1930s
      HOLC "Redlining" Maps. American Economic Journal: Economic Policy.
      https://doi.org/10.1257/pol.20190414

    - Mitchell, B., & Franco, J. (2018). HOLC "Redlining" Maps: The Persistent
      Structure of Segregation and Economic Inequality. NCRC.

Note:
    This module provides utilities for fetching and integrating external data.
    Users are responsible for complying with the terms of use for each data source.
    Some APIs may require registration or have rate limits.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pandas as pd  # noqa: F401  # availability probe

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class HOLCGrade(Enum):
    """
    Home Owners' Loan Corporation (HOLC) neighborhood grades from 1930s-1940s.

    These grades were used to determine mortgage lending risk and contributed
    to systematic discrimination against minority neighborhoods.

    Reference:
        Mapping Inequality: Redlining in New Deal America
        https://dsl.richmond.edu/panorama/redlining/
    """

    A = "A"  # "Best" - Green - Predominantly white, affluent areas
    B = "B"  # "Still Desirable" - Blue - Generally white, middle-class
    C = "C"  # "Definitely Declining" - Yellow - Immigrant/mixed areas
    D = "D"  # "Hazardous" - Red - Predominantly Black/minority (redlined)
    UNKNOWN = "Unknown"


@dataclass
class HOLCAreaInfo:
    """
    Information about a geographic area's HOLC redlining classification.

    Attributes:
        city: City name
        state: State abbreviation
        holc_grade: The HOLC grade assigned (A, B, C, or D)
        holc_id: Unique identifier for the HOLC area
        area_description: Historical description from HOLC surveyors
        coordinates: Geographic coordinates (if available)
        contemporary_demographics: Modern demographic info (if available)
        persistent_effects: Evidence of lasting effects from redlining
    """

    city: str
    state: str
    holc_grade: HOLCGrade
    holc_id: Optional[str] = None
    area_description: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    contemporary_demographics: Optional[Dict[str, Any]] = None
    persistent_effects: Optional[Dict[str, Any]] = None


@dataclass
class GeographicRiskAssessment:
    """
    Risk assessment for geographic features based on historical discrimination data.

    Attributes:
        feature_name: Name of the geographic feature analyzed
        risk_score: Overall risk score (0-1) over the rows that resolved to a
            HOLC grade; NaN when none did (could not check, not "low")
        holc_coverage: Fraction (0-1) of values with HOLC data
        grade_distribution: Distribution of HOLC grades in the data
        affected_samples: Number of samples in high-risk areas
        disparate_impact_risk: Risk of disparate impact based on geography
        recommendations: Suggested actions
        data_sources_used: Which external data sources were consulted
    """

    feature_name: str
    risk_score: float
    holc_coverage: float
    grade_distribution: Dict[str, float]
    affected_samples: int
    # 'low', 'medium', 'high', 'critical' or 'unknown'. When fewer than half of
    # the rows resolved to a HOLC grade the verdict is qualified in place, e.g.
    # "critical (on 10.0% of rows with HOLC data; 900 of 1000 unresolved)";
    # the category is always the first word.
    disparate_impact_risk: str
    recommendations: List[str]
    data_sources_used: List[str] = field(default_factory=list)


# HOLC data sources

# Cities with available HOLC data from Mapping Inequality Project
HOLC_AVAILABLE_CITIES = {
    # Format: 'city_state': 'holc_city_id'
    "Atlanta_GA": "GAAtlanta1938",
    "Baltimore_MD": "MDBaltimore1937",
    "Birmingham_AL": "ALBirmingham1937",
    "Boston_MA": "MABoston1938",
    "Brooklyn_NY": "NYBrooklyn1938",
    "Buffalo_NY": "NYBuffalo1937",
    "Chicago_IL": "ILChicago1940",
    "Cincinnati_OH": "OHCincinnati1937",
    "Cleveland_OH": "OHCleveland1939",
    "Columbus_OH": "OHColumbus1936",
    "Dallas_TX": "TXDallas1937",
    "Denver_CO": "CODenver1938",
    "Detroit_MI": "MIDetroit1939",
    "Houston_TX": "TXHouston1937",
    "Indianapolis_IN": "INIndianapolis1937",
    "Jacksonville_FL": "FLJacksonville1937",
    "Kansas City_MO": "MOKansasCity1939",
    "Los Angeles_CA": "CALosAngeles1939",
    "Louisville_KY": "KYLouisville1937",
    "Memphis_TN": "TNMemphis1937",
    "Miami_FL": "FLMiami1937",
    "Milwaukee_WI": "WIMilwaukee1937",
    "Minneapolis_MN": "MNMinneapolis1937",
    "Newark_NJ": "NJNewark1939",
    "New Orleans_LA": "LANewOrleans1939",
    "New York_NY": "NYManhattan1938",
    "Oakland_CA": "CAOakland1937",
    "Philadelphia_PA": "PAPhiladelphia1937",
    "Phoenix_AZ": "AZPhoenix1937",
    "Pittsburgh_PA": "PAPittsburgh1937",
    "Portland_OR": "ORPortland1938",
    "Richmond_VA": "VARichmond1937",
    "St. Louis_MO": "MOStLouis1937",
    "San Antonio_TX": "TXSanAntonio1937",
    "San Diego_CA": "CASanDiego1937",
    "San Francisco_CA": "CASanFrancisco1937",
    "Seattle_WA": "WASeattle1936",
    "Tampa_FL": "FLTampa1937",
    "Washington_DC": "DCWashington1937",
}

# Mapping Inequality API base URL
MAPPING_INEQUALITY_API = "https://dsl.richmond.edu/panorama/redlining/data/"


def _normalise_city_key(city: Any, state: Any) -> str:
    """The lookup key for *city* / *state*, insensitive to case and spacing.

    BGL7 NO-BATCH-b-w2, 2026-09-29. ``fetch_holc_data`` built its key as
    ``f"{city}_{state}"`` and tested it against a table whose keys are
    Title_CASE, so a caller who typed ``('detroit', 'mi')`` was told "HOLC data
    not available for detroit, mi", a statement about the WORLD, when the truth
    was a case-sensitive dictionary miss. For a redlining library that direction
    is the harmful one: it answers "this city was never redlined" to a question
    about a city with a 1939 HOLC map. Measured before the fix::

        'Detroit_MI' in table: True | 'detroit_mi' in table: False
        fetch_holc_data('Detroit', 'MI')   -> dict, warnings []
        fetch_holc_data('detroit', 'mi')   -> None, "HOLC data not available for detroit, mi."
        fetch_holc_data(' Detroit ', ' MI ') -> None, same claim

    Internal spacing is collapsed but KEPT, because two canonical keys need it
    ("Kansas City_MO", "New York_NY"), and the punctuation in "St. Louis_MO" is
    kept for the same reason. Only case and surrounding/repeated whitespace are
    discarded, which no canonical key depends on (verified: the normalised index
    below has as many entries as the table).
    """
    city_part = " ".join(str(city).split()).casefold()
    state_part = " ".join(str(state).split()).casefold()
    return f"{city_part}_{state_part}"


#: Normalised lookup key -> the canonical ``HOLC_AVAILABLE_CITIES`` key. Built
#: once, so the resolution rule lives in one place and cannot drift from the
#: table it indexes.
_HOLC_CITY_INDEX: Dict[str, str] = {
    _normalise_city_key(*key.rsplit("_", 1)): key for key in HOLC_AVAILABLE_CITIES
}


#: JSON keys under which a Mapping Inequality document carries its areas. The
#: endpoint serves GeoJSON (``features``); this function's own documented reader
#: uses ``data.get('areas', [])``. Either counts as HOLC data; anything else is
#: not, whatever status code arrived with it.
_HOLC_AREA_KEYS = ("features", "areas")

# The keys a HOLC grade is spelled under, on the record itself or inside a GeoJSON
# Feature's ``properties``. Used only to tell a GRADED document apart from a
# well-shaped one whose grades could not be found, which is a could-not-check about
# the grades and not a reason to refuse the document.
_HOLC_GRADE_KEYS = frozenset({"grade", "holc_grade", "holc_grade_letter", "label"})


def _area_has_a_readable_grade(area: Mapping[Any, Any]) -> bool:
    """True when one HOLC grade key on this area (or its ``properties``) holds text.

    The value must be non-blank text. ``.get(key, default)`` does NOT fire when the
    key is PRESENT holding None, so the test is on the VALUE: a features list of
    ``{'properties': {'grade': None}}`` is exactly the shape that makes
    ``parse_holc_grade`` raise.
    """
    scopes: List[Mapping[Any, Any]] = [area]
    props = area.get("properties")
    if isinstance(props, Mapping):
        scopes.append(props)
    for scope in scopes:
        for key in _HOLC_GRADE_KEYS:
            value = scope.get(key)
            if isinstance(value, str) and value.strip():
                return True
    return False


def get_available_holc_cities() -> List[str]:
    """
    Get list of cities with available HOLC redlining data.

    Returns:
        List of city names in 'City_State' format

    Example:
        >>> cities = get_available_holc_cities()
        >>> print(cities[:5])
        ['Atlanta_GA', 'Baltimore_MD', 'Birmingham_AL', 'Boston_MA', 'Brooklyn_NY']
    """
    return sorted(HOLC_AVAILABLE_CITIES.keys())


def fetch_holc_data(city: str, state: str) -> Optional[Dict[str, Any]]:
    """
    Fetch HOLC redlining data for a specific city from the Mapping Inequality API.

    Args:
        city: City name (e.g., 'Chicago')
        state: State abbreviation (e.g., 'IL')

    Args are matched against ``HOLC_AVAILABLE_CITIES`` ignoring case and spacing,
    so ``('detroit', 'mi')`` resolves exactly as ``('Detroit', 'MI')`` does.

    Returns:
        Dictionary with HOLC data including area boundaries and grades,
        or None if data is not available

        ``None`` is always a COULD-NOT-CHECK and never a finding about the city.
        It carries a warning naming which refusal it is: the city is not in this
        module's table, the fetch failed, the endpoint answered 200 with something
        that is not a JSON object, or it answered 200 with an object carrying no
        ``features`` / ``areas`` list. A 200 alone is NOT a document: an error body
        served with one reads as zero HOLC areas through the example below, and a
        measured zero redlined areas is the most harmful answer this module can
        give. A well-formed document holding an EMPTY area list is returned, and
        warned about, because that zero was read rather than assumed.

    Raises:
        ImportError: If requests library is not installed
        ConnectionError: If API is unreachable

    Example:
        >>> data = fetch_holc_data('Detroit', 'MI')
        >>> if data:
        ...     print(f"Found {len(data.get('areas', []))} HOLC areas")

    Note:
        This function requires internet access and the 'requests' library.
        Data is provided by the University of Richmond's Mapping Inequality Project.
        Please cite appropriately and respect rate limits.

    Citation:
        Nelson, R. K., Winling, L., Marciano, R., Connolly, N., et al.
        "Mapping Inequality." American Panorama,
        ed. Robert K. Nelson and Edward L. Ayers.
        https://dsl.richmond.edu/panorama/redlining/
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "The 'requests' library is required for fetching HOLC data. "
            "Install it with: pip install requests"
        )

    # THE TABLE IS NOT THE WORLD. The warning used to read "HOLC data not
    # available for {city}, {state}", which asserts a fact about that city, and it
    # fired for a CASE MISMATCH against a Title_CASE dictionary. See
    # :func:`_normalise_city_key` for the measurement. The lookup is now
    # normalised, and the refusal says what it actually is: this city is not in
    # THIS TABLE, which is a curated subset of the Mapping Inequality project and
    # not a census of every city HOLC surveyed.
    city_key = _HOLC_CITY_INDEX.get(_normalise_city_key(city, state))
    if city_key is None:
        warnings.warn(
            f"fetch_holc_data: '{city}, {state}' is not one of the "
            f"{len(HOLC_AVAILABLE_CITIES)} cities in this module's HOLC_AVAILABLE_CITIES "
            f"table, so NOTHING WAS FETCHED and no HOLC data was read. This is a "
            f"could-not-check about a lookup, NOT a finding that the city has no HOLC "
            f"record: the table is a curated subset of the Mapping Inequality project. "
            f"Matching ignores case and spacing. Call get_available_holc_cities() for "
            f"the full list; it begins "
            f"{', '.join(get_available_holc_cities()[:10])}...",
            UserWarning,
            stacklevel=2,
        )
        return None

    holc_id = HOLC_AVAILABLE_CITIES[city_key]
    url = f"{MAPPING_INEQUALITY_API}{holc_id}.geojson"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        # ValueError as well as RequestException: a body that is not JSON at all
        # reaches this through requests.exceptions.JSONDecodeError, which is BOTH,
        # but a non-requests transport raises the plain one.
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        warnings.warn(
            f"fetch_holc_data: failed to fetch HOLC data for {city}, {state}: {e}. "
            f"Returning None: this is a could-not-check, not a finding that the city "
            f"has no HOLC record.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # A 200 IS NOT A DOCUMENT. BGL7 NO-BATCH-b-w2, 2026-09-29. Whatever came back
    # was returned unexamined, so a failed fetch that answers 200 became HOLC data.
    # Measured before this change, each with a 200 and raise_for_status passing::
    #   {'message': 'Not Found'} -> returned {'message': 'Not Found'}, warnings []
    #   None                     -> returned None, warnings []   (collapses into the
    #                               'not available' state, which is a different fact)
    #   {}                       -> returned {}, warnings []
    #   [1, 2, 3]                -> returned [1, 2, 3], warnings []  (violates the
    #                               declared Optional[Dict] return)
    #   'not json data'          -> returned 'not json data', warnings []
    # and this function's OWN documented reader, len(data.get('areas', [])), gives 0
    # for the first three, so the example in the docstring above prints "Found 0 HOLC
    # areas" for a city whose data was never read. A measured zero redlined areas is
    # the most harmful answer this module can give, so the shape is now checked and a
    # body that is not a HOLC document is REFUSED with a warning naming what arrived.
    if not isinstance(payload, dict):
        warnings.warn(
            f"fetch_holc_data: the endpoint answered 200 for {city}, {state} with a "
            f"{type(payload).__name__}, not a JSON object, so it is not the HOLC "
            f"document this function returns. Returning None. This is a "
            f"could-not-check: it is NOT a finding that the city has no HOLC areas, "
            f"and reading a count off such a body would report a measured zero for "
            f"data that was never read.",
            UserWarning,
            stacklevel=2,
        )
        return None
    areas = next(
        (payload[key] for key in _HOLC_AREA_KEYS if isinstance(payload.get(key), list)),
        None,
    )
    if areas is None:
        warnings.warn(
            f"fetch_holc_data: the endpoint answered 200 for {city}, {state} with a JSON "
            f"object carrying none of {list(_HOLC_AREA_KEYS)} as a list, so it is not a "
            f"HOLC document. Keys received: {sorted(str(k) for k in payload)[:10]}. "
            f"Returning None. This is a could-not-check, NOT a finding of zero HOLC "
            f"areas: an error body answered with a 200 reads as zero areas through this "
            f"function's own documented reader.",
            UserWarning,
            stacklevel=2,
        )
        return None
    if not areas:
        # WELL FORMED AND EMPTY is its own state, the third one: the document was
        # read and it carries no areas. That is a real zero, but it is a zero a
        # reader must be told about rather than infer from a silent empty list.
        warnings.warn(
            f"fetch_holc_data: the HOLC document for {city}, {state} is well formed and "
            f"carries ZERO areas. The document WAS read, so this is not a failed fetch, "
            f"but a HOLC city with no graded areas is unexpected: check the response "
            f"before reporting it as a measurement of no redlining.",
            UserWarning,
            stacklevel=2,
        )
        return payload

    # THE LIST'S CONTENTS, NOT ONLY ITS TYPE AND LENGTH. BGL-F5 2026-09-30, the
    # SIBLING of the guard above. `isinstance(payload.get(key), list)` plus a
    # non-empty test is a TYPE-AND-LENGTH check, and this function's stated
    # contract is that "a body that is not a HOLC document is REFUSED". The guard
    # closed the door that produced a measured ZERO from a body never read and left
    # open the door that produces a measured NON-ZERO count of HOLC graded areas
    # from a body never read, in a redlining module, in silence, without even the
    # disclosure the empty case gets. Measured before this change, all three
    # returning the document with ZERO warnings, counted through this function's
    # OWN documented reader len(data.get('areas', [])) / len(features)::
    #   {'features': ['Not Found']}                  -> returned dict, count 1
    #   {'features': [{'foo': 1}, {'bar': 2}, None]} -> returned dict, count 3
    #   {'areas': [0, 0, 0, 0, 0]}                   -> returned dict, count 5
    # Downstream, parse_holc_grade(None) and parse_holc_grade(1) raise
    # AttributeError ("'NoneType' object has no attribute 'upper'"), so such a list
    # does not merely grade as UNKNOWN, it crashes the documented next step.
    #
    # A HOLC area is a RECORD: a GeoJSON Feature or an area object, i.e. a mapping.
    # Anything else in the list is not an area, and one is enough to say the body
    # is not a HOLC document.
    non_records = [a for a in areas if not isinstance(a, Mapping)]
    if non_records:
        warnings.warn(
            f"fetch_holc_data: the endpoint answered 200 for {city}, {state} with a list "
            f"of {len(areas)} item(s) under a HOLC area key, but {len(non_records)} of "
            f"them are not records (types: "
            f"{sorted({type(a).__name__ for a in non_records})}; first: {non_records[0]!r}). "
            f"A HOLC area is a GeoJSON Feature or an area object, so this is not a HOLC "
            f"document and it is REFUSED with None. This is a could-not-check: returning "
            f"it would publish a measured count of {len(areas)} HOLC graded areas read "
            f"off a body that was never a document, and parse_holc_grade() raises "
            f"AttributeError on such an item rather than grading it UNKNOWN.",
            UserWarning,
            stacklevel=2,
        )
        return None
    # THE DOCUMENT IS A DOCUMENT, BUT IS IT GRADED? Refusing here would reject a
    # real response that spells its grade some other way, so this is a
    # could-not-check ABOUT THE GRADES, warned and returned, not a refusal.
    if not any(_area_has_a_readable_grade(a) for a in areas):
        warnings.warn(
            f"fetch_holc_data: the HOLC document for {city}, {state} carries "
            f"{len(areas)} area record(s) and NOT ONE of them holds a readable HOLC "
            f"grade under any of {sorted(_HOLC_GRADE_KEYS)}. The document is returned "
            f"because its shape is right, but a grade distribution taken from it would "
            f"be a could-not-check, not a measurement of no redlining: keys seen on the "
            f"first record are {sorted(str(k) for k in areas[0])[:10]}.",
            UserWarning,
            stacklevel=2,
        )
    return payload


def parse_holc_grade(grade_str: str) -> HOLCGrade:
    """
    Parse a HOLC grade string into an enum value.

    Args:
        grade_str: Grade string (e.g., 'A', 'B', 'C', 'D')

    Returns:
        HOLCGrade enum value
    """
    grade_map = {
        "A": HOLCGrade.A,
        "B": HOLCGrade.B,
        "C": HOLCGrade.C,
        "D": HOLCGrade.D,
    }
    return grade_map.get(grade_str.upper(), HOLCGrade.UNKNOWN)


def get_holc_risk_level(grade: HOLCGrade) -> str:
    """
    Map HOLC grade to risk level for bias detection.

    Grade D ("Hazardous") areas were redlined and their residents
    systematically denied financial services. Using data from these
    areas in predictive models risks perpetuating historical discrimination.

    Args:
        grade: HOLC grade enum value

    Returns:
        Risk level string: 'critical', 'high', 'medium', 'low', or 'unknown'
    """
    risk_map = {
        HOLCGrade.D: "critical",  # Redlined - highest discrimination risk
        HOLCGrade.C: "high",  # "Declining" - significant discrimination
        HOLCGrade.B: "medium",  # "Desirable" - some risk
        HOLCGrade.A: "low",  # "Best" - low discrimination risk
        HOLCGrade.UNKNOWN: "unknown",
    }
    return risk_map.get(grade, "unknown")


# ZIP code to HOLC mapping (approximate)

# Note: This is a simplified mapping. For production use, integrate with
# actual HOLC boundary data and perform proper spatial joins.

KNOWN_REDLINED_ZIP_PATTERNS = {
    # Detroit, MI - historically redlined areas
    "48201": HOLCGrade.D,  # Downtown/Corktown
    "48202": HOLCGrade.D,  # North End
    "48203": HOLCGrade.D,  # Highland Park
    "48204": HOLCGrade.D,  # Brightmoor
    "48205": HOLCGrade.D,  # East Side
    "48206": HOLCGrade.D,  # Dexter-Linwood
    "48207": HOLCGrade.D,  # Jefferson-Chalmers
    "48208": HOLCGrade.D,  # Southwest Detroit
    "48209": HOLCGrade.C,  # Southwest
    "48210": HOLCGrade.C,  # West Side
    "48211": HOLCGrade.D,  # Hamtramck area
    "48212": HOLCGrade.D,  # Hamtramck
    "48213": HOLCGrade.D,  # East Side
    "48214": HOLCGrade.D,  # East Side
    "48215": HOLCGrade.C,  # Grosse Pointe border
    "48216": HOLCGrade.C,  # Southwest
    "48217": HOLCGrade.D,  # Delray
    "48219": HOLCGrade.C,  # Brightmoor
    "48221": HOLCGrade.C,  # Palmer Park
    "48224": HOLCGrade.D,  # East Side
    "48226": HOLCGrade.D,  # Downtown
    "48227": HOLCGrade.D,  # West Side
    "48228": HOLCGrade.D,  # Southwest
    "48234": HOLCGrade.D,  # East Side
    "48235": HOLCGrade.C,  # Northwest
    "48238": HOLCGrade.D,  # Livernois-McNichols
    # Chicago, IL - historically redlined areas
    "60609": HOLCGrade.D,  # Back of the Yards
    "60615": HOLCGrade.D,  # Hyde Park South
    "60619": HOLCGrade.D,  # Chatham
    "60620": HOLCGrade.D,  # Auburn Gresham
    "60621": HOLCGrade.D,  # Englewood
    "60628": HOLCGrade.D,  # Roseland
    "60629": HOLCGrade.D,  # Chicago Lawn
    "60636": HOLCGrade.D,  # West Englewood
    "60637": HOLCGrade.D,  # Greater Grand Crossing
    "60643": HOLCGrade.D,  # Morgan Park
    "60644": HOLCGrade.D,  # Austin
    "60649": HOLCGrade.D,  # South Shore
    "60651": HOLCGrade.D,  # Humboldt Park
    "60653": HOLCGrade.D,  # North Kenwood
    # Baltimore, MD - historically redlined areas
    "21201": HOLCGrade.D,  # Downtown West
    "21202": HOLCGrade.D,  # Downtown
    "21205": HOLCGrade.D,  # Middle East
    "21213": HOLCGrade.D,  # Clifton-Berea
    "21215": HOLCGrade.D,  # Northwest
    "21216": HOLCGrade.D,  # Walbrook
    "21217": HOLCGrade.D,  # Penn North
    "21218": HOLCGrade.C,  # Charles Village
    "21223": HOLCGrade.D,  # Poppleton/Union Square
    "21229": HOLCGrade.D,  # Southwest
    # Atlanta, GA - historically redlined areas
    "30301": HOLCGrade.D,  # Downtown/Sweet Auburn
    "30303": HOLCGrade.D,  # Downtown
    "30310": HOLCGrade.D,  # West End
    "30311": HOLCGrade.D,  # Southwest
    "30314": HOLCGrade.D,  # Vine City
    "30318": HOLCGrade.D,  # West Midtown
    "30331": HOLCGrade.D,  # Southwest
}


def lookup_holc_grade_by_zip(zip_code: str) -> Optional[HOLCGrade]:
    """
    Look up approximate HOLC grade for a ZIP code.

    This is an approximate mapping based on known historical redlining patterns.
    For precise analysis, use the full HOLC boundary data with proper spatial joins.

    Args:
        zip_code: 5-digit ZIP code

    Returns:
        HOLCGrade if known, None otherwise

    Warning:
        This is an approximation. ZIP code boundaries don't align with 1930s
        HOLC boundaries. For accurate analysis, use actual HOLC geospatial data.
    """
    zip_code = str(zip_code).strip()[:5]
    return KNOWN_REDLINED_ZIP_PATTERNS.get(zip_code)


def assess_geographic_feature_risk(
    values: List[str],
    feature_type: str = "zip_code",
) -> GeographicRiskAssessment:
    """
    Assess the discrimination risk of a geographic feature based on
    historical redlining data.

    Args:
        values: List of geographic values (ZIP codes, etc.)
        feature_type: Type of geographic feature ('zip_code' currently supported)

    Returns:
        GeographicRiskAssessment with risk analysis

    Example:
        >>> zip_codes = ['48201', '48202', '90210', '10001']
        >>> assessment = assess_geographic_feature_risk(zip_codes)
        >>> print(f"Risk score: {assessment.risk_score:.2f}")
        >>> print(f"Disparate impact risk: {assessment.disparate_impact_risk}")
    """
    # audit-6 lane 2 (2026-09-09): the two could-not-check returns below used to
    # carry risk_score=0.0, the "no risk" value, for a risk nobody measured.
    # They are NaN now, with the verdict 'unknown' they already had.
    if feature_type != "zip_code":
        warnings.warn(
            f"assess_geographic_feature_risk: feature_type {feature_type!r} is not "
            "supported (only 'zip_code'); risk_score is NaN and the verdict 'unknown'."
        )
        return GeographicRiskAssessment(
            feature_name=feature_type,
            risk_score=float("nan"),
            holc_coverage=0.0,
            grade_distribution={},
            affected_samples=0,
            disparate_impact_risk="unknown",
            recommendations=[f"Geographic analysis not supported for '{feature_type}'"],
            data_sources_used=[],
        )

    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "Unknown": 0}

    for val in values:
        grade = lookup_holc_grade_by_zip(str(val))
        if grade:
            grade_counts[grade.value] += 1
        else:
            grade_counts["Unknown"] += 1

    total = len(values)
    if total == 0:
        warnings.warn(
            "assess_geographic_feature_risk: no geographic values were given; "
            "risk_score is NaN and the verdict 'unknown'."
        )
        return GeographicRiskAssessment(
            feature_name=feature_type,
            risk_score=float("nan"),
            holc_coverage=0.0,
            grade_distribution={},
            affected_samples=0,
            disparate_impact_risk="unknown",
            recommendations=["No geographic values to analyze"],
            data_sources_used=[],
        )

    known_count = sum(grade_counts[g] for g in ["A", "B", "C", "D"])
    holc_coverage = known_count / total
    unresolved = total - known_count

    grade_distribution = {g: count / total for g, count in grade_counts.items() if count > 0}

    # D = 1.0, C = 0.7, B = 0.3, A = 0.0
    grade_weights = {"D": 1.0, "C": 0.7, "B": 0.3, "A": 0.0}
    # audit-6 lane 2 (2026-09-09): the score used to divide by `total`, which
    # INCLUDES the "Unknown" rows, so every ZIP the HOLC table could not resolve
    # entered the average with the grade-A weight of 0. Measured: 100 known
    # grade-D rows -> 1.000 'critical'; the same 100 rows plus 900 unresolved ->
    # 0.100 'medium'; 50 unresolved rows alone -> 0.000 'low'. An unresolved ZIP
    # is could-not-check, not grade A: the score is the mean over the RESOLVED
    # rows only, NaN when there are none, and a verdict that rests on a minority
    # of rows says so in place instead of being downgraded.
    if known_count == 0:
        risk_score = float("nan")
        base_risk = "unknown"
        warnings.warn(
            f"assess_geographic_feature_risk: none of the {total} value(s) resolved to a "
            "HOLC grade; risk_score is NaN and the verdict is 'unknown', not 'low'."
        )
    else:
        risk_score = sum(grade_counts[g] * w for g, w in grade_weights.items()) / known_count
        if risk_score >= 0.5:
            base_risk = "critical"
        elif risk_score >= 0.3:
            base_risk = "high"
        elif risk_score >= 0.1:
            base_risk = "medium"
        else:
            base_risk = "low"

    # Count affected samples (C or D grades)
    affected_samples = grade_counts["C"] + grade_counts["D"]

    if base_risk != "unknown" and holc_coverage < 0.5:
        disparate_impact_risk = (
            f"{base_risk} (on {holc_coverage:.1%} of rows with HOLC data; "
            f"{unresolved} of {total} unresolved)"
        )
    else:
        disparate_impact_risk = base_risk

    recommendations = []
    if grade_counts["D"] > 0:
        pct_d = grade_counts["D"] / total * 100
        pct_d_resolved = grade_counts["D"] / known_count * 100
        recommendations.append(
            f"{pct_d:.1f}% of samples ({pct_d_resolved:.1f}% of those with HOLC data) are "
            "from historically redlined (Grade D) areas. Using this feature may "
            "perpetuate historical discrimination."
        )

    if base_risk in ["critical", "high"]:
        recommendations.extend(
            [
                "Consider removing or aggregating this geographic feature",
                "Test model predictions for geographic disparities",
                "Apply fairness constraints if feature must be used",
            ]
        )
    elif base_risk == "medium":
        recommendations.extend(
            [
                "Monitor for geographic disparities in model outcomes",
                "Consider broader geographic aggregation",
            ]
        )

    if holc_coverage < 0.5:
        recommendations.append(
            f"Only {holc_coverage:.1%} of values have HOLC data; the verdict rests on "
            f"{known_count} resolved row(s) out of {total}. Consider supplementing with "
            "other geographic risk indicators."
        )

    return GeographicRiskAssessment(
        feature_name=feature_type,
        risk_score=risk_score,
        holc_coverage=holc_coverage,
        grade_distribution=grade_distribution,
        affected_samples=affected_samples,
        disparate_impact_risk=disparate_impact_risk,
        recommendations=recommendations,
        data_sources_used=["HOLC Redlining Maps (Mapping Inequality Project)"],
    )


# CDC Social Vulnerability Index (SVI)

SVI_THEMES = {
    "socioeconomic": "Socioeconomic Status (below poverty, unemployed, low income, no high school diploma)",
    "household_composition": "Household Composition & Disability (aged 65+, under 17, disability, single parent)",
    "minority_language": "Minority Status & Language (minority, limited English)",
    "housing_transportation": "Housing Type & Transportation (multi-unit, mobile homes, crowding, no vehicle, group quarters)",
}


def get_svi_data_url(year: int = 2022) -> str:
    """
    Get the URL for downloading CDC Social Vulnerability Index data.

    Args:
        year: Year of SVI data (2010-2022 available)

    Returns:
        URL for SVI data download

    Note:
        Data is provided at census tract level and includes
        vulnerability scores across multiple themes.
    """
    return f"https://www.atsdr.cdc.gov/placeandhealth/svi/data_{year}_download.html"


# Data quality and limitations

DATA_QUALITY_NOTES = """
Data Quality and Limitations
============================

1. HOLC Redlining Maps
   - Created 1935-1940 by the Home Owners' Loan Corporation
   - Available for ~200+ US cities
   - Boundaries are hand-drawn and may not align with modern geographic units
   - ZIP codes are an approximation; use actual HOLC boundaries for precision

2. Temporal Considerations
   - HOLC maps are 80+ years old
   - Neighborhood demographics have changed significantly
   - Effects of redlining persist but vary by location

3. Spatial Precision
   - HOLC areas don't align with modern ZIP codes or census tracts
   - Proper analysis requires spatial joins with original HOLC boundary data
   - This module provides approximations for demonstration

4. Missing Data
   - Not all cities have HOLC data
   - Rural areas were generally not surveyed
   - Some maps are incomplete or damaged

5. Recommended Usage
   - Use as one signal among many in bias detection
   - Always validate with local knowledge
   - Consider contemporary data (SVI, ACS) alongside historical

References:
   - https://dsl.richmond.edu/panorama/redlining/
   - Aaronson et al. (2021). Effects of 1930s HOLC "Redlining" Maps. AEJ: EP.
   - NCRC (2018). HOLC Redlining Maps: Persistent Structure of Segregation.
"""


def print_data_quality_notes():
    """Print data quality and limitations notes."""
    print(DATA_QUALITY_NOTES)
