"""Shared base classes for production-quality result types."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import vfairness

logger = logging.getLogger(__name__)


def _plain_scalar(value):
    """A numpy scalar as its plain Python equivalent, or the value unchanged.

    Ordered so a boolean is caught FIRST: ``np.bool_`` is not a Python bool, and
    letting it fall through to the float branch turns a recorded yes/no into a 1.0
    measurement (the rule ``_triage.is_flag`` exists for).
    """
    import numpy as np

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _plain_containers(value):
    """Recursively turn numpy scalars and arrays inside containers into plain Python.

    G10, 2026-09-30. ``to_dict`` converted only the TOP-LEVEL numpy scalars and
    never looked inside a dict or a nested list, and the docstring's promise is a
    "plain dict for database storage or API response". ``RunMetadata.parameters``
    is exactly such a dict and it rides on every result, so a threshold recorded
    as ``np.float32(0.3)`` stayed a numpy scalar through ``to_dict()`` and then
    ``json.dumps(..., default=str)`` rendered it as the STRING ``"0.3"``: a number
    published as text, which is the ``"1" == 1 is False`` door the campaign keeps
    finding. ``np.int64(3)`` became ``"3"`` and ``np.bool_(True)`` became
    ``"True"`` by the same route.

    NaN is deliberately PRESERVED here. ``to_dict()`` is the in-process surface and
    its consumers test it with ``np.isfinite``; the JSON boundary is where a
    non-finite value becomes ``null`` (see :func:`_json_safe`). That split is the
    one ``CalibrationReport.to_json`` already established for G005-JSON.
    """
    import numpy as np

    if isinstance(value, dict):
        return {k: _plain_containers(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        # A 0-d array, and ``np.ma.masked`` (which IS an ndarray subclass), give a
        # SCALAR back from tolist(), and masked gives None. Iterating that raises,
        # so check the shape of what came back rather than assuming a list. The
        # pre-G10 code called tolist() at the top level only and so returned None
        # for masked by accident; that behaviour is preserved deliberately.
        listed = value.tolist()
        if not isinstance(listed, list):
            return _plain_scalar(listed)
        return [_plain_containers(v) for v in listed]
    if isinstance(value, (list, tuple)):
        return [v.to_dict() if hasattr(v, "to_dict") else _plain_containers(v) for v in value]
    return _plain_scalar(value)


def _json_safe(value):
    """Recursively make a value writable as STRICT JSON, third state included.

    G10, 2026-09-30. ``to_json`` was ``json.dumps(self.to_dict(), default=str)``,
    which fails two different ways on exactly the values that mean "not measured":

    * A non-finite float is written as the bare token ``NaN`` / ``Infinity``.
      That is not JSON. Python's own ``json.loads`` accepts it as an extension and
      every strict parser rejects the WHOLE document, so the one result that had
      something important to say, that its number could not be measured, is the
      one a browser consumer cannot read at all, and the cheapest local repair for
      whoever hits it is to put a ``0.0`` back. Measured on this mixin before the
      fix, a ``GroundednessResult`` carrying a NaN faithfulness serialised to
      ``"faithfulness": NaN`` and ``json.loads(..., parse_constant=raise)``
      refused it.
    * ``default=str`` MINTS CONTENT from an absence sentinel. Measured:
      ``pd.NA`` became the string ``"<NA>"``, ``pd.NaT`` became ``"NaT"`` and
      ``np.ma.masked`` inside a dict became ``"--"``. A group named ``'<NA>'``
      once carried a maximal finding of discrimination against a group that is not
      a group; here the same mint reaches the published envelope.

    ``None`` is the encoding this repo already uses for the third state at a JSON
    boundary, and the KEY STAYS PRESENT holding null, so null is still
    distinguishable from a measured number and from an absent field.
    """
    import numpy as np
    import pandas as pd

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    # ``np.ma.masked`` answers ``pd.isna(...) -> masked``, whose ``bool()`` is
    # False, so the pandas test at the bottom of this function cannot see it. That
    # is why ``multi_agent.harness._label_not_recorded`` carries the same identity
    # line.
    #
    # HERE IT IS DEFENCE IN DEPTH, NOT LOAD-BEARING, and the distinction is
    # recorded because a redundant guard looks identical to one that fired.
    # Measured 2026-09-30: removing this line left the whole G10 pin file GREEN
    # (48 passed), because ``np.ma.masked`` IS an ``np.ndarray`` subclass and the
    # array branch above it already answers ``tolist() -> None``, which then falls
    # through to ``pd.isna(None) -> True``. It is kept because the redundancy costs
    # nothing and a numpy that stopped making masked an ndarray would otherwise
    # publish the string ``'--'``; that assumption is pinned in
    # tests/test_bgl_g10_base_and_exceptions.py so the day it changes is a red test
    # rather than a silent change of meaning.
    if value is np.ma.masked:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    # A str is never absent for this purpose: 'None', 'nan' and 'NA' are all
    # strings a caller may legitimately mean, and rewriting them would be the
    # over-correction. Only the SENTINELS become null.
    if isinstance(value, (str, bytes)):
        return value
    try:
        absent = pd.isna(value)
    except (TypeError, ValueError):  # pragma: no cover - a value pandas cannot judge
        return value
    if np.ndim(absent) == 0 and bool(absent):
        return None
    return value


@dataclass
class RunMetadata:
    """Audit trail metadata attached to every test result.

    Required by EU AI Act Article 12 (automatic logging) to ensure
    every fairness evaluation is traceable and reproducible.
    """

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    library_version: str = field(default_factory=lambda: vfairness.__version__)
    parameters: dict = field(default_factory=dict)
    random_seed: Optional[int] = None
    system_type: str = ""
    model_name: str = ""

    def to_dict(self) -> dict:
        """The audit trail as a plain dict.

        ``parameters`` is caller-supplied and routinely holds numpy scalars (a
        threshold, a seed, a sample count read off a DataFrame). ``asdict`` copies
        them through unchanged, and the JSON layer above then rendered
        ``np.float32(0.3)`` as the string ``"0.3"``, so normalise them here (G10).
        NaN is preserved; ``SerializableMixin.to_json`` is where it becomes null.
        """
        return _plain_containers(asdict(self))


class SerializableMixin:
    """Mixin providing to_dict() and to_json() for all result dataclasses."""

    def to_dict(self) -> dict:
        """Convert result to a plain dict for database storage or API response."""
        import numpy as np

        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                # Recurse: a 2-D or object array holds values the top-level
                # tolist() leaves as numpy scalars (G10).
                result[k] = _plain_containers(v)
            elif isinstance(v, (list, tuple)):
                result[k] = _plain_containers(v)
            elif hasattr(v, "to_dict"):
                result[k] = _plain_containers(v.to_dict())
            else:
                result[k] = _plain_containers(v)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to STRICT JSON, with could-not-check as JSON ``null``.

        A non-finite number and an absence sentinel both become ``null`` rather
        than the bare ``NaN`` token or a minted ``"<NA>"`` string; see
        :func:`_json_safe` for the measurement and the reason. ``to_dict()`` is
        untouched and still carries the float NaN its in-process consumers test
        with ``np.isfinite``.
        """
        return json.dumps(_json_safe(self.to_dict()), indent=indent, default=str)
