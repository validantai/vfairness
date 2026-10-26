"""Coded group labels for the experiment-tracker writers (pre-release).

PRE-RELEASE, OUTSIDE THE FROZEN PUBLIC SURFACE. This module is private on
purpose: it has not been through the library's grading yet, so it is not part
of the public API that the capability ledger and the beta gate count, and its
names may change. The behaviour it gives the public writers
(``log_fairness_to_mlflow``, ``log_fairness_to_wandb`` and ``auto_log_fairness``)
is part of those writers and is documented on them.

WHAT IT DOES. A tracker run is read by many people, and the names of protected
groups ("Female", "Over 60", a postcode) are themselves sensitive. With
``group_labels="coded"`` (the default) the writers replace every group name
they would write with a code such as ``g_3fa91c07d2e4``:

    code = "g_" + HMAC-SHA256(key, b"vfairness.group-label.v1\\x00" + utf8(str(label)))[:12 hex]

The code is KEYED. A plain hash of "Female" is found again by hashing a short
list of likely names, so it hides nothing; without the key the code cannot be
recomputed. The same key gives the same code in every run, so two versions of a
model stay comparable on one dashboard. The key comes from ``label_key=`` or the
environment variable ``VFAIRNESS_GROUP_LABEL_KEY``; with neither, the writers
refuse before anything is written. The mapping from name to code is never
written to the tracker: ``group_label_codes`` rebuilds it locally from the key.

HOW THE REPORT IS CODED. From its structure, never by searching for strings
equal to a label: a label can be "0", "1" or "yes", and replacing every such
string would rewrite counts and statuses. The places that carry a group are
listed below in ``_ReportCoder`` and each is coded where it sits. Prose that
names groups inside sentences (``assessment.summary``, ``explanations``) cannot
be coded safely and is left out, and so is anything this module cannot read or
check; everything left out is NAMED, so a reader knows the coded report is not
the whole report. Fail closed: when in doubt a part is left out and named, never
written with a name in it.

ORDER IS NOT LEAKED EITHER. The library orders groups by their real names, so
writing them in that order would tell a reader which code has the earlier name,
and for an attribute with two well-known values that is the whole secret. Every
coded list and dict is therefore ordered by CODE, and every effect-size pair is
written as ``<lower code>_vs_<higher code>``. When the producer compared the two
groups the other way round, the pair's values are restated for this direction
(Cohen's d and h and the mean differences negated, risk and odds ratios and their
bounds inverted, the group1 and group2 fields swapped), which is the same
comparison read from the other side. A field whose direction is not known is
left out and named rather than written under a direction it may not have.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

ENV_VAR = "VFAIRNESS_GROUP_LABEL_KEY"
MODES = ("coded", "names")

_CODE_DOMAIN = b"vfairness.group-label.v1\x00"
_KEY_ID_DOMAIN = b"vfairness.group-label.key-id.v1"

LabelKey = Union[str, bytes, bytearray]


def check_mode(group_labels: Any, *, caller: str) -> str:
    """Return ``group_labels`` when it is one of the two modes, else refuse."""
    if not isinstance(group_labels, str) or group_labels not in MODES:
        raise ValueError(
            f"{caller}: group_labels must be 'coded' (the default: group names are "
            f"replaced by keyed codes) or 'names' (real group names are written on "
            f"purpose), got {group_labels!r}. Nothing was written."
        )
    return group_labels


def resolve_key(label_key: Optional[LabelKey], *, caller: str) -> bytes:
    """The key that codes group names: ``label_key``, else the environment variable.

    Refuses when neither gives a non-empty key. Called before anything is written,
    so a run without a key stays empty rather than half-written.
    """
    if label_key is None:
        from_env = os.environ.get(ENV_VAR)
        if from_env:
            return from_env.encode("utf-8")
        raise ValueError(
            f"{caller}: group names are coded by default and no key was found, so "
            f"nothing was written. Fix it in one line: set {ENV_VAR} (any secret "
            f"string, kept like a password; the same key gives the same codes in every "
            f"run, so versions stay comparable) or pass label_key=..., or pass "
            f"group_labels='names' to write the real group names on purpose."
        )
    if isinstance(label_key, str):
        key = label_key.encode("utf-8")
    elif isinstance(label_key, (bytes, bytearray)):
        key = bytes(label_key)
    else:
        raise TypeError(
            f"{caller}: label_key must be a str or bytes secret, got "
            f"{type(label_key).__name__}. Nothing was written."
        )
    if not key:
        raise ValueError(
            f"{caller}: label_key is empty, and an empty key codes nothing anybody "
            f"could not recompute. Pass a secret string, or set {ENV_VAR}. Nothing "
            f"was written."
        )
    return key


def label_code(key: bytes, label: Any) -> str:
    """The code for one group label under ``key``."""
    digest = hmac.new(key, _CODE_DOMAIN + str(label).encode("utf-8"), hashlib.sha256)
    return "g_" + digest.hexdigest()[:12]


def key_id(key: bytes) -> str:
    """Eight hex characters that tell keys apart without revealing them.

    Written on every coded run, so a team can see which key coded a run (and
    that two runs are comparable) while the key itself stays off the tracker.
    """
    return hmac.new(key, _KEY_ID_DOMAIN, hashlib.sha256).hexdigest()[:8]


def _codes_for(key: bytes, labels: Iterable[Any]) -> Dict[str, str]:
    """``str(label) -> code`` for every distinct label, refusing a shared code."""
    codes: Dict[str, str] = {}
    owner: Dict[str, str] = {}
    for label in labels:
        name = str(label)
        if name in codes:
            continue
        code = label_code(key, name)
        if code in owner and owner[code] != name:
            # Never merge two groups silently: their results would be written under
            # one name. The names stay out of the message, which may be logged.
            raise ValueError(
                f"Two different group labels get the same code {code} under this key, "
                f"so their results would be merged under one name. Use a different "
                f"key. Nothing was written."
            )
        codes[name] = code
        owner[code] = name
    return codes


def group_label_codes(
    labels: Iterable[Any], label_key: Optional[LabelKey] = None
) -> Dict[str, str]:
    """Map each group name to the code the tracker writers give it.

    Run this on your own machine to read a coded run: the tracker holds only the
    codes, never this mapping. The key is resolved exactly as the writers resolve
    it: ``label_key`` if given, else the environment variable
    ``VFAIRNESS_GROUP_LABEL_KEY``, else ``ValueError``.

    Args:
        labels: The group names (any iterable; repeats are ignored). Each label
            is coded by its text, ``str(label)``, as the writers code it.
        label_key: The secret key, as a str or bytes. Optional when the
            environment variable is set.

    Returns:
        ``{name: code}``, one entry per distinct ``str(label)``.

    Raises:
        ValueError: If no key is found, or if two different names get the same
            code (never merged silently).

    Example:
        >>> from vfairness._group_codes import group_label_codes
        >>> codes = group_label_codes(["Female", "Male"], label_key="my secret")
        >>> {code: name for name, code in codes.items()}  # read a coded run
    """
    key = resolve_key(label_key, caller="group_label_codes")
    return _codes_for(key, labels)


# A dict key that NAMES a group-carrying field but is not one of the places coded
# below is left out, because its value may be a group label this module cannot
# recognise (a numeric label is just a number). Counts of groups are the
# exception: ``n_groups``, ``n_groups_before``, ``n_valid_groups`` are numbers of
# groups, not groups.
_GROUP_COUNT_KEY = re.compile(r"^n_\w*groups?\w*$")

# Effect-size fields by how they change when the pair is read the other way round.
# Anything else in an effect-size entry is left out: its direction is not known.
_EFFECT_NEGATE = frozenset(
    {
        "cohens_d_positive_rate",
        "cohens_h_positive_rate",
        "cohens_d_predictions",
        "cohens_d_residuals",
        "mean_pred_diff",
        "mean_residual_diff",
    }
)
_EFFECT_INVERT = frozenset({"risk_ratio", "odds_ratio"})
_EFFECT_SWAP = {
    "group1_positive_rate": "group2_positive_rate",
    "group2_positive_rate": "group1_positive_rate",
    "group1_size": "group2_size",
    "group2_size": "group1_size",
}
# Graded on the magnitude of the effect, so the same whichever way it is read.
_EFFECT_SYMMETRIC = frozenset({"interpretation", "interpretation_residuals"})

# Left out on every coded run, by design: sentences that name groups.
DELIBERATE = ("assessment.summary", "explanations")

_DROP = object()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _negated(value: Any) -> Any:
    # 0.0 - x is exact in IEEE arithmetic, gives NaN for NaN, and never yields -0.0.
    return None if value is None else 0.0 - value


def _reciprocal(value: Any) -> Any:
    if value is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.float64(1.0) / np.float64(value)


class _ReportCoder:
    """Builds the coded copy of one report and records what it left out.

    Group-carrying places, coded where they sit (and ordered by code):

      group_stats, residual_bias                          keys
      data_info.valid_groups / invalid_groups / groups_dropped   items
      data_info.group_sizes                               keys
      metrics_with_ci.<metric>.metadata.group_sizes       keys
      statistical_validation.per_group_size_recommendation / per_group_methods   keys
      assessment.insufficient_evidence_groups[].group     value
      effect_sizes                                        pair keys, matched to the
                                                          producer's f"{g1}_vs_{g2}"

    Everything else is copied, and while it is copied every text value and key is
    checked against the run's labels (equal to one, or, for a label with a letter
    in it, containing it as a whole word). A hit is left out and named, never
    rewritten. A label with no letter ("0", "1", "18-25") is checked for equality
    only: inside a sentence such a string is far more often a count than a group.
    """

    def __init__(self, key: bytes, report: Mapping[str, Any]) -> None:
        self.left_out: List[str] = []
        self.unexpected: List[str] = []
        self._codes = _codes_for(key, _collect_labels(report))
        self._exact = set(self._codes)
        worded = sorted((n for n in self._codes if any(c.isalpha() for c in n)), key=len)
        self._worded = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(n) for n in reversed(worded)) + r")(?!\w)"
            )
            if worded
            else None
        )

    # The record of what was left out. ``deliberate`` marks the two prose fields
    # that are left out on every coded run; anything else is also warned about.

    def _leave_out(self, path: str, why: str = "", deliberate: bool = False) -> None:
        entry = f"{path} ({why})" if why else path
        self.left_out.append(entry)
        if not deliberate:
            self.unexpected.append(entry)

    def code(self, label: Any) -> str:
        return self._codes[str(label)]

    def names_a_group(self, text: str) -> bool:
        if text in self._exact:
            return True
        return bool(self._worded is not None and self._worded.search(text))

    # Generic copy, for every place that does not carry a group by design.

    def copy(self, value: Any, path: str) -> Any:
        if value is None or isinstance(value, (bool, np.bool_)) or _is_number(value):
            return value
        if isinstance(value, str):
            if self.names_a_group(value):
                self._leave_out(path, "text naming a group")
                return _DROP
            return value
        if isinstance(value, np.ndarray):
            return self.copy(value.tolist(), path)
        if isinstance(value, (list, tuple)):
            mark = (len(self.left_out), len(self.unexpected))
            items = [self.copy(item, f"{path}[{i}]") for i, item in enumerate(value)]
            if any(item is _DROP for item in items):
                # A list with holes would change meaning, so the whole list goes.
                del self.left_out[mark[0] :]
                del self.unexpected[mark[1] :]
                self._leave_out(path, "a list holding an item that could not be kept")
                return _DROP
            return tuple(items) if isinstance(value, tuple) else items
        if isinstance(value, Mapping):
            out: Dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    self._leave_out(path, "keys that are not text, which may be groups")
                    return _DROP
                kept = self.copy_entry(key, item, path)
                if kept is not _DROP:
                    out[key] = kept
            return out
        enum_value = getattr(value, "value", _DROP)
        if enum_value is not _DROP and not callable(enum_value):
            return self.copy(enum_value, path)
        self._leave_out(path, f"a {type(value).__name__} that cannot be checked")
        return _DROP

    def copy_entry(self, key: str, value: Any, parent: str) -> Any:
        path = f"{parent}.{key}" if parent else key
        if self.names_a_group(key):
            self._leave_out(
                f"{parent}.<a key naming a group>" if parent else "<a key naming a group>"
            )
            return _DROP
        if "group" in key.lower() and not _GROUP_COUNT_KEY.match(key):
            self._leave_out(path, "a group field this version does not code")
            return _DROP
        return self.copy(value, path)

    def copy_metric_keyed(self, value: Any, path: str) -> Any:
        """A dict keyed by METRIC name (``worst_group_accuracy`` is one)."""
        if not isinstance(value, Mapping):
            return self.copy(value, path)
        out: Dict[str, Any] = {}
        for metric, item in value.items():
            if not isinstance(metric, str) or self.names_a_group(metric):
                self._leave_out(f"{path}.<a key naming a group>")
                continue
            kept = self.copy(item, f"{path}.{metric}")
            if kept is not _DROP:
                out[metric] = kept
        return out

    # The group-carrying places.

    def code_keys(self, value: Any, path: str) -> Any:
        if not isinstance(value, Mapping):
            self._leave_out(path, f"a {type(value).__name__}, not a dict keyed by group")
            return _DROP
        out: Dict[str, Any] = {}
        for label, item in value.items():
            code = self.code(label)
            if code in out:
                raise ValueError(
                    f"Two different group labels in {path} have the same text, so they "
                    f"would share the code {code} and their results would be merged. "
                    f"Nothing was written."
                )
            kept = self.copy(item, f"{path}.{code}")
            if kept is not _DROP:
                out[code] = kept
        return dict(sorted(out.items()))

    def code_list(self, value: Any, path: str) -> Any:
        if not isinstance(value, (list, tuple)):
            self._leave_out(path, f"a {type(value).__name__}, not a list of groups")
            return _DROP
        codes = [self.code(label) for label in value]
        if len(set(codes)) != len(codes):
            raise ValueError(
                f"Two group labels in {path} share one code, so their results would "
                f"be merged. Nothing was written."
            )
        return sorted(codes)

    def code_report(self, report: Mapping[str, Any]) -> Dict[str, Any]:
        handlers = {
            "group_stats": self.code_keys,
            "residual_bias": self.code_keys,
            "data_info": self._data_info,
            "assessment": self._assessment,
            "metrics": self.copy_metric_keyed,
            "thresholds_used": self.copy_metric_keyed,
            "metrics_with_ci": self._metrics_with_ci,
            "effect_sizes": self._effect_sizes,
            "statistical_validation": self._statistical_validation,
        }
        coded: Dict[str, Any] = {}
        for section, value in report.items():
            if not isinstance(section, str):
                self._leave_out("<a section whose name is not text>")
                continue
            if section == "explanations":
                self._leave_out(section, "sentences naming groups", deliberate=True)
                continue
            handler = handlers.get(section)
            kept = handler(value, section) if handler else self.copy_entry(section, value, "")
            if kept is not _DROP:
                coded[section] = kept
        return coded

    def _with(self, value: Any, path: str, places: Dict[str, Any]) -> Any:
        """Copy a dict, coding the named keys with their handlers."""
        if not isinstance(value, Mapping):
            return self.copy(value, path)
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key in places:
                handler = places[key]
                kept = _DROP if handler is None else handler(item, f"{path}.{key}")
            elif isinstance(key, str):
                kept = self.copy_entry(key, item, path)
            else:
                self._leave_out(path, "keys that are not text, which may be groups")
                return _DROP
            if kept is not _DROP:
                out[key] = kept
        return out

    def _data_info(self, value: Any, path: str) -> Any:
        return self._with(
            value,
            path,
            {
                "valid_groups": self.code_list,
                "invalid_groups": self.code_list,
                "groups_dropped": self.code_list,
                "group_sizes": self.code_keys,
            },
        )

    def _assessment(self, value: Any, path: str) -> Any:
        def summary(_item: Any, item_path: str) -> Any:
            self._leave_out(item_path, "a sentence naming groups", deliberate=True)
            return _DROP

        return self._with(
            value,
            path,
            {
                "summary": summary,
                "insufficient_evidence_groups": self._insufficient,
                "passed_metrics": self.copy,
                "failed_metrics": self.copy,
                "not_assessable_metrics": self.copy,
            },
        )

    def _insufficient(self, value: Any, path: str) -> Any:
        if not isinstance(value, (list, tuple)):
            self._leave_out(path, f"a {type(value).__name__}, not a list of records")
            return _DROP
        records = []
        for i, record in enumerate(value):
            if not isinstance(record, Mapping) or "group" not in record:
                self._leave_out(f"{path}[{i}]", "a record without a group field")
                continue
            coded = self._with(record, f"{path}[{i}]", {"group": lambda g, _p: self.code(g)})
            if coded is not _DROP:
                records.append(coded)
        return sorted(records, key=lambda r: r["group"])

    def _metrics_with_ci(self, value: Any, path: str) -> Any:
        if not isinstance(value, Mapping):
            return self.copy(value, path)
        out: Dict[str, Any] = {}
        for metric, entry in value.items():
            if not isinstance(metric, str) or self.names_a_group(metric):
                self._leave_out(f"{path}.<a key naming a group>")
                continue
            entry_path = f"{path}.{metric}"
            kept = self._with(
                entry,
                entry_path,
                {
                    "metadata": lambda m, p: self._with(m, p, {"group_sizes": self.code_keys}),
                },
            )
            if kept is not _DROP:
                out[metric] = kept
        return out

    def _statistical_validation(self, value: Any, path: str) -> Any:
        return self._with(
            value,
            path,
            {
                "per_group_size_recommendation": self.code_keys,
                "per_group_methods": self.code_keys,
            },
        )

    def _effect_sizes(self, value: Any, path: str) -> Any:
        if not isinstance(value, Mapping):
            self._leave_out(path, f"a {type(value).__name__}, not a dict keyed by pair")
            return _DROP
        # The producer builds each key as f"{g1}_vs_{g2}" over the report's groups
        # (classification.compute_effect_sizes, regression.compute_regression_effect_sizes).
        # A key is read back by building those strings from the groups this report
        # names, NEVER by splitting on "_vs_": a group name may itself contain it.
        names = sorted(self._codes)
        out: Dict[str, Any] = {}
        claimed: Dict[str, int] = {}
        for pair_key, entry in value.items():
            matches = _pair_matches(str(pair_key), names)
            if len(matches) != 1:
                why = (
                    "a pair key that matches no pair of this report's groups"
                    if not matches
                    else "a pair key that more than one pair of groups would produce"
                )
                self._leave_out(f"{path}.<a pair key>", why)
                continue
            first, second = matches[0]
            code_a, code_b = self._codes[first], self._codes[second]
            low, high = sorted((code_a, code_b))
            coded_key = f"{low}_vs_{high}"
            claimed[coded_key] = claimed.get(coded_key, 0) + 1
            entry_path = f"{path}.{coded_key}"
            if not isinstance(entry, Mapping):
                # Its fields cannot be named, so they cannot be turned to code order.
                self._leave_out(entry_path, f"a {type(entry).__name__} the writers cannot read")
                continue
            kept = self._effect_entry(entry, entry_path, flip=code_a > code_b)
            out[coded_key] = kept
        for coded_key, count in claimed.items():
            if count > 1 and coded_key in out:
                # Two entries for one pair of groups (A_vs_B and B_vs_A): which one to
                # keep is not this module's call, so neither is written.
                del out[coded_key]
                self._leave_out(f"{path}.{coded_key}", f"{count} entries for one pair of groups")
        return dict(sorted(out.items()))

    def _effect_entry(self, entry: Mapping[str, Any], path: str, *, flip: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for field, item in entry.items():
            field_path = f"{path}.{field}"
            if not isinstance(field, str) or self.names_a_group(field):
                self._leave_out(f"{path}.<a key naming a group>")
                continue
            if field in _EFFECT_SYMMETRIC:
                kept = self.copy(item, field_path)
                if kept is not _DROP:
                    out[field] = kept
            elif field in _EFFECT_NEGATE:
                if item is not None and not _is_number(item):
                    self._leave_out(field_path, "not a number, so it cannot be put in code order")
                    continue
                out[field] = _negated(item) if flip else item
            elif field in _EFFECT_INVERT:
                restated = _restated_interval(item, flip)
                if restated is _DROP:
                    self._leave_out(field_path, "not a (value, lower, upper) of numbers")
                    continue
                out[field] = restated
            elif field in _EFFECT_SWAP:
                if item is not None and not _is_number(item):
                    self._leave_out(field_path, "not a number")
                    continue
                out[_EFFECT_SWAP[field] if flip else field] = item
            else:
                self._leave_out(field_path, "a field whose direction this version does not know")
        return out


def _restated_interval(value: Any, flip: bool) -> Any:
    """A ratio, or its ``(value, lower, upper)``, read for the reversed pair.

    The same check runs whichever way the pair is read, so whether a field is
    written never depends on the direction (that would leak the order too).
    """
    if value is None or _is_number(value):
        return _reciprocal(value) if flip else value
    if isinstance(value, (tuple, list)) and len(value) in (1, 3):
        if not all(v is None or _is_number(v) for v in value):
            return _DROP
        if not flip:
            return value
        if len(value) == 1:
            restated = [_reciprocal(value[0])]
        else:
            point, lower, upper = value
            restated = [_reciprocal(point), _reciprocal(upper), _reciprocal(lower)]
        return tuple(restated) if isinstance(value, tuple) else restated
    return _DROP


def _pair_matches(pair_key: str, names: List[str]) -> List[Tuple[str, str]]:
    """Every (g1, g2) of the report's groups whose f"{g1}_vs_{g2}" is ``pair_key``."""
    found = []
    for first in names:
        head = first + "_vs_"
        if pair_key.startswith(head):
            second = pair_key[len(head) :]
            if second != first and second in names:
                found.append((first, second))
    return found


def _collect_labels(report: Mapping[str, Any]) -> List[Any]:
    """Every group label the report names in a group-carrying place."""
    labels: List[Any] = []

    def keys_of(value: Any) -> None:
        if isinstance(value, Mapping):
            labels.extend(value.keys())

    def items_of(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            labels.extend(value)

    keys_of(report.get("group_stats"))
    keys_of(report.get("residual_bias"))
    info = report.get("data_info")
    if isinstance(info, Mapping):
        for field in ("valid_groups", "invalid_groups", "groups_dropped"):
            items_of(info.get(field))
        keys_of(info.get("group_sizes"))
    assessment = report.get("assessment")
    if isinstance(assessment, Mapping):
        records = assessment.get("insufficient_evidence_groups")
        if isinstance(records, (list, tuple)):
            labels.extend(r["group"] for r in records if isinstance(r, Mapping) and "group" in r)
    validation = report.get("statistical_validation")
    if isinstance(validation, Mapping):
        keys_of(validation.get("per_group_size_recommendation"))
        keys_of(validation.get("per_group_methods"))
    with_ci = report.get("metrics_with_ci")
    if isinstance(with_ci, Mapping):
        for entry in with_ci.values():
            if isinstance(entry, Mapping) and isinstance(entry.get("metadata"), Mapping):
                keys_of(entry["metadata"].get("group_sizes"))
    return labels


def code_report(
    report: Mapping[str, Any], key: bytes
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """The coded copy of ``report``, what was left out, and what of that was unexpected.

    Raises ValueError before anything is written when two different labels would
    share a code.
    """
    coder = _ReportCoder(key, report)
    coded = coder.code_report(report)
    return coded, coder.left_out, coder.unexpected


def left_out_text(left_out: List[str], limit: int = 4000) -> str:
    """The left-out list as one tag value, cut at ``limit`` characters with a count.

    The exact number travels in its own tag, so a cut list is never read as the
    whole of it.
    """
    text = str(left_out)
    if len(text) <= limit:
        return text
    kept: List[str] = []
    for entry in left_out:
        if len(str(kept + [entry])) > limit - 40:
            break
        kept.append(entry)
    return f"{kept} and {len(left_out) - len(kept)} more"
