"""Conservative parsing of OSM tags; ambiguous measurements stay unknown."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite, radians, tan
from types import MappingProxyType

from .errors import DataError


@dataclass(frozen=True, slots=True, eq=False, init=False)
class _FrozenTags(Mapping[str, str]):
    """An owned, validated snapshot that can safely be shared between records."""

    _values: Mapping[str, str]

    def __init__(self, tags: Mapping[str, object]) -> None:
        result: dict[str, str] = {}
        for key, value in tags.items():
            if not isinstance(key, str) or not key.strip():
                raise DataError("OSM tag keys must be nonempty strings")
            if not isinstance(value, (str, int, float, bool)):
                raise DataError("OSM tag values must be strings, numbers, or booleans")
            if isinstance(value, float) and not isfinite(value):
                raise DataError("OSM tags cannot contain non-finite numbers")
            if isinstance(value, bool):
                result[key] = "yes" if value else "no"
            else:
                try:
                    result[key] = str(value).strip()
                except ValueError as exc:
                    raise DataError(
                        "OSM tag value cannot be represented as a string"
                    ) from exc
        object.__setattr__(
            self, "_values", MappingProxyType(dict(sorted(result.items())))
        )

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


_EMPTY_TAGS = _FrozenTags({})


def normalize_tags(tags: Mapping[str, object] | None) -> Mapping[str, str]:
    """Copy external mappings, including proxies over mutable dictionaries."""
    if tags is None:
        return _EMPTY_TAGS
    if isinstance(tags, _FrozenTags):
        return tags
    if not isinstance(tags, Mapping):
        raise DataError("OSM tags must be a mapping")
    return _FrozenTags(tags) if tags else _EMPTY_TAGS


def metres(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(
        r"(\d+(?:[.,]\d+)?)\s*(m|cm|ft|feet|')?", str(value).strip().lower()
    )
    if not match:
        return None
    number = float(match[1].replace(",", "."))
    factor = {"cm": 0.01, "ft": 0.3048, "feet": 0.3048, "'": 0.3048}.get(match[2], 1.0)
    result = number * factor
    return result if isfinite(result) and result >= 0 else None


def incline_percent(value: str | None) -> float | None:
    """Signed grade; qualitative up/down cannot supply a measured magnitude."""
    if value is None:
        return None
    match = re.fullmatch(r"([+-]?\d+(?:[.,]\d+)?)\s*(%|°)?", str(value).strip())
    if not match:
        return None
    number = float(match[1].replace(",", "."))
    if not isfinite(number):
        return None
    if match[2] == "°":
        if abs(number) >= 90:
            return None
        number = tan(radians(number)) * 100
    return number if isfinite(number) else None


def reverse_incline(value: str) -> str:
    if not isinstance(value, str):
        raise DataError("Incline tag must be a string")
    direction = value.strip().lower()
    if direction in {"up", "down"}:
        return "down" if direction == "up" else "up"
    number = incline_percent(value)
    if number is None:
        return value
    if number == 0:
        return "0%"
    # Preserve the float value without emitting unsupported exponent notation.
    reversed_number = format(Decimal(str(-number)), "f")
    if "." in reversed_number:
        reversed_number = reversed_number.rstrip("0").rstrip(".")
    return f"{reversed_number}%"


def is_crossing(tags: Mapping[str, str]) -> bool:
    tags = normalize_tags(tags)
    return tags.get("highway") == "crossing" or tags.get("footway") == "crossing" or (
        "crossing" in tags and tags["crossing"] != "no"
    )
