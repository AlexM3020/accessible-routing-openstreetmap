"""Spherical distances for WGS84 latitude/longitude coordinates."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from .errors import DataError
from .models import Coordinate

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(a: Coordinate, b: Coordinate) -> float:
    if not isinstance(a, Coordinate) or not isinstance(b, Coordinate):
        raise DataError("Haversine endpoints must be Coordinate records")
    lat1, lat2 = radians(a.latitude), radians(b.latitude)
    d_lat = lat2 - lat1
    d_lon = radians(b.longitude - a.longitude)
    value = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(max(0.0, min(1.0, value))))
