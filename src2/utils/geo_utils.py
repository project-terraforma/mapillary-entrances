# geo_utils.py

import math
from typing import Dict
from .constants import EARTH_RADIUS_M

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def deg_per_meter(lat_deg: float):
    dlat = 1.0 / 111_320.0
    dlon = dlat / max(0.01, abs(math.cos(math.radians(lat_deg))))
    return dlat, dlon

def robust_radius_to_bbox(lat: float, lon: float, radius_m: float):
    """
    Convert a (lat, lon, radius_m) into a bbox = (xmin, ymin, xmax, ymax)
    using a local equirectangular approximation.
    """
    delta_lat = (radius_m / EARTH_RADIUS_M) * (180 / math.pi)
    delta_lon = (radius_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180 / math.pi)
    return lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat

def simple_radius_to_bbox(lon: float, lat: float, meters: float = 80.0) -> Dict[str, float]:
    # simple meters to deg approximation for small areas
    dy = meters / 111_320.0
    dx = meters / (111_320.0 * math.cos(math.radians(lat)))
    return {"xmin": lon - dx, "ymin": lat - dy, "xmax": lon + dx, "ymax": lat + dy}
