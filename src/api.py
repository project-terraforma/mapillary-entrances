# --- new file: src/api.py ---
# interface for single building near a (lat, lon) point. uses same pipeline
# pieces but returns results in-memory as a dict 

# example usage

"""
PYTHONPATH=. python3 - <<'PY'
from src.api import get_building_package_for_point
lat, lon = 37.789606, -122.396844   # Salesforce Tower
pkg = get_building_package_for_point(lat, lon)
print("✅", pkg["building_id"], "images:", len(pkg["evan_images"]))
print("Sample wall:", pkg["walls"][0])
PY 
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import math
import pandas as pd

from .utils import polygon_vertices_from_wkt
from .buildings import get_buildings
from .places import join_buildings_places
from .selection import select_best_place_for_building
from .imagery import fetch_and_slice_for_building, write_candidates_json
from .sources import resolve_sources

def _point_bbox(lon: float, lat: float, meters: float = 80.0) -> Dict[str, float]:
    # simple meters→deg approximation for small areas
    dy = meters / 111_320.0
    dx = meters / (111_320.0 * math.cos(math.radians(lat)))
    return {"xmin": lon - dx, "ymin": lat - dy, "xmax": lon + dx, "ymax": lat + dy}

def _nearest_building(bdf: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    if bdf.empty:
        raise RuntimeError("No buildings in bbox")
    d2 = (bdf["lat"] - lat)**2 + (bdf["lon"] - lon)**2
    return bdf.loc[d2.idxmin()]

def get_building_package_for_point(
    lat: float,
    lon: float,
    *,
    search_radius_m: int = 120,
    place_radius_m: int = 60,
    max_images_per_building: int = 8,
    min_capture_date: Optional[str] = None,
    prefer_360: bool = True,
    fov_half_angle: float = 25.0,
    apply_fov: bool = True,
    src_mode: str = "auto",
) -> Dict[str, Any]:
    """
    Single-point adapter for inference.py:
      - picks nearest building to (lat,lon)
      - joins places & selects best single place
      - fetches/slices imagery (returns Evan schema)
      - returns polygon [[lon,lat], ...] and wall segments [[lon,lat], [lon,lat]]...
    """
    # get nearest building
    bbox_small = _point_bbox(lon, lat, meters=search_radius_m)
    b_src, p_src = resolve_sources(bbox_small, src_mode)
    bdf = get_buildings(bbox_small, b_src, limit_hint=50)
    b = _nearest_building(bdf, lat, lon)

    # join building with place data 
    links = join_buildings_places(bdf, bbox_small, p_src, radius_m=place_radius_m)
    best_place = select_best_place_for_building(
        links[links["building_id"] == b["id"]],
        building_id=b["id"],
        max_dist_m=place_radius_m
    )

    # imagery (returns detailed saved records)
    saved = fetch_and_slice_for_building(
        b,
        radius_m=search_radius_m,
        min_capture_date=min_capture_date,
        apply_fov=apply_fov,
        max_images_per_building=max_images_per_building,
        prefer_360=prefer_360,
        fov_half_angle=fov_half_angle,
    )

    # write candidates.json (also returns jpg→json list)
    pairs = write_candidates_json(b, best_place, saved)

    # polygon + simple wall segments (edges of the exterior ring)
    polygon = polygon_vertices_from_wkt(b["wkt"])  # [[lon,lat], ...]
    walls: List[List[List[float]]] = []
    if len(polygon) >= 2:
        for i in range(len(polygon)):
            a = polygon[i]
            bpt = polygon[(i + 1) % len(polygon)]
            walls.append([a, bpt])

    # build list to be used in inference.py
    evan_images: List[Dict[str, Any]] = []
    for rec in saved:
        evan_images.append({
            "image_path": rec.get("path") or rec.get("jpg_path"),
            "compass_angle": rec["compass_angle"],
            "coordinates": (
                rec.get("coordinates")
                or (rec.get("computed_geometry", {}).get("coordinates")
                    if isinstance(rec.get("computed_geometry"), dict) else None)
                or [rec.get("lon"), rec.get("lat")]
            ),

        })

    return {
        "building_id": b["id"],
        "building_center": [float(b["lat"]), float(b["lon"])],
        "polygon": polygon, # [[lon,lat], ...]
        "walls": walls, # [ [[lon,lat],[lon,lat]], ...]
        "place": best_place or None,
        "evan_images": evan_images,
        "pairs": pairs,  # [{jpg: json}, ...]
    }

