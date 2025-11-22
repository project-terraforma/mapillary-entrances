# api.py
from typing import Optional, Dict, Any, List
from utils.geo_utils import simple_radius_to_bbox
from utils.matching_utils import resolve_sources, select_best_place_for_building
from utils.duck_db_utils import get_buildings, join_buildings_places
from utils.polygon_utils import polygon_vertices_from_wkt
from utils.mapillary_utils import _extract_lon_lat
from imagery import fetch_and_slice_for_building


def get_buildings_and_imagery_in_radius(
    lat: float,
    lon: float,
    search_radius_m: int,
    place_radius_m: int,
    max_images_total: int,
    min_capture_date: Optional[str],
    prefer_360: bool,
    src_mode: str,
) -> Dict[str, Any]:
    """
    Multi-building + shared imagery adapter for inference.py

    - Finds *all* buildings within search_radius_m of (lat, lon)
    - Joins each with its best nearby place (optional)
    - Fetches a *single shared* imagery set centered on (lat, lon)
    - Returns:
        {
          "input_coordinates": [lon, lat],
          "building_polygons": {building_id: [[lon,lat], ...], ...},
          "building_walls": {building_id: [[[lon,lat],[lon,lat]], ...], ...},
          "places": {building_id: place_info or None, ...},
          "image_dicts": [ {...}, {...}, ... ]   # all images in radius
        }
    """

    # define bounding box and load building and place data
    bbox = simple_radius_to_bbox(lon, lat, meters=search_radius_m)
    b_src, p_src = resolve_sources(bbox, src_mode)
    bdf = get_buildings(bbox, b_src, limit_hint=200)
    if bdf is None or len(bdf) == 0:
        print("[WARN] No buildings found in radius.")
        return {
            "input_coordinates": [lon, lat],
            "building_polygons": {},
            "building_walls": {},
            "places": {},
            "image_dicts": []
        }

    print(f"[INFO] Found {len(bdf)} buildings within {search_radius_m} m")

    # join once with places
    links = join_buildings_places(bdf, bbox, p_src, radius_m=place_radius_m)

    building_polygons = {}
    building_walls = {}
    building_places = {}

    for _, b in bdf.iterrows():
        bid = b["id"]

        # polygon + wall segments
        polygon = polygon_vertices_from_wkt(b["wkt"])  # [[lon,lat], ...]
        building_polygons[bid] = polygon

        walls = []
        if len(polygon) >= 2:
            for i in range(len(polygon)):
                a = polygon[i]
                bpt = polygon[(i + 1) % len(polygon)]
                walls.append([a, bpt])
        building_walls[bid] = walls

        # best place (if any)
        best_place = None
        if "building_id" in links.columns:
            subset = links[links["building_id"] == bid]
            if len(subset) > 0:
                best_place = select_best_place_for_building(
                    subset,
                    building_id=bid,
                    max_dist_m=place_radius_m,
                )
        building_places[bid] = best_place or None

    # fetch imagery *once* for the entire area around (lat, lon)
    print(f"[INFO] Fetching imagery around ({lat:.6f}, {lon:.6f}) within {search_radius_m} m")

    temp_building = {
        "id": "shared_area",
        "lat": lat,
        "lon": lon,
        "wkt": None, # unused
    }

    saved = fetch_and_slice_for_building(
        temp_building,
        radius_m=search_radius_m,
        min_capture_date=min_capture_date,
        max_images_per_building=max_images_total,
        prefer_360=prefer_360,
    )

    if not saved:
        print("[WARN] No imagery fetched for area.")
        saved = []

    # build unified image metadata list
    image_data: List[Dict[str, Any]] = []
    for rec in saved:
        lon_rec, lat_rec = _extract_lon_lat(rec, lon, lat)
        image_data.append({
            "image_path": rec.get("path") or rec.get("jpg_path"),
            "compass_angle": rec.get("compass_angle"),
            "coordinates": [lon_rec, lat_rec],
            "is_360": rec.get("is_360", False),
            "camera_type": rec.get("camera_type"),
        })

    # return unified dictionary
    return {
        "input_coordinates": [lon, lat],
        "building_polygons": building_polygons,
        "building_walls": building_walls,
        "places": building_places,
        "image_dicts": image_data,
    }
