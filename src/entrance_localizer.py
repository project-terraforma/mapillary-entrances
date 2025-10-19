# entrance_localizer.py
import json, os, glob
from pathlib import Path
from shapely.geometry import shape, LineString, Point, Polygon
from src import geo_estimator as geo

HFOV_DEG = 70.0  # rough assumption; you can tune later

def bbox_center_bearing(base_heading_deg: float, img_width_px: int, x_center_px: int) -> float:
    # map pixel offset to degrees: offset_norm * HFOV
    offset = (x_center_px / img_width_px) - 0.5  # [-0.5 .. +0.5]
    return (base_heading_deg + offset * HFOV_DEG) % 360

def build_ray_from_meta(meta: dict, bearing_deg: float, length_m=60.0) -> LineString:
    cam_lon, cam_lat = (
        (meta.get("computed_geometry") or meta.get("geometry"))["coordinates"]
    )

    return geo.single_image_line(None, meta, 'line', None, None, None)  # or implement your own small helper

def localize_building_entrance(building_poly: Polygon, image_dir: Path) -> dict | None:
    lines = []
    for meta_file in sorted(image_dir.glob("*.txt")):
        with open(meta_file) as f:
            meta = json.load(f)
        det_file = meta_file.with_suffix(".detections.json")
        if not det_file.exists():
            continue
        dets = json.loads(det_file.read_text())
        door_like = [d for d in dets if d["cls"] in ("door","storefront","building")]  # starter filter
        if not door_like:
            continue
        # pick highest conf
        d = max(door_like, key=lambda x: x["conf"])
        x_center = int(d["bbox"][0])  # xywh center-x in px
        width_px = meta.get("width", meta.get("thumb_width", 1024))  # fallback
        heading = meta.get("compass_angle") or meta.get("computed_compass_angle")
        if heading is None:
            continue
        bearing = bbox_center_bearing(heading, width_px, x_center)
        ray = build_ray_from_meta(meta, bearing)
        lines.append(ray)

    if not lines:
        return None

    # If 1 ray: intersect with building boundary
    if len(lines) == 1:
        inter = lines[0].intersection(building_poly.boundary)
        pt = inter.geoms[0] if hasattr(inter, "geoms") else inter
        return {"lon": pt.x, "lat": pt.y, "n_images": 1}

    # If ≥2 rays: triangulate (use geo_estimator.triangulate if it takes lines)
    # Otherwise, fall back to nearest boundary point of first ray intersection
