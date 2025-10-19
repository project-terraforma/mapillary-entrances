import os, json
from pathlib import Path
from typing import Dict, List, Optional
from shapely.wkt import loads as load_wkt
from shapely.geometry import Polygon
from .config import RESULTS_ROOT
from .pano_slices import slice_equirectangular
from . import mly_utils as mly

def _download_image(url: str, out_path: Path) -> None:
    import requests
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

def view_cone_polygon(cam_lon, cam_lat, heading_deg, half_angle_deg, length_m) -> Polygon:
    from math import radians, cos, sin
    from shapely.affinity import rotate, translate
    dlat = 1/111_320.0
    dlon = dlat / max(0.01, abs(cos(radians(cam_lat))))
    Lx = length_m * dlon
    Ly = length_m * dlat
    tri = Polygon([(0,0), (-Lx*sin(radians(half_angle_deg)), Ly*cos(radians(half_angle_deg))),
                           ( Lx*sin(radians(half_angle_deg)), Ly*cos(radians(half_angle_deg)))])
    tri = rotate(tri, angle=heading_deg, origin=(0,0), use_radians=False)
    return translate(tri, xoff=cam_lon, yoff=cam_lat)

def fetch_and_slice_for_building(
    b_row,
    radius_m: int = 120,
    min_capture_date: Optional[str] = None,
    apply_fov: bool = True,
    max_images_per_building: int = 8,
    prefer_360: bool = True,
    fov_half_angle: float = 25.0,
) -> List[Dict]:
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing")

    b_id = b_row["id"]
    lat = float(b_row["lat"]); lon = float(b_row["lon"])
    out_dir = RESULTS_ROOT / str(b_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = mly.fetch_images(
        token=token, lat=lat, lon=lon, radius_m=radius_m,
        min_capture_date_filter=min_capture_date, prefer_360=prefer_360
    )
    if not images:
        print("  ↳ no images nearby"); return []

    if apply_fov:
        def fov_pass(im):
            try:
                compass = im.get("compass_angle") or im.get("computed_compass_angle")
                geom = im.get("computed_geometry") or im.get("geometry")
                if compass is None or geom is None: return False
                cam_lon, cam_lat = geom["coordinates"]
                b = mly._bearing(cam_lat, cam_lon, lat, lon)
                diff = abs((b - compass + 180) % 360 - 180)
                return diff <= fov_half_angle
            except Exception:
                return False
        images = sorted(images, key=lambda im: not fov_pass(im))

    building_polygon = None
    wkt_str = b_row.get("wkt")
    if wkt_str:
        try: building_polygon = load_wkt(wkt_str)
        except Exception: pass

    if building_polygon is not None:
        filtered = []
        for im in images:
            compass = im.get("compass_angle") or im.get("computed_compass_angle")
            geom = im.get("computed_geometry") or im.get("geometry")
            if not (compass and geom and isinstance(geom.get("coordinates"), (list, tuple))):
                continue
            cam_lon, cam_lat = geom["coordinates"]
            cone = view_cone_polygon(cam_lon, cam_lat, compass, fov_half_angle, radius_m)
            if cone.intersects(building_polygon):
                filtered.append(im)
        images = filtered

    images = images[:max_images_per_building]
    print(f"  ↳ saving {len(images)} images to {out_dir}")

    saved, slice_saved = [], []
    for idx, im in enumerate(images, start=1):
        meta_path = out_dir / f"{idx:02d}.json"
        meta_path.write_text(json.dumps(im, indent=2), encoding="utf-8")
        url = (im.get("thumb_1024_url") or im.get("thumb_2048_url") or
               im.get("thumb_256_url")  or im.get("thumb_original_url"))
        img_path = out_dir / f"{idx:02d}.jpg" if url else None
        if url: _download_image(url, img_path)
        saved.append({"file": img_path.name if img_path else None, "meta": meta_path.name})

        if img_path and ((im.get("camera_type") or "").lower() in ("spherical","equirectangular","panorama","panoramic","360")):
            chips = slice_equirectangular(
                img_path=img_path, out_dir=out_dir / "slices",
                slice_deg=int(os.getenv("PANO_SLICE_DEG","60")),
                overlap_deg=int(os.getenv("PANO_SLICE_OVERLAP","0")),
            )
            compass = im.get("compass_angle") or im.get("computed_compass_angle")
            geom = im.get("computed_geometry") or im.get("geometry")
            if compass is None or geom is None:
                for ch in chips:
                    slice_saved.append({"source": saved[-1]["file"], "slice_path": ch["path"]})
            else:
                cam_lon, cam_lat = geom["coordinates"]
                bearing_to_building = mly._bearing(cam_lat, cam_lon, lat, lon)
                slice_deg = int(os.getenv("PANO_SLICE_DEG","60")); half_w = slice_deg/2
                for ch in chips:
                    center_local = ch["center_local_deg"]
                    center_abs   = (compass + (center_local - 180.0)) % 360.0
                    diff = abs((bearing_to_building - center_abs + 180) % 360 - 180)
                    if diff <= half_w:
                        slice_saved.append({
                            "source": saved[-1]["file"], "slice_path": ch["path"],
                            "center_abs_deg": center_abs,
                            "bearing_to_building_deg": bearing_to_building,
                            "angle_diff_deg": diff
                        })
    if slice_saved:
        (out_dir / "slices_manifest.json").write_text(json.dumps(slice_saved, indent=2), encoding="utf-8")

    print(f"  ✓ done building {b_id} ({len(saved)} originals, {len(slice_saved)} kept 360-slices)")
    return saved

def write_candidates_json(b_row, best_place, saved_images):
    out_dir = (RESULTS_ROOT / str(b_row["id"]))
    out = {
        "building": {"id": b_row["id"], "lon": float(b_row["lon"]), "lat": float(b_row["lat"]), "wkt": b_row.get("wkt", None)},
        "place": best_place,
        "images": saved_images,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  ✓ wrote {out_dir/'candidates.json'}")
