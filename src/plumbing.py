#!/usr/bin/env python3
"""
Plumbing: Overture Buildings + Places → Mapillary imagery (download + metadata).

- Input: bbox (xmin,ymin,xmax,ymax)
- For each building in the bbox (capped by --limit-buildings):
    * find nearby Places (spatial join within --place-radius-m)
    * fetch Mapillary images near the building centroid
    * write per-building folder in results/buildings/<building_id>/
      - NN.jpg + NN.json (image + metadata)
      - candidates.json (building, nearby places, image list)

Run:
  python -m src.plumbing --bbox="-122.4350,37.7829,-122.4315,37.7853" \
    --radius-m=40 --place-radius-m=60 --limit-buildings=5 --max-images-per-building=12

Prereqs:
  - MAPILLARY_ACCESS_TOKEN in .env or environment
  - src/mly_utils.py present (with fetch_images, _bearing)
"""

from __future__ import annotations

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import duckdb
import pandas as pd
import dotenv
import requests

import src.mly_utils as mly  # Sid's helpers (with attribution in that file)

dotenv.load_dotenv()

# ---------- Config ----------
RELEASE = "2025-09-24.0"
S3_URL_BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
PLACES_URL       = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"

RESULTS_ROOT = Path("results/buildings")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------- Utilities ----------
def _open_duckdb():
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_use_ssl=true;")
    con.execute("SET enable_object_cache=true;")
    con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")
    return con


def _download_image(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            return
        except Exception as e:
            if attempt == 0:
                print(f"      retry after error: {e}")
            else:
                print(f"      failed: {e}")


def parse_bbox_string(bbox_str: str) -> Dict[str, float]:
    parts = [float(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError('bbox must be "xmin,ymin,xmax,ymax"')
    return {"xmin": parts[0], "ymin": parts[1], "xmax": parts[2], "ymax": parts[3]}


# ---------- Overture: Buildings & Places ----------
def query_buildings_with_wkt(bbox: Dict[str, float]) -> pd.DataFrame:
    """
    Return buildings in bbox with centroid and WKT geometry.
    """
    sql = f"""
    SELECT
      id,
      geometry,
      bbox,
      ST_AsText(geometry) AS wkt,
      ST_X(ST_Centroid(geometry)) AS lon,
      ST_Y(ST_Centroid(geometry)) AS lat
    FROM read_parquet('{S3_URL_BUILDINGS}', hive_partitioning=1)
    WHERE
      struct_extract(bbox,'xmin') BETWEEN {bbox['xmin']} AND {bbox['xmax']}
      AND struct_extract(bbox,'ymin') BETWEEN {bbox['ymin']} AND {bbox['ymax']};
    """
    con = _open_duckdb()
    con.execute(f"SELECT COUNT(*) FROM glob('{S3_URL_BUILDINGS}')").fetchone()
    df = con.execute(sql).fetchdf()
    con.close()
    return df


def query_buildings_places_join(bbox: Dict[str, float], radius_m: int = 30) -> pd.DataFrame:
    """
    Spatial join: link Places to Buildings if inside OR within ~radius_m (meters).
    Returns rows with building_id, place_id, place lon/lat, names, categories.
    """
    mean_lat = (bbox["ymin"] + bbox["ymax"]) / 2.0
    deg_lat = radius_m / 111_320.0
    deg_lon = deg_lat / max(0.01, abs(math.cos(math.radians(mean_lat))))
    # expand bbox a hair so near-outside places are included
    bbx = {
        "xmin": bbox["xmin"] - deg_lon,
        "xmax": bbox["xmax"] + deg_lon,
        "ymin": bbox["ymin"] - deg_lat,
        "ymax": bbox["ymax"] + deg_lat,
    }

    con = _open_duckdb()
    con.execute(f"SELECT COUNT(*) FROM glob('{S3_URL_BUILDINGS}')").fetchone()
    con.execute(f"SELECT COUNT(*) FROM glob('{PLACES_URL}')").fetchone()

    sql = f"""
    WITH buildings AS (
      SELECT
        id AS building_id,
        geometry AS bgeom,
        struct_extract(bbox,'xmin') AS b_xmin,
        struct_extract(bbox,'xmax') AS b_xmax,
        struct_extract(bbox,'ymin') AS b_ymin,
        struct_extract(bbox,'ymax') AS b_ymax,
        ST_Centroid(geometry) AS bcentroid
      FROM read_parquet('{S3_URL_BUILDINGS}', hive_partitioning=1)
      WHERE
        struct_extract(bbox,'xmin') BETWEEN {bbox['xmin']} AND {bbox['xmax']}
        AND struct_extract(bbox,'ymin') BETWEEN {bbox['ymin']} AND {bbox['ymax']}
    ),
    places AS (
      SELECT
        id AS place_id,
        geometry AS pgeom,
        struct_extract(bbox,'xmin') AS p_xmin,
        struct_extract(bbox,'xmax') AS p_xmax,
        struct_extract(bbox,'ymin') AS p_ymin,
        struct_extract(bbox,'ymax') AS p_ymax,
        names,
        categories,
        ST_Centroid(geometry) AS pcentroid
      FROM read_parquet('{PLACES_URL}', hive_partitioning=1)
      WHERE
        struct_extract(bbox,'xmin') BETWEEN {bbx['xmin']} AND {bbx['xmax']}
        AND struct_extract(bbox,'ymin') BETWEEN {bbx['ymin']} AND {bbx['ymax']}
    ),
    cand AS (
      SELECT *
      FROM places p
      JOIN buildings b
        ON p.p_xmax >= b.b_xmin AND p.p_xmin <= b.b_xmax
       AND p.p_ymax >= b.b_ymin AND p.p_ymin <= b.b_ymax
    )
    SELECT
      b.building_id,
      p.place_id,
      ST_X(p.pcentroid) AS place_lon,
      ST_Y(p.pcentroid) AS place_lat,
      p.names,
      p.categories,
      ST_Contains(b.bgeom, p.pgeom) AS inside,
      ST_DWithin(p.pgeom, b.bgeom, {deg_lat}) AS within
    FROM cand p
    JOIN buildings b ON p.building_id = b.building_id
    WHERE ST_Contains(b.bgeom, p.pgeom) OR ST_DWithin(p.pgeom, b.bgeom, {deg_lat});
    """
    df = con.execute(sql).fetchdf()
    con.close()
    return df

def _deg_per_meter(lat_deg: float):
    # meters → degrees for small radii; longitude scales by cos(lat)
    dlat = 1.0 / 111_320.0
    dlon = dlat / max(0.01, abs(math.cos(math.radians(lat_deg))))
    return dlat, dlon

def get_places_near_building_centroid(
    lon: float,
    lat: float,
    radius_m: int,
    max_places: int = 20,
    category_allow: tuple[str, ...] = ("restaurant", "cafe", "food", "bar", "shop", "retail", "store", "service"),
) -> pd.DataFrame:
    """
    Return up to `max_places` Places within ~radius_m of (lon,lat),
    filtered to frontage-relevant categories and sorted by distance to the building centroid.
    """
    dlat = 1.0 / 111_320.0
    dlon = dlat / max(0.01, abs(math.cos(math.radians(lat))))
    bbx = {
        "xmin": lon - radius_m * dlon,
        "xmax": lon + radius_m * dlon,
        "ymin": lat - radius_m * dlat,
        "ymax": lat + radius_m * dlat,
    }

    # simple category predicate: names/categories may be arrays or strings in Overture.
    # We'll do a lowercase string search on categories JSON text for now.
    cat_pred = " OR ".join([f"lower(cats) LIKE '%{c}%'" for c in category_allow])

    con = _open_duckdb()
    con.execute(f"SELECT COUNT(*) FROM glob('{PLACES_URL}')").fetchone()

    sql = f"""
    WITH raw AS (
      SELECT
        id AS place_id,
        geometry AS pgeom,
        names,
        categories,
        CAST(categories AS VARCHAR) AS cats, -- to text for LIKE filtering
        ST_X(geometry) AS place_lon,
        ST_Y(geometry) AS place_lat
      FROM read_parquet('{PLACES_URL}', hive_partitioning=1)
      WHERE
        struct_extract(bbox,'xmin') BETWEEN {bbx['xmin']} AND {bbx['xmax']}
        AND struct_extract(bbox,'ymin') BETWEEN {bbx['ymin']} AND {bbx['ymax']}
    ),
    filtered AS (
      SELECT *
      FROM raw
      WHERE ({cat_pred})
    ),
    with_dist AS (
      SELECT
        *,
        /* degree-distance approx in meters (good for ≤200 m) */
        SQRT( POW( (place_lat - {lat}) * 111320.0, 2 )
            + POW( (place_lon - {lon}) * 111320.0 * COS(RADIANS({lat})), 2 ) ) AS dist_m
      FROM filtered
      WHERE ST_DWithin(pgeom, ST_Point({lon}, {lat}), {radius_m} / 111320.0)
    )
    SELECT
      place_id, place_lon, place_lat, names, categories, dist_m
    FROM with_dist
    ORDER BY dist_m ASC
    LIMIT {max_places};
    """
    df = con.execute(sql).fetchdf()
    con.close()
    return df


# ---------- Mapillary fetching ----------
def fetch_images_for_building_row(
    b_row: pd.Series,
    radius_m: int = 40,
    min_capture_date: Optional[str] = None,
    apply_fov: bool = True,
    max_images_per_building: int = 12,
) -> List[Dict]:
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing (set in .env or env)")

    b_id = b_row["id"]
    lat = float(b_row["lat"])
    lon = float(b_row["lon"])
    out_dir = RESULTS_ROOT / str(b_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Building {b_id} @ ({lat:.6f},{lon:.6f}) – radius {radius_m} m")
    images = mly.fetch_images(
        token=token,
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        min_capture_date_filter=min_capture_date,
    )
    if not images:
        print("  ↳ no images nearby")
        return []

    if apply_fov:
        half_angle = 50
        def fov_pass(im):
            try:
                compass = im.get("compass_angle") or im.get("computed_compass_angle")
                geom = im.get("computed_geometry") or im.get("geometry")
                if compass is None or geom is None:
                    return False
                cam_lon, cam_lat = geom["coordinates"]
                b = mly._bearing(cam_lat, cam_lon, lat, lon)
                diff = abs((b - compass + 180) % 360 - 180)
                return diff <= half_angle
            except Exception:
                return False
        images = sorted(images, key=lambda im: not fov_pass(im))

    images = images[:max_images_per_building]
    print(f"  ↳ saving {len(images)} images to {out_dir}")

    saved = []
    for idx, im in enumerate(images, start=1):
        meta_path = out_dir / f"{idx:02d}.json"
        meta_path.write_text(json.dumps(im, indent=2), encoding="utf-8")
        url = (
            im.get("thumb_1024_url")
            or im.get("thumb_2048_url")
            or im.get("thumb_256_url")
            or im.get("thumb_original_url")
        )
        img_path = out_dir / f"{idx:02d}.jpg" if url else None
        if url:
            _download_image(url, img_path)
        saved.append({"file": img_path.name if img_path else None, "meta": meta_path.name})
    print(f"  ✓ done building {b_id}")
    return saved


def write_candidates_json(
    building_row: pd.Series,
    places_df: pd.DataFrame,
    saved_images: List[Dict],
) -> None:
    out_dir = RESULTS_ROOT / str(building_row["id"])
    out = {
        "building": {
            "id": building_row["id"],
            "lon": float(building_row["lon"]),
            "lat": float(building_row["lat"]),
            "wkt": building_row.get("wkt", None),
        },
        "places_nearby": [
            {
                "id": str(r["place_id"]),
                "lon": float(r["place_lon"]),
                "lat": float(r["place_lat"]),
                "names": r.get("names", None),
                "categories": r.get("categories", None),
                "distance_m": float(r.get("dist_m", 0.0)),
            } for _, r in places_df.iterrows()
        ],

        "images": saved_images,
    }
    (out_dir / "candidates.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  ✓ wrote {out_dir/'candidates.json'}")


# ---------- CLI (unified writer) ----------
def main():
    ap = argparse.ArgumentParser(description="Fetch Overture Buildings/Places and Mapillary images; write results per building.")
    ap.add_argument("--bbox", required=True, help='xmin,ymin,xmax,ymax (e.g. "-122.4350,37.7829,-122.4315,37.7853")')
    ap.add_argument("--radius-m", type=int, default=40, help="Mapillary search radius around building centroid (m)")
    ap.add_argument("--place-radius-m", type=int, default=60, help="Places-to-building spatial join threshold (m)")
    ap.add_argument("--limit-buildings", type=int, default=10)
    ap.add_argument("--max-images-per-building", type=int, default=12)
    ap.add_argument("--min-capture-date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--no-fov", action="store_true", help="Disable FOV/heading filter before saving images")
    args = ap.parse_args()

    bbox = parse_bbox_string(args.bbox)

    # 1) Buildings in bbox (with wkt)
    bdf = query_buildings_with_wkt(bbox)
    if args.limit_buildings and len(bdf) > args.limit_buildings:
        bdf = bdf.sample(args.limit_buildings, random_state=42).reset_index(drop=True)
    print(f"Found {len(bdf)} buildings in bbox.")

    # 2) Places×Buildings spatial links for the bbox
    links = query_buildings_places_join(bbox, radius_m=args.place_radius_m)
    print(f"Found {len(links)} place↔building links in area.")

    # 3) For each building: subset places, fetch images, write candidates.json
    for _, b in bdf.iterrows():
        b_places = get_places_near_building_centroid(
            lon=float(b["lon"]),
            lat=float(b["lat"]),
            radius_m=max(40, args.place_radius_m),  # try 40–60
            max_places=20
        )

        saved = fetch_images_for_building_row(
            b,
            radius_m=args.radius_m,
            min_capture_date=args.min_capture_date,
            apply_fov=not args.no_fov,
            max_images_per_building=args.max_images_per_building,     
        )
        write_candidates_json(b, b_places, saved)


if __name__ == "__main__":
    main()
