#!/usr/bin/env python3
"""
Plumbing: Overture Buildings → Mapillary imagery (download + metadata).

- Queries Overture Buildings from public S3 for a bbox (via DuckDB + spatial ext).
- For each building, fetches nearby Mapillary images (radius, optional date filter).
- Saves images as JPG + their metadata JSON into results/buildings/<building_id>/.

Run:
  python -m src.plumbing --bbox="-122.42,37.77,-122.40,37.78" --radius-m=40 --limit-buildings=50

Notes:
- Requires: MAPILLARY_ACCESS_TOKEN in .env or environment.
- Depends on Sid's helpers in src/mly_utils.py (fetch_images, quality filters, etc.).
- Keep bbox small at first to avoid pulling too much data.
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict

import duckdb
import pandas as pd
import dotenv
import requests

import src.mly_utils as mly  # noqa: E402

dotenv.load_dotenv()

# ---------- Config ----------
RELEASE = "2025-09-24.0"
S3_URL_BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"

# Where to save results
RESULTS_ROOT = Path("results/buildings")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- Small utils ----------

def _download_image(url: str, out_path: Path) -> None:
    """Stream download with a tiny retry loop."""
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




# ---------- Core: Overture query ----------

def query_buildings(bbox: Dict[str, float]) -> pd.DataFrame:
    """
    Return buildings df with id, lon, lat (centroid), geometry, bbox inside bbox.

    Strategy:
      1) S3 via DuckDB httpfs (public Overture bucket)
      2) Local parquet fallback under data/buildings.parquet (if you've downloaded once)
    """
    RELEASE = "2025-09-24.0"
    S3_URL = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"

    sql_tpl = """
    SELECT
      id,
      geometry,
      bbox,
      ST_X(ST_Centroid(geometry)) AS lon,
      ST_Y(ST_Centroid(geometry)) AS lat
    FROM read_parquet('{PATH}', hive_partitioning=1)
    WHERE
      bbox.xmin BETWEEN {xmin} AND {xmax}
      AND bbox.ymin BETWEEN {ymin} AND {ymax};
    """

    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # --- Try S3 first ---
    e_s3 = None
    try:
        # Required for public S3 access
        con.execute("SET s3_region='us-west-2';")
        con.execute("SET s3_url_style='path';")
        con.execute("SET s3_use_ssl=true;")
        con.execute("SET enable_object_cache=true;")
        con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")

        # Sanity check the path resolves
        con.execute(f"SELECT COUNT(*) FROM glob('{S3_URL}')").fetchone()

        sql = sql_tpl.format(PATH=S3_URL, **bbox)
        df = con.execute(sql).fetchdf()
        con.close()
        return df
    except Exception as ex:
        e_s3 = ex
        print(f"[info] S3 access failed ({ex}); trying local parquet under data/…")

    # --- Local fallback (download once with overturemaps CLI) ---
    e_local = None
    try:
        LOCAL_PATH = "data/buildings.parquet"
        sql = sql_tpl.format(PATH=LOCAL_PATH, **bbox)
        df = con.execute(sql).fetchdf()
        con.close()
        return df
    except Exception as ex:
        e_local = ex
        con.close()
        raise RuntimeError(
            "All sources failed.\n"
            f"S3 error: {e_s3}\n"
            f"Local error: {e_local}\n\n"
            "Fix options:\n"
            "  1) Ensure DuckDB httpfs is loaded and you can reach public S3.\n"
            "  2) Download a local parquet once with the CLI:\n"
            f"     overturemaps download --bbox={bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']} "
            " -f parquet --type=building -o data/buildings.parquet\n"
        )



# ---------- Core: Mapillary fetch per building ----------

def fetch_images_for_buildings(
    buildings_df: pd.DataFrame,
    radius_m: int = 40,
    min_capture_date: Optional[str] = None,
    apply_fov: bool = True,
    max_images_per_building: int = 12,
) -> None:
    """
    For each building (expects columns: id, lat, lon), fetch nearby Mapillary images.
    Saves NN.jpg and NN.json metadata files into results/buildings/<building_id>/.
    """
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing (set in .env or env)")

    # Optional: parse min_capture_date (YYYY-MM-DD) in mly_utils if supported
    min_date = None
    if min_capture_date:
        # mly.fetch_images accepts date string in Sid's version; pass through as-is
        min_date = min_capture_date

    for _, row in buildings_df.iterrows():
        b_id = row["id"]
        lat = float(row["lat"])
        lon = float(row["lon"])

        out_dir = RESULTS_ROOT / str(b_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 Building {b_id} @ ({lat:.6f},{lon:.6f}) – radius {radius_m} m")
        images = mly.fetch_images(
            token=token,
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            min_capture_date_filter=min_date,
        )
        if not images:
            print("  ↳ no images nearby")
            continue

        # If FOV filtering is desired, mark pass/fail and keep passers first
        if apply_fov:
            half_angle = 50  # deg (±50°). Tune later.
            images_with_flags = []
            for im in images:
                bearing_ok = False
                try:
                    compass = im.get("compass_angle") or im.get("computed_compass_angle")
                    geom = im.get("computed_geometry") or im.get("geometry")
                    if compass is not None and geom is not None:
                        cam_lon, cam_lat = geom["coordinates"]
                        b = mly._bearing(cam_lat, cam_lon, lat, lon)
                        diff = abs((b - compass + 180) % 360 - 180)
                        bearing_ok = diff <= half_angle
                except Exception:
                    bearing_ok = False
                im["fov_pass"] = bearing_ok
                images_with_flags.append(im)

            pass_images = [im for im in images_with_flags if im["fov_pass"]]
            fail_images = [im for im in images_with_flags if not im["fov_pass"]]
            ordered = pass_images + fail_images
        else:
            ordered = images

        # Cap number to keep it light while prototyping
        ordered = ordered[:max_images_per_building]

        print(f"  ↳ saving {len(ordered)} images to {out_dir}")
        for idx, im in enumerate(ordered, start=1):
            # save metadata
            meta_path = out_dir / f"{idx:02d}.json"
            meta_path.write_text(json.dumps(im, indent=2), encoding="utf-8")

            # pick a thumbnail url
            url = (
                im.get("thumb_1024_url")
                or im.get("thumb_2048_url")
                or im.get("thumb_256_url")
                or im.get("thumb_original_url")
            )
            if not url:
                print("    • no thumbnail url; skipping download")
                continue

            img_path = out_dir / f"{idx:02d}.jpg"
            _download_image(url, img_path)

        print(f"  ✓ done building {b_id}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Query Overture Buildings and fetch nearby Mapillary images."
    )
    ap.add_argument("--bbox", required=True, help='xmin,ymin,xmax,ymax (e.g. "-122.42,37.77,-122.40,37.78")')
    ap.add_argument("--radius-m", type=int, default=40)
    ap.add_argument("--limit-buildings", type=int, default=50)
    ap.add_argument("--min-capture-date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--no-fov", action="store_true", help="Disable FOV filtering")
    args = ap.parse_args()

    bbox = parse_bbox_string(args.bbox)
    bdf = query_buildings(bbox)

    if args.limit_buildings and len(bdf) > args.limit_buildings:
        bdf = bdf.sample(args.limit_buildings, random_state=42).reset_index(drop=True)

    print(f"Found {len(bdf)} buildings in bbox.")
    fetch_images_for_buildings(
        bdf,
        radius_m=args.radius_m,
        min_capture_date=args.min_capture_date,
        apply_fov=not args.no_fov,
    )

if __name__ == "__main__":
    main()
