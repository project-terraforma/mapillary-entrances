import argparse
from pathlib import Path
from shapely.geometry import box
from src.db import open_duckdb
from src.config import RELEASE

"""
PYTHONPATH=. python3 -m src.download_bbox_extract \
  --bbox="-122.4106,37.7918,-122.4028,37.7980" \
  --out-buildings data/buildings_union.parquet \
  --out-places    data/places_union.parquet

"""


S3_BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
S3_PLACES    = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"

def bbox_to_wkt(b):
    xmin, ymin, xmax, ymax = b
    return box(xmin, ymin, xmax, ymax).wkt

def _try(con, stmt: str):
    try:
        con.execute(stmt)
    except Exception as e:
        print(f"[warn] skipped setting: {stmt} ({e})")

def main():
    ap = argparse.ArgumentParser(description="Download bbox extract from Overture S3 → local parquet")
    ap.add_argument("--bbox", required=True, help="xmin,ymin,xmax,ymax (lon/lat)")
    ap.add_argument("--out-buildings", default="data/buildings_bbox.parquet")
    ap.add_argument("--out-places", default="data/places_bbox.parquet")
    args = ap.parse_args()

    xmin, ymin, xmax, ymax = [float(x) for x in args.bbox.split(",")]
    aoi_wkt = bbox_to_wkt((xmin, ymin, xmax, ymax))

    out_b = Path(args.out_buildings); out_b.parent.mkdir(parents=True, exist_ok=True)
    out_p = Path(args.out_places);    out_p.parent.mkdir(parents=True, exist_ok=True)

    con = open_duckdb()

    # Extensions
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")

    # S3/http settings – keep it simple & tolerant across versions
    _try(con, "SET s3_region='us-west-2'")
    _try(con, "SET s3_url_style='path'")
    _try(con, "SET s3_endpoint='s3.us-west-2.amazonaws.com'")
    _try(con, "SET enable_object_cache=true")
    _try(con, "SET http_keep_alive=true")
    _try(con, "SET http_timeout=30")        # <- works on your build

    # A little parallelism helps, but don’t go crazy
    _try(con, "SET threads=8")

    # AOI table
    con.execute(f"CREATE OR REPLACE TABLE aoi AS SELECT ST_GeomFromText('{aoi_wkt}') AS g;")

    # Coarse bbox filter (fast) + exact geometry filter (accurate)
    # This uses the row-level 'bbox' struct to prune most data before ST_Intersects.
    bfilter = f"""
      struct_extract(bbox,'xmax') >= {xmin} AND
      struct_extract(bbox,'xmin') <= {xmax} AND
      struct_extract(bbox,'ymax') >= {ymin} AND
      struct_extract(bbox,'ymin') <= {ymax}
    """

    # BUILDINGS
    con.execute(f"""
      COPY (
        WITH src AS (
          SELECT id, geometry, bbox
          FROM read_parquet('{S3_BUILDINGS}')
          WHERE {bfilter}
        )
        SELECT
          id,
          ST_AsText(geometry) AS wkt,
          ST_X(ST_Centroid(geometry)) AS lon,
          ST_Y(ST_Centroid(geometry)) AS lat
        FROM src
        WHERE ST_Intersects(geometry, (SELECT g FROM aoi))
      ) TO '{out_b.as_posix()}' (FORMAT 'PARQUET');
    """)

    # PLACES
    con.execute(f"""
      COPY (
        WITH src AS (
          SELECT id, geometry, names, categories, bbox
          FROM read_parquet('{S3_PLACES}')
          WHERE {bfilter}
        )
        SELECT id, geometry, names, categories, bbox
        FROM src
        WHERE ST_Intersects(geometry, (SELECT g FROM aoi))
      ) TO '{out_p.as_posix()}' (FORMAT 'PARQUET');
    """)

    con.close()
    print(f"✓ wrote {out_b}")
    print(f"✓ wrote {out_p}")

if __name__ == "__main__":
    main()
