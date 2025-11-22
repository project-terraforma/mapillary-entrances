# src/download_v2.py

from pathlib import Path
from utils.constants import S3_BUILDINGS, S3_PLACES
from utils.duck_db_utils import open_duckdb, _try
from utils.polygon_utils import bbox_to_wkt
from utils.geo_utils import robust_radius_to_bbox


def download_overture_radius(lat: float, lon: float, radius_m: int,
                             out_buildings: str = "../data/buildings_radius.parquet",
                             out_places: str = "../data/places_radius.parquet"):
    """
    Download Overture buildings & places within *radius_m* of (lat, lon)
    and save to local parquet files.
    """
    bbox = robust_radius_to_bbox(lat, lon, radius_m)
    xmin, ymin, xmax, ymax = bbox
    aoi_wkt = bbox_to_wkt(bbox)

    out_b = Path(out_buildings); out_b.parent.mkdir(parents=True, exist_ok=True)
    out_p = Path(out_places);    out_p.parent.mkdir(parents=True, exist_ok=True)

    con = open_duckdb()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")

    _try(con, "SET s3_region='us-west-2'")
    _try(con, "SET s3_url_style='path'")
    _try(con, "SET s3_endpoint='s3.us-west-2.amazonaws.com'")
    _try(con, "SET enable_object_cache=true")
    _try(con, "SET http_keep_alive=true")
    _try(con, "SET http_timeout=30")
    _try(con, "SET threads=8")

    con.execute(f"CREATE OR REPLACE TABLE aoi AS SELECT ST_GeomFromText('{aoi_wkt}') AS g;")

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

