# duck_db_utils.py

import duckdb
import pandas as pd
import time
import math
from typing import Dict

def open_duckdb():
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_use_ssl=true;")
    con.execute("SET enable_object_cache=true;")
    con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")
    con.execute("PRAGMA memory_limit='1GB';")
    con.execute("PRAGMA threads=4;")
    return con

def read_parquet_expr(path: str) -> str:
    # for local single files we don’t need hive_partitioning
    hp = ", hive_partitioning=1" if path.startswith("s3://") else ""
    return f"read_parquet('{path}'{hp})"

def _try(con, stmt: str):
    try:
        con.execute(stmt)
    except Exception as e:
        print(f"[warn] skipped setting: {stmt} ({e})")

def get_buildings(bbox: Dict[str,float], buildings_src: str, limit_hint: int = 10, id_multiplier: int = 5) -> pd.DataFrame:
    t0 = time.perf_counter()
    print(f"[buildings/A] bbox={bbox} (ids only) src={buildings_src}", flush=True)
    limit_ids = max(20, min(1000, (limit_hint or 10) * id_multiplier))
    con = open_duckdb()

    rb = read_parquet_expr(buildings_src)

    # Detect available columns to choose the right WHERE clause
    cols = con.execute(f"SELECT * FROM {rb} LIMIT 0").fetchdf().columns.tolist()
    has_bbox = "bbox" in cols
    has_lonlat = "lon" in cols and "lat" in cols
    has_wkt = "wkt" in cols
    has_geometry = "geometry" in cols

    # Prefer bbox if present; else use lon/lat; else fall back to centroid of wkt/geometry
    if has_bbox:
        where_ids = f"""
          struct_extract(bbox,'xmax') >= {bbox['xmin']} AND
          struct_extract(bbox,'xmin') <= {bbox['xmax']} AND
          struct_extract(bbox,'ymax') >= {bbox['ymin']} AND
          struct_extract(bbox,'ymin') <= {bbox['ymax']}
        """
    elif has_lonlat:
        where_ids = f"""
          lon BETWEEN {bbox['xmin']} AND {bbox['xmax']} AND
          lat BETWEEN {bbox['ymin']} AND {bbox['ymax']}
        """
    elif has_wkt:
        # compute centroid on the fly
        where_ids = f"""
          ST_X(ST_Centroid(ST_GeomFromText(wkt))) BETWEEN {bbox['xmin']} AND {bbox['xmax']} AND
          ST_Y(ST_Centroid(ST_GeomFromText(wkt))) BETWEEN {bbox['ymin']} AND {bbox['ymax']}
        """
    elif has_geometry:
        where_ids = f"""
          ST_X(ST_Centroid(geometry)) BETWEEN {bbox['xmin']} AND {bbox['xmax']} AND
          ST_Y(ST_Centroid(geometry)) BETWEEN {bbox['ymin']} AND {bbox['ymax']}
        """
    else:
        con.close()
        raise RuntimeError("buildings parquet missing usable columns (need bbox OR (lon,lat) OR wkt/geometry)")

    # Stage A: collect candidate IDs (fast)
    sql_ids = f"""
    WITH q AS (
      SELECT {bbox['xmin']} AS xmin, {bbox['ymin']} AS ymin,
             {bbox['xmax']} AS xmax, {bbox['ymax']} AS ymax
    )
    SELECT id
    FROM {rb}
    WHERE {where_ids}
    LIMIT {limit_ids}
    """
    ids_df = con.execute(sql_ids).fetchdf()
    print(f"[buildings/A] candidate ids: {len(ids_df)}  [{time.perf_counter() - t0:0.2f}s]")

    if ids_df.empty:
        con.close()
        return pd.DataFrame(columns=["id","lon","lat","wkt"])

    # Stage B: fetch full rows for those IDs
    ids_list = ",".join([f"'{i}'" for i in ids_df["id"].astype(str).tolist()])
    if has_wkt and has_lonlat:
        sql_rows = f"""
          SELECT id, wkt, lon, lat
          FROM {rb}
          WHERE id IN ({ids_list})
        """
    elif has_geometry:
        # produce consistent columns
        sql_rows = f"""
          SELECT
            id,
            ST_AsText(geometry) AS wkt,
            ST_X(ST_Centroid(geometry)) AS lon,
            ST_Y(ST_Centroid(geometry)) AS lat
          FROM {rb}
          WHERE id IN ({ids_list})
        """
    else:
        # last resort: compute lon/lat from wkt if lon/lat missing
        sql_rows = f"""
          SELECT
            id,
            wkt,
            ST_X(ST_Centroid(ST_GeomFromText(wkt))) AS lon,
            ST_Y(ST_Centroid(ST_GeomFromText(wkt))) AS lat
          FROM {rb}
          WHERE id IN ({ids_list})
        """

    rows_df = con.execute(sql_rows).fetchdf()
    print(f"[buildings/B] fetched {len(rows_df)} rows with geometry", flush=True)
    con.close()
    return rows_df

def join_buildings_places(bdf: pd.DataFrame, bbox: Dict[str,float], places_src: str, radius_m: int = 60) -> pd.DataFrame:
    t0 = time.perf_counter()
    print(f"[places-join] radius_m={radius_m}", flush=True)

    mean_lat = (bbox["ymin"] + bbox["ymax"]) / 2.0
    deg_lat  = radius_m / 111_320.0
    deg_lon  = deg_lat / max(0.01, abs(math.cos(math.radians(mean_lat))))
    bbx = {
        "xmin": bbox["xmin"] - deg_lon,
        "xmax": bbox["xmax"] + deg_lon,
        "ymin": bbox["ymin"] - deg_lat,
        "ymax": bbox["ymax"] + deg_lat,
    }

    slim = bdf[["id","wkt","lon","lat"]].copy()
    con = open_duckdb()
    con.register("buildings_df", slim)
    rp = read_parquet_expr(places_src)

    # detect columns by selecting zero rows from the table function
    cols = con.execute(f"SELECT * FROM {rp} LIMIT 0").fetchdf().columns.tolist()
    has_bbox = "bbox" in cols

    bbox_where = ""
    if has_bbox:
        bbox_where = f"""
        AND struct_extract(bbox,'xmax') >= {bbx['xmin']}
        AND struct_extract(bbox,'xmin') <= {bbx['xmax']}
        AND struct_extract(bbox,'ymax') >= {bbx['ymin']}
        AND struct_extract(bbox,'ymin') <= {bbx['ymax']}
        """

    sql = f"""
    WITH buildings AS (
      SELECT id AS building_id, ST_GeomFromText(wkt) AS bgeom, lon AS b_lon, lat AS b_lat
      FROM buildings_df
    ),
    places AS (
      SELECT id AS place_id, geometry AS pgeom, names, categories, ST_Centroid(geometry) AS pcentroid
      FROM {rp}
      WHERE 1=1
      {bbox_where}
    )
    SELECT
      b.building_id, p.place_id,
      ST_X(p.pcentroid) AS place_lon, ST_Y(p.pcentroid) AS place_lat,
      p.names, p.categories,
      ST_Contains(b.bgeom, p.pgeom) AS inside,
      ST_DWithin(p.pgeom, b.bgeom, {deg_lat}) AS within,
      SQRT( POW((ST_Y(p.pcentroid) - b.b_lat) * 111320.0, 2) +
            POW((ST_X(p.pcentroid) - b.b_lon) * 111320.0 * COS(RADIANS(b.b_lat)), 2) ) AS dist_m
    FROM places p, buildings b
    WHERE ST_Contains(b.bgeom, p.pgeom) OR ST_DWithin(p.pgeom, b.bgeom, {deg_lat});
    """
    df = con.execute(sql).fetchdf()
    con.close()
    print(f"[places-join] links={len(df)}", flush=True)
    return df