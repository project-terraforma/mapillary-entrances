# Reads buildings parquet and returns a DataFrame with id, lon, lat, wkt

import os, time, pandas as pd
from typing import Dict
from .db import open_duckdb, read_parquet_expr

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
