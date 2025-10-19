import os, time, pandas as pd
from typing import Dict
from .db import open_duckdb, read_parquet_expr

def get_buildings(bbox: Dict[str,float], buildings_src: str, limit_hint: int = 10, id_multiplier: int = 5) -> pd.DataFrame:
    t0 = time.perf_counter()
    print(f"[buildings/A] bbox={bbox} (ids only) src={buildings_src}", flush=True)
    limit_ids = max(20, min(1000, (limit_hint or 10) * id_multiplier))
    con = open_duckdb()

    rb = read_parquet_expr(buildings_src)
    sql_ids = f"""
    WITH q AS (
      SELECT {bbox['xmin']} AS xmin, {bbox['ymin']} AS ymin,
             {bbox['xmax']} AS xmax, {bbox['ymax']} AS ymax,
             ({(bbox['xmin']+bbox['xmax'])/2.0}) AS qlon,
             ({(bbox['ymin']+bbox['ymax'])/2.0}) AS qlat
    ),
    cand AS (
      SELECT id,
             (struct_extract(bbox,'xmin') + struct_extract(bbox,'xmax'))/2.0 AS lon_mid,
             (struct_extract(bbox,'ymin') + struct_extract(bbox,'ymax'))/2.0 AS lat_mid
      FROM {rb}
      WHERE struct_extract(bbox,'xmax') >= (SELECT xmin FROM q)
        AND struct_extract(bbox,'xmin') <= (SELECT xmax FROM q)
        AND struct_extract(bbox,'ymax') >= (SELECT ymin FROM q)
        AND struct_extract(bbox,'ymin') <= (SELECT ymax FROM q)
    )
    SELECT id
    FROM (
      SELECT id,
             SQRT( POW((lat_mid - (SELECT qlat FROM q)) * 111320.0, 2) +
                   POW((lon_mid - (SELECT qlon FROM q)) * 111320.0 * COS(RADIANS((SELECT qlat FROM q))), 2 ) ) AS dist_m
      FROM cand
    )
    ORDER BY dist_m ASC
    LIMIT {limit_ids};
    """
    ids_df = con.execute(sql_ids).fetchdf()
    print(f"[buildings/A] candidate ids: {len(ids_df)}  [{time.perf_counter()-t0:0.2f}s]", flush=True)
    if ids_df.empty:
        con.close()
        return ids_df

    con.register("cand_ids", ids_df[["id"]])
    sql_rows = f"""
      SELECT b.id, b.geometry, b.bbox,
             ST_AsText(b.geometry) AS wkt,
             ST_X(ST_Centroid(b.geometry)) AS lon,
             ST_Y(ST_Centroid(b.geometry)) AS lat
      FROM {rb} b JOIN cand_ids c ON b.id = c.id
    """
    df = con.execute(sql_rows).fetchdf()
    con.close()
    print(f"[buildings/B] fetched {len(df)} rows with geometry", flush=True)
    return df
