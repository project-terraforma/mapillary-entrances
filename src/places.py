# places.py
import time, math, pandas as pd
from typing import Dict
from .db import open_duckdb, read_parquet_expr

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

    # --- Detect columns by selecting zero rows from the table function ---
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
