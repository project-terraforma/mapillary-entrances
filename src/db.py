import duckdb

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
