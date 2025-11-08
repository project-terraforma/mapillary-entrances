# Chooses where to load data from

from pathlib import Path
from typing import Dict, Tuple
from .config import LOCAL_BUILDINGS, LOCAL_PLACES, RELEASE

S3_BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
S3_PLACES    = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"

def resolve_sources(_: Dict[str,float], src_mode: str = "auto") -> Tuple[str, str]:
    lb = LOCAL_BUILDINGS if LOCAL_BUILDINGS.exists() else None
    lp = LOCAL_PLACES    if LOCAL_PLACES.exists()    else None

    if src_mode == "s3":
        return S3_BUILDINGS, S3_PLACES

    if src_mode == "local":
        if not (lb and lp):
            raise RuntimeError("local mode but local geoparquet missing.")
        return str(lb), str(lp)

    # auto: prefer local, else S3
    return (str(lb) if lb else S3_BUILDINGS,
            str(lp) if lp else S3_PLACES)
