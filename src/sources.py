from pathlib import Path
from typing import Dict, Tuple
from .config import LOCAL_BUILDINGS, LOCAL_PLACES, BUILDINGS_SRC, PLACES_SRC, RELEASE

S3_BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
S3_PLACES    = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"

def resolve_sources(_: Dict[str,float], src_mode: str = "auto") -> Tuple[str,str]:
    """auto: prefer local files if they exist, else S3. local: require local. s3: force S3."""
    lb = LOCAL_BUILDINGS if LOCAL_BUILDINGS.exists() else None
    lp = LOCAL_PLACES    if LOCAL_PLACES.exists()    else None
    if src_mode == "s3":
        return BUILDINGS_SRC, PLACES_SRC
    if src_mode == "local":
        if not (lb and lp):
            raise RuntimeError("local mode but local geoparquet missing.")
        return str(lb), str(lp)
    # auto
    return (str(lb) if lb else S3_BUILDINGS, str(lp) if lp else S3_PLACES)
