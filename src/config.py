# ---------- Config ----------
from pathlib import Path

RELEASE = "2025-09-24.0"

# Prefer local extracts if present; else use S3
LOCAL_BUILDINGS = Path("data/buildings_soma.parquet")
LOCAL_PLACES    = Path("data/places_soma.parquet")

BUILDINGS_SRC = str(LOCAL_BUILDINGS) if LOCAL_BUILDINGS.exists() else \
    f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"

PLACES_SRC = str(LOCAL_PLACES) if LOCAL_PLACES.exists() else \
    f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"
RESULTS_ROOT = Path("results/buildings")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)