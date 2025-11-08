# src/selection.py
import math
from typing import Any, Dict, Optional
import pandas as pd

# Simple category prior — tweak as you like
_CAT_PRIOR = {
    "art_museum": 1.2,
    "museum": 1.1,
    "library": 1.1,
    "school": 1.1,
    "university": 1.1,
    "hotel": 1.05,
    "restaurant": 1.05,
}

def _cat_weight(categories: Any) -> float:
    """
    categories is typically a dict like:
      {"primary": "art_museum", "alternate": ["museum", ...]}
    but may be None / string / JSON; be defensive.
    """
    try:
        primary = None
        if isinstance(categories, dict):
            primary = categories.get("primary") or None
        # fallbacks — very conservative
        if primary and isinstance(primary, str):
            return float(_CAT_PRIOR.get(primary, 1.0))
    except Exception:
        pass
    return 1.0

def _extract_name(names) -> str | None:
    # Overture often stores names as nested dicts, e.g., {"common":{"en":"Foo"}}
    try:
        if isinstance(names, dict):
            # Try common buckets first
            for key in ("primary", "common", "official", "label", "name"):
                v = names.get(key)
                if isinstance(v, dict):
                    return v.get("en") or next((x for x in v.values() if isinstance(x, str)), None)
                if isinstance(v, str):
                    return v
            # Fallback: scan any nested dicts for an EN name or any string
            for v in names.values():
                if isinstance(v, dict):
                    cand = v.get("en") or next((x for x in v.values() if isinstance(x, str)), None)
                    if cand:
                        return cand
        elif isinstance(names, str):
            return names
    except Exception:
        pass
    return None


def _as_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    try:
        return float(val) if val is not None else float(default)
    except Exception:
        return float(default)

def _as_bool(row: pd.Series, key: str) -> bool:
    return bool(row.get(key, False))

def _geom_weight(row: pd.Series, max_dist_m: float) -> float:
    """
    Robust geometric score that works even if many columns are missing.
    Components (each optional):
      +1.0 if inside
      +0.5 if within (buffer)
      + up to +0.5 from overlap_ratio (if present)
      + up to +0.5 from touch_len_m/10 (if present; 5m+ saturates)
      + [0..1] from proximity: (1 - dist / max_dist_m)
    """
    w = 0.0

    if _as_bool(row, "inside"):
        w += 1.0
    if _as_bool(row, "within"):
        w += 0.5

    # Optional extras, harmless if absent:
    w += min(0.5, _as_float(row, "overlap_ratio", 0.0))
    w += min(0.5, _as_float(row, "touch_len_m", 0.0) / 10.0)

    d = _as_float(row, "dist_m", float("inf"))
    if math.isfinite(d) and max_dist_m > 0:
        w += max(0.0, 1.0 - (d / max_dist_m))

    return w

def select_best_place_for_building(
    links_df: pd.DataFrame,
    building_id: Optional[str] = None,
    max_dist_m: float = 60.0
) -> Optional[Dict[str, Any]]:
    """
    Picks a single 'best' place link for a building.
    Works even if the links_df lacks fancy geometric columns.

    Returns a dict with keys:
      place_id, name, categories, lon, lat, inside, dist_m, score
    or None if nothing reasonable is found.
    """
    if links_df is None or links_df.empty:
        return None

    df = links_df
    # If the caller passed a superset, filter here (safe no-op otherwise)
    if building_id is not None and "building_id" in df.columns:
        df = df[df["building_id"] == building_id]
        if df.empty:
            return None

    # Hard distance gate (unless 'inside' is true)
    def _keep(row: pd.Series) -> bool:
        if _as_bool(row, "inside"):
            return True
        d = _as_float(row, "dist_m", float("inf"))
        return d <= max_dist_m

    df = df[df.apply(_keep, axis=1)]
    if df.empty:
        return None

    # Score = geom * category prior
    scores = []
    for _, r in df.iterrows():
        g = _geom_weight(r, max_dist_m)
        c = _cat_weight(r.get("categories", None))
        scores.append(g * c)

    df = df.copy()
    df["__score"] = scores

    best = df.loc[df["__score"].idxmax()]

    # names can be nested; pick something printable
    name = None
    try:
        nm = best.get("names", None)
        if isinstance(nm, dict):
            # Overture names often like {'common': {'en': 'Foo'}}
            name = (nm.get("common") or {}).get("en") or None
        elif isinstance(nm, str):
            name = nm
    except Exception:
        pass

    return {
        "place_id": best.get("place_id"),
        "name": _extract_name(best.get("names")),
        "categories": best.get("categories"),
        "lon": _as_float(best, "place_lon", None),
        "lat": _as_float(best, "place_lat", None),
        "inside": _as_bool(best, "inside"),
        "dist_m": _as_float(best, "dist_m", float("nan")),
        "score": float(best["__score"]),
    }
