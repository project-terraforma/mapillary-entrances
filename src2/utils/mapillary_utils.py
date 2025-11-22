#mapillary_utils.py

from typing import Optional, List, Dict
from datetime import datetime, date
from pathlib import Path
import requests
from .geo_utils import _haversine, robust_radius_to_bbox

def _is_360(img: dict) -> bool:
    ct = (img.get("camera_type") or "").lower()
    return ct in {"spherical", "equirectangular", "panorama", "panoramic", "360"}

def fetch_images(
    token: str,
    lat: float,
    lon: float,
    radius_m: float,
    fields: Optional[List[str]] = None,
    min_capture_date_filter: Optional[datetime.date] = None,
    prefer_360: bool = False,
) -> List[Dict]:
    """Query Mapillary Graph API for images inside *radius_m* of *lat*, *lon*.

    The function returns a list of dictionaries, one for every image whose centre point
    falls inside the requested radius.
    Optionally filters images by a minimum capture date.
    """
    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN (token) must be provided.")

    if fields is None:
        fields = [
            "id", "computed_geometry", "captured_at", "compass_angle",
            "thumb_256_url", "thumb_1024_url", "thumb_2048_url", "thumb_original_url",
            "camera_type"
        ]
    if isinstance(min_capture_date_filter, str):
        try:
            min_capture_date_filter = date.fromisoformat(min_capture_date_filter)
        except ValueError:
            min_capture_date_filter = None
            
    min_lon, min_lat, max_lon, max_lat = robust_radius_to_bbox(lat, lon, radius_m)
    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "fields": ",".join(fields),
        "limit": 2000,
    }
    headers = {"Authorization": f"OAuth {token}"}
    resp = requests.get("https://graph.mapillary.com/images", params=params, headers=headers, timeout=90)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # If Mapillary has a temporary 5xx hiccup, just skip this building gracefully
        if resp is not None and 500 <= resp.status_code < 600:
            print("    [mly] Server 5xx from Mapillary; skipping this batch.")
            return []
        raise

    candidates = resp.json().get("data", [])
    in_radius: List[Dict] = []
    for img in candidates:
        geometry = img.get("computed_geometry") or img.get("geometry")
        if not geometry or geometry.get("type") != "Point":
            continue
        img_lon, img_lat = geometry["coordinates"]
        dist = _haversine(lat, lon, img_lat, img_lon)
        if dist <= radius_m:
            img["distance_m"] = dist
            in_radius.append(img)
    
    # Filter by camera_type to exclude *only* fisheye lenses
    initial_count = len(in_radius)
    in_radius = [img for img in in_radius if (img.get('camera_type') or '').lower() != 'fisheye']
    filtered_count = initial_count - len(in_radius)
    if filtered_count > 0:
        print(f"  [utils] {filtered_count} fisheye images filtered out. {len(in_radius)} remaining.")
    # Prefer 360° images if available
    if prefer_360:
        only_360 = [i for i in in_radius if _is_360(i)]
        if only_360:
            in_radius = only_360

    # Filter by min_capture_date if specified
    if min_capture_date_filter and in_radius:
        initial_image_count = len(in_radius)
        # Mapillary's captured_at is in milliseconds since epoch
        dated_images = []
        for img in in_radius:
            captured_at_ms = img.get('captured_at')
            if captured_at_ms:
                img_capture_date = datetime.fromtimestamp(captured_at_ms / 1000).date()
                if img_capture_date >= min_capture_date_filter:
                    dated_images.append(img)
            # else: image without captured_at is kept if no date filter, or implicitly dropped here if date filter exists
        
        filtered_count = initial_image_count - len(dated_images)
        if filtered_count > 0:
            print(f"  [utils] {filtered_count} images filtered out by min_capture_date ({min_capture_date_filter.strftime('%Y-%m-%d')}). {len(dated_images)} remaining.")
        elif initial_image_count > 0: # Only print if there were images to begin with
            print(f"  [utils] No images filtered by min_capture_date ({min_capture_date_filter.strftime('%Y-%m-%d')}).")
        in_radius = dated_images # Update in_radius with date-filtered images

    return in_radius


def download_image(meta_or_url, path: Path):
    """
    Download an image to the given path.
    Accepts either a URL string or a metadata dict with one of the keys:
      thumb_1024_url, thumb_original_url, or similar.
    """
    if isinstance(meta_or_url, dict):
        url = (
            meta_or_url.get("thumb_1024_url")
            or meta_or_url.get("thumb_original_url")
            or meta_or_url.get("url")
        )
    else:
        url = meta_or_url

    if not url:
        print(f"[WARN] No image URL provided for {path}")
        return

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")

def _extract_lon_lat(rec, fallback_lon, fallback_lat):
    """
    Try multiple fields to recover (lon, lat). Return floats.
    Priority:
      1) rec['lon'], rec['lat']
      2) rec['lng'], rec['lat']               # sometimes 'lng' is used
      3) rec['image_lon'], rec['image_lat']   # upstream variants
      4) rec['orig_lon'], rec['orig_lat']     # originals before slicing
      5) rec['coordinates'] = [lon, lat]
      6) fallback (building center)
    """
    def _ok(a, b):
        return isinstance(a, (int, float)) and isinstance(b, (int, float))

    cand_pairs = [
        ("lon", "lat"),
        ("lng", "lat"),
        ("image_lon", "image_lat"),
        ("orig_lon", "orig_lat"),
    ]
    for lo_k, la_k in cand_pairs:
        lo, la = rec.get(lo_k), rec.get(la_k)
        if _ok(lo, la):
            return float(lo), float(la)

    coords = rec.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        lo, la = coords[0], coords[1]
        if _ok(lo, la):
            return float(lo), float(la)

    # last resort – building center (ensures numbers, never nulls)
    return float(fallback_lon), float(fallback_lat)