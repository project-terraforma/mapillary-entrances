# Adapted from: Siddarth Mamidanna, Spring 2025 Innovation Lab (Project F)
# Source: https://github.com/project-terraforma/Automated-Imagery-Feature-Annotation/blob/main/README.md
# Modifications by Julien Howard & Evan Rantala, Fall 2025 (Project E - Entrances)


import os
import math
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime, date # Added for date operations

import requests
import cv2 # type: ignore
import numpy as np # type: ignore
from pathlib import Path

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6378137  # mean Earth radius in metres

# ── GEOSPATIAL HELPERS ────────────────────────────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _bbox_around(lat: float, lon: float, radius_m: float) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) bounding box that encloses a circle."""
    delta_lat = (radius_m / EARTH_RADIUS_M) * (180 / math.pi)
    delta_lon = (radius_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180 / math.pi)
    return lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat

def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing in degrees from point 1 to point 2 (0° = North)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360
def _is_360(img: dict) -> bool:
    ct = (img.get("camera_type") or "").lower()
    return ct in {"spherical", "equirectangular", "panorama", "panoramic", "360"}

# ── MAPILLARY API HELPERS ────────────────────────────────────────────────────
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
            
    min_lon, min_lat, max_lon, max_lat = _bbox_around(lat, lon, radius_m)
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
        print(f"  [mly_utils] {filtered_count} fisheye images filtered out. {len(in_radius)} remaining.")
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
            print(f"  [mly_utils] {filtered_count} images filtered out by min_capture_date ({min_capture_date_filter.strftime('%Y-%m-%d')}). {len(dated_images)} remaining.")
        elif initial_image_count > 0: # Only print if there were images to begin with
            print(f"  [mly_utils] No images filtered by min_capture_date ({min_capture_date_filter.strftime('%Y-%m-%d')}).")
        in_radius = dated_images # Update in_radius with date-filtered images

    return in_radius

def filter_images_fov(
    images: List[Dict],
    target_lat: float,
    target_lon: float,
    fov_half_angle: float = 30.0
) -> List[Dict]:
    """Return subset of *images* whose camera orientation looks towards *target*."""
    passing = []
    for img in images:
        compass = img.get("compass_angle") or img.get("computed_compass_angle")
        geometry = img.get("computed_geometry") or img.get("geometry")
        if compass is None or geometry is None:
            continue
        cam_lon, cam_lat = geometry["coordinates"]
        bearing = _bearing(cam_lat, cam_lon, target_lat, target_lon)
        diff = abs((bearing - compass + 180) % 360 - 180)
        if diff <= fov_half_angle:
            img["bearing_to_target"] = bearing
            img["angle_diff"] = diff
            passing.append(img)
    return passing

# ── OSRM (ROUTING) HELPERS ────────────────────────────────────────────────────
def snap_to_street(lat: float, lon: float) -> Tuple[float, float]:
    """Return the (lat, lon) of the nearest road centre-line using OSRM's *nearest* service.
    Falls back to the original coordinate if the service fails.
    """
    try:
        url = f"https://router.project-osrm.org/nearest/v1/driving/{lon},{lat}?number=1"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        loc = js["waypoints"][0]["location"]  # [lon, lat]
        return loc[1], loc[0]
    except Exception as exc:
        print(f"[WARN] snap_to_street failed: {exc} – using original coordinate")
        return lat, lon

# ── IMAGE QUALITY HEURISTICS ────────────────────────────────────────────────
def is_sharp(img: np.ndarray, thresh: float = 100.0) -> bool:
    """Check image sharpness using the variance of the Laplacian."""
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img, cv2.CV_64F).var() > thresh

def has_enough_resolution(img: np.ndarray, min_width: int = 300, min_height: int = 300) -> bool:
    """Check if image meets minimum width and height requirements."""
    h, w = img.shape[:2]
    return (w >= min_width) and (h >= min_height)

def is_well_exposed(img: np.ndarray, dark_thresh: float = 0.05, bright_thresh: float = 0.95) -> bool:
    """Check image exposure based on the proportion of very dark or very bright pixels."""
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = img.flatten() / 255.0
    return ((flat < dark_thresh).mean() < 0.5 and
            (flat > bright_thresh).mean() < 0.5)

def filter_images_by_quality(
    folder_path: str,
    sharpness_thresh: float = 100.0,
    min_width: int = 300,
    min_height: int = 300,
    dark_thresh: float = 0.05,
    bright_thresh: float = 0.95
) -> List[Path]:
    """Apply quality filters to images in a folder, deleting low-quality ones.
    Returns a list of paths to the images that passed all filters.
    """
    good_images = []
    for path_obj in Path(folder_path).glob("*.jpg"):
        try:
            img = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
            if img is None:
                 print(f"[WARN] Could not read image file {path_obj} – skipping quality check")
                 continue
            if (has_enough_resolution(img, min_width, min_height) and
                is_sharp(img, sharpness_thresh) and
                is_well_exposed(img, dark_thresh, bright_thresh)):
                good_images.append(path_obj)
            else:
                os.remove(path_obj)
                print(f"  • filtered out low-quality image: {path_obj}")
        except Exception as e:
             print(f"[ERROR] Error processing image {path_obj}: {e} – skipping quality check")
    return good_images