# Adapted from: Siddarth Mamidanna, Spring 2025 Innovation Lab (Project F)
# Source: https://github.com/project-terraforma/Automated-Imagery-Feature-Annotation/blob/main/README.md
# Modifications by Julien Howard & Evan Rantala, Fall 2025 (Project E - Entrances)

import math
import os
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import re
import difflib

import numpy as np  # type: ignore

# Optional heavy deps
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import easyocr  # type: ignore
except ImportError:
    easyocr = None

EARTH_RADIUS_M = 6378137.0  # WGS-84 mean

# ---------------------------------------------------------------------------
# Utility geo helpers
# ---------------------------------------------------------------------------

def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def _rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def bearing_from_pixel(x_norm: float, heading_deg: float, hfov_deg: float) -> float:
    """Given normalised pixel x ∈ [0,1], camera heading and HFOV, return bearing."""
    delta = (x_norm - 0.5) * hfov_deg  # positive to the right
    bearing = (heading_deg + delta) % 360.0
    print(f"    [geo] Bearing calc: x_norm={x_norm:.2f}, heading={heading_deg:.1f}, hfov={hfov_deg:.1f} -> delta={delta:.1f}, final_bearing={bearing:.1f}")
    return bearing


def lonlat_to_xy_m(lon: float, lat: float, lon0: float, lat0: float) -> Tuple[float, float]:
    """Simple equirectangular projection in metres around (lon0, lat0)."""
    x = _deg2rad(lon - lon0) * EARTH_RADIUS_M * math.cos(_deg2rad(lat0))
    y = _deg2rad(lat - lat0) * EARTH_RADIUS_M
    return x, y


def xy_m_to_lonlat(x: float, y: float, lon0: float, lat0: float) -> Tuple[float, float]:
    lon = lon0 + _rad2deg(x / (EARTH_RADIUS_M * math.cos(_deg2rad(lat0))))
    lat = lat0 + _rad2deg(y / EARTH_RADIUS_M)
    return lon, lat

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize text for better matching: lowercase, remove punctuation and common suffixes."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    
    suffixes = ['restaurant', 'cafe', 'bar', 'grill', 'inc', 'llc', 'co', 'ltd', 'shop', 'store']
    # A regex to match whole words for suffixes
    for suffix in suffixes:
        text = re.sub(r'\b' + suffix + r'\b', '', text)
        
    return text.strip()

def _detect_text_bbox(img_path: str, poi_name: str) -> Optional[Tuple[float, float]]:
    """Return centre x,y (norm 0-1) of bbox containing the best text match for poi_name."""
    if easyocr is None:
        print("    [geo] easyocr not installed, cannot detect text.")
        return None
    print(f"    [geo] Running text detection on {os.path.basename(img_path)} for '{poi_name}'...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    results = reader.readtext(img_path, detail=1)

    cleaned_poi_name = _clean_text(poi_name)
    print(f"    [geo] Cleaned POI name to: '{cleaned_poi_name}'")

    best_match_score = 0.0
    best_match_bbox = None
    best_match_text = ""

    if not results:
        print("    [geo] No text found in image.")
        return None

    for (bbox, text, conf) in results:  # type: ignore
        cleaned_text = _clean_text(text)
        if not cleaned_text:
            continue
            
        similarity = difflib.SequenceMatcher(None, cleaned_poi_name, cleaned_text).ratio()
        score = similarity * conf
        
        # Keep track of the best match
        if score > best_match_score:
            best_match_score = score
            best_match_bbox = bbox
            best_match_text = text

    # Use the best match only if it's above a reasonable threshold
    MIN_MATCH_THRESHOLD = 0.5
    if best_match_score > MIN_MATCH_THRESHOLD:
        print(f"    [geo] Best match found: '{best_match_text}' with score {best_match_score:.2f} (similarity: {best_match_score/conf:.2f}, conf: {conf:.2f})")
        xs = [p[0] for p in best_match_bbox]
        ys = [p[1] for p in best_match_bbox]
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0
        h, w = _img_shape(img_path)
        return cx / w, cy / h
    else:
        print(f"    [geo] No suitable text match found (best score {best_match_score:.2f} was below threshold {MIN_MATCH_THRESHOLD}).")
        return None


def _img_shape(img_path: str) -> Tuple[int, int]:
    if cv2 is None:
        raise RuntimeError("cv2 missing – cannot read image")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Cannot read image {img_path}")
    h, w = img.shape[:2]
    return h, w

# Placeholder for object detection bbox centre

def _detect_object_bbox(img_path: str) -> Optional[Tuple[float, float]]:
    # Stub: simply return image centre for now.
    h, w = _img_shape(img_path)
    return 0.5, 0.5

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def single_image_line(img_path: str, img_meta: Dict, detect_method: str, poi_name: str, lon0: float, lat0: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (point p, direction d) in local XY metres representing sight-line.

    p – origin (2-D)
    d – unit direction vector length 1
    """
    compass = img_meta.get('compass_angle') or img_meta.get('computed_compass_angle')
    geom = img_meta.get('computed_geometry') or img_meta.get('geometry')
    if compass is None or geom is None:
        print(f"    [geo] Missing compass ({compass}) or geometry ({geom}) for {os.path.basename(img_path)}. Cannot create sight-line.")
        return None
    cam_lon, cam_lat = geom['coordinates']

    # Determine HFOV
    cam_type = img_meta.get('camera_type', '')
    h_fov_lookup = {
        'smartphone': 60,
        'hemispherical': 180,
        'equirectangular': 90,
        '': 60,
    }
    hfov = h_fov_lookup.get(cam_type, 60)
    print(f"    [geo] Camera type '{cam_type}' -> HFOV: {hfov} deg")

    # Detect where POI is in image
    print(f"    [geo] Detection method: '{detect_method}'")
    if detect_method == 'text':
        centre = _detect_text_bbox(img_path, poi_name)
    elif detect_method == 'object':
        centre = _detect_object_bbox(img_path)
    else:
        centre = (0.5, 0.5)

    if centre is None:
        print("    [geo] Detection failed, falling back to image center.")
        centre = (0.5, 0.5)
    x_norm, _ = centre
    bearing = bearing_from_pixel(x_norm, compass, hfov)

    # Convert to local XY and unit dir vector
    px, py = lonlat_to_xy_m(cam_lon, cam_lat, lon0, lat0)
    # Bearing 0=north → vector
    theta = _deg2rad(bearing)
    dx = math.sin(theta)
    dy = math.cos(theta)
    d_norm = math.hypot(dx, dy)
    line = (np.array([px, py]), np.array([dx / d_norm, dy / d_norm]))
    print(f"    [geo] Created sight-line from {os.path.basename(img_path)}: p=({px:.1f}, {py:.1f}), d=({line[1][0]:.2f}, {line[1][1]:.2f})")
    return line


def triangulate(lines: List[Tuple[np.ndarray, np.ndarray]], lon0: float, lat0: float) -> Optional[Tuple[float, float]]:
    """Least-squares intersection of ≥2 sight-lines in 2-D. Returns (lon, lat)."""
    if len(lines) < 2:
        return None
    
    print(f"\n    [geo] Triangulating with {len(lines)} sight-lines...")

    # Solve A x = b where A = sum(I - dd^T), b = sum((I - dd^T) p)
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, d in lines:
        d = d / np.linalg.norm(d)
        I_minus = np.eye(2) - np.outer(d, d)
        A += I_minus
        b += I_minus @ p
    try:
        est_xy = np.linalg.solve(A, b)
        print(f"    [geo] Solved for estimated local XY: ({est_xy[0]:.2f}, {est_xy[1]:.2f})")
    except np.linalg.LinAlgError:
        print("    [geo] Triangulation failed (matrix is singular).")
        return None
    est_lon, est_lat = xy_m_to_lonlat(est_xy[0], est_xy[1], lon0, lat0)
    print(f"    [geo] Converted estimate to lon/lat: ({est_lon:.6f}, {est_lat:.6f})")
    return est_lat, est_lon 
