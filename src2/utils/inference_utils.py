# inference_utils.py

import pyproj
from pyproj import CRS, Transformer
import math
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Tuple

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False


def _is_360(img: dict) -> bool:
    ct = (img.get("camera_type") or "").lower()
    return ct in {"spherical", "equirectangular", "panorama", "panoramic", "360"}


def make_local_proj(lat0, lon0):
    lat0 = float(lat0)
    lon0 = float(lon0)
    return CRS.from_user_input(
        f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +ellps=WGS84 +units=m +no_defs"
    )

def to_local_xy(lon, lat, crs_local):
    lat = float(lat)
    lon = float(lon)
    transformer = Transformer.from_crs("EPSG:4326", crs_local, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.array([x, y])

def to_lonlat_xy(xy, crs_local):
    """
    Convert local (x, y) coordinates back to geographic (lon, lat)
    using the same local Azimuthal Equidistant CRS.
    """
    x, y = map(float, xy)
    transformer = Transformer.from_crs(crs_local, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)


def get_fov_half_angle(img_dict):
    # check is_360 field first (slices have is_360=False even if camera_type is spherical)
    is_pano = img_dict.get("is_360", False)
    if is_pano:
        return 90.0
    elif (img_dict.get("camera_type") or "").lower() in ("perspective", "planar"):
        return 45.0
    else:
        return 45.0

# Image quality filters

def is_sharp(img: np.ndarray, thresh: float = 100.0) -> bool:
    # check image sharpness using the variance of the Laplacian
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img, cv2.CV_64F).var() > thresh


def is_well_exposed(img: np.ndarray, dark_thresh: float = 0.05, bright_thresh: float = 0.95) -> bool:
    # check image exposure based on the proportion of very dark or very bright pixels
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = img.flatten() / 255.0
    return ((flat < dark_thresh).mean() < 0.5 and
            (flat > bright_thresh).mean() < 0.5)


def filter_images_by_quality(
    images : List,
    sharpness_thresh: float = 100.0,
    dark_thresh: float = 0.05,
    bright_thresh: float = 0.95
) -> List:
    """Apply quality filters to list of images pulled from mapillary, removing low-quality ones.
    Returns the list of images that passed quality filters.
    """
    good_images = []

    for img_path in images:
        img = cv2.imread(img_path['image_path'])
        if img is None:
            print(f"[WARN] Could not read {img_path['image_path']}")
            continue

        if (is_sharp(img, sharpness_thresh) and
            is_well_exposed(img, dark_thresh, bright_thresh)):
            good_images.append(img_path)

    return good_images


# Vision model functions
# Run model on images to detect entrance

def load_yolo_model(model_path: str, device: Optional[str] = None):

    if not _HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics not installed. `pip install ultralytics`")

    model = YOLO(model_path)  # ultralytics auto-selects device unless specified
    if device is not None:
        # The ultralytics API selects device at predict() time; we’ll pass device then.
        model.overrides = model.overrides or {}
        model.overrides['device'] = device
    return model


def run_yolo_on_image(
    model,
    img: np.ndarray,
    conf_thr: float = 0.35,
    iou_thr: float = 0.5,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Run YOLO on a full image and return list of detections:
      [{ 'conf': float, 'bbox': (x1,y1,x2,y2), 'cls_id': int, 'cls_name': str }]
    """
    # Ultralytics expects BGR numpy or path
    results = model.predict(source=img, conf=conf_thr, iou=iou_thr, verbose=False, device=device)
    dets: List[Dict] = []
    if not results:
        return dets

    res = results[0]
    names = res.names
    if res.boxes is None:
        return dets

    for b in res.boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        conf = float(b.conf[0].cpu().numpy())
        cls_id = int(b.cls[0].cpu().numpy()) if b.cls is not None else -1
        cls_name = names.get(cls_id, str(cls_id))
        dets.append({"conf": conf, "bbox": tuple(xyxy.tolist()), "cls_id": cls_id, "cls_name": cls_name})
    return dets


# visualize detections
def _draw_dets(img: np.ndarray, dets: List[Dict], color=(0, 200, 0)) -> np.ndarray:
    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        conf = d['conf']
        label = d.get("cls_name", "obj")
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def validate_folder_with_seg_and_yolo(
    images_dir: Path,
    yolo_weights_path: str,
    conf_thr: float = 0.5,
    iou_thr: float = 0.5,
    device: Optional[str] = None,
    save_dir: Optional[Path] = None
)-> List[Tuple]:
    '''
    run YOLO (on ROIs or full image)
    save/print results for quick sanity check
    return image with bbox around detected doors as dictionary
    '''
    # Load YOLO door model
    model = load_yolo_model(yolo_weights_path, device=device)

    if save_dir:
        save_vis_dir = Path(save_dir) / "visualizations"

    img = cv2.imread(str(images_dir), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read {images_dir}")
        return None

    dets = run_yolo_on_image(model, img, conf_thr=conf_thr, iou_thr=iou_thr, device=device)
    # Keep only door class if model has multiple classes; otherwise keep all
    entrance_dets = [d for d in dets if d.get("cls_name", "").lower().find("entrance") != -1 or True]

    print(f"detections: {len(entrance_dets)}")
    #assume for now there will only be one door detection in an image
    if len(dets) > 0:
        if save_dir:
            Path(save_vis_dir).mkdir(parents=True, exist_ok=True)
            vis = _draw_dets(img, entrance_dets)
            name = Path(images_dir).stem
            out_path = Path(save_vis_dir) / f"{name}_vis.jpg"
            ok = cv2.imwrite(str(out_path), vis)
            if not ok:
                print(f"[WARN] cv2.imwrite failed: {out_path}")
            else:
                print(f"[OK] wrote {out_path}")
    print(f"Number of detections:{len(entrance_dets)}")
    if len(entrance_dets) == 0:
        return []
    else:
        entrance_list = []
        for i in range(len(entrance_dets)):
            entrance_list.append(entrance_dets[i]['bbox'])
        return entrance_list # [(x1, y1, x2, y2), ... ]


# Coordinate Extraction Functions
# Given an image with a detected entrance, these functions will find the entrance point to the building

def horizontal_fov_to_fx(img_w, hfov_deg):
    # pinhole: fx = (W/2)/tan(hfov/2)
    return (img_w * 0.5) / math.tan((math.pi*(hfov_deg * 0.5))/180)


def extract_bbox_coordinates(
    image_dict: Dict,
    bbox,
    proj_local,
    hfov_deg=45.0,
):
    """
    from a detection bbox on an image, build the local (x,y) camera ray:
    ray_origin = camera position in local XY
    direction  = unit vector in local XY pointing where the bbox bottom-center subtends
    returns (ray_origin_xy, direction_xy)
    """
    # camera origin in local XY
    cam_lon, cam_lat = image_dict['coordinates'][0], image_dict['coordinates'][1]
    C = to_local_xy(cam_lon, cam_lat, proj_local)  # shape (2,)

    # load image to get width for intrinsics
    img = cv2.imread(str(image_dict['image_path']), cv2.IMREAD_COLOR)
    
    H, W = img.shape[:2]
    # get bottom-center pixel of bbox
    # support xyxy (x1,y1,x2,y2) and xywh-center (xc,yc,w,h)
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        u = 0.5 * (x1 + x2)
        v = float(y2)
    else:
        # unsupported structure; best effort with center
        xc, yc, bw, bh = bbox
        u = float(xc)
        v = float(yc + 0.5 * bh)

    # turn pixel u into a yaw offset using a pinhole model
    fx = horizontal_fov_to_fx(W, hfov_deg)
    cx = 0.5 * W
    if _is_360(image_dict):
        yaw_offset_deg = (u / W) * 360.0 - 180.0
    else:
        fx = horizontal_fov_to_fx(W, hfov_deg)
        cx = 0.5 * W
        yaw_offset_rad = math.atan2((u - cx), fx)
        yaw_offset_deg = math.degrees(yaw_offset_rad)

    compass_deg = float(image_dict.get("compass_angle") or 0.0)
    bearing_deg = (compass_deg + yaw_offset_deg) % 360.0

    theta = math.radians(bearing_deg)
    d = np.array([math.sin(theta), math.cos(theta)], dtype=float)
    d /= np.linalg.norm(d)

    return C, d


def _ray_segment_intersection(C, d, A, B, t_min=0.0):
    """
    solve C + t d = A + u (B - A), t >= t_min, u in [0,1].
    returns (hit_xy, t, hit_bool).
    """
    AB = B - A
    M = np.stack([d, -AB], axis=1)
    rhs = A - C
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if abs(det) < 1e-12:
        return None, None, False
    inv = np.array([[ M[1,1], -M[0,1] ],
                    [ -M[1,0], M[0,0] ]], dtype=float) / det
    t, u = (inv @ rhs)
    if t is not None and u is not None and t >= t_min and 0.0 <= u <= 1.0:
        return (C + t * d), float(t), True
    return None, None, False


def match_entrance_to_building(ray, buildings_xy, max_range_m=100.0, spread_deg=5.0):
    """
    cast a small fan of rays (±spread_deg) around the main direction vector `d`.
    Returns the closest intersection across all building polygons.
    """
    C, d = ray
    best_bid = None
    best_t = float("inf")
    best_hit = None
    candidates = {}

    # try slight angular offsets to make matching robust
    for delta in [-spread_deg, 0, spread_deg]:
        theta = math.atan2(d[1], d[0]) + math.radians(delta)
        d_rot = np.array([math.cos(theta), math.sin(theta)], dtype=float)

        for bid, poly in buildings_xy.items():
            for i in range(len(poly)):
                A = np.asarray(poly[i], dtype=float)
                B = np.asarray(poly[(i + 1) % len(poly)], dtype=float)
                hit, t, ok = _ray_segment_intersection(C, d_rot, A, B)
                if ok and 0.1 < t < best_t and t <= max_range_m:
                    candidates.setdefault(bid, []).append(t)

    # if no valid intersections, break
    if not candidates:
        return None, None

    # choose the building whose median hit distance is the smallest
    best_bid = min(candidates, key=lambda k: np.median(candidates[k]))
    best_t = np.median(candidates[best_bid])

    # compute the intersection point in local XY
    best_hit = C + best_t * d
    return best_bid, best_hit