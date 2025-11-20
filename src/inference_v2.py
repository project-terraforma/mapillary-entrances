import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import cv2
import pyproj
from pyproj import CRS, Transformer
import shutil
import json
from .utils import _is_360

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

#----------------------------------
# Basic helper functions

# shared local projection utilities
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

# ----------------------------------
# Step 1:
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
    bad_images = []

    bad_images_dir = Path("geometry_checking_visuals") / "filtered_out_images_by_quality"
    bad_images_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        img = cv2.imread(img_path['image_path'])
        if img is None:
            print(f"[WARN] Could not read {img_path['image_path']}")
            continue

        if (is_sharp(img, sharpness_thresh) and
            is_well_exposed(img, dark_thresh, bright_thresh)):
            good_images.append(img_path)
        else:
            bad_images.append(img_path)

            # copy bad image to diagnostics folder
            src = Path(img_path['image_path'])
            dst = bad_images_dir / src.name
            try:
                shutil.move(src, dst)
                print(f"[INFO] moved bad image → {dst}")
            except Exception as e:
                print(f"[WARN] Failed to move {src}: {e}")
    return good_images


#-----------------------------
# Step 2:
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
    save_vis_dir: Optional[str] = None
)-> List[Tuple]:
    '''
    run YOLO (on ROIs or full image)
    save/print results for quick sanity check
    return image with bbox around detected doors as dictionary
    '''
    # Load YOLO door model
    model = load_yolo_model(yolo_weights_path, device=device)

    if save_vis_dir:
        Path(save_vis_dir).mkdir(parents=True, exist_ok=True)

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
        if save_vis_dir:
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


#---------------------------------
# Step 3:
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



def write_geojson_for_verification(
    building_entrances: Dict[str, Tuple[float, float]],
    buildings_lat_lon: Dict[str, List[List[float]]],
    place_names: Dict[str, Dict[str, Any]],
    image_detections: Dict[str, List[Tuple[float, float]]],
    output_name: str,
):
    """
    Write a GeoJSON file containing:
      - Building polygons (blue)
      - Predicted entrance points (red stars)
      - Place points (green circles)
      - Image detection coordinates (orange diamonds)

    """
    features = []

    # building polygons (blue)
    for bid, polygon in buildings_lat_lon.items():
        # ensure ring is closed
        if polygon and polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]

        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Building {bid}",
                "stroke": "#1f77b4",
                "stroke-width": 2,
                "fill": "#1f77b4",
                "fill-opacity": 0.1
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            }
        })

    # entrance detections (red stars)
    for bid, entrance_coords in building_entrances.items():
        lon, lat = entrance_coords
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Predicted Entrance for Building {bid}",
                "marker-color": "#ff0000",
                "marker-symbol": "star",
                "marker-size": "medium"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        })

    # place points (green circles)
    if place_names:
        for bid, place in place_names.items():
            if not isinstance(place, dict):
                continue
            lon = place.get("lon")
            lat = place.get("lat")
            if lon is not None and lat is not None:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "name": f"Place {place.get('place_id', '')} ({place.get('name', '')})",
                        "marker-color": "#00cc00",
                        "marker-symbol": "circle",
                        "marker-size": "small"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lon), float(lat)]
                    }
                })

    # iage detections (orange diamonds)
    if image_detections:
        for img_path, lonlat in image_detections.items():
            lon, lat = float(lonlat[0]), float(lonlat[1])
            features.append({
                "type": "Feature",
                "properties": {
                    "name": f"Detection from {Path(img_path).name}",
                    "marker-color": "#ffa500",
                    "marker-symbol": "diamond",
                    "marker-size": "small"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            })

    # write file
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    out_path = Path("outputs/geojson_verifications") / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"[OK] Wrote GeoJSON verification file: {out_path.resolve()}")
    print("→ Open this file at https://geojson.io to visualize results.")

def run_inference(data, yolo_weights, conf, iou, device, save_vis):

    if not _HAS_ULTRALYTICS:
        raise SystemExit("ERROR: ultralytics not installed. Try: pip install ultralytics")

    centroid_lon, centroid_lat = data['input_coordinates'][0], data['input_coordinates'][1]
    all_images = data['image_dicts']

    candidate_images_dir = Path("candidate_images") / "run_candidates"
    candidate_images_dir.mkdir(parents=True, exist_ok=True)

    # image_compass_angles -> {img_path : angle, ...}
    image_compass_angles = {}
    for img_path in all_images:
        # build dictionary of all compass angles to be easily accessed later
        path = img_path['image_path']
        if _is_360(img_path):
            image_compass_angles[path] = "360"
        else:
            image_compass_angles[path] = img_path['compass_angle']

        # (when optionally checking ) put candidate images in other directory for validation
        src = Path(path)
        dst = candidate_images_dir / src.name

        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[WARN] Failed to copy {src}: {e}")

    
    place_names = data['places']
    
    # buildings : {building_id: [(wall_point_lon, wall_point_lat), ...], building_id: [...]}
    buildings_lat_lon = data['building_polygons']

    # perform basic filtering based on brightness/quality
    print(f"All images pre image filtering by quality: {len(all_images)}")
    all_images = filter_images_by_quality(all_images, sharpness_thresh = 100.0, 
                                        dark_thresh = 0.05, bright_thresh = 0.95)
    print(f" All images Dictionary post filtering by quality: {len(all_images)}")


    # create full x,y approximation of images and building polygons coordinates
    # use input coordinates as reference (0,0)
    proj_local = make_local_proj(centroid_lat, centroid_lon)
    
    # convert all building polygons to local coordinates
    # buildings_xy -> {building_id_i : [(wall_point_1_x, wall_point_1_y), (wall_point_2_x, wall_point_2_y), ...], building_id_j : {...}}
    buildings_xy = {}
    for id in buildings_lat_lon.keys():
        polygon_xy = [to_local_xy(wall_tuple[0], wall_tuple[1], proj_local) for wall_tuple in buildings_lat_lon[id]]
        buildings_xy[id] = polygon_xy


    # convert all image coordinates to local coordinates
    # images_xy -> {image_path : (x, y), image_path : (x, y), ...}
    images_xy = {}
    for img in all_images:
        images_xy[img['image_path']] = to_local_xy(img['coordinates'][0], img['coordinates'][1], proj_local)

    # main loop
    # run model on each image get entrance detections
    # for each entrance detected, map x,y approximation to associated building polygon, 
    # entrances are matched with nearest building polygon, iff the image was pointing towards that building
    # convert entrance point back to lon,lat, attach to entrances dictionary with associated building id
    # building_entrances -> {building_id : (lon,lat), ...}

    building_entrances = {}
    images_with_detections = {}
    for img in all_images:
        path = img['image_path']
        detections = validate_folder_with_seg_and_yolo(path, yolo_weights, conf, iou, device, save_vis)
        # when no detections are made, move to next image
        if detections is None:
            continue
        print(f"Found {len(detections)} in image: {path}")
        entrances_xy = []
        for d in detections:
            C, dir_xy = extract_bbox_coordinates(img, d, proj_local, get_fov_half_angle(img))
            print(f"C = {C}, dir_xy = {dir_xy}")
            bid, hit_xy = match_entrance_to_building((C, dir_xy), buildings_xy, max_range_m=120.0)
            print(f"bid: {bid}, hit_xy:{hit_xy}")
            if bid is None or hit_xy is None:
                continue
            # shift hit slightly outward toward camera to account for building polygons -> roof of building
            #hit_xy = hit_xy - 6.0 * dir_xy
            print("Building matched")
            entrances_xy.append((bid, hit_xy))
            print(f"Image coordinates: {img['coordinates']}")
            images_with_detections[path] = img['coordinates']

        for (building_id, e_xy) in entrances_xy:
            entrance_lon, entrance_lat = to_lonlat_xy(e_xy, proj_local)
            building_entrances[building_id] = (entrance_lon, entrance_lat)
       
    return building_entrances, buildings_lat_lon, place_names, images_with_detections