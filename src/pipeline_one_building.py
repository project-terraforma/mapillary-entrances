# pipeline_one_building.py
# currently using api.py for single building functionality 
import argparse
from .api import get_building_package_for_point
from .inference import run_inference
from .utils import parse_bbox_string

'''
Example Usage:
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m src.pipeline_one_building \
  --point="37.7860,-122.4005" \
  --search_radius=120 \
  --place_radius=60 \
  --limit_buildings=1 \
  --max_images_per_building=5 \
  --prefer_360 \
  --src_mode=auto \
  --model="./"yolov8s.pt"" \
  --device="cuda" \
  --hfov=45.0 \
  --facade_tau=0.35 \
  --conf=0.4 \
  --iou=0.5 \
  --save-vis="./outputs/visualizations"
'''

def build_parser():
    p = argparse.ArgumentParser(...)
    # data fetching args for api.py
    p.add_argument("--bbox", type=str, help="minLon,minLat,maxLon,maxLat")
    p.add_argument("--point", type=str, help="lat,lon for single-building mode")
    p.add_argument("--search_radius", type=int, default=120)
    p.add_argument("--place_radius", type=int, default=60)
    p.add_argument("--limit_buildings", type=int)
    p.add_argument("--max_images_per_building", type=int, default=8)
    p.add_argument("--prefer_360", action="store_true")
    p.add_argument("--src_mode", choices=["auto","local","s3"], default="auto")
    p.add_argument("--min_capture_date", type=str)
    p.add_argument("--fov_half_angle", type=float, default=45.0)
    p.add_argument("--apply_fov", type=bool)

    # inference args for inference.py
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use_rois", type=bool, default = False)
    p.add_argument("--save-vis", type=str)
    p.add_argument("--hfov", type=float, default=45.0)
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--facade_tau", type=float, default = 0.5) # change iou conf and facade_tau default values

    return p

def main():
    args = build_parser().parse_args()

    # parse bbox/point strings into tuples once
    bbox = parse_bbox_string(args.bbox) if args.bbox else None
    lat, lon = args.point.split(",") if args.point else None
    lat = float(lat)
    lon = float(lon)

    # data gathering
    print("Gathering Data")
    data = get_building_package_for_point(lat, lon, args.search_radius,
                                          args.place_radius, args.max_images_per_building,
                                          args.min_capture_date, args.prefer_360,
                                          args.fov_half_angle, args.apply_fov, args.src_mode)
    
    print("Finished Gathering Data")
    
    # inference
    results = run_inference(data, args.hfov, 
                  args.model, args.facade_tau, args.use_rois, 
                  args.conf, args.iou, args.device, args.save_vis)
    print("Finished Inference")
if __name__ == "__main__":
    main()
