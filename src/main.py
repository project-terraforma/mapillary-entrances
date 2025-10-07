# main.py
import argparse
from src.plumbing import query_buildings, fetch_images_for_buildings
from src.yolo_infer import run_yolo_on_results
from src.entrance_localizer import localize_building_entrances_for_bbox

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True, help="xmin,ymin,xmax,ymax")
    ap.add_argument("--radius-m", type=int, default=40)
    args = ap.parse_args()

    b = {k: float(v) for k,v in zip(["xmin","ymin","xmax","ymax"], args.bbox.split(","))}
    bdf = query_buildings(b)
    fetch_images_for_buildings(bdf, radius_m=args.radius_m, apply_fov=True)
    run_yolo_on_results(root="results/buildings", weights="yolov8n.pt", conf=0.25)
    # implement a wrapper that loops buildings and writes entrances.csv/geojson
    # localize_building_entrances_for_bbox(b)

if __name__ == "__main__":
    main()
