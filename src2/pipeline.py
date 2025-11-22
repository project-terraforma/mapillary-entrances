# pipeline.py

import argparse
from utils.io_utils import write_geojson_for_verification
from api import get_buildings_and_imagery_in_radius
from inference import run_inference
from download import download_overture_radius
import dotenv; dotenv.load_dotenv()

'''
Example Usage:

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m src.pipeline \
  --input_point="47.610,-122.341" \
  --search_radius=100 \
  --place_radius=100 \
  --max_images=50 \
  --prefer_360 \
  --src_mode=local \
  --model="yolo_weights_500_image_set.pt" \
  --device="cpu" \
  --conf=0.60 \
  --iou=0.50 \
  --save="../outputs"
'''

def build_parser():
    p = argparse.ArgumentParser(...)
    # data fetching args for api.py
    p.add_argument("--input_point", type=str, help="lat,lon")
    p.add_argument("--search_radius", type=int, default=120)
    p.add_argument("--place_radius", type=int, default=120)
    p.add_argument("--limit_buildings", type=int)
    p.add_argument("--max_images", type=int, default=20)
    p.add_argument("--prefer_360", action="store_true")
    p.add_argument("--src_mode", choices=["auto","local","s3"], default="auto")
    p.add_argument("--min_capture_date", type=str)
 
    # inference args for inference.py
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save", type=str)
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)

    return p

def main():
    args = build_parser().parse_args()

    # parse bbox/point strings into tuples once
    lat, lon = args.input_point.split(",") if args.input_point else None
    lat = float(lat)
    lon = float(lon)

    print(f"Downloading Overture data within {args.search_radius} m of ({lat}, {lon})...")

    download_overture_radius(lat, lon, args.search_radius,
                            out_buildings="../data/buildings_local.parquet",
                            out_places="../data/places_local.parquet"
                            )
    
    print("Relevant Buildings/Places Data Downloaded")

    # data gathering
    print("Gathering Imagery")
    data = get_buildings_and_imagery_in_radius(lat, lon, args.search_radius,
                                          args.place_radius, args.max_images,
                                          args.min_capture_date, args.prefer_360,
                                          args.src_mode)
    
    # inference
    if len(data['image_dicts']) > 0:
        print("Finished Gathering Data")
        print("Starting Inference")

        building_entrances, buildings_lat_lon, place_names, images_with_detections = run_inference(data, args.model, 
                                                                                                   args.conf, args.iou, 
                                                                                                   args.device, args.save)
        
        print("Finished Inference")

        if len(building_entrances) == 0:
            print("Could Not Find Any Entrances")

        for id in building_entrances:
            print(f"Building ID: {id}, Entrance: {building_entrances[id]}, Place ID: {place_names[id]['place_id']}")
            print(f"Building polygon: {buildings_lat_lon[id]}")
            print(f"Entrance in lat,lon format: {building_entrances[id][1],building_entrances[id][0]}")
        
        write_geojson_for_verification(
            building_entrances,
            buildings_lat_lon,
            place_names,
            images_with_detections,
            output_dir = args.save,
            output_name = f"{lat:.5f}_{lon:.5f}.geojson"
        )
if __name__ == "__main__":
    main()
