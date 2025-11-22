# mapillary-entrances

Julien Howard and Evan Rantala

This project integrates **Overture Buildings** and **Overture Places** with **Mapillary** street-level imagery to detect and map building entrances.  
The pipeline combines:

- Overture → DuckDB geospatial queries  
- Mapillary imagery (360° preferred)   
- YOLOv8 entrance detection  
- Geometry-based triangulation to estimate entrance coordinates  

## Pipeline Structure

- src/
   - `pipeline.py`            → End-to-end runner (download → imagery → inference → output)
   - `api.py`                 → API wrapper for programmatic access
   - `download.py`            → Overture data download for specified radius
   - `imagery.py`             → Mapillary queries, image preparation
   - `inference.py`           → YOLOv8 detection and entrance extraction

   - utils/
      - `constants.py`         → Release versions, S3 paths, global constants
      - `geo_utils.py`         → Haversine, bounding boxes
      - `polygon_utils.py`     → Shapely/WKT parsing, polygon → walls, vertices
      - `duckdb_utils.py`      → DuckDB connection, parquet querying helpers
      - `mapillary_utils.py`   → Graph API fetching, image download helpers
      - `matching_utils.py`    → Building and place matching, scoring logic
      - `io_utils.py`          → File I/O, logging, directory helpers
      - `inference_utils.py`   → Image filtering, YOLO model helpers, entrance point extraction


## Setup

### 1. Clone the repository
```bash
git clone https://github.com/project-terraforma/mapillary-entrances
cd mapillary-entrances/src
```
### 2. Create a virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
### 3. Environment variables
Create a .env file
``` bash
MAPILLARY_ACCESS_TOKEN=YOUR_TOKEN_HERE
```

## Running the pipeline
To run the pipeline, use the file `pipeline.py`

Running `pipeline.py` performs the full workflow automatically:

1. **Download Overture data**  
   Uses `download.py` to fetch Buildings and Places for the area around `--input_point` and `--search_radius`.

2. **Collect Mapillary imagery**  
   `imagery.py` gathers nearby imagery with corresponding coordinates and compass angles, prefers 360° when available.

3. **Run YOLOv8 inference**  
   Passes the gathered imagery and buildings to `inference.py`, which detects potential entrances and finds precise coordinates.

4. **Save data with entrance locations**  
   Saves output geojson file and images with detection to results directory.


Example Usage:

```bash
  python3 pipeline.py \
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
```
# Notes on arguments

- `--input_point="lat,lon"` → Coordinates to seed the search  
- `--search_radius` → Radius to collect buildings + imagery  
- `--place_radius` → Radius around each building to search for place  
- `--max_images` → Limit on total Mapillary images  
- `--prefer_360` → Prefer panorama imagery when available  
- `--model` → YOLOv8 model weights  
- `--device` → CPU or GPU, (use GPU when available)
- `--conf`, `--iou` → Detection thresholds  
- `--save` → Directory to save annotated images and geojson output

# Output Structure

Each run of `pipeline.py` produces the following:

- `outputs/geojson_verifications/`
   - `<lat>_<lon>.geojson`
- `outputs/visualizations/`
   - `<image_id>_vis.jpg`

In `outputs/geojson_verifications/`, `<lat>_<lon>.geojson` contains the geojson file that can be copy and pasted into: https://geojson.io/#map=2/0/20, for visualizing the data. This geojson contains:

   - Polygons for every building inside the search radius (marked with black lines)
   - The Overture Places that correspond to each building (marked as green circles)
   - Camera positions of the Mapillary images that produced valid detections (marked as orange circles)
   - Predicted entrance coordinates (marked as red circles)

In `outputs/visualizations/`, `<image_id>_vis.jpg` corresponds to a Mapillary image that resulted in an entrance detection. In this file, the resulting bounding box of the entrance is included in the original image, along with its confidence score. 