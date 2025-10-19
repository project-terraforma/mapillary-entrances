import time, math
from typing import Dict

def parse_bbox_string(bbox_str: str) -> Dict[str, float]:
    parts = [float(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError('bbox must be "xmin,ymin,xmax,ymax"')
    return {"xmin": parts[0], "ymin": parts[1], "xmax": parts[2], "ymax": parts[3]}

def deg_per_meter(lat_deg: float):
    dlat = 1.0 / 111_320.0
    dlon = dlat / max(0.01, abs(math.cos(math.radians(lat_deg))))
    return dlat, dlon

def tlog(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"[{t1 - t0:6.2f}s] {label}", flush=True)
    return t1
