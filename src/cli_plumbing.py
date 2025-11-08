# User CLI for bbox workflow
# Flow: resolve sources -> gets buildings -> joins places -> loops buildings -> calls imagery pipeline -> writes to candidates.json per building

# example usage 
"""
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m src.cli_plumbing \
  --bbox="-122.4106,37.7918,-122.4028,37.7980" \
  --radius-m=50 --place-radius-m=10 \
  --limit-buildings=10 --max-images-per-building=8 \
  --prefer-360 --src-mode=local
"""

import math
import argparse, os, time, pandas as pd
from .utils import parse_bbox_string, tlog
from .sources import resolve_sources
from .buildings import get_buildings
from .places import join_buildings_places
from .selection import select_best_place_for_building
from .imagery import fetch_and_slice_for_building, write_candidates_json
import dotenv; dotenv.load_dotenv()

def main():
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description="Overture Buildings+Places → Mapillary → slices")
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--radius-m", type=int, default=120)
    ap.add_argument("--place-radius-m", type=int, default=60)
    ap.add_argument("--limit-buildings", type=int, default=10)
    ap.add_argument("--max-images-per-building", type=int, default=8)
    ap.add_argument("--min-capture-date", type=str, default=None)
    ap.add_argument("--no-fov", action="store_true")
    ap.add_argument("--prefer-360", action="store_true")
    ap.add_argument("--fov-half-angle", type=float, default=45.0)
    ap.add_argument("--src-mode", choices=["auto","local","s3"], default="auto")
    args = ap.parse_args()
    print(f"[args] {args}", flush=True)

    bbox = parse_bbox_string(args.bbox)
    b_src, p_src = resolve_sources(bbox, args.src_mode)
    print(f"[sources] buildings={b_src}\n[sources] places={p_src}")

    os.environ["LIMIT_BUILDINGS_HINT"] = str(args.limit_buildings or 10)
    tlog("[stage] buildings query…", t0)
    bdf = get_buildings(bbox, b_src, limit_hint=args.limit_buildings)
    if args.limit_buildings and len(bdf) > args.limit_buildings:
        bdf = bdf.sample(args.limit_buildings, random_state=42).reset_index(drop=True)
    print(f"[stage] buildings kept={len(bdf)}")
    if bdf.empty:
        print("[exit] No buildings"); return

    tlog("[stage] places join…", t0)
    links = join_buildings_places(bdf, bbox, p_src, radius_m=args.place_radius_m)
    print(f"[stage] links={len(links)}")

    for i, (_, b) in enumerate(bdf.iterrows(), start=1):
        bt = time.perf_counter()
        print(f"[building {i}/{len(bdf)}] id={b['id']} @ ({b['lat']:.6f},{b['lon']:.6f})")
        b_links = links[links["building_id"] == b["id"]] if not links.empty else pd.DataFrame()
        best_place = select_best_place_for_building(b_links, building_id=b["id"], max_dist_m=60)
        if best_place:
            nm = best_place.get("name") or "<unnamed>"
            d  = best_place.get("dist_m")
            if d is not None and math.isfinite(d):
                print(f"  ↳ place: {nm} (dist {d:.1f} m)")
            else:
                print(f"  ↳ place: {nm}")
        else:
            print("  ↳ place: <none>")

        saved = fetch_and_slice_for_building(
            b, radius_m=args.radius_m, min_capture_date=args.min_capture_date,
            apply_fov=not args.no_fov, max_images_per_building=args.max_images_per_building,
            prefer_360=args.prefer_360, fov_half_angle=args.fov_half_angle,
        )
        write_candidates_json(b, best_place, saved)
        tlog("  ✓ building done", bt)

    tlog("[done] all buildings", t0)

if __name__ == "__main__":
    main()