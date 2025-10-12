#!/usr/bin/env python3
import argparse, json, math, os, base64
from pathlib import Path

import folium
from folium.features import DivIcon
from shapely.geometry import mapping
from shapely import wkt as _wkt

# --- small helpers ---
def meters_to_deg(lat_deg: float):
    dlat = 1.0 / 111_320.0
    dlon = dlat / max(0.01, abs(math.cos(math.radians(lat_deg))))
    return dlat, dlon

def offset_latlon(lat, lon, forward_m, bearing_deg):
    # simple equirectangular-ish step for tiny distances
    dlat, dlon = meters_to_deg(lat)
    rad = math.radians(bearing_deg)
    dy = forward_m * math.cos(rad)
    dx = forward_m * math.sin(rad)
    return lat + dy * dlat, lon + dx * dlon

def try_get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def load_json(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return json.load(f)

def popup_html_for_place(row):
    names = row.get("names")
    cats  = row.get("categories")
    dist  = row.get("distance_m")
    return f"""
    <b>Place</b><br/>
    <b>names:</b> {names}<br/>
    <b>categories:</b> {cats}<br/>
    <b>distance_m:</b> {dist if dist is not None else "—"}<br/>
    """.strip()

def popup_html_for_image(img_file, meta_file, meta):
    cap = meta.get("captured_at") or meta.get("created_at") or "—"
    comp = try_get(meta, "compass_angle", "computed_compass_angle")
    geom = try_get(meta, "computed_geometry", "geometry")
    coords = geom.get("coordinates") if geom else None
    return f"""
    <b>Image</b><br/>
    <b>file:</b> {img_file}<br/>
    <b>meta:</b> {meta_file}<br/>
    <b>captured:</b> {cap}<br/>
    <b>compass:</b> {comp if comp is not None else "—"}<br/>
    <b>camera coords:</b> {coords if coords else "—"}<br/>
    """.strip()

def main():
    ap = argparse.ArgumentParser(description="Visualize candidates.json as an interactive map.")
    ap.add_argument("--file", required=True, help="Path to results/.../candidates.json")
    ap.add_argument("--out", default="map.html", help="Output HTML path")
    ap.add_argument("--arrow-m", type=float, default=20.0, help="Length of heading arrows (meters)")
    ap.add_argument("--show-labels", action="store_true", help="Label places with rank numbers")
    args = ap.parse_args()

    cj_path = Path(args.file).resolve()
    root = cj_path.parent

    data = load_json(cj_path)
    b = data["building"]
    places = data.get("places_nearby", [])
    images = data.get("images", [])

    # Center map on building centroid
    center = (b["lat"], b["lon"])
    m = folium.Map(location=center, zoom_start=19, tiles="CartoDB Positron")

    # Draw building polygon (handles Polygon or MultiPolygon)
    if b.get("wkt"):
        geom = _wkt.loads(b["wkt"])
        geojson_obj = mapping(geom)  # shapely → geojson dict
        folium.GeoJson(
            data=geojson_obj,
            name="building",
            style_function=lambda x: {
                "color": "#2E86AB",
                "weight": 3,
                "fillColor": "#2E86AB",
                "fillOpacity": 0.15,
            },
            tooltip=f"Building {b['id']}",
        ).add_to(m)


    # Plot places (closest first if distance_m provided)
    if places and isinstance(places, list):
        # sort by distance_m if present
        try:
            places_sorted = sorted(places, key=lambda r: float(r.get("distance_m", 1e9)))
        except Exception:
            places_sorted = places
        for idx, r in enumerate(places_sorted, start=1):
            plat, plon = float(r["lat"]), float(r["lon"])
            popup = folium.Popup(popup_html_for_place(r), max_width=320)
            folium.CircleMarker(
                location=(plat, plon),
                radius=5,
                color="#E67E22",
                fill=True,
                fill_opacity=0.9,
            ).add_child(popup).add_to(m)
            if args.show_labels:
                folium.map.Marker(
                    [plat, plon],
                    icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0),
                                 html=f'<div style="font-size:10pt;color:#E67E22">{idx}</div>')
                ).add_to(m)

    # Plot camera positions + heading arrows
    for im in images:
        meta_path = root / im["meta"]
        img_path  = root / im["file"] if im.get("file") else None
        if not meta_path.exists():
            continue
        meta = load_json(meta_path)

        geom = try_get(meta, "computed_geometry", "geometry")
        if not geom or "coordinates" not in geom:
            continue
        cam_lon, cam_lat = geom["coordinates"]
        compass = try_get(meta, "compass_angle", "computed_compass_angle")

        # point marker
        folium.CircleMarker(
            location=(cam_lat, cam_lon),
            radius=4,
            color="#1B998B",
            fill=True,
            fill_opacity=1.0,
        ).add_to(m)

        # heading arrow (short line)
        if compass is not None:
            end_lat, end_lon = offset_latlon(cam_lat, cam_lon, args.arrow_m, float(compass))
            folium.PolyLine(
                locations=[(cam_lat, cam_lon), (end_lat, end_lon)],
                color="#1B998B",
                weight=3,
                opacity=0.8,
            ).add_to(m)

        # popup with info (and local file names)
        popup = folium.Popup(popup_html_for_image(im.get("file"), im.get("meta"), meta), max_width=340)
        folium.Marker(
            location=(cam_lat, cam_lon),
            icon=folium.Icon(icon="camera", prefix="fa", color="green")
        ).add_to(m).add_child(popup)

    folium.LayerControl().add_to(m)
    m.save(args.out)
    print(f"✓ Wrote {args.out}. Open it in your browser.")
    print("Tip: if markers look squished, zoom out one level or set --arrow-m 10")

if __name__ == "__main__":
    main()
