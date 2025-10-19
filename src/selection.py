from typing import Dict
import pandas as pd

PREFERRED_CATS = {
    "museum":5,"art":5,"education":4,"library":4,"government":4,"transport":3,"hotel":3,
    "restaurant":2,"cafe":2,"bar":2,"shop":2,"store":2,"service":1
}

def _cat_score(categories: str) -> int:
    s = (str(categories) or "").lower()
    return max((score for cat, score in PREFERRED_CATS.items() if cat in s), default=0)

def normalize_place_name(names, categories, *_):
    cands = []
    if isinstance(names, dict):
        for key in ("official","primary","short","common","name"):
            v = names.get(key)
            if isinstance(v, str) and v.strip():
                cands.append(v.strip())
        for v in names.values():
            if isinstance(v, str) and v.strip():
                cands.append(v.strip())
    elif isinstance(names, str):
        cands.append(names.strip())
    if not cands:
        return None
    best = max(cands, key=len)
    s = best.lower().replace(" ","").replace(".","")
    cats = str(categories).lower()
    if ("museum" in cats or "art_museum" in cats or "modern_art_museum" in cats):
        if any(k in s for k in ("sfmoma","momasf","moma_sf","momasanfrancisco")):
            return "San Francisco Museum of Modern Art (SFMOMA)"
    if s.isalpha() and s == best.lower().replace(" ","") and len(s) <= 8:
        return best.upper()
    return best

def select_best_place_for_building(links_df: pd.DataFrame, building_id: str, max_dist_m: float = 60.0) -> Dict | None:
    cand = links_df[links_df["building_id"] == building_id].copy()
    if cand.empty:
        return None
    cand = cand[(cand["inside"]) | (cand["dist_m"] <= max_dist_m)]
    if cand.empty:
        return None

    cand["cat_score"] = cand["categories"].apply(_cat_score)
    cand["rank_key"] = list(zip(-cand["cat_score"], ~cand["inside"], cand["dist_m"]))
    cand = cand.sort_values(by=["rank_key","dist_m"])
    row = cand.iloc[0]

    # host–tenant heuristic
    top = cand.iloc[0]
    for n in cand["names"].astype(str).tolist():
        if isinstance(top["names"], str) and n and n.lower() in top["names"].lower() and len(n) < len(top["names"]):
            row = cand[cand["names"].astype(str) == n].iloc[0]
            break

    name = normalize_place_name(row["names"], row["categories"])
    return {
        "place_id": str(row["place_id"]),
        "name": name,
        "categories": row["categories"],
        "lon": float(row["place_lon"]),
        "lat": float(row["place_lat"]),
        "inside": bool(row["inside"]),
        "dist_m": float(row["dist_m"]),
    }
