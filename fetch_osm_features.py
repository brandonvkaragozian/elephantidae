"""
fetch_osm_features.py
=====================
Query the OpenStreetMap Overpass API for water bodies and human settlements /
agricultural land in the Walayar Range Forest bounding box.

Results are cached to  osm_features.json  so the trajectory script does not
need network access on every run.

Usage
-----
    python fetch_osm_features.py

Output
------
    osm_features.json   — dict with keys:
        "water_points"    : list of (lon, lat, name, type) dicts
        "water_polygons"  : list of polygon dicts with "coords" key
        "settlements"     : list of (lon, lat, name, place_type) dicts
        "crop_polygons"   : list of polygon dicts with "coords" key
        "bounding_box"    : [min_lat, min_lon, max_lat, max_lon]

The Overpass API is publicly available without authentication.
Rate limit: 1 request per ~10 seconds is polite; results are cached.
"""

import json
import math
import urllib.request
import urllib.parse
import sys

# ---------------------------------------------------------------------------
# Walayar Range bounding box (slightly larger than forest extent to capture
# features at edges — settlements are often just outside forest boundary)
# ---------------------------------------------------------------------------
BBOX = {
    "min_lat": 10.73,
    "min_lon": 76.58,
    "max_lat": 10.95,
    "max_lon": 76.90,
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OUTPUT_FILE  = "osm_features.json"


def overpass_query(query_body):
    """
    POST a query to the Overpass API and return parsed JSON.
    Raises on HTTP error or timeout.
    """
    payload = urllib.parse.urlencode({"data": query_body}).encode()
    req = urllib.request.Request(
        OVERPASS_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": "ElephantTrajectoryResearch/1.0 (academic)"},
        method="POST",
    )
    print(f"  Querying Overpass API …", end=" ", flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    print(f"OK ({len(data.get('elements', []))} elements)")
    return data


def centroid(coords):
    """Compute centroid of a list of (lon, lat) pairs."""
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    return (cx, cy)


def area_km2(coords):
    """
    Approximate area in km² using the shoelace formula with local flat-Earth.
    """
    if len(coords) < 3:
        return 0.0
    lat_m = 111320.0
    mid_lat = sum(c[1] for c in coords) / len(coords)
    lon_m = 111320.0 * math.cos(math.radians(mid_lat))
    pts_m = [(c[0] * lon_m, c[1] * lat_m) for c in coords]
    n = len(pts_m)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts_m[i][0] * pts_m[j][1]
        area -= pts_m[j][0] * pts_m[i][1]
    return abs(area) / 2.0 / 1e6   # m² → km²


def nodes_to_coords(node_map, way_nodes):
    """Convert a list of node IDs to (lon, lat) pairs using node_map."""
    coords = []
    for nid in way_nodes:
        if nid in node_map:
            n = node_map[nid]
            coords.append((n["lon"], n["lat"]))
    return coords


# ---------------------------------------------------------------------------
# WATER FEATURES QUERY
# ---------------------------------------------------------------------------

def fetch_water_features(bbox):
    """
    Fetch water bodies, rivers, streams, canals, reservoirs, and springs.
    Handles node points, closed way polygons, linear ways, AND relation
    multipolygons (e.g. Malampuzha Reservoir which is stored as a relation).

    Returns
    -------
    water_points   : list of {"lon","lat","name","type"} dicts
    water_polygons : list of {"coords","name","type","area_km2","centroid"} dicts
    """
    b = f"{bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']}"
    query = f"""
[out:json][timeout:120];
(
  node["natural"="spring"]({b});
  node["natural"="water"]({b});
  node["waterway"~"waterfall|weir"]({b});
  way["natural"="water"]({b});
  way["waterway"~"river|stream|canal|drain"]({b});
  way["landuse"="reservoir"]({b});
  way["water"~"reservoir|lake|pond|oxbow|river"]({b});
  relation["natural"="water"]({b});
  relation["water"~"reservoir|lake|river"]({b});
  relation["landuse"="reservoir"]({b});
  relation["waterway"="riverbank"]({b});
);
out body;
>;
out skel qt;
"""
    data = overpass_query(query)
    elements = data.get("elements", [])

    node_map = {e["id"]: e for e in elements if e["type"] == "node"}
    way_map  = {e["id"]: e for e in elements if e["type"] == "way"}

    water_points   = []
    water_polygons = []

    for el in elements:
        if el["type"] == "node":
            tags = el.get("tags", {})
            nat = tags.get("natural", "")
            ww  = tags.get("waterway", "")
            if nat in ("spring", "water") or ww:
                water_points.append({
                    "lon" : el["lon"],
                    "lat" : el["lat"],
                    "name": tags.get("name", ""),
                    "type": nat or ww,
                })

        elif el["type"] == "way":
            tags   = el.get("tags", {})
            nat    = tags.get("natural", "")
            ww     = tags.get("waterway", "")
            lu     = tags.get("landuse",  "")
            wt     = tags.get("water",    "")
            coords = nodes_to_coords(node_map, el.get("nodes", []))
            if not coords:
                continue
            nodes_ids = el.get("nodes", [])
            is_closed = len(nodes_ids) > 2 and nodes_ids[0] == nodes_ids[-1]
            if is_closed:
                a = area_km2(coords)
                cx, cy = centroid(coords)
                water_polygons.append({
                    "coords"   : coords,
                    "name"     : tags.get("name", ""),
                    "type"     : nat or ww or lu or wt or "water",
                    "area_km2" : a,
                    "centroid" : (cx, cy),
                })
            else:
                mid = coords[len(coords) // 2]
                water_points.append({
                    "lon" : mid[0],
                    "lat" : mid[1],
                    "name": tags.get("name", ""),
                    "type": nat or ww or "waterway",
                })

        elif el["type"] == "relation":
            tags = el.get("tags", {})
            name = tags.get("name", "")
            wtype = (tags.get("natural") or tags.get("water") or
                     tags.get("landuse") or "water")
            # Collect all nodes from outer member ways to form the polygon
            all_coords = []
            for member in el.get("members", []):
                if member.get("type") == "way" and member.get("role") in ("outer", ""):
                    way = way_map.get(member["ref"])
                    if way:
                        all_coords.extend(
                            nodes_to_coords(node_map, way.get("nodes", [])))
            if len(all_coords) >= 3:
                a = area_km2(all_coords)
                cx, cy = centroid(all_coords)
                # Only store if area is meaningful
                if a > 0.01:
                    water_polygons.append({
                        "coords"   : all_coords,
                        "name"     : name,
                        "type"     : wtype,
                        "area_km2" : a,
                        "centroid" : (cx, cy),
                    })

    return water_points, water_polygons


# ---------------------------------------------------------------------------
# SETTLEMENT & AGRICULTURAL LAND QUERY
# ---------------------------------------------------------------------------

def fetch_settlement_features(bbox):
    """
    Fetch human settlements (towns, villages, hamlets, residential areas)
    and agricultural land (farmland, plantation, orchard, paddy) from OSM.

    Returns
    -------
    settlements   : list of {"lon", "lat", "name", "place_type"} dicts
    crop_polygons : list of {"coords", "name", "type", "area_km2",
                              "centroid"} dicts
    """
    b = f"{bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']}"
    query = f"""
[out:json][timeout:90];
(
  node["place"~"city|town|village|hamlet|suburb|neighbourhood|isolated_dwelling"]({b});
  way["place"~"city|town|village|hamlet"]({b});
  way["landuse"~"residential|commercial|industrial"]({b});
  way["building"~"yes|residential|house|apartments"]({b});
  way["landuse"~"farmland|farm|orchard|plantation|vineyard|paddy|agricultural"]({b});
  way["landuse"="greenhouse_horticulture"]({b});
  way["crop"]({b});
);
out body;
>;
out skel qt;
"""
    data = overpass_query(query)
    elements = data.get("elements", [])

    node_map = {e["id"]: e for e in elements if e["type"] == "node"}

    settlements   = []
    crop_polygons = []

    # density classification: "dense" = city/town/suburb/residential area (elephant avoids always)
    # "sparse" = hamlet/isolated dwelling (mild avoidance only)
    DENSE_TYPES = {"city", "town", "suburb", "neighbourhood",
                   "residential", "commercial", "industrial"}

    for el in elements:
        if el["type"] == "node":
            tags  = el.get("tags", {})
            place = tags.get("place", "")
            if place:
                settlements.append({
                    "lon"       : el["lon"],
                    "lat"       : el["lat"],
                    "name"      : tags.get("name", ""),
                    "place_type": place,
                    "dense"     : place in DENSE_TYPES,
                })

        elif el["type"] == "way":
            tags   = el.get("tags", {})
            lu     = tags.get("landuse", "")
            place  = tags.get("place",   "")
            coords = nodes_to_coords(node_map, el.get("nodes", []))
            if not coords:
                continue

            a = area_km2(coords)
            cx, cy = centroid(coords)

            if lu in ("farmland", "farm", "orchard", "plantation",
                      "vineyard", "paddy", "agricultural",
                      "greenhouse_horticulture") or "crop" in tags:
                crop_polygons.append({
                    "coords"   : coords,
                    "name"     : tags.get("name", ""),
                    "type"     : lu or "farmland",
                    "area_km2" : a,
                    "centroid" : (cx, cy),
                })
            elif lu in ("residential", "commercial", "industrial") or place:
                if a > 0:
                    settlements.append({
                        "lon"       : cx,
                        "lat"       : cy,
                        "name"      : tags.get("name", ""),
                        "place_type": place or lu,
                        "dense"     : (place or lu) in DENSE_TYPES,
                    })

    return settlements, crop_polygons


# ---------------------------------------------------------------------------
# SAVE & REPORT
# ---------------------------------------------------------------------------

def save_features(water_pts, water_polys, settlements, crop_polys):
    out = {
        "bounding_box"   : [BBOX["min_lat"], BBOX["min_lon"],
                             BBOX["max_lat"], BBOX["max_lon"]],
        "water_points"   : water_pts,
        "water_polygons" : [
            {**p, "coords": p["coords"]}  # already serialisable
            for p in water_polys
        ],
        "settlements"    : settlements,
        "crop_polygons"  : crop_polys,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUTPUT_FILE}")


def report(water_pts, water_polys, settlements, crop_polys):
    print("\n─── Summary ───────────────────────────────────────────")
    print(f"  Water points      : {len(water_pts)}")
    if water_pts[:5]:
        for w in water_pts[:5]:
            print(f"    ({w['lon']:.4f},{w['lat']:.4f})  {w['type']:12s}  {w['name']}")

    print(f"  Water polygons    : {len(water_polys)}")
    for w in sorted(water_polys, key=lambda x: x['area_km2'], reverse=True)[:8]:
        print(f"    centroid ({w['centroid'][0]:.4f},{w['centroid'][1]:.4f})"
              f"  area={w['area_km2']:.3f} km²  {w['type']:12s}  {w['name']}")

    print(f"  Settlements       : {len(settlements)}")
    for s in settlements[:8]:
        print(f"    ({s['lon']:.4f},{s['lat']:.4f})  {s['place_type']:14s}  {s['name']}")

    print(f"  Crop polygons     : {len(crop_polys)}")
    total_crop = sum(p['area_km2'] for p in crop_polys)
    print(f"    Total agricultural area: {total_crop:.1f} km²")
    for c in sorted(crop_polys, key=lambda x: x['area_km2'], reverse=True)[:6]:
        print(f"    centroid ({c['centroid'][0]:.4f},{c['centroid'][1]:.4f})"
              f"  area={c['area_km2']:.3f} km²  {c['type']:14s}  {c['name']}")
    print("────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("─" * 60)
    print("OSM Feature Extraction — Walayar Range")
    print(f"Bounding box: lat [{BBOX['min_lat']}, {BBOX['max_lat']}]  "
          f"lon [{BBOX['min_lon']}, {BBOX['max_lon']}]")
    print("─" * 60)

    print("\n[1] Fetching water features …")
    water_pts, water_polys = fetch_water_features(BBOX)

    print("\n[2] Fetching settlement and crop features …")
    settlements, crop_polys = fetch_settlement_features(BBOX)

    report(water_pts, water_polys, settlements, crop_polys)
    save_features(water_pts, water_polys, settlements, crop_polys)

    print(f"\nRun overlay_trajectories.py to regenerate trajectories using these features.")


if __name__ == "__main__":
    main()
