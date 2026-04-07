#!/usr/bin/env python3
"""
walayar_osm_grid_kml.py
========================
Generate a new Walayar Range Forest KML overlay with:
  - 250 m (N-S height) × 500 m (E-W width) grid cells
  - OSM feature layers: water bodies, crop fields, settlements,
    roads and railways (Palakkad Railway)
  - Proper polygon extraction of large water bodies (Malampuzha Reservoir,
    Walayar Lake) from osm_features.json
  - New OSM fetch for roads / railways cached to osm_roads.json

Run:
    python walayar_osm_grid_kml.py

Output:
    Walayar_Range_Grid_OSM.kml  (open in Google Earth)
"""

import json
import math
import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

WALAYAR_KML   = "Walayar_Range_clean.kml"
OSM_FEATURES    = "osm_features.json"
OSM_ROADS       = "osm_roads.json"
OSM_SETTLE_POLYS = "osm_settle_polys.json"
OUTPUT_KML      = "Walayar_Range_Grid_OSM.kml"

KML_NS        = "http://www.opengis.net/kml/2.2"
LAT_M         = 111320.0          # metres per degree latitude

GRID_H_M      = 500.0             # grid cell height (N-S, metres) — square cells
GRID_W_M      = 500.0             # grid cell width  (E-W, metres) — must equal GRID_H_M

OVERPASS_URL  = "https://overpass-api.de/api/interpreter"

# Bounding box for OSM queries (same as fetch_osm_features.py)
BBOX = dict(min_lat=10.73, min_lon=76.58, max_lat=10.95, max_lon=76.90)

# KML colors  AABBGGRR (KML byte order)
COL = {
    # feature polygon / line fills
    "water_fill"    : "99ff6600",   # semi-transparent amber-blue (water)
    "water_border"  : "ffff6600",
    "crop_fill"     : "6600d4ff",   # semi-transparent yellow
    "crop_border"   : "ff00aaff",
    "settle_fill"   : "660055ff",   # semi-transparent orange
    "settle_border" : "ff0055ff",
    "rail_line"     : "ff222222",   # near-black for railway
    "road_line"     : "ff505050",   # dark grey for roads
    # grid cell shading (very transparent overlays)
    "cell_water"    : "44ff6600",
    "cell_crop"     : "3300d4ff",
    "cell_settle"   : "330055ff",
    "cell_road"     : "33505050",
    "cell_forest"   : "22006600",
    # grid line
    "grid_line"     : "ff0000ff",   # red
    # forest sections
    "forest_border" : "ff005500",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def lon_m_per_deg(lat_deg):
    return LAT_M * math.cos(math.radians(lat_deg))


def deg_lat(metres):
    return metres / LAT_M


def deg_lon(metres, lat_deg):
    return metres / lon_m_per_deg(lat_deg)


# ---------------------------------------------------------------------------
# 1. PARSE WALAYAR FOREST KML
# ---------------------------------------------------------------------------

def parse_forest_kml(kml_file):
    """Return list of {name, coords[(lon,lat)]} for every forest section."""
    tree   = ET.parse(kml_file)
    root   = tree.getroot()
    ns     = {"k": KML_NS}
    sections = []
    for pm in root.findall(".//k:Placemark", ns):
        name_el = pm.find("k:name", ns)
        if name_el is None:
            continue
        coords_el = pm.find(".//k:coordinates", ns)
        if coords_el is None or not coords_el.text:
            continue
        pts = []
        for tok in coords_el.text.strip().split():
            parts = tok.split(",")
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        if len(pts) >= 3:
            sections.append({"name": name_el.text or "", "coords": pts})
    if not sections:
        raise ValueError(f"No Placemarks found in {kml_file}")
    return sections


# ---------------------------------------------------------------------------
# 2. LOAD OSM FEATURES (water, crops, settlements)
# ---------------------------------------------------------------------------

def area_km2_coords(coords):
    """Shoelace area in km² for a list of (lon,lat) pairs."""
    if len(coords) < 3:
        return 0.0
    mid_lat = sum(c[1] for c in coords) / len(coords)
    lm = LAT_M
    sm = lon_m_per_deg(mid_lat)
    pts = [(c[0] * sm, c[1] * lm) for c in coords]
    n   = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0 / 1e6


def coords_to_shapely(coords):
    """(lon,lat) list → Shapely Polygon, or None if invalid."""
    if len(coords) < 3:
        return None
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


def load_osm_features():
    """
    Load and return feature geometries from osm_features.json.

    Returns dict with:
      water_polys  : list of Shapely Polygons for water bodies
      water_info   : parallel list of {name, area_km2, centroid}
      crop_polys   : list of Shapely Polygons for crop / farmland
      crop_info    : parallel list of {name, area_km2, centroid}
      settlements  : list of {lon, lat, name, place_type}
    """
    if not os.path.exists(OSM_FEATURES):
        print(f"  WARNING: {OSM_FEATURES} not found – water/crop/settlement layers skipped.")
        return dict(water_polys=[], water_info=[],
                    crop_polys=[], crop_info=[], settlements=[])

    with open(OSM_FEATURES) as f:
        data = json.load(f)

    water_polys, water_info = [], []
    for p in data.get("water_polygons", []):
        coords = [(c[0], c[1]) for c in p["coords"]]
        geom = coords_to_shapely(coords)
        if geom is None or p.get("area_km2", 0) < 0.01:
            continue
        water_polys.append(geom)
        water_info.append({
            "name"    : p.get("name", "") or "",
            "area_km2": p.get("area_km2", 0),
            "centroid": p.get("centroid", (0, 0)),
        })

    crop_polys, crop_info = [], []
    for p in data.get("crop_polygons", []):
        coords = [(c[0], c[1]) for c in p["coords"]]
        geom = coords_to_shapely(coords)
        if geom is None or p.get("area_km2", 0) < 0.005:
            continue
        crop_polys.append(geom)
        crop_info.append({
            "name"    : p.get("name", "") or "",
            "area_km2": p.get("area_km2", 0),
            "centroid": p.get("centroid", (0, 0)),
        })

    settlements = data.get("settlements", [])

    print(f"  Loaded {len(water_polys)} water polygons, "
          f"{len(crop_polys)} crop polygons, "
          f"{len(settlements)} settlements")
    return dict(water_polys=water_polys, water_info=water_info,
                crop_polys=crop_polys, crop_info=crop_info,
                settlements=settlements)


# ---------------------------------------------------------------------------
# 3. FETCH / LOAD ROADS & RAILWAYS
# ---------------------------------------------------------------------------

def _overpass_query(query_body):
    payload = urllib.parse.urlencode({"data": query_body}).encode()
    req = urllib.request.Request(
        OVERPASS_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": "WalayarForestResearch/1.0 (academic)"},
        method="POST",
    )
    print("  Querying Overpass API …", end=" ", flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    n = len(data.get("elements", []))
    print(f"OK ({n} elements)")
    return data


def fetch_roads_railways(bbox):
    """
    Fetch primary/secondary/tertiary roads and railway lines from OSM.
    Returns dict saved to osm_roads.json.
    """
    b = f"{bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']}"
    query = f"""
[out:json][timeout:120];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|road"]({b});
  way["railway"~"rail|narrow_gauge|tram|monorail|funicular"]({b});
  relation["railway"~"rail|narrow_gauge"]({b});
);
out body;
>;
out skel qt;
"""
    data = _overpass_query(query)
    elements = data.get("elements", [])

    node_map = {e["id"]: e for e in elements if e["type"] == "node"}

    roads = []
    railways = []

    for el in elements:
        if el["type"] != "way":
            continue
        tags  = el.get("tags", {})
        nodes = el.get("nodes", [])
        coords = []
        for nid in nodes:
            if nid in node_map:
                n = node_map[nid]
                coords.append([n["lon"], n["lat"]])
        if len(coords) < 2:
            continue

        hw  = tags.get("highway", "")
        rw  = tags.get("railway", "")
        name = tags.get("name", "") or tags.get("ref", "")

        if rw:
            railways.append({"coords": coords, "name": name, "type": rw})
        elif hw:
            roads.append({"coords": coords, "name": name, "type": hw})

    result = {"roads": roads, "railways": railways}
    with open(OSM_ROADS, "w") as f:
        json.dump(result, f)
    print(f"  Saved {len(roads)} road segments + {len(railways)} railway segments → {OSM_ROADS}")
    return result


def load_roads_railways():
    if os.path.exists(OSM_ROADS):
        print(f"  Using cached {OSM_ROADS}")
        with open(OSM_ROADS) as f:
            return json.load(f)
    print("  Fetching roads & railways from OSM …")
    return fetch_roads_railways(BBOX)


def roads_to_shapely(data):
    """Convert roads/railways dict → {roads: [LineString], railways: [LineString]}."""
    def to_lines(items):
        out = []
        for seg in items:
            pts = [(c[0], c[1]) for c in seg["coords"]]
            if len(pts) >= 2:
                try:
                    out.append((LineString(pts), seg["name"], seg["type"]))
                except Exception:
                    pass
        return out

    return dict(roads=to_lines(data.get("roads", [])),
                railways=to_lines(data.get("railways", [])))


# ---------------------------------------------------------------------------
# 3b. BUILD SETTLEMENT POLYGONS
# ---------------------------------------------------------------------------

# Buffer radius in metres by OSM place_type.
# Sized to cover the built-up footprint typically mapped in OSM for each type.
_SETTLE_BUF_M = {
    "city"              : 1200,
    "town"              : 600,
    "suburb"            : 350,
    "neighbourhood"     : 250,
    "village"           : 250,
    "residential"       : 250,
    "commercial"        : 200,
    "industrial"        : 200,
    "hamlet"            : 120,
    "isolated_dwelling" :  60,
}
_DEFAULT_BUF_M = 150


def _buf_deg(metres):
    """Convert a buffer radius in metres to degrees latitude (conservative)."""
    return metres / LAT_M


def build_settlement_polygons(settlements, water_polys, crop_polys):
    """
    Generate merged settlement area polygons:
      1. Buffer each settlement point by its type-appropriate radius.
      2. Apply a proximity-join buffer (200 m) so nearby points connect
         into one continuous area, then unary_union all buffers — nearby
         settlements merge into single larger polygons organically.
      3. Shrink back by the same join buffer to restore approximate
         true footprint boundaries (morphological closing).
      4. Clip out water bodies and crop fields so those layers stay clear.

    Returns list of (Shapely geom, name, place_type).
    """
    JOIN_M  = 200.0          # extra buffer to bridge nearby settlements
    join_d  = _buf_deg(JOIN_M)

    # Pre-build exclusion union once
    excl_geoms = list(water_polys) + list(crop_polys)
    exclusion  = unary_union(excl_geoms) if excl_geoms else None

    # Step 1 – collect per-point buffers enlarged by join distance
    expanded = []
    labels   = []           # (name, place_type) per raw buffer
    for s in settlements:
        ptype  = s.get("place_type", "")
        radius = _buf_deg(_SETTLE_BUF_M.get(ptype, _DEFAULT_BUF_M)) + join_d
        pt     = Point(s["lon"], s["lat"])
        buf    = pt.buffer(radius, resolution=32)
        expanded.append(buf)
        labels.append((s.get("name", "") or "", ptype))

    # Step 2 – union: overlapping/touching expanded circles merge
    merged = unary_union(expanded)
    if merged.is_empty:
        return []

    # Step 3 – shrink back by join distance (undo the expansion bridge)
    shrunk = merged.buffer(-join_d)
    if shrunk.is_empty or not shrunk.is_valid:
        shrunk = merged.buffer(0)

    # Collect individual polygons from the result
    if hasattr(shrunk, "geoms"):
        pieces = list(shrunk.geoms)
    else:
        pieces = [shrunk]

    # Step 4 – clip exclusion zones, label each merged piece with dominant name
    # Find the name of the settlement point whose centre lands inside each piece
    result = []
    for piece in pieces:
        if piece.is_empty or piece.area == 0:
            continue
        if not piece.is_valid:
            piece = piece.buffer(0)

        # Label: find closest/dominant settlement point inside this piece
        best_name, best_type = "", ""
        for s, (name, ptype) in zip(settlements, labels):
            if piece.contains(Point(s["lon"], s["lat"])):
                # prefer largest-type name
                if not best_name and name:
                    best_name, best_type = name, ptype
                rank_map = {"city":0,"town":1,"suburb":2,"village":3,"hamlet":4}
                if rank_map.get(ptype, 9) < rank_map.get(best_type, 9):
                    best_name, best_type = name or best_name, ptype

        # Clip water + crop
        if exclusion is not None:
            try:
                clipped = piece.difference(exclusion)
            except Exception:
                clipped = piece
        else:
            clipped = piece

        if clipped is None or clipped.is_empty:
            continue
        if not clipped.is_valid:
            clipped = clipped.buffer(0)
        if clipped.is_empty:
            continue

        # Emit sub-polygons if the clip split the piece
        if hasattr(clipped, "geoms"):
            for part in clipped.geoms:
                if not part.is_empty:
                    result.append((part, best_name, best_type))
        else:
            result.append((clipped, best_name, best_type))

    return result


# ---------------------------------------------------------------------------
# 4. BUILD GRID & CLASSIFY CELLS
# ---------------------------------------------------------------------------

def build_grid(lat_min, lat_max, lon_min, lon_max, mid_lat):
    """
    Return list of (cell_box, row, col) where cell_box is a Shapely box.
    Cell size: GRID_H_M metres (lat) × GRID_W_M metres (lon).
    """
    dlat = deg_lat(GRID_H_M)
    dlon = deg_lon(GRID_W_M, mid_lat)

    cells = []
    row = 0
    lat = lat_min
    while lat < lat_max - dlat * 0.01:
        col = 0
        lon = lon_min
        while lon < lon_max - dlon * 0.01:
            cell = box(lon, lat, lon + dlon, lat + dlat)
            cells.append((cell, row, col))
            col += 1
            lon += dlon
        row += 1
        lat += dlat

    return cells, dlat, dlon


def classify_cells(cells, forest_geom, water_polys, crop_polys,
                   settlements, road_lines, rail_lines):
    """
    For each grid cell determine dominant feature type.
    Priority: water > settlement > road/rail > crop > forest > empty.

    Returns list of (cell_box, feature_type) in same order as cells.
    """
    # Build settlement buffers (50 m radius points)
    settle_geoms = []
    mid_lat = forest_geom.centroid.y
    buf_deg = deg_lat(50.0)
    for s in settlements:
        p = Point(s["lon"], s["lat"])
        settle_geoms.append(p.buffer(buf_deg))
    settle_union = unary_union(settle_geoms) if settle_geoms else None

    # Merge rail lines into buffered union (10 m buffer)
    rail_buf = deg_lat(10.0)
    rail_union = (unary_union([g.buffer(rail_buf) for g, *_ in rail_lines])
                  if rail_lines else None)

    # Merge road lines into buffered union (8 m)
    road_buf = deg_lat(8.0)
    road_union = (unary_union([g.buffer(road_buf) for g, *_ in road_lines])
                  if road_lines else None)

    # Merge all water polygons
    water_union = unary_union(water_polys) if water_polys else None

    # Merge all crop polygons
    crop_union = unary_union(crop_polys) if crop_polys else None

    results = []
    for cell, row, col in cells:
        cell_area = cell.area
        if cell_area == 0:
            results.append((cell, row, col, "empty"))
            continue

        # Water
        if water_union is not None:
            inter = cell.intersection(water_union)
            if inter.area / cell_area > 0.08:
                results.append((cell, row, col, "water"))
                continue

        # Settlement
        if settle_union is not None and cell.intersects(settle_union):
            results.append((cell, row, col, "settlement"))
            continue

        # Railway
        if rail_union is not None and cell.intersects(rail_union):
            results.append((cell, row, col, "railway"))
            continue

        # Road
        if road_union is not None and cell.intersects(road_union):
            results.append((cell, row, col, "road"))
            continue

        # Crop
        if crop_union is not None:
            inter = cell.intersection(crop_union)
            if inter.area / cell_area > 0.10:
                results.append((cell, row, col, "crop"))
                continue

        # Default: forest
        results.append((cell, row, col, "forest"))

    return results


# ---------------------------------------------------------------------------
# 5. KML BUILDING HELPERS
# ---------------------------------------------------------------------------

def _kml_coords(ring_coords, alt=0):
    """Shapely exterior coords → KML coordinates text."""
    return "\n          ".join(
        f"{lon:.7f},{lat:.7f},{alt}" for lon, lat in ring_coords
    )


def _add_style(parent, sid, fill_color, line_color, line_width=1, fill=1):
    s = ET.SubElement(parent, "Style", id=sid)
    ps = ET.SubElement(s, "PolyStyle")
    ET.SubElement(ps, "color").text = fill_color
    ET.SubElement(ps, "fill").text  = str(fill)
    ls = ET.SubElement(s, "LineStyle")
    ET.SubElement(ls, "color").text = line_color
    ET.SubElement(ls, "width").text = str(line_width)
    return s


def _add_line_style(parent, sid, line_color, line_width=2):
    s = ET.SubElement(parent, "Style", id=sid)
    ls = ET.SubElement(s, "LineStyle")
    ET.SubElement(ls, "color").text = line_color
    ET.SubElement(ls, "width").text = str(line_width)
    ET.SubElement(ET.SubElement(s, "PolyStyle"), "fill").text = "0"
    return s


def _placemark_polygon(folder, name, description, style_url, poly):
    """Add a Polygon Placemark to folder from a Shapely polygon."""
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    ET.SubElement(pm, "description").text = description
    ET.SubElement(pm, "styleUrl").text = style_url
    pg = ET.SubElement(pm, "Polygon")
    ET.SubElement(pg, "tessellate").text = "1"
    ob = ET.SubElement(pg, "outerBoundaryIs")
    lr = ET.SubElement(ob, "LinearRing")
    # close the ring
    ext = list(poly.exterior.coords)
    ET.SubElement(lr, "coordinates").text = "\n          " + _kml_coords(ext)


def _placemark_line(folder, name, description, style_url, linestring):
    """Add a LineString Placemark to folder."""
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    ET.SubElement(pm, "description").text = description
    ET.SubElement(pm, "styleUrl").text = style_url
    ls = ET.SubElement(pm, "LineString")
    ET.SubElement(ls, "tessellate").text = "1"
    pts = list(linestring.coords)
    ET.SubElement(ls, "coordinates").text = "\n          " + _kml_coords(pts)


# ---------------------------------------------------------------------------
# 6. GENERATE KML
# ---------------------------------------------------------------------------

SECTION_COLORS = ["2ecc71", "3498db", "9b59b6", "e74c3c", "f39c12", "1abc9c"]

CELL_STYLE_MAP = {
    "water"     : ("#cellWater",     COL["cell_water"],  COL["water_border"], 1),
    "crop"      : ("#cellCrop",      COL["cell_crop"],   COL["crop_border"],  1),
    "settlement": ("#cellSettle",    COL["cell_settle"], COL["settle_border"],1),
    "railway"   : ("#cellRail",      COL["cell_road"],   COL["rail_line"],    1),
    "road"      : ("#cellRoad",      COL["cell_road"],   COL["road_line"],    1),
    "forest"    : ("#cellForest",    COL["cell_forest"], COL["forest_border"],1),
    "empty"     : ("#cellForest",    COL["cell_forest"], COL["forest_border"],0),
}


def build_kml(sections, osm, roads_data, settle_polys_shapely,
              classified_cells, dlat, dlon,
              lat_min, lat_max, lon_min, lon_max, mid_lat, grid_lat_max):

    ET.register_namespace("", KML_NS)
    kml  = ET.Element("kml", xmlns=KML_NS)
    doc  = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = "Walayar Range — 250m×500m Grid + OSM Features"
    ET.SubElement(doc, "description").text = (
        "Walayar Forest sections with 250m(N-S)×500m(E-W) grid covering the "
        "southern half of the range. Cells coloured by dominant OSM feature: "
        "water, crop, settlement, road/railway, forest."
    )

    # ── Styles ──────────────────────────────────────────────────────────────
    for idx, hx in enumerate(SECTION_COLORS):
        _add_style(doc, f"sec{idx}", f"7f{hx}", f"ff{hx}", line_width=2)

    _add_style(doc, "cellWater",  COL["cell_water"],  COL["water_border"], 1)
    _add_style(doc, "cellCrop",   COL["cell_crop"],   COL["crop_border"],  1)
    _add_style(doc, "cellSettle", COL["cell_settle"], COL["settle_border"],1)
    _add_style(doc, "cellRail",   COL["cell_road"],   COL["rail_line"],    1)
    _add_style(doc, "cellRoad",   COL["cell_road"],   COL["road_line"],    1)
    _add_style(doc, "cellForest", COL["cell_forest"], COL["forest_border"],1)

    _add_style(doc, "waterPoly",   COL["water_fill"],  COL["water_border"],  2)
    _add_style(doc, "cropPoly",    COL["crop_fill"],   COL["crop_border"],   1)
    _add_style(doc, "settlePoly",  COL["settle_fill"], COL["settle_border"], 2)
    _add_line_style(doc, "railLine", COL["rail_line"], 3)
    _add_line_style(doc, "roadLine", COL["road_line"], 2)

    # ── Forest Sections ──────────────────────────────────────────────────────
    sec_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(sec_folder, "name").text = "Forest Sections"
    for idx, sec in enumerate(sections):
        pm   = ET.SubElement(sec_folder, "Placemark")
        ET.SubElement(pm, "name").text        = sec["name"]
        ET.SubElement(pm, "styleUrl").text    = f"#sec{idx % len(SECTION_COLORS)}"
        ET.SubElement(pm, "description").text = (
            f"Forest Section: {sec['name']}\n"
            f"Boundary points: {len(sec['coords'])}"
        )
        mg = ET.SubElement(pm, "MultiGeometry")
        pg = ET.SubElement(mg, "Polygon")
        ob = ET.SubElement(pg, "outerBoundaryIs")
        lr = ET.SubElement(ob, "LinearRing")
        ET.SubElement(lr, "coordinates").text = (
            "\n          " + _kml_coords(sec["coords"])
        )

    # ── Grid Cells ───────────────────────────────────────────────────────────
    grid_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(grid_folder, "name").text = "500m×500m Grid Cells (bottom half)"
    ET.SubElement(grid_folder, "description").text = (
        "Each cell is 500 m × 500 m (square). "
        "Colour: blue=water, yellow=crop, orange=settlement, "
        "grey=road/railway, green=forest."
    )

    style_map = {
        "water"     : "#cellWater",
        "crop"      : "#cellCrop",
        "settlement": "#cellSettle",
        "railway"   : "#cellRail",
        "road"      : "#cellRoad",
        "forest"    : "#cellForest",
        "empty"     : "#cellForest",
    }

    feature_counts = {}
    for cell, row, col, ftype in classified_cells:
        feature_counts[ftype] = feature_counts.get(ftype, 0) + 1
        pm = ET.SubElement(grid_folder, "Placemark")
        ET.SubElement(pm, "name").text = f"R{row:03d}C{col:03d}"
        ET.SubElement(pm, "styleUrl").text = style_map[ftype]
        pg = ET.SubElement(pm, "Polygon")
        ET.SubElement(pg, "tessellate").text = "1"
        ob = ET.SubElement(pg, "outerBoundaryIs")
        lr = ET.SubElement(ob, "LinearRing")
        ext = list(cell.exterior.coords)
        ET.SubElement(lr, "coordinates").text = "\n          " + _kml_coords(ext)

    # ── Grid Lines overlay ───────────────────────────────────────────────────
    line_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(line_folder, "name").text = "Grid Lines (500m×500m)"
    ET.SubElement(line_folder, "visibility").text = "0"   # hidden by default

    # horizontal lines — span only the grid's lat extent (bottom half)
    lat = lat_min
    i = 0
    while lat <= grid_lat_max + dlat * 0.01:
        pm = ET.SubElement(line_folder, "Placemark")
        ET.SubElement(pm, "name").text = f"H{i}"
        st = ET.SubElement(pm, "Style")
        ls = ET.SubElement(st, "LineStyle")
        ET.SubElement(ls, "color").text = COL["grid_line"]
        ET.SubElement(ls, "width").text = "1"
        lstr = ET.SubElement(pm, "LineString")
        ET.SubElement(lstr, "coordinates").text = (
            f"\n          {lon_min:.7f},{lat:.7f},0"
            f"\n          {lon_max:.7f},{lat:.7f},0"
        )
        lat += dlat
        i += 1

    # vertical lines — span only the grid's lat extent (bottom half)
    lon = lon_min
    j = 0
    while lon <= lon_max + dlon * 0.01:
        pm = ET.SubElement(line_folder, "Placemark")
        ET.SubElement(pm, "name").text = f"V{j}"
        st = ET.SubElement(pm, "Style")
        ls = ET.SubElement(st, "LineStyle")
        ET.SubElement(ls, "color").text = COL["grid_line"]
        ET.SubElement(ls, "width").text = "1"
        lstr = ET.SubElement(pm, "LineString")
        ET.SubElement(lstr, "coordinates").text = (
            f"\n          {lon:.7f},{lat_min:.7f},0"
            f"\n          {lon:.7f},{grid_lat_max:.7f},0"
        )
        lon += dlon
        j += 1

    # ── Water Bodies ─────────────────────────────────────────────────────────
    water_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(water_folder, "name").text = "Water Bodies"
    for geom, info in zip(osm["water_polys"], osm["water_info"]):
        name = info["name"] or "Water Body"
        desc = f"Area: {info['area_km2']:.3f} km²"
        if isinstance(geom, MultiPolygon):
            for part in geom.geoms:
                _placemark_polygon(water_folder, name, desc, "#waterPoly", part)
        else:
            _placemark_polygon(water_folder, name, desc, "#waterPoly", geom)

    # ── Crop Fields ──────────────────────────────────────────────────────────
    crop_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(crop_folder, "name").text = "Crop Fields"
    for geom, info in zip(osm["crop_polys"], osm["crop_info"]):
        name = info["name"] or "Farmland"
        desc = f"Type: farmland/crop\nArea: {info['area_km2']:.4f} km²"
        if isinstance(geom, MultiPolygon):
            for part in geom.geoms:
                _placemark_polygon(crop_folder, name, desc, "#cropPoly", part)
        else:
            _placemark_polygon(crop_folder, name, desc, "#cropPoly", geom)

    # ── Settlements (buffered polygons clipped against water+crop) ──────────
    settle_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(settle_folder, "name").text = "Settlements"
    ET.SubElement(settle_folder, "description").text = (
        "Settlement area polygons buffered from OSM place markers. "
        "Water bodies and crop fields have been clipped out."
    )
    for geom, name, ptype in settle_polys_shapely:
        label = name or ptype or "Settlement"
        desc  = f"Type: {ptype}"
        if isinstance(geom, MultiPolygon):
            for part in geom.geoms:
                _placemark_polygon(settle_folder, label, desc, "#settlePoly", part)
        else:
            _placemark_polygon(settle_folder, label, desc, "#settlePoly", geom)

    # ── Roads ────────────────────────────────────────────────────────────────
    road_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(road_folder, "name").text = "Roads"
    for line, name, rtype in roads_data["roads"]:
        desc = f"Highway type: {rtype}"
        _placemark_line(road_folder, name or rtype, desc, "#roadLine", line)

    # ── Railways ─────────────────────────────────────────────────────────────
    rail_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(rail_folder, "name").text = "Railways (Palakkad Railway)"
    for line, name, rtype in roads_data["railways"]:
        desc = f"Railway type: {rtype}"
        _placemark_line(rail_folder, name or "Railway", desc, "#railLine", line)

    return kml, feature_counts


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Walayar Range — 250m×500m Grid + OSM KML Generator")
    print("=" * 60)

    # 1. Parse forest sections
    print(f"\n[1] Parsing {WALAYAR_KML} …")
    sections = parse_forest_kml(WALAYAR_KML)
    print(f"  {len(sections)} forest sections loaded")

    all_pts  = [pt for s in sections for pt in s["coords"]]
    lons_all = [p[0] for p in all_pts]
    lats_all = [p[1] for p in all_pts]
    lat_min, lat_max = min(lats_all), max(lats_all)
    lon_min, lon_max = min(lons_all), max(lons_all)
    mid_lat  = (lat_min + lat_max) / 2.0

    print(f"  Bounds: lat [{lat_min:.4f}, {lat_max:.4f}]  "
          f"lon [{lon_min:.4f}, {lon_max:.4f}]")

    # 2. Load OSM features
    print(f"\n[2] Loading OSM features from {OSM_FEATURES} …")
    osm = load_osm_features()

    # 3. Load / fetch roads & railways
    print(f"\n[3] Roads & railways …")
    roads_raw  = load_roads_railways()
    roads_data = roads_to_shapely(roads_raw)
    print(f"  {len(roads_data['roads'])} road segments, "
          f"{len(roads_data['railways'])} railway segments")

    # 3b. Settlement polygons — buffer point markers, clip out water+crop
    print(f"\n[3b] Building settlement polygons from {len(osm['settlements'])} markers …")
    settle_shapely = build_settlement_polygons(
        osm["settlements"], osm["water_polys"], osm["crop_polys"]
    )
    print(f"  {len(settle_shapely)} settlement polygons generated")

    # 4. Build forest union geometry
    print(f"\n[4] Building forest geometry …")
    forest_polys = []
    for s in sections:
        try:
            p = Polygon(s["coords"])
            if not p.is_valid:
                p = p.buffer(0)
            forest_polys.append(p)
        except Exception:
            pass
    forest_geom = unary_union(forest_polys) if forest_polys else None
    print(f"  Forest union: {forest_geom.area:.6f} sq-degrees")

    # 5. Build grid — bottom half of region only
    grid_lat_max = (lat_min + lat_max) / 2.0   # southern half upper bound
    print(f"\n[5] Building 500m×500m square grid (bottom half: {lat_min:.4f}–{grid_lat_max:.4f}) …")
    cells, dlat, dlon = build_grid(lat_min, grid_lat_max, lon_min, lon_max, mid_lat)
    print(f"  Grid: {len(cells)} cells  "
          f"(dlat={dlat*111320:.1f} m, dlon={dlon*lon_m_per_deg(mid_lat):.1f} m)")

    # 6. Classify cells
    print(f"\n[6] Classifying cells by OSM feature …")
    classified = classify_cells(
        cells, forest_geom,
        osm["water_polys"], osm["crop_polys"], osm["settlements"],
        roads_data["roads"], roads_data["railways"],
    )

    type_counts = {}
    for _, _, _, ft in classified:
        type_counts[ft] = type_counts.get(ft, 0) + 1
    for ft, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ft:12s}: {cnt:5d} cells")

    # 7. Build KML
    print(f"\n[7] Building KML …")
    kml_root, _ = build_kml(
        sections, osm, roads_data, settle_shapely,
        classified, dlat, dlon,
        lat_min, lat_max, lon_min, lon_max, mid_lat, grid_lat_max,
    )

    # 8. Write file
    tree = ET.ElementTree(kml_root)
    with open(OUTPUT_KML, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    print(f"\n  Saved → {OUTPUT_KML}")

    print(f"\n{'='*60}")
    print(f"KML contents:")
    print(f"  Forest sections : {len(sections)}")
    print(f"  Grid cells      : {len(classified)}")
    print(f"  Water polygons  : {len(osm['water_polys'])}")
    print(f"  Crop polygons   : {len(osm['crop_polys'])}")
    print(f"  Settlement polys: {len(settle_shapely)} (from {len(osm['settlements'])} markers)")
    print(f"  Road segments   : {len(roads_data['roads'])}")
    print(f"  Railway segments: {len(roads_data['railways'])}")
    print(f"\nOpen {OUTPUT_KML} in Google Earth.")


if __name__ == "__main__":
    main()
