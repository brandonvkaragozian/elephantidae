#!/usr/bin/env python3
"""
south_africa_osm_kml.py
========================
Fetch OSM features (water bodies, crop fields, settlements, roads, railways)
for the Kruger National Park region where elephant GPS data was collected,
and generate a KML overlay combining elephant trajectories with OSM layers.

Run:
    python south_africa_osm_kml.py

Output:
    S_Africa_Elephants_OSM.kml   (open in Google Earth)
    south_africa_osm_cache.json  (cached OSM data)
"""

import csv
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

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
ELEPHANT_KML    = os.path.join(SCRIPT_DIR, "S. Africa Elephants.kml")
ELEPHANT_CSV    = os.path.join(SCRIPT_DIR, "ThermochronTracking Elephants Kruger 2007.csv")
OSM_CACHE       = os.path.join(SCRIPT_DIR, "south_africa_osm_cache.json")
OUTPUT_KML      = os.path.join(SCRIPT_DIR, "S_Africa_Elephants_OSM.kml")

# 7 spatially diverse elephants chosen for broad coverage of the Kruger region.
# Selection rationale:
#   AM107  – far west/south   (center -24.85, 31.38)
#   AM253  – southernmost     (center -24.93, 31.73)
#   AM254  – central-south    (center -24.71, 31.59)
#   AM308  – mid-west         (center -24.41, 31.50)
#   AM307  – northwest        (center -24.31, 31.36)
#   AM91   – northeast        (center -24.29, 31.80), largest home range
#   AM93   – east-central     (center -24.36, 31.81)
SELECTED_ELEPHANTS = None  # None = include all elephants that have data in the date range
TRAJECTORY_START   = "2007-08-13"   # first timestamp in dataset
TRAJECTORY_END     = "2008-08-14"   # one year window
TRAJECTORY_STEP    = 3              # keep every Nth point (downsample)

# Douglas-Peucker tolerance for OSM geometry simplification (degrees).
# ~0.0003 deg ≈ 33 m — keeps shape fidelity while cutting vertices heavily.
SIMPLIFY_TOLERANCE = 0.0003

KML_NS          = "http://www.opengis.net/kml/2.2"
GX_NS           = "http://www.google.com/kml/ext/2.2"
LAT_M           = 111320.0          # metres per degree latitude

OVERPASS_URL    = "https://overpass-api.de/api/interpreter"

# Bounding box derived from CSV data with buffer
# Actual data: lat -25.377 to -23.979, lon 31.063 to 32.004
BBOX = dict(min_lat=-25.45, min_lon=31.00, max_lat=-23.90, max_lon=32.07)

# KML colors (AABBGGRR format — KML byte order)
COL = {
    "water_fill"    : "99ff6600",   # semi-transparent blue
    "water_border"  : "ffff6600",
    "crop_fill"     : "6600d4ff",   # semi-transparent yellow
    "crop_border"   : "ff00aaff",
    "settle_fill"   : "660055ff",   # semi-transparent orange-red
    "settle_border" : "ff0055ff",
    "rail_line"     : "ff222222",   # near-black
    "road_primary"  : "ff0000aa",   # dark red
    "road_secondary": "ff0066cc",   # orange
    "road_tertiary" : "ff505050",   # dark grey
    "road_track"    : "ff808080",   # light grey
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def lon_m_per_deg(lat_deg):
    return LAT_M * math.cos(math.radians(lat_deg))


def _overpass_query(query_body, max_retries=2):
    """Send a query to Overpass API and return parsed JSON."""
    payload = urllib.parse.urlencode({"data": query_body}).encode()

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                OVERPASS_URL,
                data=payload,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "ElephantTrajectoryResearch/1.0 (academic)",
                },
                method="POST",
            )
            label = f"(attempt {attempt+1})" if attempt > 0 else ""
            print(f"    Querying Overpass API {label}...", end=" ", flush=True)
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            n = len(data.get("elements", []))
            print(f"OK ({n} elements)")
            return data
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries:
                wait = 10 * (attempt + 1)
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    All retries failed.")
                return {"elements": []}


def _geom_elements_to_features(data):
    """
    Convert Overpass `out geom;` response into
    list of {coords: [(lon, lat)], name: str, tags: dict}.

    `out geom` embeds geometry directly in each way element.
    """
    results = []
    for el in data.get("elements", []):
        if el["type"] == "way" and "geometry" in el:
            tags = el.get("tags", {})
            coords = [(node["lon"], node["lat"]) for node in el["geometry"]]
            if len(coords) < 2:
                continue
            name = tags.get("name", "") or tags.get("ref", "") or ""
            results.append({"coords": coords, "tags": tags, "name": name})
    return results


# ---------------------------------------------------------------------------
# 1. PARSE ELEPHANT KML
# ---------------------------------------------------------------------------

def parse_elephant_kml(kml_file):
    """Parse elephant trajectory KML, return list of placemarks with coords."""
    tree = ET.parse(kml_file)
    root = tree.getroot()
    ns = {"k": KML_NS, "gx": GX_NS}

    placemarks = []
    for pm in root.findall(".//{%s}Placemark" % KML_NS):
        name_el = pm.find("{%s}name" % KML_NS)
        name = name_el.text if name_el is not None and name_el.text else "Unnamed"

        # Collect all coordinate strings from LineString, Point, etc.
        all_coords = []
        for coords_el in pm.findall(".//{%s}coordinates" % KML_NS):
            if coords_el.text:
                for tok in coords_el.text.strip().split():
                    parts = tok.split(",")
                    if len(parts) >= 2:
                        try:
                            all_coords.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass

        # Get style info
        style_el = pm.find("{%s}styleUrl" % KML_NS)
        style_url = style_el.text if style_el is not None else ""

        if all_coords:
            placemarks.append({
                "name": name,
                "coords": all_coords,
                "style_url": style_url,
            })

    return placemarks


# ---------------------------------------------------------------------------
# 1b. LOAD TRAJECTORIES FROM CSV (filtered by year + subset of elephants)
# ---------------------------------------------------------------------------

def load_trajectories_from_csv(csv_file, selected_ids, start_date, end_date, step=1):
    """
    Read elephant GPS data from CSV, filter to date range and optionally
    selected IDs (None = all elephants). Downsample by keeping every
    `step`-th point.
    Returns list of {name, coords: [(lon, lat)], count_raw, count_kept}.
    """
    from collections import OrderedDict
    tracks = OrderedDict()
    if selected_ids is not None:
        for eid in selected_ids:
            tracks[eid] = {"raw": [], "name": eid}

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["individual-local-identifier"]
            ts = row.get("timestamp", "")
            # Filter by date range (string comparison works for ISO format)
            if ts < start_date or ts > end_date:
                continue
            if selected_ids is not None and eid not in tracks:
                continue
            if eid not in tracks:
                tracks[eid] = {"raw": [], "name": eid}
            try:
                lon = float(row["location-long"])
                lat = float(row["location-lat"])
                tracks[eid]["raw"].append((lon, lat))
            except (ValueError, KeyError):
                pass

    result = []
    for eid, data in tracks.items():
        raw = data["raw"]
        if not raw:
            continue
        # Downsample
        kept = raw[::step]
        # Always include the last point
        if kept[-1] != raw[-1]:
            kept.append(raw[-1])
        result.append({
            "name": eid,
            "coords": kept,
            "count_raw": len(raw),
            "count_kept": len(kept),
        })
    return result


# ---------------------------------------------------------------------------
# 1c. GEOMETRY SIMPLIFICATION
# ---------------------------------------------------------------------------

def simplify_coords(coords, tolerance=SIMPLIFY_TOLERANCE):
    """
    Simplify a coordinate list using Douglas-Peucker via Shapely.
    Works for both open lines and closed polygons.
    Returns simplified list of (lon, lat) tuples.
    """
    if len(coords) < 3:
        return coords

    # Check if closed polygon
    is_closed = (coords[0] == coords[-1]) or (
        len(coords) >= 4 and
        math.sqrt((coords[0][0]-coords[-1][0])**2 +
                  (coords[0][1]-coords[-1][1])**2) < 0.0001
    )

    try:
        if is_closed and len(coords) >= 4:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            simplified = poly.simplify(tolerance, preserve_topology=True)
            if simplified.is_empty:
                return coords
            result = list(simplified.exterior.coords)
        else:
            line = LineString(coords)
            simplified = line.simplify(tolerance, preserve_topology=True)
            if simplified.is_empty:
                return coords
            result = list(simplified.coords)
        return result if len(result) >= 2 else coords
    except Exception:
        return coords


# ---------------------------------------------------------------------------
# 2. FETCH OSM FEATURES
# ---------------------------------------------------------------------------

def _split_bbox_halves(bbox):
    """Split a bbox into 2 halves along latitude."""
    mid_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
    return [
        dict(min_lat=bbox["min_lat"], min_lon=bbox["min_lon"],
             max_lat=mid_lat,         max_lon=bbox["max_lon"]),
        dict(min_lat=mid_lat,         min_lon=bbox["min_lon"],
             max_lat=bbox["max_lat"], max_lon=bbox["max_lon"]),
    ]


def _bbox_str(bbox):
    return f"{bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']}"


def _fetch_geom(label, query_template, bbox, split_if_fail=True):
    """
    Fetch features using `out geom;` for a given bbox.
    If it times out and split_if_fail is True, split the bbox in half and retry.
    """
    b = _bbox_str(bbox)
    query = query_template.replace("{BBOX}", b)
    data = _overpass_query(query)
    elements = data.get("elements", [])

    # If we got 0 elements and query didn't explicitly fail, that's fine
    # If we got a timeout (empty due to retry exhaustion) and can split, do so
    if len(elements) == 0 and split_if_fail:
        # Check if we should try splitting
        lat_span = bbox["max_lat"] - bbox["min_lat"]
        if lat_span > 0.3:  # only split if region is still sizable
            print(f"    Splitting {label} bbox into halves for retry...")
            halves = _split_bbox_halves(bbox)
            all_elements = []
            for i, half in enumerate(halves):
                time.sleep(5)
                sub_data = _fetch_geom(f"{label} (half {i+1})", query_template,
                                        half, split_if_fail=False)
                all_elements.extend(sub_data.get("elements", []))
            return {"elements": all_elements}

    return data


def fetch_all_osm(bbox):
    """Fetch water, crops, settlements, roads, railways from OSM Overpass API."""
    result = {"water": [], "crops": [], "settlements": [],
              "roads": [], "railways": [], "settlement_points": []}

    # --- Water bodies (split into 2 queries to avoid timeout) ---
    print("  [Water bodies]")
    water_template = """
[out:json][timeout:300];
(
  way["natural"="water"]({BBOX});
  way["waterway"="river"]({BBOX});
  way["waterway"="stream"]({BBOX});
  way["waterway"="canal"]({BBOX});
  way["landuse"="reservoir"]({BBOX});
);
out geom;
"""
    for i, half_bbox in enumerate(_split_bbox_halves(bbox)):
        print(f"    Sub-region {i+1}/2:")
        data = _fetch_geom("water", water_template, half_bbox)
        features = _geom_elements_to_features(data)
        for w in features:
            wtype = (w["tags"].get("natural", "") or
                     w["tags"].get("waterway", "") or
                     w["tags"].get("landuse", "") or "water")
            result["water"].append({
                "coords": w["coords"], "name": w["name"], "type": wtype,
            })
        time.sleep(5)
    print(f"    Total water features: {len(result['water'])}")

    # --- Crop fields / farmland ---
    print("  [Crop fields]")
    crop_template = """
[out:json][timeout:300];
(
  way["landuse"="farmland"]({BBOX});
  way["landuse"="orchard"]({BBOX});
  way["landuse"="vineyard"]({BBOX});
);
out geom;
"""
    for i, half_bbox in enumerate(_split_bbox_halves(bbox)):
        print(f"    Sub-region {i+1}/2:")
        data = _fetch_geom("crops", crop_template, half_bbox)
        features = _geom_elements_to_features(data)
        for w in features:
            ctype = w["tags"].get("landuse", "farmland")
            result["crops"].append({
                "coords": w["coords"], "name": w["name"], "type": ctype,
            })
        time.sleep(5)
    print(f"    Total crop features: {len(result['crops'])}")

    # --- Settlements (nodes for points + ways for residential areas) ---
    print("  [Settlements]")
    settle_template = """
[out:json][timeout:300];
(
  node["place"~"city|town|village|hamlet|suburb"]({BBOX});
  way["landuse"="residential"]({BBOX});
);
out geom;
"""
    for i, half_bbox in enumerate(_split_bbox_halves(bbox)):
        print(f"    Sub-region {i+1}/2:")
        data = _fetch_geom("settlements", settle_template, half_bbox)

        for el in data.get("elements", []):
            if el["type"] == "node" and "tags" in el:
                tags = el.get("tags", {})
                if tags.get("place"):
                    result["settlement_points"].append({
                        "lon": el["lon"], "lat": el["lat"],
                        "name": tags.get("name", ""),
                        "place_type": tags.get("place", ""),
                    })

        features = _geom_elements_to_features(data)
        for w in features:
            result["settlements"].append({
                "coords": w["coords"], "name": w["name"], "type": "residential",
            })
        time.sleep(5)
    print(f"    Settlement points: {len(result['settlement_points'])}, "
          f"residential areas: {len(result['settlements'])}")

    # --- Roads (only major roads to keep manageable) ---
    print("  [Roads]")
    road_template = """
[out:json][timeout:300];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary"]({BBOX});
);
out geom;
"""
    for i, half_bbox in enumerate(_split_bbox_halves(bbox)):
        print(f"    Sub-region {i+1}/2:")
        data = _fetch_geom("roads", road_template, half_bbox)
        features = _geom_elements_to_features(data)
        for w in features:
            rtype = w["tags"].get("highway", "road")
            result["roads"].append({
                "coords": w["coords"], "name": w["name"], "type": rtype,
            })
        time.sleep(5)
    print(f"    Total road features: {len(result['roads'])}")

    # --- Railways ---
    print("  [Railways]")
    rail_template = """
[out:json][timeout:300];
(
  way["railway"~"rail|narrow_gauge"]({BBOX});
);
out geom;
"""
    data = _fetch_geom("railways", rail_template, bbox)
    features = _geom_elements_to_features(data)
    for w in features:
        rtype = w["tags"].get("railway", "rail")
        result["railways"].append({
            "coords": w["coords"], "name": w["name"], "type": rtype,
        })
    print(f"    Total railway features: {len(result['railways'])}")

    return result


def load_or_fetch_osm(bbox):
    """Load cached OSM data or fetch fresh."""
    if os.path.exists(OSM_CACHE):
        print(f"  Using cached {OSM_CACHE}")
        with open(OSM_CACHE) as f:
            data = json.load(f)
        # Verify it has actual data
        total = sum(len(v) for k, v in data.items() if isinstance(v, list))
        if total > 0:
            return data
        print("  Cache is empty, re-fetching...")

    data = fetch_all_osm(bbox)

    with open(OSM_CACHE, "w") as f:
        json.dump(data, f)
    print(f"  Cached OSM data -> {OSM_CACHE}")
    return data


# ---------------------------------------------------------------------------
# 3. KML BUILDING HELPERS
# ---------------------------------------------------------------------------

def _kml_coords(coords, alt=0):
    """List of (lon, lat) -> KML coordinates text."""
    return "\n          ".join(
        f"{lon:.7f},{lat:.7f},{alt}" for lon, lat in coords
    )


def _add_style(parent, sid, fill_color, line_color, line_width=1, fill=1):
    s = ET.SubElement(parent, "Style", id=sid)
    ps = ET.SubElement(s, "PolyStyle")
    ET.SubElement(ps, "color").text = fill_color
    ET.SubElement(ps, "fill").text = str(fill)
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


def _add_icon_style(parent, sid, icon_color, scale=0.8):
    s = ET.SubElement(parent, "Style", id=sid)
    ics = ET.SubElement(s, "IconStyle")
    ET.SubElement(ics, "color").text = icon_color
    ET.SubElement(ics, "scale").text = str(scale)
    icon = ET.SubElement(ics, "Icon")
    ET.SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png"
    ls = ET.SubElement(s, "LabelStyle")
    ET.SubElement(ls, "scale").text = "0.7"
    return s


def _placemark_polygon(folder, name, description, style_url, coords):
    """Add a Polygon Placemark from a list of (lon, lat) coords."""
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    ET.SubElement(pm, "description").text = description
    ET.SubElement(pm, "styleUrl").text = style_url
    pg = ET.SubElement(pm, "Polygon")
    ET.SubElement(pg, "tessellate").text = "1"
    ob = ET.SubElement(pg, "outerBoundaryIs")
    lr = ET.SubElement(ob, "LinearRing")
    ET.SubElement(lr, "coordinates").text = "\n          " + _kml_coords(coords)


def _placemark_line(folder, name, description, style_url, coords):
    """Add a LineString Placemark from a list of (lon, lat) coords."""
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    ET.SubElement(pm, "description").text = description
    ET.SubElement(pm, "styleUrl").text = style_url
    ls = ET.SubElement(pm, "LineString")
    ET.SubElement(ls, "tessellate").text = "1"
    ET.SubElement(ls, "coordinates").text = "\n          " + _kml_coords(coords)


def _placemark_point(folder, name, description, style_url, lon, lat):
    """Add a Point Placemark."""
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    ET.SubElement(pm, "description").text = description
    ET.SubElement(pm, "styleUrl").text = style_url
    pt = ET.SubElement(pm, "Point")
    ET.SubElement(pt, "coordinates").text = f"{lon:.7f},{lat:.7f},0"


# ---------------------------------------------------------------------------
# 4. BUILD KML
# ---------------------------------------------------------------------------

# Elephant trajectory colors (distinct, visible on satellite imagery)
ELEPHANT_COLORS = [
    "ff00ff00",  # green
    "ff0000ff",  # red
    "ffffff00",  # cyan
    "ffff00ff",  # magenta
    "ff00ffff",  # yellow
    "ffff6600",  # blue-orange
    "ff6600ff",  # purple
    "ff00cc66",  # teal-green
    "ff3399ff",  # orange
    "ffcc33ff",  # pink
    "ff66ffcc",  # light green
    "ff9966ff",  # light purple
    "ffff9933",  # light blue
    "ff33cccc",  # sea green
]


def _emit_osm_features(folder, features, style_poly, style_line, label_default):
    """
    Add OSM features to a KML folder, applying Douglas-Peucker simplification.
    Returns (poly_count, line_count, total_vertices).
    """
    poly_count = line_count = total_verts = 0
    for feat in features:
        coords = [tuple(c) for c in feat["coords"]]
        name = feat["name"] or label_default
        desc = f"Type: {feat.get('type', '')}"

        # Simplify geometry
        coords = simplify_coords(coords)

        is_closed = (len(coords) >= 4 and coords[0] == coords[-1])
        if not is_closed and len(coords) >= 3:
            d = math.sqrt((coords[0][0]-coords[-1][0])**2 +
                          (coords[0][1]-coords[-1][1])**2)
            if d < 0.001:
                coords = list(coords) + [coords[0]]
                is_closed = True

        if is_closed and len(coords) >= 4:
            _placemark_polygon(folder, name, desc, style_poly, coords)
            poly_count += 1
        elif len(coords) >= 2:
            _placemark_line(folder, name, desc, style_line, coords)
            line_count += 1
        else:
            continue

        total_verts += len(coords)
    return poly_count, line_count, total_verts


def build_kml(trajectories, osm_data):
    """Build complete KML with elephant trajectories + OSM features."""

    ET.register_namespace("", KML_NS)
    ET.register_namespace("gx", GX_NS)
    kml = ET.Element("kml", xmlns=KML_NS)
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = "S. Africa Elephants + OSM Features"
    ET.SubElement(doc, "description").text = (
        f"Elephant GPS trajectories ({TRAJECTORY_START} to {TRAJECTORY_END}) from Kruger National Park "
        f"({len(trajectories)} selected elephants) overlaid with OSM features: "
        "water bodies, crop fields, settlements, roads, and railways. "
        "OSM geometries simplified with Douglas-Peucker."
    )

    total_vertices = 0

    # ── Styles ──────────────────────────────────────────────────────────────
    for idx, color in enumerate(ELEPHANT_COLORS):
        _add_line_style(doc, f"elephant{idx}", color, line_width=2)
    _add_style(doc, "waterPoly", COL["water_fill"], COL["water_border"], 2)
    _add_style(doc, "cropPoly", COL["crop_fill"], COL["crop_border"], 1)
    _add_style(doc, "settlePoly", COL["settle_fill"], COL["settle_border"], 2)
    _add_line_style(doc, "roadPrimary", COL["road_primary"], 3)
    _add_line_style(doc, "roadSecondary", COL["road_secondary"], 2)
    _add_line_style(doc, "roadTertiary", COL["road_tertiary"], 1)
    _add_line_style(doc, "roadTrack", COL["road_track"], 1)
    _add_line_style(doc, "railLine", COL["rail_line"], 3)
    _add_icon_style(doc, "settlePoint", "ff0055ff", 0.9)
    _add_line_style(doc, "waterLine", COL["water_border"], 2)

    # ── Elephant Trajectories ───────────────────────────────────────────────
    eleph_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(eleph_folder, "name").text = "Elephant Trajectories"
    ET.SubElement(eleph_folder, "description").text = (
        f"GPS tracking data ({TRAJECTORY_START} to {TRAJECTORY_END}) from {len(trajectories)} "
        "elephants in Kruger National Park, South Africa. "
        f"Downsampled to every {TRAJECTORY_STEP}rd point."
    )

    for idx, trk in enumerate(trajectories):
        style_id = f"#elephant{idx % len(ELEPHANT_COLORS)}"
        if len(trk["coords"]) >= 2:
            _placemark_line(
                eleph_folder, trk["name"],
                f"Elephant: {trk['name']}\n"
                f"Year: {TRAJECTORY_START} to {TRAJECTORY_END}\n"
                f"Raw GPS fixes: {trk['count_raw']}\n"
                f"Points shown: {trk['count_kept']}",
                style_id, trk["coords"],
            )
            total_vertices += trk["count_kept"]

    # ── Water Bodies ────────────────────────────────────────────────────────
    water_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(water_folder, "name").text = "Water Bodies"
    wp, wl, wv = _emit_osm_features(
        water_folder, osm_data.get("water", []),
        "#waterPoly", "#waterLine", "Water")
    total_vertices += wv

    # ── Crop Fields ─────────────────────────────────────────────────────────
    crop_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(crop_folder, "name").text = "Crop Fields / Farmland"
    cp, cl, cv = _emit_osm_features(
        crop_folder, osm_data.get("crops", []),
        "#cropPoly", "#cropPoly", "Farmland")
    total_vertices += cv

    # ── Settlements ─────────────────────────────────────────────────────────
    settle_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(settle_folder, "name").text = "Settlements"

    settle_pts = osm_data.get("settlement_points", [])
    for sp in settle_pts:
        label = sp["name"] or sp["place_type"].capitalize()
        desc = f"Place type: {sp['place_type']}"
        _placemark_point(settle_folder, label, desc,
                         "#settlePoint", sp["lon"], sp["lat"])
    total_vertices += len(settle_pts)

    sp2, sl2, sv = _emit_osm_features(
        settle_folder, osm_data.get("settlements", []),
        "#settlePoly", "#settlePoly", "Residential Area")
    total_vertices += sv

    # ── Roads ───────────────────────────────────────────────────────────────
    road_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(road_folder, "name").text = "Roads"

    road_style_map = {
        "motorway": "#roadPrimary", "trunk": "#roadPrimary",
        "primary": "#roadPrimary", "secondary": "#roadSecondary",
        "tertiary": "#roadTertiary", "unclassified": "#roadTertiary",
        "track": "#roadTrack",
    }

    road_count = road_verts = 0
    for feat in osm_data.get("roads", []):
        coords = simplify_coords([tuple(c) for c in feat["coords"]])
        name = feat["name"] or feat["type"].capitalize()
        desc = f"Highway type: {feat['type']}"
        style = road_style_map.get(feat["type"], "#roadTertiary")
        if len(coords) >= 2:
            _placemark_line(road_folder, name, desc, style, coords)
            road_count += 1
            road_verts += len(coords)
    total_vertices += road_verts

    # ── Railways ────────────────────────────────────────────────────────────
    rail_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(rail_folder, "name").text = "Railways"

    rail_count = rail_verts = 0
    for feat in osm_data.get("railways", []):
        coords = simplify_coords([tuple(c) for c in feat["coords"]])
        name = feat["name"] or "Railway"
        desc = f"Railway type: {feat['type']}"
        if len(coords) >= 2:
            _placemark_line(rail_folder, name, desc, "#railLine", coords)
            rail_count += 1
            rail_verts += len(coords)
    total_vertices += rail_verts

    # ── 5 km Grid Overlay ─────────────────────────────────────────────────
    grid_folder = ET.SubElement(doc, "Folder")
    ET.SubElement(grid_folder, "name").text = "5 km Training Grid"
    ET.SubElement(grid_folder, "description").text = (
        "5 km × 5 km grid used for GAN training feature extraction. "
        "Each cell is classified by dominant OSM feature."
    )
    ET.SubElement(grid_folder, "visibility").text = "0"  # hidden by default

    _add_style(doc, "gridCell", "22ffffff", "88ffffff", 1, 1)

    grid_cell_km = 5.0
    mid_lat_grid = (BBOX["min_lat"] + BBOX["max_lat"]) / 2.0
    grid_dlat = (grid_cell_km * 1000) / 111320.0
    grid_dlon = (grid_cell_km * 1000) / (111320.0 * math.cos(math.radians(mid_lat_grid)))

    grid_verts = 0
    lat = BBOX["min_lat"]
    row = 0
    while lat < BBOX["max_lat"]:
        lon = BBOX["min_lon"]
        col = 0
        while lon < BBOX["max_lon"]:
            lat1 = min(lat + grid_dlat, BBOX["max_lat"])
            lon1 = min(lon + grid_dlon, BBOX["max_lon"])
            cell_coords = [
                (lon, lat), (lon1, lat), (lon1, lat1), (lon, lat1), (lon, lat)
            ]
            _placemark_polygon(
                grid_folder, f"R{row}C{col}",
                f"Grid cell ({row}, {col})\n"
                f"Lat: {lat:.4f}-{lat1:.4f}\n"
                f"Lon: {lon:.4f}-{lon1:.4f}",
                "#gridCell", cell_coords,
            )
            grid_verts += 5
            col += 1
            lon += grid_dlon
        row += 1
        lat += grid_dlat

    total_vertices += grid_verts

    counts = {
        "elephants": len(trajectories),
        "elephant_vertices": sum(t["count_kept"] for t in trajectories),
        "water_polygons": wp, "water_lines": wl, "water_vertices": wv,
        "crop_features": cp + cl, "crop_vertices": cv,
        "settlement_points": len(settle_pts),
        "settlement_polygons": sp2, "settlement_vertices": sv,
        "roads": road_count, "road_vertices": road_verts,
        "railways": rail_count, "rail_vertices": rail_verts,
        "grid_cells": row * col, "grid_vertices": grid_verts,
        "TOTAL_VERTICES": total_vertices,
    }

    return kml, counts


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("S. Africa Elephants — OSM Feature Extraction + KML Generator")
    print("=" * 65)

    # 1. Load elephant trajectories from CSV (filtered)
    print(f"\n[1] Loading elephant trajectories from CSV")
    sel_label = ', '.join(SELECTED_ELEPHANTS) if SELECTED_ELEPHANTS else "all with data"
    print(f"    Selected elephants: {sel_label}")
    print(f"    Date range: {TRAJECTORY_START} to {TRAJECTORY_END}")
    print(f"    Downsample: every {TRAJECTORY_STEP}rd point")
    trajectories = load_trajectories_from_csv(
        ELEPHANT_CSV, SELECTED_ELEPHANTS, TRAJECTORY_START, TRAJECTORY_END, TRAJECTORY_STEP
    )
    for trk in trajectories:
        print(f"      {trk['name']}: {trk['count_raw']} raw -> "
              f"{trk['count_kept']} kept")
    total_eleph = sum(t["count_kept"] for t in trajectories)
    print(f"    Total elephant vertices: {total_eleph}")

    # 2. Fetch / load OSM features
    print(f"\n[2] Loading OSM features for Kruger NP region")
    print(f"    Bbox: {BBOX['min_lat']:.2f} to {BBOX['max_lat']:.2f} lat, "
          f"{BBOX['min_lon']:.2f} to {BBOX['max_lon']:.2f} lon")
    osm_data = load_or_fetch_osm(BBOX)

    print(f"\n    OSM Feature Summary (raw):")
    for key in ["water", "crops", "settlements", "roads", "railways"]:
        feats = osm_data.get(key, [])
        verts = sum(len(f["coords"]) for f in feats)
        print(f"      {key:20s}: {len(feats):>5d} features, {verts:>7d} vertices")
    sp_count = len(osm_data.get("settlement_points", []))
    print(f"      {'settlement_points':20s}: {sp_count:>5d}")

    print(f"\n    Douglas-Peucker tolerance: {SIMPLIFY_TOLERANCE} deg "
          f"(~{SIMPLIFY_TOLERANCE * 111320:.0f} m)")

    # 3. Build KML
    print(f"\n[3] Building KML...")
    kml_root, counts = build_kml(trajectories, osm_data)

    # 4. Write KML
    tree = ET.ElementTree(kml_root)
    ET.indent(tree, space="  ")
    with open(OUTPUT_KML, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    print(f"    Saved -> {OUTPUT_KML}")

    # Final summary
    print(f"\n{'=' * 65}")
    print("KML Vertex Budget:")
    for key, val in counts.items():
        marker = " <<<" if key == "TOTAL_VERTICES" else ""
        print(f"  {key:25s}: {val:>8,}{marker}")

    tv = counts["TOTAL_VERTICES"]
    if tv <= 250000:
        print(f"\n  PASS: {tv:,} vertices (under 250,000 limit)")
    else:
        print(f"\n  WARNING: {tv:,} vertices (OVER 250,000 limit!)")
        print(f"  Consider increasing SIMPLIFY_TOLERANCE or TRAJECTORY_STEP")

    print(f"\nOpen {OUTPUT_KML} in Google Earth.")
    print("=" * 65)


if __name__ == "__main__":
    main()
