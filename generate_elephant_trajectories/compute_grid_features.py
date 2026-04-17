#!/usr/bin/env python3
"""
compute_grid_features.py
========================
Compute per-grid-cell features from FINAL WALAYAR MAP.kml and trajectory data.

Generates a dataset with:
  - Land-cover/geometry features (crops, settlements, water, roads, railways)
  - Trajectory-based metrics (visit counts, entry counts, centrality, etc.)

Output: grid_features_dataset.csv
"""

import csv
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

KML_MAP = os.path.join(SCRIPT_DIR, "FINAL WALAYAR MAP.kml")
TRAJECTORIES_KML = os.path.join(SCRIPT_DIR, "generated_walayar_trajectories.kml")
OSM_FEATURES = os.path.join(ROOT_DIR, "osm_features.json")
OSM_ROADS = os.path.join(ROOT_DIR, "osm_roads.json")

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "grid_features_dataset.csv")

LAT_M = 111320.0  # metres per degree latitude
LON_M = 40075000.0 / 360.0  # metres per degree longitude at equator

# ---------------------------------------------------------------------------
# GEOMETRY UTILITIES
# ---------------------------------------------------------------------------

def lat_lon_to_meters(lat: float, lon: float) -> Tuple[float, float]:
    """Convert lat/lon to approximate meters (for local distance calculations)."""
    x = lon * LON_M * math.cos(math.radians(lat))
    y = lat * LAT_M
    return x, y

def meters_to_lat_lon(x: float, y: float, origin_lat: float) -> Tuple[float, float]:
    """Convert meters back to lat/lon."""
    lat = y / LAT_M
    lon = x / (LON_M * math.cos(math.radians(origin_lat)))
    return lat, lon

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    lat, lon = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > lon) != (yj > lon)) and (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Compute approximate area of polygon in square meters using Shoelace formula."""
    # Convert to meters first
    polygon_m = [lat_lon_to_meters(lat, lon) for lat, lon in polygon]
    n = len(polygon_m)
    area = 0.0
    for i in range(n):
        x1, y1 = polygon_m[i]
        x2, y2 = polygon_m[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def line_length(line: List[Tuple[float, float]]) -> float:
    """Compute total length of a line in meters."""
    total_length = 0.0
    for i in range(len(line) - 1):
        lat1, lon1 = line[i]
        lat2, lon2 = line[i + 1]
        x1, y1 = lat_lon_to_meters(lat1, lon1)
        x2, y2 = lat_lon_to_meters(lat2, lon2)
        total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total_length

def distance_point_to_line(point: Tuple[float, float], line: List[Tuple[float, float]]) -> float:
    """Compute minimum distance from point to line in meters."""
    if len(line) < 2:
        return float('inf')
    
    lat_p, lon_p = point
    x_p, y_p = lat_lon_to_meters(lat_p, lon_p)
    
    min_dist = float('inf')
    for i in range(len(line) - 1):
        lat1, lon1 = line[i]
        lat2, lon2 = line[i + 1]
        x1, y1 = lat_lon_to_meters(lat1, lon1)
        x2, y2 = lat_lon_to_meters(lat2, lon2)
        
        # Distance from point to line segment
        dx = x2 - x1
        dy = y2 - y1
        t = max(0, min(1, ((x_p - x1) * dx + (y_p - y1) * dy) / (dx * dx + dy * dy + 1e-6)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        dist = math.sqrt((x_p - closest_x) ** 2 + (y_p - closest_y) ** 2)
        min_dist = min(min_dist, dist)
    
    return min_dist

def distance_point_to_polygon_edge(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> float:
    """Compute distance from point to nearest edge of polygon."""
    if len(polygon) < 2:
        return float('inf')
    return distance_point_to_line(point, polygon + [polygon[0]])

def polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute centroid of polygon."""
    n = len(polygon)
    lat = sum(p[0] for p in polygon) / n
    lon = sum(p[1] for p in polygon) / n
    return lat, lon

# ---------------------------------------------------------------------------
# KML PARSING
# ---------------------------------------------------------------------------

def parse_kml_placemarks(kml_path: str) -> List[Dict]:
    """Parse KML file and extract Placemarks with coordinates."""
    placemarks = []
    tree = ET.parse(kml_path)
    root = tree.getroot()
    
    # Define namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    for placemark in root.findall('.//kml:Placemark', ns):
        name = placemark.findtext('kml:name', '', ns) or 'Unnamed'
        description = placemark.findtext('kml:description', '', ns) or ''
        
        # Try Polygon
        polygon_elem = placemark.find('kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
        if polygon_elem is not None and polygon_elem.text:
            coords_text = polygon_elem.text.strip()
            coords = []
            for coord in coords_text.split():
                parts = coord.split(',')
                if len(parts) >= 2:
                    coords.append((float(parts[1]), float(parts[0])))  # lat, lon
            if coords:
                placemarks.append({
                    'name': name,
                    'description': description,
                    'type': 'Polygon',
                    'coordinates': coords
                })
        
        # Try LineString
        line_elem = placemark.find('kml:LineString/kml:coordinates', ns)
        if line_elem is not None and line_elem.text:
            coords_text = line_elem.text.strip()
            coords = []
            for coord in coords_text.split():
                parts = coord.split(',')
                if len(parts) >= 2:
                    coords.append((float(parts[1]), float(parts[0])))  # lat, lon
            if coords:
                placemarks.append({
                    'name': name,
                    'description': description,
                    'type': 'LineString',
                    'coordinates': coords
                })
        
        # Try Point
        point_elem = placemark.find('kml:Point/kml:coordinates', ns)
        if point_elem is not None and point_elem.text:
            parts = point_elem.text.strip().split(',')
            if len(parts) >= 2:
                lat, lon = float(parts[1]), float(parts[0])
                placemarks.append({
                    'name': name,
                    'description': description,
                    'type': 'Point',
                    'coordinates': (lat, lon)
                })
    
    return placemarks

def extract_grid_cells(placemarks: List[Dict]) -> List[Dict]:
    """Extract rectangular grid cells from placemarks."""
    grid_cells = []
    for pm in placemarks:
        # Grid cells are named like R000C000, R000C001, etc.
        if pm['type'] == 'Polygon' and pm['name'].startswith('R') and 'C' in pm['name']:
            try:
                parts = pm['name'].split('C')
                int(parts[0][1:])  # Row number
                int(parts[1])      # Column number
                grid_cells.append(pm)
            except (IndexError, ValueError):
                pass
    return grid_cells

def extract_trajectories(kml_path: str) -> List[List[Tuple[float, float]]]:
    """Extract trajectory LineStrings from KML."""
    placemarks = parse_kml_placemarks(kml_path)
    trajectories = []
    for pm in placemarks:
        if pm['type'] == 'LineString':
            trajectories.append(pm['coordinates'])
    return trajectories

# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def load_osm_features() -> Dict:
    """Load OSM features (polygons for land cover types)."""
    if os.path.exists(OSM_FEATURES):
        with open(OSM_FEATURES) as f:
            return json.load(f)
    return {'crops': [], 'settlements': [], 'water': [], 'forest': []}

def load_osm_roads() -> Dict:
    """Load OSM roads (LineStrings for roads/railways)."""
    if os.path.exists(OSM_ROADS):
        with open(OSM_ROADS) as f:
            return json.load(f)
    return {'roads': [], 'railways': []}

def compute_land_cover_features(cell: Dict, osm_features: Dict, osm_roads: Dict) -> Dict:
    """Compute land-cover features for a grid cell."""
    if cell['type'] != 'Polygon':
        return {}
    
    polygon = cell['coordinates']
    cell_centroid = polygon_centroid(polygon)
    cell_area = polygon_area(polygon)
    
    features = {
        'cell_id': cell['name'],
        'centroid_lat': cell_centroid[0],
        'centroid_lon': cell_centroid[1],
        'area_m2': cell_area,
    }
    
    # Compute land-cover percentages
    crop_area = 0.0
    settlement_area = 0.0
    water_area = 0.0
    forest_area = 0.0
    
    for crop_poly in osm_features.get('crops', []):
        if isinstance(crop_poly, dict):
            crop_poly = crop_poly.get('coordinates', [])
        # Approximate: check if any vertex is in cell
        for point in crop_poly:
            if point_in_polygon(point, polygon):
                crop_area += polygon_area(crop_poly) if len(crop_poly) > 2 else 0
                break
    
    for settle_poly in osm_features.get('settlements', []):
        if isinstance(settle_poly, dict):
            settle_poly = settle_poly.get('coordinates', [])
        for point in settle_poly:
            if point_in_polygon(point, polygon):
                settlement_area += polygon_area(settle_poly) if len(settle_poly) > 2 else 0
                break
    
    for water_poly in osm_features.get('water', []):
        if isinstance(water_poly, dict):
            water_poly = water_poly.get('coordinates', [])
        for point in water_poly:
            if point_in_polygon(point, polygon):
                water_area += polygon_area(water_poly) if len(water_poly) > 2 else 0
                break
    
    for forest_poly in osm_features.get('forest', []):
        if isinstance(forest_poly, dict):
            forest_poly = forest_poly.get('coordinates', [])
        for point in forest_poly:
            if point_in_polygon(point, polygon):
                forest_area += polygon_area(forest_poly) if len(forest_poly) > 2 else 0
                break
    
    features['pct_crops'] = (crop_area / cell_area * 100) if cell_area > 0 else 0
    features['pct_settlements'] = (settlement_area / cell_area * 100) if cell_area > 0 else 0
    features['pct_water'] = (water_area / cell_area * 100) if cell_area > 0 else 0
    features['pct_forest'] = (forest_area / cell_area * 100) if cell_area > 0 else 0
    
    # Compute road/railway lengths
    total_road_length = 0.0
    total_rail_length = 0.0
    
    for road in osm_roads.get('roads', []):
        if isinstance(road, dict):
            road = road.get('coordinates', [])
        # Check if road intersects cell
        intersects = any(point_in_polygon(pt, polygon) for pt in road)
        if intersects:
            total_road_length += line_length(road)
    
    for rail in osm_roads.get('railways', []):
        if isinstance(rail, dict):
            rail = rail.get('coordinates', [])
        intersects = any(point_in_polygon(pt, polygon) for pt in rail)
        if intersects:
            total_rail_length += line_length(rail)
    
    features['road_length_m'] = total_road_length
    features['rail_length_m'] = total_rail_length
    
    # Distances to nearest features
    road_data = osm_roads.get('roads', [])
    first_road = road_data[0] if road_data else []
    if isinstance(first_road, dict):
        first_road = first_road.get('coordinates', [])
    features['dist_to_road_m'] = distance_point_to_line(cell_centroid, first_road) if first_road else float('inf')
    
    water_data = osm_features.get('water', [])
    first_water = water_data[0] if water_data else []
    if isinstance(first_water, dict):
        first_water = first_water.get('coordinates', [])
    features['dist_to_water_m'] = distance_point_to_polygon_edge(cell_centroid, first_water) if first_water else float('inf')
    
    settle_data = osm_features.get('settlements', [])
    first_settle = settle_data[0] if settle_data else []
    if isinstance(first_settle, dict):
        first_settle = first_settle.get('coordinates', [])
    features['dist_to_settlement_m'] = distance_point_to_polygon_edge(cell_centroid, first_settle) if first_settle else float('inf')
    
    return features

def compute_trajectory_features(cell: Dict, trajectories: List[List[Tuple[float, float]]]) -> Dict:
    """Compute trajectory-based features for a grid cell."""
    if cell['type'] != 'Polygon':
        return {}
    
    polygon = cell['coordinates']
    cell_centroid = polygon_centroid(polygon)
    
    features = {}
    
    visit_count = 0
    unique_trajectories_visiting = set()
    entry_count = 0
    corridor_cell_count = 0  # Simplified centrality measure
    
    for traj_idx, trajectory in enumerate(trajectories):
        points_in_cell = []
        previous_outside = True
        
        for point in trajectory:
            in_cell = point_in_polygon(point, polygon)
            if in_cell:
                points_in_cell.append(point)
                visit_count += 1
                if previous_outside:
                    entry_count += 1
                previous_outside = False
            else:
                previous_outside = True
        
        if len(points_in_cell) > 0:
            unique_trajectories_visiting.add(traj_idx)
    
    features['visit_count'] = visit_count
    features['unique_trajectory_count'] = len(unique_trajectories_visiting)
    features['entry_count'] = entry_count
    features['avg_points_per_visit'] = (visit_count / len(unique_trajectories_visiting)) if unique_trajectories_visiting else 0
    
    return features

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    
    # Parse map KML
    print(f"Parsing {KML_MAP}...")
    map_placemarks = parse_kml_placemarks(KML_MAP)
    grid_cells = extract_grid_cells(map_placemarks)
    print(f"Found {len(grid_cells)} grid cells")
    
    # Parse trajectory KML
    print(f"Parsing {TRAJECTORIES_KML}...")
    trajectories = extract_trajectories(TRAJECTORIES_KML)
    print(f"Found {len(trajectories)} trajectories")
    
    # Load OSM data
    print("Loading OSM features...")
    osm_features = load_osm_features()
    osm_roads = load_osm_roads()
    
    # Compute features for each cell
    print("Computing features...")
    dataset = []
    
    for cell in grid_cells:
        row = {}
        
        # Land-cover features
        lc_features = compute_land_cover_features(cell, osm_features, osm_roads)
        row.update(lc_features)
        
        # Trajectory features
        traj_features = compute_trajectory_features(cell, trajectories)
        row.update(traj_features)
        
        dataset.append(row)
    
    # Write to CSV
    print(f"Writing output to {OUTPUT_CSV}...")
    if dataset:
        fieldnames = sorted(dataset[0].keys())
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)
        print(f"✓ Wrote {len(dataset)} rows to {OUTPUT_CSV}")
    else:
        print("✗ No data to write")

if __name__ == '__main__':
    main()
