#!/usr/bin/env python3
"""
compute_advanced_grid_features.py
==================================
Compute advanced grid cell features including:
  - Edge density and fragmentation metrics
  - Corridor centrality based on trajectory graph
  - Boundary proximity to forest/cropland interfaces
  - Crossing intensity near human features
  - Seasonal/scenario-based metrics if available

Output: grid_advanced_features_dataset.csv
"""

import csv
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(SCRIPT_DIR, "generate_elephant_trajectories")

KML_MAP = os.path.join(GEN_DIR, "FINAL WALAYAR MAP.kml")
TRAJECTORIES_KML = os.path.join(GEN_DIR, "generated_walayar_trajectories.kml")
OSM_FEATURES = os.path.join(SCRIPT_DIR, "osm_features.json")
OSM_ROADS = os.path.join(SCRIPT_DIR, "osm_roads.json")

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "grid_advanced_features_dataset.csv")

LAT_M = 111320.0
LON_M = 40075000.0 / 360.0

# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS (shared with main script)
# ---------------------------------------------------------------------------

def lat_lon_to_meters(lat: float, lon: float) -> Tuple[float, float]:
    """Convert lat/lon to approximate meters."""
    x = lon * LON_M * math.cos(math.radians(lat))
    y = lat * LAT_M
    return x, y

def polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute centroid of polygon."""
    n = len(polygon)
    lat = sum(p[0] for p in polygon) / n
    lon = sum(p[1] for p in polygon) / n
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

def polygon_perimeter(polygon: List[Tuple[float, float]]) -> float:
    """Compute perimeter of polygon in meters."""
    perimeter = 0.0
    polygon_m = [lat_lon_to_meters(lat, lon) for lat, lon in polygon]
    n = len(polygon_m)
    for i in range(n):
        x1, y1 = polygon_m[i]
        x2, y2 = polygon_m[(i + 1) % n]
        perimeter += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return perimeter

def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Compute area of polygon in square meters."""
    polygon_m = [lat_lon_to_meters(lat, lon) for lat, lon in polygon]
    n = len(polygon_m)
    area = 0.0
    for i in range(n):
        x1, y1 = polygon_m[i]
        x2, y2 = polygon_m[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def parse_kml_placemarks(kml_path: str) -> List[Dict]:
    """Parse KML file and extract Placemarks."""
    placemarks = []
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    for placemark in root.findall('.//kml:Placemark', ns):
        name = placemark.findtext('kml:name', '', ns) or 'Unnamed'
        
        # Polygon
        polygon_elem = placemark.find('kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
        if polygon_elem is not None and polygon_elem.text:
            coords_text = polygon_elem.text.strip()
            coords = []
            for coord in coords_text.split():
                parts = coord.split(',')
                if len(parts) >= 2:
                    coords.append((float(parts[1]), float(parts[0])))
            if coords:
                placemarks.append({'name': name, 'type': 'Polygon', 'coordinates': coords})
        
        # LineString
        line_elem = placemark.find('kml:LineString/kml:coordinates', ns)
        if line_elem is not None and line_elem.text:
            coords_text = line_elem.text.strip()
            coords = []
            for coord in coords_text.split():
                parts = coord.split(',')
                if len(parts) >= 2:
                    coords.append((float(parts[1]), float(parts[0])))
            if coords:
                placemarks.append({'name': name, 'type': 'LineString', 'coordinates': coords})
    
    return placemarks

def extract_grid_cells(placemarks: List[Dict]) -> List[Dict]:
    """Extract grid cells named R###C###."""
    grid_cells = []
    for pm in placemarks:
        if pm['type'] == 'Polygon' and pm['name'].startswith('R') and 'C' in pm['name']:
            try:
                parts = pm['name'].split('C')
                int(parts[0][1:])
                int(parts[1])
                grid_cells.append(pm)
            except (IndexError, ValueError):
                pass
    return grid_cells

def load_osm_features() -> Dict:
    """Load OSM features."""
    if os.path.exists(OSM_FEATURES):
        with open(OSM_FEATURES) as f:
            return json.load(f)
    return {'crops': [], 'settlements': [], 'water': [], 'forest': []}

def extract_trajectories(kml_path: str) -> List[List[Tuple[float, float]]]:
    """Extract trajectories from KML."""
    placemarks = parse_kml_placemarks(kml_path)
    trajectories = []
    for pm in placemarks:
        if pm['type'] == 'LineString':
            trajectories.append(pm['coordinates'])
    return trajectories

# ---------------------------------------------------------------------------
# ADVANCED METRICS
# ---------------------------------------------------------------------------

def compute_edge_density_and_fragmentation(cell: Dict, osm_features: Dict) -> Dict:
    """
    Compute edge density (perimeter-to-area ratio) and land fragmentation.
    Higher edge density indicates more fragmented landscape.
    """
    if cell['type'] != 'Polygon':
        return {}
    
    polygon = cell['coordinates']
    area = polygon_area(polygon)
    perimeter = polygon_perimeter(polygon)
    
    features = {}
    
    # Edge density = perimeter / area
    features['edge_density'] = (perimeter / area) if area > 0 else 0
    
    # Count distinct patches of each type within cell
    features['num_crop_patches'] = 0
    features['num_settlement_patches'] = 0
    features['num_water_patches'] = 0
    features['num_forest_patches'] = 0
    
    for crop_poly in osm_features.get('crops', []):
        if isinstance(crop_poly, dict):
            crop_poly = crop_poly.get('coordinates', [])
        for point in crop_poly:
            if point_in_polygon(point, polygon):
                features['num_crop_patches'] += 1
                break
    
    for settle_poly in osm_features.get('settlements', []):
        if isinstance(settle_poly, dict):
            settle_poly = settle_poly.get('coordinates', [])
        for point in settle_poly:
            if point_in_polygon(point, polygon):
                features['num_settlement_patches'] += 1
                break
    
    for water_poly in osm_features.get('water', []):
        if isinstance(water_poly, dict):
            water_poly = water_poly.get('coordinates', [])
        for point in water_poly:
            if point_in_polygon(point, polygon):
                features['num_water_patches'] += 1
                break
    
    for forest_poly in osm_features.get('forest', []):
        if isinstance(forest_poly, dict):
            forest_poly = forest_poly.get('coordinates', [])
        for point in forest_poly:
            if point_in_polygon(point, polygon):
                features['num_forest_patches'] += 1
                break
    
    return features

def compute_boundary_metrics(cell: Dict, osm_features: Dict) -> Dict:
    """
    Compute proportion of cell near boundaries between different land types.
    Forest-crop and forest-settlement boundaries are critical for movement.
    """
    if cell['type'] != 'Polygon':
        return {}
    
    polygon = cell['coordinates']
    area = polygon_area(polygon)
    
    features = {}
    
    # Simple heuristic: count vertices near boundaries
    boundary_distance = 500  # meters
    boundary_count = 0
    
    for vertex in polygon:
        # Check if near forest
        near_forest = any(
            distance_to_polygon_edge(vertex, p.get('coordinates', p) if isinstance(p, dict) else p) < boundary_distance
            for p in osm_features.get('forest', [])
        )
        
        # Check if near crop or settlement
        near_human = any(
            distance_to_polygon_edge(vertex, p.get('coordinates', p) if isinstance(p, dict) else p) < boundary_distance
            for p in osm_features.get('crops', []) + osm_features.get('settlements', [])
        )
        
        if near_forest and near_human:
            boundary_count += 1
    
    features['boundary_vertex_proportion'] = (boundary_count / len(polygon)) if len(polygon) > 0 else 0
    
    return features

def distance_to_polygon_edge(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> float:
    """Compute distance from point to polygon edge."""
    if len(polygon) < 2:
        return float('inf')
    
    lat_p, lon_p = point
    x_p, y_p = lat_lon_to_meters(lat_p, lon_p)
    min_dist = float('inf')
    
    for i in range(len(polygon)):
        lat1, lon1 = polygon[i]
        lat2, lon2 = polygon[(i + 1) % len(polygon)]
        x1, y1 = lat_lon_to_meters(lat1, lon1)
        x2, y2 = lat_lon_to_meters(lat2, lon2)
        
        dx = x2 - x1
        dy = y2 - y1
        t = max(0, min(1, ((x_p - x1) * dx + (y_p - y1) * dy) / (dx * dx + dy * dy + 1e-6)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        dist = math.sqrt((x_p - closest_x) ** 2 + (y_p - closest_y) ** 2)
        min_dist = min(min_dist, dist)
    
    return min_dist

def compute_corridor_centrality(cell: Dict, trajectories: List[List[Tuple[float, float]]], 
                                grid_cells: List[Dict]) -> Dict:
    """
    Compute centrality metrics based on trajectory patterns.
    Cells that are crossed frequently by trajectories have higher centrality.
    Simplified betweenness-like score.
    """
    if cell['type'] != 'Polygon':
        return {}
    
    polygon = cell['coordinates']
    cell_id = cell['name']
    
    features = {}
    
    # Build a simple movement graph: which cells do trajectories connect?
    cell_by_id = {c['name']: c for c in grid_cells}
    
    # Count crossing paths (trajectories that enter, pass through, and exit)
    crossing_count = 0
    pass_through_count = 0
    
    for trajectory in trajectories:
        entered = False
        exited = False
        passed_through = False
        
        for i, point in enumerate(trajectory):
            if point_in_polygon(point, polygon):
                if not entered:
                    entered = True
                passed_through = True
            else:
                if entered and not exited:
                    exited = True
        
        if entered and exited:
            crossing_count += 1
            pass_through_count += len([p for p in trajectory if point_in_polygon(p, polygon)])
    
    features['crossing_intensity'] = crossing_count
    features['pass_through_points'] = pass_through_count
    
    return features

def compute_crossing_intensity_near_features(cell: Dict, trajectories: List[List[Tuple[float, float]]],
                                            osm_features: Dict) -> Dict:
    """
    Compute trajectory visits near specific features.
    High values indicate frequent elephant use near human/natural features.
    """
    if cell['type'] != 'Polygon':
        return {}
    
    polygon = cell['coordinates']
    feature_distance = 200  # meters
    
    features = {}
    
    visits_near_crops = 0
    visits_near_settlements = 0
    visits_near_water = 0
    
    for trajectory in trajectories:
        for point in trajectory:
            if point_in_polygon(point, polygon):
                # Check proximity to features
                near_crop = any(
                    distance_to_polygon_edge(point, p.get('coordinates', p) if isinstance(p, dict) else p) < feature_distance
                    for p in osm_features.get('crops', [])
                )
                near_settlement = any(
                    distance_to_polygon_edge(point, p.get('coordinates', p) if isinstance(p, dict) else p) < feature_distance
                    for p in osm_features.get('settlements', [])
                )
                near_water_body = any(
                    distance_to_polygon_edge(point, p.get('coordinates', p) if isinstance(p, dict) else p) < feature_distance
                    for p in osm_features.get('water', [])
                )
                
                if near_crop:
                    visits_near_crops += 1
                if near_settlement:
                    visits_near_settlements += 1
                if near_water_body:
                    visits_near_water += 1
    
    features['visits_near_crops'] = visits_near_crops
    features['visits_near_settlements'] = visits_near_settlements
    features['visits_near_water'] = visits_near_water
    
    return features

def compute_first_passage_frequency(cell: Dict, trajectories: List[List[Tuple[float, float]]]) -> Dict:
    """
    Compute how often elephants first appear in this cell (entry from outside).
    First-passage events indicate cells that are initial destinations or turning points.
    """
    if cell['type'] != 'Polygon':
        return {}
    
    polygon = cell['coordinates']
    features = {}
    
    first_passage_count = 0
    
    # For each trajectory, count first entry into this cell
    for trajectory in trajectories:
        first_entry = False
        for i, point in enumerate(trajectory):
            if point_in_polygon(point, polygon):
                if not first_entry:
                    first_passage_count += 1
                    first_entry = True
                break
    
    features['first_passage_frequency'] = first_passage_count
    
    return features

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    
    map_placemarks = parse_kml_placemarks(KML_MAP)
    grid_cells = extract_grid_cells(map_placemarks)
    print(f"Found {len(grid_cells)} grid cells")
    
    trajectories = extract_trajectories(TRAJECTORIES_KML)
    print(f"Found {len(trajectories)} trajectories")
    
    osm_features = load_osm_features()
    
    print("Computing advanced features...")
    dataset = []
    
    for cell in grid_cells:
        row = {'cell_id': cell['name']}
        
        # Edge density and fragmentation
        row.update(compute_edge_density_and_fragmentation(cell, osm_features))
        
        # Boundary metrics
        row.update(compute_boundary_metrics(cell, osm_features))
        
        # Corridor centrality
        row.update(compute_corridor_centrality(cell, trajectories, grid_cells))
        
        # Crossing intensity near features
        row.update(compute_crossing_intensity_near_features(cell, trajectories, osm_features))
        
        # First passage frequency
        row.update(compute_first_passage_frequency(cell, trajectories))
        
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
