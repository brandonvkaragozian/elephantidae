#!/usr/bin/env python3
"""
extract_kml_features.py
=======================
Extract water bodies, crops, settlements, roads, and railways from KML
and save them as structured data for feature computation.
"""

import xml.etree.ElementTree as ET
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KML_PATH = os.path.join(SCRIPT_DIR, "FINAL WALAYAR MAP.kml")

OUTPUT_WATER = os.path.join(SCRIPT_DIR, "kml_water_bodies.json")
OUTPUT_CROPS = os.path.join(SCRIPT_DIR, "kml_crops.json")
OUTPUT_SETTLEMENTS = os.path.join(SCRIPT_DIR, "kml_settlements.json")
OUTPUT_ROADS = os.path.join(SCRIPT_DIR, "kml_roads.json")
OUTPUT_RAILWAYS = os.path.join(SCRIPT_DIR, "kml_railways.json")
OUTPUT_FOREST = os.path.join(SCRIPT_DIR, "kml_forest.json")

def parse_kml_coordinates(coords_text):
    """Parse KML coordinates string into list of (lat, lon) tuples."""
    coords = []
    for coord in coords_text.strip().split():
        parts = coord.split(',')
        if len(parts) >= 2:
            coords.append((float(parts[1]), float(parts[0])))  # lat, lon
    return coords

def extract_features_by_type(kml_path, feature_keywords, geometry_type='Polygon'):
    """Extract placemarks matching keywords and geometry type."""
    features = []
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    for placemark in root.findall('.//kml:Placemark', ns):
        name = placemark.findtext('kml:name', '', ns)
        
        # Check if name matches keywords
        matches = any(keyword.lower() in name.lower() for keyword in feature_keywords)
        if not matches:
            continue
        
        # Extract geometry based on type
        if geometry_type == 'Polygon':
            poly_elem = placemark.find('kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
            if poly_elem is not None and poly_elem.text:
                coords = parse_kml_coordinates(poly_elem.text)
                if coords:
                    features.append({
                        'name': name,
                        'type': 'Polygon',
                        'coordinates': coords
                    })
        
        elif geometry_type == 'LineString':
            line_elem = placemark.find('kml:LineString/kml:coordinates', ns)
            if line_elem is not None and line_elem.text:
                coords = parse_kml_coordinates(line_elem.text)
                if coords:
                    features.append({
                        'name': name,
                        'type': 'LineString',
                        'coordinates': coords
                    })
        
        elif geometry_type == 'Point':
            point_elem = placemark.find('kml:Point/kml:coordinates', ns)
            if point_elem is not None and point_elem.text:
                parts = point_elem.text.strip().split(',')
                if len(parts) >= 2:
                    lat, lon = float(parts[1]), float(parts[0])
                    features.append({
                        'name': name,
                        'type': 'Point',
                        'coordinates': (lat, lon)
                    })
    
    return features

def main():
    print("Extracting features from FINAL WALAYAR MAP.kml...\n")
    
    # Extract water bodies (polygons only - skip points for now)
    water = extract_features_by_type(KML_PATH, ['water', 'lake', 'river', 'kulam', 'tank'], 'Polygon')
    water = [feat['coordinates'] for feat in water]
    with open(OUTPUT_WATER, 'w') as f:
        json.dump(water, f)
    print(f"✓ Extracted {len(water)} water bodies (polygons) → {OUTPUT_WATER}")
    
    # Extract crop fields (polygons)
    crops = extract_features_by_type(KML_PATH, ['crop', 'field', 'paddy'], 'Polygon')
    crops = [feat['coordinates'] for feat in crops]
    with open(OUTPUT_CROPS, 'w') as f:
        json.dump(crops, f)
    print(f"✓ Extracted {len(crops)} crop fields → {OUTPUT_CROPS}")
    
    # Extract settlements (polygons)
    settlements = extract_features_by_type(KML_PATH, ['settlement', 'town', 'village', 'city'], 'Polygon')
    settlements = [feat['coordinates'] for feat in settlements]
    with open(OUTPUT_SETTLEMENTS, 'w') as f:
        json.dump(settlements, f)
    print(f"✓ Extracted {len(settlements)} settlements → {OUTPUT_SETTLEMENTS}")
    
    # Extract forest sections (polygons)
    forest = extract_features_by_type(KML_PATH, ['forest', 'section', 'reserve', 'wildlife', 'walayar section'], 'Polygon')
    forest = [feat['coordinates'] for feat in forest]
    with open(OUTPUT_FOREST, 'w') as f:
        json.dump(forest, f)
    print(f"✓ Extracted {len(forest)} forest sections → {OUTPUT_FOREST}")
    
    # Extract roads (linestrings)
    roads = extract_features_by_type(KML_PATH, ['road', 'nh', 'highway', 'service', 'street', 'bypass'], 'LineString')
    roads = [feat['coordinates'] for feat in roads]
    with open(OUTPUT_ROADS, 'w') as f:
        json.dump(roads, f)
    print(f"✓ Extracted {len(roads)} roads → {OUTPUT_ROADS}")
    
    # Extract railways (linestrings)
    railways = extract_features_by_type(KML_PATH, ['railway', 'rail', 'train'], 'LineString')
    railways = [feat['coordinates'] for feat in railways]
    with open(OUTPUT_RAILWAYS, 'w') as f:
        json.dump(railways, f)
    print(f"✓ Extracted {len(railways)} railways → {OUTPUT_RAILWAYS}")
    
    print(f"\n=== Summary ===")
    print(f"Water bodies: {len(water)}")
    print(f"Crop fields: {len(crops)}")
    print(f"Settlements: {len(settlements)}")
    print(f"Forest sections: {len(forest)}")
    print(f"Roads: {len(roads)}")
    print(f"Railways: {len(railways)}")

if __name__ == '__main__':
    main()
