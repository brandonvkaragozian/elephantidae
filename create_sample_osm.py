#!/usr/bin/env python3
"""Generate sample OSM features for Walayar Range (offline version)"""

import json
import folium
import numpy as np

print("="*70)
print("GENERATING SAMPLE OSM FEATURES FOR WALAYAR")
print("="*70)

# Walayar bounds
lat_min, lat_max = 10.7498, 10.9305
lon_min, lon_max = 76.6225, 76.8539

# Create sample roads
roads = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"type": "Roads", "highway": "primary"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[76.65, 10.75], [76.75, 10.85], [76.85, 10.90]]
            }
        },
        {
            "type": "Feature",
            "properties": {"type": "Secondary Roads", "highway": "secondary"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[76.70, 10.76], [76.78, 10.82], [76.80, 10.88]]
            }
        },
        {
            "type": "Feature",
            "properties": {"type": "Secondary Roads", "highway": "secondary"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[76.72, 10.78], [76.73, 10.85], [76.82, 10.92]]
            }
        }
    ]
}

# Create sample water bodies
water = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"type": "Water Bodies", "natural": "water"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[76.68, 10.80], [76.70, 10.82], [76.72, 10.81]]
            }
        },
        {
            "type": "Feature",
            "properties": {"type": "Rivers", "waterway": "river"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[76.75, 10.78], [76.76, 10.80], [76.78, 10.84]]
            }
        },
        {
            "type": "Feature",
            "properties": {"type": "Rivers", "waterway": "river"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[76.80, 10.75], [76.82, 10.80], [76.83, 10.87]]
            }
        }
    ]
}

# Create sample natural areas
natural = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"type": "Forests", "natural": "forest"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[76.63, 10.82], [76.65, 10.85], [76.67, 10.88], [76.65, 10.90]]
            }
        },
        {
            "type": "Feature",
            "properties": {"type": "Forests", "natural": "forest"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[76.78, 10.75], [76.80, 10.78], [76.83, 10.82], [76.85, 10.85]]
            }
        }
    ]
}

# Create sample settlements
settlements = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"type": "Settlements", "place": "village", "name": "Walayar Village"},
            "geometry": {
                "type": "Point",
                "coordinates": [76.74, 10.84]
            }
        },
        {
            "type": "Feature",
            "properties": {"type": "Settlements", "place": "village", "name": "Chittur"},
            "geometry": {
                "type": "Point",
                "coordinates": [76.68, 10.77]
            }
        },
        {
            "type": "Feature",
            "properties": {"type": "Settlements", "place": "village", "name": "Nemmara"},
            "geometry": {
                "type": "Point",
                "coordinates": [76.80, 10.88]
            }
        }
    ]
}

# Save all features
features = {
    'roads': roads,
    'water': water,
    'natural': natural,
    'settlements': settlements
}

for feature_type, geojson in features.items():
    filename = f'/Users/brandonk28/milind/osm_{feature_type}.geojson'
    with open(filename, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    num_features = len(geojson['features'])
    print(f"✓ Created: osm_{feature_type}.geojson ({num_features} features)")

print("\n" + "="*70)
print("✅ SAMPLE OSM FEATURES CREATED")
print("="*70)
