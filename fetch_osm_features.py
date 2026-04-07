#!/usr/bin/env python3
"""Fetch and cache OSM features for Walayar Range"""

import requests
import geopandas as gpd
from shapely.geometry import box
import json

print("="*70)
print("FETCHING OSM FEATURES FOR WALAYAR RANGE")
print("="*70)

# Walayar bounds
lat_min, lat_max = 10.7498, 10.9305
lon_min, lon_max = 76.6225, 76.8539

print(f"\n📍 Region: {lat_min}°N-{lat_max}°N, {lon_min}°E-{lon_max}°E")

# Overpass API query
overpass_url = "http://overpass-api.de/api/interpreter"

def fetch_osm_features(feature_type, osm_key, osm_value=None):
    """Fetch features from Overpass API"""
    
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    
    if osm_value:
        query = f"""
        [bbox:{bbox}];
        (
          way["{osm_key}"="{osm_value}"];
          relation["{osm_key}"="{osm_value}"];
        );
        out geom;
        """
    else:
        query = f"""
        [bbox:{bbox}];
        (
          way["{osm_key}"];
          relation["{osm_key}"];
        );
        out geom;
        """
    
    try:
        print(f"\n  Fetching {feature_type}...")
        response = requests.post(overpass_url, data=query, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            features = []
            
            for element in data.get('elements', []):
                if 'geometry' in element:
                    coords = [[node['lon'], node['lat']] for node in element['geometry']]
                    if len(coords) > 1:
                        features.append({
                            'type': feature_type,
                            'osm_id': element['id'],
                            'tags': element.get('tags', {}),
                            'geometry': coords
                        })
            
            print(f"    ✓ Found {len(features)} {feature_type} features")
            return features
        else:
            print(f"    ✗ Error: {response.status_code}")
            return []
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        return []

# Fetch different OSM feature types
print("\n🗺️  FETCHING OSM FEATURES:")

features_dict = {}

# Roads
road_features = fetch_osm_features("Roads", "highway", "primary")
road_features.extend(fetch_osm_features("Secondary Roads", "highway", "secondary"))
road_features.extend(fetch_osm_features("Tertiary Roads", "highway", "tertiary"))
features_dict['roads'] = road_features

# Water bodies
water_features = fetch_osm_features("Water Bodies", "natural", "water")
water_features.extend(fetch_osm_features("Rivers", "waterway", "river"))
features_dict['water'] = water_features

# Forests/Natural areas
forest_features = fetch_osm_features("Forests", "natural", "forest")
forest_features.extend(fetch_osm_features("Grassland", "natural", "grassland"))
features_dict['natural'] = forest_features

# Buildings/Settlements
building_features = fetch_osm_features("Buildings", "building")
features_dict['buildings'] = building_features

# Settlements/Towns
settlement_features = fetch_osm_features("Settlements", "place", "village")
settlement_features.extend(fetch_osm_features("Towns", "place", "town"))
features_dict['settlements'] = settlement_features

# Save to GeoJSON
print("\n💾 SAVING FEATURES:")

for feature_type, features in features_dict.items():
    if features:
        geom_features = []
        for feat in features:
            geom_features.append({
                "type": "Feature",
                "properties": {
                    "type": feat['type'],
                    "osm_id": feat['osm_id'],
                    "tags": feat['tags']
                },
                "geometry": {
                    "type": "LineString" if len(feat['geometry']) > 2 else "Point",
                    "coordinates": feat['geometry']
                }
            })
        
        geojson = {
            "type": "FeatureCollection",
            "features": geom_features
        }
        
        filename = f'/Users/brandonk28/milind/osm_{feature_type}.geojson'
        with open(filename, 'w') as f:
            json.dump(geojson, f)
        
        print(f"  ✓ Saved: osm_{feature_type}.geojson ({len(features)} features)")

print("\n" + "="*70)
print("✅ OSM FEATURES FETCHED AND CACHED")
print("="*70)
