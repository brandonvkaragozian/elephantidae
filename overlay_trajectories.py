#!/usr/bin/env python3
"""Overlay Elephant Trajectories on Walayar Map with OSM Features"""

import xml.etree.ElementTree as ET
import folium
from folium import plugins
import numpy as np
import json
import os

print("="*70)
print("OVERLAYING ELEPHANT TRAJECTORIES")
print("="*70)

# Walayar center
center_lat = 10.84
center_lon = 76.73

# Parse KML for elephant trajectories
print("\n📂 Loading Elephant Trajectories KML...")

kml_files = [
    '/Users/brandonk28/milind/Walayar_with_Elephant_Trajectories.kml',
    '/Users/brandonk28/milind/elephantidae/Walayar_with_Elephant_Trajectories.kml'
]

trajectories = []
kml_found = False

for kml_file in kml_files:
    if os.path.exists(kml_file):
        kml_found = True
        print(f"✓ Found: {kml_file}")
        
        try:
            tree = ET.parse(kml_file)
            root = tree.getroot()
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            
            # Extract LineStrings (trajectories)
            for linestring in root.findall('.//kml:LineString', ns):
                coords_elem = linestring.find('kml:coordinates', ns)
                if coords_elem is not None and coords_elem.text:
                    coords_text = coords_elem.text.strip()
                    coords = []
                    for coord in coords_text.split():
                        parts = coord.split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                coords.append([lat, lon])
                            except ValueError:
                                continue
                    if coords:
                        trajectories.append(coords)
            
            # Extract Points (start/end/waypoints)
            points = []
            for point in root.findall('.//kml:Point', ns):
                coords_elem = point.find('kml:coordinates', ns)
                if coords_elem is not None and coords_elem.text:
                    parts = coords_elem.text.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            name_elem = point.getparent().find('kml:name', ns)
                            name = name_elem.text if name_elem is not None else "Point"
                            points.append({
                                'lat': lat,
                                'lon': lon,
                                'name': name
                            })
                        except ValueError:
                            continue
            
            print(f"✓ Loaded {len(trajectories)} trajectories with {len(points)} waypoints")
            break
        except Exception as e:
            print(f"✗ Error parsing KML: {e}")

if not kml_found:
    print("⚠️  KML file not found - creating demo trajectories...")
    # Create demo elephant trajectories
    trajectories = [
        [[10.78, 76.72], [10.80, 76.73], [10.82, 76.74], [10.84, 76.75]],
        [[10.88, 76.78], [10.86, 76.77], [10.84, 76.76], [10.82, 76.75]],
        [[10.80, 76.80], [10.82, 76.79], [10.84, 76.78], [10.86, 76.79]]
    ]
    points = [
        {'lat': 10.78, 'lon': 76.72, 'name': 'Start 1'},
        {'lat': 10.84, 'lon': 76.75, 'name': 'End 1'},
        {'lat': 10.88, 'lon': 76.78, 'name': 'Start 2'},
        {'lat': 10.82, 'lon': 76.75, 'name': 'End 2'},
    ]

# Create base map
print("\n🗺️  Creating interactive map...")
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles='OpenStreetMap'
)

# Add base layers
folium.TileLayer('OpenTopoMap', name='🏔️ Topographic').add_to(m)
folium.TileLayer('CartoDB positron', name='🗺️ Light Map').add_to(m)

# Elephant trajectories layer
traj_layer = folium.FeatureGroup(name='🐘 Elephant Trajectories', show=True)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for idx, trajectory in enumerate(trajectories):
    color = colors[idx % len(colors)]
    
    # Draw trajectory line
    folium.PolyLine(
        locations=trajectory,
        color=color,
        weight=3,
        opacity=0.8,
        popup=f"Trajectory {idx+1}",
        tooltip=f"Elephant Path {idx+1}"
    ).add_to(traj_layer)
    
    # Add arrow decorators
    for i in range(len(trajectory) - 1):
        start = trajectory[i]
        end = trajectory[i + 1]
        
        # Midpoint for arrow
        mid_lat = (start[0] + end[0]) / 2
        mid_lon = (start[1] + end[1]) / 2
        
        # Calculate bearing
        lat1, lon1 = np.radians(start[0]), np.radians(start[1])
        lat2, lon2 = np.radians(end[0]), np.radians(end[1])
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(y, x))
        
        folium.RegularPolygonMarker(
            location=[mid_lat, mid_lon],
            fill_color=color,
            number_of_sides=3,
            radius=6,
            rotation=bearing,
            popup=f"Direction: {bearing:.1f}°"
        ).add_to(traj_layer)

traj_layer.add_to(m)

# Waypoints layer
waypoint_layer = folium.FeatureGroup(name='📍 Waypoints', show=True)

for point in points:
    folium.CircleMarker(
        location=[point['lat'], point['lon']],
        radius=8,
        popup=f"<b>{point['name']}</b><br>Lat: {point['lat']:.4f}<br>Lon: {point['lon']:.4f}",
        color='#2C3E50',
        fill=True,
        fillColor='#E74C3C',
        fillOpacity=0.8,
        weight=2
    ).add_to(waypoint_layer)

waypoint_layer.add_to(m)

# Load and add OSM features if available
print("\n📍 Adding OSM Features...")

osm_files = {
    'roads': ('🛣️ Roads', '#FF4444', 2),
    'water': ('💧 Water', '#4444FF', 2),
    'natural': ('🌲 Natural Areas', '#44AA44', 1),
    'settlements': ('🏘️ Settlements', '#FFAA00', 2)
}

for osm_type, (display_name, color, weight) in osm_files.items():
    geojson_file = f'/Users/brandonk28/milind/osm_{osm_type}.geojson'
    
    if os.path.exists(geojson_file):
        try:
            osm_layer = folium.FeatureGroup(name=display_name, show=(osm_type in ['water', 'natural']))
            
            with open(geojson_file, 'r') as f:
                geojson_data = json.load(f)
            
            for feature in geojson_data['features']:
                coords = feature['geometry']['coordinates']
                
                if feature['geometry']['type'] == 'LineString':
                    folium.PolyLine(
                        locations=[[lat, lon] for lon, lat in coords],
                        color=color,
                        weight=weight,
                        opacity=0.6
                    ).add_to(osm_layer)
                elif feature['geometry']['type'] == 'Point':
                    folium.CircleMarker(
                        location=[coords[1], coords[0]],
                        radius=4,
                        color=color,
                        fill=True,
                        fillOpacity=0.7
                    ).add_to(osm_layer)
            
            osm_layer.add_to(m)
            print(f"  ✓ Added {display_name}")
        except Exception as e:
            print(f"  ✗ Error loading {osm_type}: {e}")
    else:
        print(f"  ⚠️  {osm_type} data not found (run fetch_osm_features.py first)")

# Statistics panel
stats_html = f"""
<div style="position: fixed; bottom: 50px; right: 50px; width: 300px; background-color: white; 
border:2px solid #2C3E50; z-index:9999; font-size:11px; padding: 12px; border-radius: 6px; 
font-weight: bold;">
    <div style="font-size:13px; color:#2C3E50; margin-bottom:8px;">🐘 ELEPHANT TRAJECTORY ANALYSIS</div>
    <hr style="margin: 5px 0;">
    <div style="margin:5px 0;"><b>Trajectories:</b> {len(trajectories)}</div>
    <div style="margin:5px 0;"><b>Total Waypoints:</b> {len(points)}</div>
    <div style="margin:5px 0;"><b>Region:</b> Walayar Range, Kerala</div>
    <hr style="margin: 5px 0;">
    <div style="font-size:10px; color:#555;">Use layer control to toggle overlays</div>
</div>
"""

m.get_root().html.add_child(folium.Element(stats_html))

# Layer control
folium.LayerControl(position='topleft').add_to(m)

# Save map
output_file = '/Users/brandonk28/milind/elephant_trajectories_map.html'
m.save(output_file)

print(f"\n✅ Map created: elephant_trajectories_map.html")
print(f"   Trajectories: {len(trajectories)}")
print(f"   Waypoints: {len(points)}")

print("\n" + "="*70)
print("✅ ELEPHANT TRAJECTORY VISUALIZATION COMPLETE")
print("="*70)
print("\n📥 View map at: localhost:8000/elephant_trajectories_map.html")
