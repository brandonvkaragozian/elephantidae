#!/usr/bin/env python3
"""Integrated Master Map: Elephant Trajectories + Crop Field Predictions"""

import folium
from folium import plugins
import geopandas as gpd
import json
import os

print("="*70)
print("CREATING INTEGRATED MASTER MAP")
print("="*70)

center_lat = 10.84
center_lon = 76.73

# Create base map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles='OpenStreetMap'
)

print("\n🗺️  Building map layers:")

# 1. Forest Sections
print("  • Forest sections...", end='')
try:
    import xml.etree.ElementTree as ET
    kml_file = '/Users/brandonk28/milind/Walayar_Range_clean.kml'
    with open(kml_file, 'r') as f:
        kml_content = f.read()
    
    root = ET.fromstring(kml_content)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    forest_layer = folium.FeatureGroup(name='🌲 Forest Sections (6)', show=True)
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    
    placemarks = root.findall('.//kml:Placemark', ns)
    for idx, placemark in enumerate(placemarks):
        coords_elem = placemark.find('.//kml:coordinates', ns)
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
                folium.Polygon(
                    locations=coords,
                    color=colors[idx % len(colors)],
                    fill=True,
                    fillColor=colors[idx % len(colors)],
                    fillOpacity=0.2,
                    weight=2
                ).add_to(forest_layer)
    
    forest_layer.add_to(m)
    print(" ✓")
except Exception as e:
    print(f" ✗ ({e})")

# 2. Water Bodies
print("  • Water bodies...", end='')
try:
    water_layer = folium.FeatureGroup(name='💧 Water Sources', show=True)
    water_locs = [[10.78, 76.75], [10.82, 76.78], [10.88, 76.80], [10.87, 76.72]]
    
    for coord in water_locs:
        folium.CircleMarker(
            location=coord,
            radius=8,
            popup=f"Water source",
            color='#0066cc',
            fill=True,
            fillColor='#00ccff',
            fillOpacity=0.7,
            weight=2
        ).add_to(water_layer)
    
    water_layer.add_to(m)
    print(" ✓")
except Exception as e:
    print(f" ✗ ({e})")

# 3. Predicted Crop Fields K=7
print("  • Predicted crop fields (K=7, 642 fields)...", end='')
try:
    gdf_k7 = gpd.read_file('/Users/brandonk28/milind/predicted_crop_fields_k7.geojson')
    crop_layer_k7 = folium.FeatureGroup(name='🌱 Predicted Crops (K=7)', show=False)
    
    zone_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#95a5a6']
    
    for idx, row in gdf_k7.iterrows():
        geom = row['geometry']
        cluster = row.get('cluster', 0)
        coords = [[lat, lon] for lon, lat in geom.exterior.coords]
        
        folium.Polygon(
            locations=coords,
            color=zone_colors[cluster % len(zone_colors)],
            fill=True,
            fillColor=zone_colors[cluster % len(zone_colors)],
            fillOpacity=0.3,
            weight=0.5,
            popup=f"Field {idx+1} | Zone {cluster+1} | Area: {row['area']:.0f} m²"
        ).add_to(crop_layer_k7)
    
    crop_layer_k7.add_to(m)
    print(" ✓")
except Exception as e:
    print(f" ✗ ({e})")

# 4. K=7 Zone Centers
print("  • Zone centers & labels (K=7)...", end='')
try:
    zone_layer = folium.FeatureGroup(name='🎯 Zone Centers (K=7)', show=True)
    gdf_k7 = gpd.read_file('/Users/brandonk28/milind/predicted_crop_fields_k7.geojson')
    zone_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#95a5a6']
    
    zone_centers = {}
    for idx, row in gdf_k7.iterrows():
        cluster = row.get('cluster', 0)
        if cluster not in zone_centers:
            zone_centers[cluster] = []
        geom = row['geometry']
        center_coords = geom.centroid
        zone_centers[cluster].append([center_coords.y, center_coords.x])
    
    for zone_id, coords_list in sorted(zone_centers.items()):
        if coords_list:
            mean_lat = sum([c[0] for c in coords_list]) / len(coords_list)
            mean_lon = sum([c[1] for c in coords_list]) / len(coords_list)
            
            folium.CircleMarker(
                location=[mean_lat, mean_lon],
                radius=12,
                popup=f"Zone {zone_id+1}",
                color=zone_colors[zone_id],
                fill=True,
                fillColor=zone_colors[zone_id],
                fillOpacity=0.9,
                weight=2
            ).add_to(zone_layer)
    
    zone_layer.add_to(m)
    print(" ✓")
except Exception as e:
    print(f" ✗ ({e})")

# 5. OSM Features
print("  • OSM features (roads, water, natural areas)...", end='')
try:
    osm_configs = {
        'roads': ('🛣️ Roads', '#FF4444', 2, False),
        'water': ('💧 Water Bodies (OSM)', '#4444FF', 2, True),
        'natural': ('🌳 Natural Areas (OSM)', '#44AA44', 1, True),
        'settlements': ('🏘️ Settlements', '#FFAA00', 2, False)
    }
    
    for osm_type, (display_name, color, weight, show) in osm_configs.items():
        geojson_file = f'/Users/brandonk28/milind/osm_{osm_type}.geojson'
        
        if os.path.exists(geojson_file):
            osm_layer = folium.FeatureGroup(name=display_name, show=show)
            
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
                    props = feature['properties']
                    name = props.get('name', 'Point')
                    folium.CircleMarker(
                        location=[coords[1], coords[0]],
                        radius=5,
                        color=color,
                        fill=True,
                        fillOpacity=0.7,
                        popup=name
                    ).add_to(osm_layer)
            
            osm_layer.add_to(m)
    
    print(" ✓")
except Exception as e:
    print(f" ✗ ({e})")

# 6. Elephant Trajectories
print("  • Elephant trajectories (demo)...", end='')
try:
    trajectories = [
        [[10.78, 76.72], [10.80, 76.73], [10.82, 76.74], [10.84, 76.75]],
        [[10.88, 76.78], [10.86, 76.77], [10.84, 76.76], [10.82, 76.75]],
        [[10.80, 76.80], [10.82, 76.79], [10.84, 76.78], [10.86, 76.79]]
    ]
    
    traj_layer = folium.FeatureGroup(name='🐘 Elephant Trajectories', show=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, traj in enumerate(trajectories):
        folium.PolyLine(
            locations=traj,
            color=colors[idx],
            weight=3,
            opacity=0.8,
            popup=f"Elephant Path {idx+1}"
        ).add_to(traj_layer)
        
        # Start point
        folium.CircleMarker(
            location=traj[0],
            radius=6,
            color=colors[idx],
            fill=True,
            fillColor=colors[idx],
            fillOpacity=0.9,
            popup=f"Path {idx+1} Start",
            weight=2
        ).add_to(traj_layer)
        
        # End point
        folium.CircleMarker(
            location=traj[-1],
            radius=6,
            color='#000',
            fill=True,
            fillColor=colors[idx],
            fillOpacity=0.9,
            popup=f"Path {idx+1} End",
            weight=2
        ).add_to(traj_layer)
    
    traj_layer.add_to(m)
    print(" ✓")
except Exception as e:
    print(f" ✗ ({e})")

# Add master legend
legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; width: 380px; background-color: rgba(255,255,255,0.95); 
border:3px solid #2c3e50; z-index:9999; font-size:11px; padding: 12px; border-radius: 6px; font-weight: bold;">
    <div style="font-size:14px; color:#2c3e50; margin-bottom:8px;">🗺️  WALAYAR INTEGRATED ANALYSIS</div>
    <hr style="margin: 5px 0; border-color:#bdc3c7;">
    <div style="margin:5px 0; color:#27ae60;">🌲 6 Forest Sections</div>
    <div style="margin:5px 0; color:#2980b9;">💧 Water Sources</div>
    <div style="margin:5px 0; color:#e74c3c;">🌱 642 Predicted Crop Fields (K=7)</div>
    <div style="margin:5px 0; color:#f39c12;">🎯 7 Agricultural Zones</div>
    <div style="margin:5px 0; color:#FF6B6B;">🐘 Elephant Movement Paths</div>
    <hr style="margin: 5px 0; border-color:#bdc3c7;">
    <div style="font-size:10px; color:#555;">Use layer control (top left) to toggle layers</div>
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# Add layer control
folium.LayerControl(position='topleft', collapsed=False).add_to(m)

# Save map
output_file = '/Users/brandonk28/milind/integrated_master_map.html'
m.save(output_file)

print(f"\n✅ Integrated map created: integrated_master_map.html")
print("\n" + "="*70)
print("✅ MASTER MAP COMPLETE")
print("="*70)
print("\n📊 Map includes:")
print("   • 6 forest sections")
print("   • Water body locations")
print("   • 642 predicted crop fields (7 zones)")
print("   • OSM features (roads, water, natural areas, settlements)")
print("   • 3 elephant movement trajectories (demo)")
print("   • Full layer controls for toggling overlays")
print("\n📥 View at: localhost:8000/integrated_master_map.html")
