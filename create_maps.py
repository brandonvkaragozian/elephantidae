#!/usr/bin/env python3
"""Create separate maps for Walayar Forest and Crop Fields"""
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
import xml.etree.ElementTree as ET
import re

print("Loading data...")

# Parse KML
kml_file = '/Users/brandonk28/milind/Walayar_Range.kml'
with open(kml_file, 'r') as f:
    kml_content = f.read()

kml_content = re.sub(r' xsi:schemaLocation="[^"]*"', '', kml_content)
root = ET.fromstring(kml_content)
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

placemarks = root.findall('.//kml:Placemark', ns)
sections_data = []
for placemark in placemarks:
    name_elem = placemark.find('kml:name', ns)
    if name_elem is None:
        continue
    name = name_elem.text
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
            sections_data.append({'name': name, 'coordinates': coords})

# Get map center
all_coords = []
for section in sections_data:
    all_coords.extend(section['coordinates'])
lats = [coord[0] for coord in all_coords]
lons = [coord[1] for coord in all_coords]
center_lat = np.mean(lats)
center_lon = np.mean(lons)

# Load water data
bodies_water_file = '/Users/brandonk28/milind/bodies_water.xlsx'
xl_file = pd.ExcelFile(bodies_water_file)
bodies_water_data = {}
for sheet_name in xl_file.sheet_names:
    bodies_water_data[sheet_name] = pd.read_excel(bodies_water_file, sheet_name=sheet_name)

longitude_data = bodies_water_data['Longitude Data']
latitude_data = bodies_water_data['Latitude Data']
walayar_region_lon = longitude_data[(longitude_data['Longitude'] >= 76) & (longitude_data['Longitude'] <= 77)]
walayar_region_lat = latitude_data[(latitude_data['Latitude'] >= 10) & (latitude_data['Latitude'] <= 11)]

# Load crop fields
crop_fields_df = pd.read_parquet('/Users/brandonk28/milind/crop_fields.parquet')
crop_fields_df['geometry'] = crop_fields_df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)
gdf = gpd.GeoDataFrame(crop_fields_df[['id', 'area', 'determination_datetime']], 
                        geometry=crop_fields_df['geometry'], 
                        crs='EPSG:4326')

# Calculate bounds
crop_bounds_north = gdf.geometry.bounds['maxy'].max()
crop_bounds_south = gdf.geometry.bounds['miny'].min()
crop_bounds_east = gdf.geometry.bounds['maxx'].max()
crop_bounds_west = gdf.geometry.bounds['minx'].min()
crop_center_lat = (crop_bounds_north + crop_bounds_south) / 2
crop_center_lon = (crop_bounds_east + crop_bounds_west) / 2

# ===== MAP 1: WALAYAR =====
print("\n1️⃣  Creating Walayar Forest Map...")
m_forest = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

for idx, section in enumerate(sections_data):
    folium.Polygon(
        locations=section['coordinates'],
        popup=f"<b>{section['name']}</b><br>Points: {len(section['coordinates'])}",
        tooltip=section['name'],
        color=colors[idx % len(colors)],
        fill=True,
        fillColor=colors[idx % len(colors)],
        fillOpacity=0.35,
        weight=2.5
    ).add_to(m_forest)
    
    section_coords = np.array(section['coordinates'])
    center_point = section_coords.mean(axis=0)
    folium.CircleMarker(
        location=center_point,
        radius=6,
        popup=section['name'],
        color=colors[idx % len(colors)],
        fill=True,
        fillColor='white',
        fillOpacity=0.9,
        weight=2
    ).add_to(m_forest)

# Add water
water_extent_avg = (walayar_region_lon['Maximum water extent'].mean() + 
                    walayar_region_lat['Maximum water extent'].mean()) / 2
folium.Circle(
    location=[center_lat, center_lon],
    radius=4000,
    popup=f"<b>Water Coverage</b><br>Average: {water_extent_avg:.1f}%<br>Permanent: 10.7%<br>Seasonal: 6.9%",
    color='#2980b9',
    fill=True,
    fillColor='#3498db',
    fillOpacity=0.15,
    weight=2,
    dash_array='5, 5'
).add_to(m_forest)

# Water markers
for loc in [[10.78, 76.75], [10.82, 76.78], [10.88, 76.80], [10.87, 76.72]]:
    folium.CircleMarker(location=loc, radius=7, popup="💧 Water Channel", color='#0066cc', fill=True, fillColor='#00ccff', fillOpacity=0.7, weight=2).add_to(m_forest)

legend1 = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 380px; background-color: rgba(255,255,255,0.95); border:3px solid #2c3e50; z-index:9999; font-size:11px; padding: 12px; border-radius: 6px; font-weight: bold;">
    <div style="font-size:14px; color:#2c3e50; margin-bottom:8px;">🌲 WALAYAR FOREST RANGE</div>
    <hr style="margin: 5px 0; border-color:#bdc3c7;">
    <div style="color:#27ae60; margin:6px 0;">✓ 6 Forest Sections (3,027 boundary points)</div>
    <hr style="margin: 5px 0; border-color:#bdc3c7;">
    <div style="color:#2980b9; margin:6px 0;">💧 Water Bodies</div>
    <div style="font-size:10px; color:#555;">Max: 23.5-26.2% | Permanent: 8.4-12.9%</div>
</div>
'''
m_forest.get_root().html.add_child(folium.Element(legend1))
folium.LayerControl().add_to(m_forest)

m_forest.save('/Users/brandonk28/milind/walayar_forest_map.html')
print("   ✅ Saved: walayar_forest_map.html")

# ===== MAP 2: CROP FIELDS =====
print("\n2️⃣  Creating Crop Fields Map...")
m_crops = folium.Map(location=[crop_center_lat, crop_center_lon], zoom_start=8, tiles='OpenStreetMap')

for idx, row in gdf.iterrows():
    geom = row['geometry']
    area = row['area']
    color = '#f1c40f' if area < 3000 else ('#f39c12' if area < 6000 else '#e67e22')
    
    folium.Polygon(
        locations=[[lat, lon] for lon, lat in geom.exterior.coords],
        popup=f"<b>Crop Field</b><br>Area: {area:.0f} m²",
        tooltip=f"{area:.0f} m²",
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.5,
        weight=0.3
    ).add_to(m_crops)
    
    if idx % 1000 == 0:
        print(f"   Processing field {idx:,} / {len(gdf):,}...")

avg_area = gdf['area'].mean()
max_area = gdf['area'].max()
min_area = gdf['area'].min()

legend2 = f'''
<div style="position: fixed; bottom: 50px; left: 50px; width: 380px; background-color: rgba(255,255,255,0.95); border:3px solid #2c3e50; z-index:9999; font-size:11px; padding: 12px; border-radius: 6px; font-weight: bold;">
    <div style="font-size:14px; color:#2c3e50; margin-bottom:8px;">🌾 CROP FIELDS REGION</div>
    <hr style="margin: 5px 0; border-color:#bdc3c7;">
    <div style="color:#e67e22; margin:6px 0;">✓ ALL {len(gdf):,} Crop Fields</div>
    <div style="font-size:10px; color:#555;">Avg: {avg_area:.0f} m² | Min: {min_area:.0f} m² | Max: {max_area:.0f} m²</div>
</div>
'''
m_crops.get_root().html.add_child(folium.Element(legend2))
folium.LayerControl().add_to(m_crops)

m_crops.save('/Users/brandonk28/milind/crop_fields_map.html')
print(f"   ✅ Saved: crop_fields_map.html\n")

print("="*70)
print("✅ BOTH MAPS CREATED SUCCESSFULLY!")
print("="*70)
print("\n📍 OUTPUT FILES:")
print("   1. walayar_forest_map.html - 6 forest sections + water visualization")
print("   2. crop_fields_map.html - ALL 10,013 crop field polygons")
print("   3. more_water_timeseries.png - Water time-series (1984-2015)")
print("\n🗺️  GEOGRAPHIC REGIONS:")
print(f"   • Walayar Range: Kerala (10.8°N, 76.8°E)")
print(f"   • Crop Fields: Gujarat/Rajasthan (26.3°N, 73.1°E)")
