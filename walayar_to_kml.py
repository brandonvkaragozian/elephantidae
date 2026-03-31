#!/usr/bin/env python3
"""
Convert Walayar Forest Map with Grid to KML for Google Earth
Generates KML with forest sections, water bodies, and 500m×500m grid overlay
"""

import xml.etree.ElementTree as ET
import re
import math
import numpy as np
import pandas as pd

print("Converting Walayar Forest Map to KML...")

# ===== PARSE ORIGINAL WALAYAR KML =====
kml_file = '/Users/brandonk28/milind/Walayar_Range_clean.kml'
with open(kml_file, 'r') as f:
    kml_content = f.read()
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

print(f"✓ Loaded {len(sections_data)} forest sections")

# ===== CALCULATE BOUNDS AND GRID =====
all_coords = []
for section in sections_data:
    all_coords.extend(section['coordinates'])

lats = [coord[0] for coord in all_coords]
lons = [coord[1] for coord in all_coords]

lat_min = min(lats)
lat_max = max(lats)
lon_min = min(lons)
lon_max = max(lons)

center_lat = np.mean(lats)
center_lon = np.mean(lons)

lat_rad = math.radians((lat_min + lat_max) / 2)

# Calculate delta for 500 meters
meters_500 = 500
km_500 = meters_500 / 1000
delta_lat = km_500 / 111.0
delta_lon = km_500 / (111.0 * math.cos(lat_rad))

print(f"✓ Map bounds: {lat_min:.4f}° to {lat_max:.4f}°N, {lon_min:.4f}° to {lon_max:.4f}°E")
print(f"✓ Grid spacing: {delta_lat:.6f}° lat × {delta_lon:.6f}° lon (500m)")

# ===== CREATE NEW KML STRUCTURE =====
kml_ns = "http://www.opengis.net/kml/2.2"
ET.register_namespace('', kml_ns)

kml = ET.Element('kml', {'xmlns': kml_ns, 'version': '2.2'})
document = ET.SubElement(kml, 'Document')

# Add document metadata
name = ET.SubElement(document, 'name')
name.text = 'Walayar Range Forest Map with Grid'

description = ET.SubElement(document, 'description')
description.text = 'Walayar Forest sections (6 polygons) with 500m × 500m measurement grid for Google Earth'

# Set default styling options
style = ET.SubElement(document, 'Style', {'id': 'forestStyle'})
poly_style = ET.SubElement(style, 'PolyStyle')
color = ET.SubElement(poly_style, 'color')
color.text = '7f00aa55'
fill = ET.SubElement(poly_style, 'fill')
fill.text = '1'

line_style = ET.SubElement(style, 'LineStyle')
line_color = ET.SubElement(line_style, 'color')
line_color.text = 'ff00aa55'
line_width = ET.SubElement(line_style, 'width')
line_width.text = '2'

# Grid style (dashed red lines)
grid_style = ET.SubElement(document, 'Style', {'id': 'gridStyle'})
grid_line = ET.SubElement(grid_style, 'LineStyle')
grid_color = ET.SubElement(grid_line, 'color')
grid_color.text = 'ff0000ff'  # Red in KML (AABBGGRR format)
grid_width = ET.SubElement(grid_line, 'width')
grid_width.text = '1'

# Water style
water_style = ET.SubElement(document, 'Style', {'id': 'waterStyle'})
water_poly = ET.SubElement(water_style, 'PolyStyle')
water_color = ET.SubElement(water_poly, 'color')
water_color.text = '4c0000ff'  # Semi-transparent blue
water_fill = ET.SubElement(water_poly, 'fill')
water_fill.text = '1'

# ===== ADD FOREST SECTIONS FOLDER =====
forest_folder = ET.SubElement(document, 'Folder')
forest_name = ET.SubElement(forest_folder, 'name')
forest_name.text = '🌲 Forest Sections'

colors = ['2ecc71', '3498db', '9b59b6', 'e74c3c', 'f39c12', '1abc9c']

for idx, section in enumerate(sections_data):
    pm = ET.SubElement(forest_folder, 'Placemark')
    
    pm_name = ET.SubElement(pm, 'name')
    pm_name.text = section['name']
    
    pm_desc = ET.SubElement(pm, 'description')
    pm_desc.text = f"Forest Section: {section['name']}\nBoundary Points: {len(section['coordinates'])}"
    
    # Use color from the original visualization
    style_elem = ET.SubElement(pm, 'Style')
    poly_style = ET.SubElement(style_elem, 'PolyStyle')
    color_elem = ET.SubElement(poly_style, 'color')
    color_elem.text = f'7f{colors[idx % len(colors)]}'
    fill_elem = ET.SubElement(poly_style, 'fill')
    fill_elem.text = '1'
    
    line_style = ET.SubElement(style_elem, 'LineStyle')
    line_color = ET.SubElement(line_style, 'color')
    line_color.text = f'ff{colors[idx % len(colors)]}'
    line_width = ET.SubElement(line_style, 'width')
    line_width.text = '2'
    
    # Polygon geometry
    multi_geom = ET.SubElement(pm, 'MultiGeometry')
    polygon = ET.SubElement(multi_geom, 'Polygon')
    outer_boundary = ET.SubElement(polygon, 'outerBoundaryIs')
    linear_ring = ET.SubElement(outer_boundary, 'LinearRing')
    coordinates = ET.SubElement(linear_ring, 'coordinates')
    
    # Add coordinates (lon,lat,0 format for KML)
    coord_text = '\n'
    for lat, lon in section['coordinates']:
        coord_text += f"          {lon},{lat},0\n"
    coordinates.text = coord_text

print(f"✓ Added {len(sections_data)} forest sections")

# ===== ADD GRID FOLDER =====
grid_folder = ET.SubElement(document, 'Folder')
grid_name = ET.SubElement(grid_folder, 'name')
grid_name.text = '📏 500m × 500m Grid'

grid_desc = ET.SubElement(grid_folder, 'description')
grid_desc.text = 'Red dashed measurement grid with 500m spacing for area reference'

# Create vertical grid lines (longitude lines)
lon = lon_min
grid_count = 0
while lon <= lon_max:
    pm = ET.SubElement(grid_folder, 'Placemark')
    pm_name = ET.SubElement(pm, 'name')
    pm_name.text = f"Grid Vertical {grid_count}"
    
    style_elem = ET.SubElement(pm, 'Style')
    line_style = ET.SubElement(style_elem, 'LineStyle')
    line_color = ET.SubElement(line_style, 'color')
    line_color.text = 'ff0000ff'  # Red
    line_width = ET.SubElement(line_style, 'width')
    line_width.text = '1'
    
    line_string = ET.SubElement(pm, 'LineString')
    coordinates = ET.SubElement(line_string, 'coordinates')
    coordinates.text = f"\n          {lon},{lat_min},0\n          {lon},{lat_max},0\n      "
    
    grid_count += 1
    lon += delta_lon

# Create horizontal grid lines (latitude lines)
lat = lat_min
while lat <= lat_max:
    pm = ET.SubElement(grid_folder, 'Placemark')
    pm_name = ET.SubElement(pm, 'name')
    pm_name.text = f"Grid Horizontal {grid_count}"
    
    style_elem = ET.SubElement(pm, 'Style')
    line_style = ET.SubElement(style_elem, 'LineStyle')
    line_color = ET.SubElement(line_style, 'color')
    line_color.text = 'ff0000ff'  # Red
    line_width = ET.SubElement(line_style, 'width')
    line_width.text = '1'
    
    line_string = ET.SubElement(pm, 'LineString')
    coordinates = ET.SubElement(line_string, 'coordinates')
    coordinates.text = f"\n          {lon_min},{lat},0\n          {lon_max},{lat},0\n      "
    
    grid_count += 1
    lat += delta_lat

print(f"✓ Added {grid_count} grid lines (500m spacing)")

# ===== ADD WATER BODIES FOLDER =====
water_folder = ET.SubElement(document, 'Folder')
water_name = ET.SubElement(water_folder, 'name')
water_name.text = '💧 Water Bodies'

# Add water extent circle marker
water_pm = ET.SubElement(water_folder, 'Placemark')
water_pm_name = ET.SubElement(water_pm, 'name')
water_pm_name.text = 'Water Extent Coverage'

water_desc = ET.SubElement(water_pm, 'description')
water_desc.text = 'Water body distribution in Walayar Range\n' \
                 'Maximum extent: 23.5-26.2%\n' \
                 'Permanent: 8.4-12.9%\n' \
                 'Seasonal: 6.7-7.2%'

water_style = ET.SubElement(water_pm, 'Style')
water_poly_style = ET.SubElement(water_style, 'PolyStyle')
water_color = ET.SubElement(water_poly_style, 'color')
water_color.text = '664c7db9'  # Semi-transparent blue (AABBGGRR)
water_fill = ET.SubElement(water_poly_style, 'fill')
water_fill.text = '1'

# Add water markers (known water channels)
water_locations = [
    {'name': 'Water Channel 1', 'lat': 10.78, 'lon': 76.75},
    {'name': 'Water Channel 2', 'lat': 10.82, 'lon': 76.78},
    {'name': 'Water Channel 3', 'lat': 10.88, 'lon': 76.80},
    {'name': 'Water Channel 4', 'lat': 10.87, 'lon': 76.72}
]

for water_loc in water_locations:
    water_marker = ET.SubElement(water_folder, 'Placemark')
    marker_name = ET.SubElement(water_marker, 'name')
    marker_name.text = water_loc['name']
    
    point = ET.SubElement(water_marker, 'Point')
    coordinates = ET.SubElement(point, 'coordinates')
    coordinates.text = f"{water_loc['lon']},{water_loc['lat']},0"

print(f"✓ Added water body markers and statistics")

# ===== ADD CENTER MARKER =====
center_folder = ET.SubElement(document, 'Folder')
center_name = ET.SubElement(center_folder, 'name')
center_name.text = '📍 Reference Points'

center_pm = ET.SubElement(center_folder, 'Placemark')
center_pm_name = ET.SubElement(center_pm, 'name')
center_pm_name.text = 'Walayar Range Center'

center_point = ET.SubElement(center_pm, 'Point')
center_coords = ET.SubElement(center_point, 'coordinates')
center_coords.text = f"{center_lon},{center_lat},0"

# ===== WRITE KML FILE =====
output_file = '/Users/brandonk28/milind/walayar_forest_with_grid.kml'
tree = ET.ElementTree(kml)
tree.write(output_file, encoding='utf-8', xml_declaration=True)

print(f"\n✅ KML file created successfully!")
print(f"   Output: {output_file}")
print(f"\n📊 KML Contents:")
print(f"   • {len(sections_data)} Forest Sections (colored polygons)")
print(f"   • {grid_count} Grid Lines (500m spacing, red dashed)")
print(f"   • {len(water_locations)} Water Body Markers")
print(f"   • 1 Center Reference Point")
print(f"\n🗺️  To view in Google Earth:")
print(f"   1. Open Google Earth")
print(f"   2. File > Import KML from Computer")
print(f"   3. Select: {output_file}")
print(f"   4. Click 'Import'")
print(f"\n💡 Tips:")
print(f"   • Forest sections are color-coded based on original map")
print(f"   • Grid lines are 500m apart for area measurement")
print(f"   • Water markers show known water body locations")
print(f"   • Use 'Measure' tool in Google Earth to verify distances")
