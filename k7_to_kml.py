#!/usr/bin/env python3
"""Convert K=7 Predicted Crop Fields to KML for Google Earth"""

import geopandas as gpd
import xml.etree.ElementTree as ET

print("="*70)
print("CONVERTING K=7 PREDICTIONS TO KML")
print("="*70)

# Load K=7 predictions
print("\n📂 Loading K=7 predictions...")
gdf_k7 = gpd.read_file('/Users/brandonk28/milind/predicted_crop_fields_k7.geojson')
print(f"✓ Loaded {len(gdf_k7)} predicted crop fields")

# Create KML document
print("\n📝 Creating KML document...")

kml = ET.Element('kml')
kml.set('xmlns', 'http://www.opengis.net/kml/2.2')

document = ET.SubElement(kml, 'Document')
name = ET.SubElement(document, 'name')
name.text = 'Walayar Range - Predicted Crop Fields (K=7 Zones)'

description = ET.SubElement(document, 'description')
description.text = f'Predicted crop field locations based on K=7 suitability clustering\nTotal fields: {len(gdf_k7)}'

# Define zone styles with colors
zone_colors = [
    'ffff0000',  # Zone 1: Red
    'ff00ff00',  # Zone 2: Green
    'ff0000ff',  # Zone 3: Blue
    'ffffff00',  # Zone 4: Cyan
    'ffff00ff',  # Zone 5: Magenta
    'ff00ffff',  # Zone 6: Yellow
    'ff808080',  # Zone 7: Gray
]

zone_names = [
    'Zone 1 - Northwest',
    'Zone 2 - West',
    'Zone 3 - Central-South',
    'Zone 4 - East',
    'Zone 5 - Central-North',
    'Zone 6 - Central',
    'Zone 7 - Southeast'
]

for zone_id in range(7):
    style = ET.SubElement(document, 'Style')
    style.set('id', f'zone{zone_id+1}')
    
    poly_style = ET.SubElement(style, 'PolyStyle')
    color = ET.SubElement(poly_style, 'color')
    color.text = zone_colors[zone_id]
    
    fill = ET.SubElement(poly_style, 'fill')
    fill.text = '1'
    
    outline = ET.SubElement(poly_style, 'outline')
    outline.text = '1'
    
    line_style = ET.SubElement(style, 'LineStyle')
    line_color = ET.SubElement(line_style, 'color')
    line_color.text = 'ff000000'
    width = ET.SubElement(line_style, 'width')
    width.text = '1'

# Create folder for each zone
zone_count = {i: 0 for i in range(7)}

for zone_id in range(7):
    folder = ET.SubElement(document, 'Folder')
    zone_name = ET.SubElement(folder, 'name')
    zone_name.text = zone_names[zone_id]
    
    zone_desc = ET.SubElement(folder, 'description')
    zone_desc.text = f'Predicted crop fields in {zone_names[zone_id]}'
    
    # Add fields for this zone
    zone_fields = gdf_k7[gdf_k7['cluster'] == zone_id]
    zone_count[zone_id] = len(zone_fields)
    
    for idx, row in zone_fields.iterrows():
        placemark = ET.SubElement(folder, 'Placemark')
        
        pm_name = ET.SubElement(placemark, 'name')
        pm_name.text = f"Field {idx+1}"
        
        pm_desc = ET.SubElement(placemark, 'description')
        area_val = row['area']
        suitability_val = row['suitability']
        pm_desc.text = f"<![CDATA[<b>Predicted Crop Field</b><br>Zone: {zone_id+1}<br>Area: {area_val:,.0f} m²<br>Suitability: {suitability_val:.3f}]]>"
        
        # Style reference
        style_url = ET.SubElement(placemark, 'styleUrl')
        style_url.text = f'#zone{zone_id+1}'
        
        # Geometry
        polygon = ET.SubElement(placemark, 'Polygon')
        
        outer = ET.SubElement(polygon, 'outerBoundaryIs')
        linear_ring = ET.SubElement(outer, 'LinearRing')
        
        coordinates_elem = ET.SubElement(linear_ring, 'coordinates')
        geom = row['geometry']
        coords_list = []
        for lon, lat in geom.exterior.coords:
            coords_list.append(f"{lon},{lat},0")
        coordinates_elem.text = '\n'.join(coords_list)

# Summary folder
summary_folder = ET.SubElement(document, 'Folder')
summary_name = ET.SubElement(summary_folder, 'name')
summary_name.text = '📊 Zone Summary'

summary_desc = ET.SubElement(summary_folder, 'description')
summary_lines = ['Walayar Range K=7 Crop Field Predictions']
for zone_id in range(7):
    summary_lines.append(f'{zone_names[zone_id]}: {zone_count[zone_id]} fields')
summary_desc.text = '<br>'.join(summary_lines)

summary_placemark = ET.SubElement(summary_folder, 'Placemark')
summary_pm_name = ET.SubElement(summary_placemark, 'name')
summary_pm_name.text = 'Model Summary'

summary_pm_desc = ET.SubElement(summary_placemark, 'description')
summary_text = f"""<![CDATA[
<b>Walayar Range - K=7 Crop Field Predictions</b><br>
Total Predicted Fields: {len(gdf_k7)}<br>
Total Predicted Area: {gdf_k7['area'].sum():,.0f} m² ({gdf_k7['area'].sum()/1e6:.2f} km²)<br>
Average Field Area: {gdf_k7['area'].mean():,.0f} m²<br>
Average Suitability: {gdf_k7['suitability'].mean():.3f}<br>
<br>
<b>Zone Distribution:</b><br>
"""
for zone_id in range(7):
    summary_text += f'{zone_names[zone_id]}: {zone_count[zone_id]} fields<br>'

summary_text += """
<b>Model Details:</b><br>
- 7 agricultural zones based on suitability clustering<br>
- Suitability factors: Water proximity (35%), Forest edge (35%), Terrain (30%)<br>
- Field sizes follow observed crop field distribution<br>
]]>"""

summary_pm_desc.text = summary_text

# Format and save
tree = ET.ElementTree(kml)
tree.write('/Users/brandonk28/milind/predicted_crop_fields_k7.kml', 
          encoding='utf-8', xml_declaration=True)

print(f"\n✅ KML created successfully!")
print(f"\n📊 Zone Summary:")
for zone_id in range(7):
    print(f"   {zone_names[zone_id]}: {zone_count[zone_id]} fields")

print(f"\n🗺️  File saved: predicted_crop_fields_k7.kml")
print(f"   Total fields: {len(gdf_k7)}")
print(f"   Total area: {gdf_k7['area'].sum()/1e6:.2f} km²")

print("\n" + "="*70)
print("✅ K=7 PREDICTIONS CONVERTED TO KML")
print("="*70)
print("\n📥 To import into Google Earth:")
print("   1. Open Google Earth")
print("   2. File → Import KML file")
print("   3. Select: predicted_crop_fields_k7.kml")
print("   4. Color-coded zones will appear on the map")
