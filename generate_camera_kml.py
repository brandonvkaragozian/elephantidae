#!/usr/bin/env python3
"""
Generate KML file for 16 camera placements to visualize on Google Maps.
"""

import pandas as pd
from lxml import etree

# Load camera placement results
cameras = pd.read_csv('camera_placement_enhanced_mip_16_cameras.csv')

print("Generating KML for 16 camera placements...")

# KML namespace
ns = 'http://www.opengis.net/kml/2.2'

# Create KML structure
kml = etree.Element('kml', xmlns=ns, version='2.2')
document = etree.SubElement(kml, 'Document')

# Document metadata
name = etree.SubElement(document, 'name')
name.text = 'Walayar Wildlife Sanctuary - 16 Camera Placements'

description = etree.SubElement(document, 'description')
description.text = 'Optimal camera placement for elephant monitoring. Generated via MIP optimization.'

# Add open in Google Earth
open_elem = etree.SubElement(document, 'open')
open_elem.text = '1'

# Define styles for different camera tiers
style_tier1 = etree.SubElement(document, 'Style', id='Tier1')
icon_style1 = etree.SubElement(style_tier1, 'IconStyle')
icon1 = etree.SubElement(icon_style1, 'Icon')
icon1_href = etree.SubElement(icon1, 'href')
icon1_href.text = 'http://maps.google.com/mapfiles/kml/shapes/camera.png'
scale1 = etree.SubElement(icon_style1, 'scale')
scale1.text = '1.2'
color1 = etree.SubElement(icon_style1, 'color')
color1.text = 'ff0000ff'  # Red (ABGR format)

label_style1 = etree.SubElement(style_tier1, 'LabelStyle')
label_color1 = etree.SubElement(label_style1, 'color')
label_color1.text = 'ff0000ff'
label_scale1 = etree.SubElement(label_style1, 'scale')
label_scale1.text = '1.0'

style_tier2 = etree.SubElement(document, 'Style', id='Tier2')
icon_style2 = etree.SubElement(style_tier2, 'IconStyle')
icon2 = etree.SubElement(icon_style2, 'Icon')
icon2_href = etree.SubElement(icon2, 'href')
icon2_href.text = 'http://maps.google.com/mapfiles/kml/shapes/camera.png'
scale2 = etree.SubElement(icon_style2, 'scale')
scale2.text = '1.0'
color2 = etree.SubElement(icon_style2, 'color')
color2.text = 'ff00ff00'  # Green

label_style2 = etree.SubElement(style_tier2, 'LabelStyle')
label_color2 = etree.SubElement(label_style2, 'color')
label_color2.text = 'ff00ff00'
label_scale2 = etree.SubElement(label_style2, 'scale')
label_scale2.text = '0.9'

style_tier3 = etree.SubElement(document, 'Style', id='Tier3')
icon_style3 = etree.SubElement(style_tier3, 'IconStyle')
icon3 = etree.SubElement(icon_style3, 'Icon')
icon3_href = etree.SubElement(icon3, 'href')
icon3_href.text = 'http://maps.google.com/mapfiles/kml/shapes/camera.png'
scale3 = etree.SubElement(icon_style3, 'scale')
scale3.text = '0.9'
color3 = etree.SubElement(icon_style3, 'color')
color3.text = 'ffff0000'  # Blue

label_style3 = etree.SubElement(style_tier3, 'LabelStyle')
label_color3 = etree.SubElement(label_style3, 'color')
label_color3.text = 'ffff0000'
label_scale3 = etree.SubElement(label_style3, 'scale')
label_scale3.text = '0.8'

# Create folders for each tier
folder_tier1 = etree.SubElement(document, 'Folder')
name_t1 = etree.SubElement(folder_tier1, 'name')
name_t1.text = 'Tier 1: High-Coverage Anchors (Northern Boundary)'
desc_t1 = etree.SubElement(folder_tier1, 'description')
desc_t1.text = '4 cameras providing complete geographic coverage'
open_t1 = etree.SubElement(folder_tier1, 'open')
open_t1.text = '1'

folder_tier2 = etree.SubElement(document, 'Folder')
name_t2 = etree.SubElement(folder_tier2, 'name')
name_t2.text = 'Tier 2: Perimeter Reinforcement (Eastern Boundary)'
desc_t2 = etree.SubElement(folder_tier2, 'description')
desc_t2.text = '9 cameras providing redundancy and coverage depth'
open_t2 = etree.SubElement(folder_tier2, 'open')
open_t2.text = '1'

folder_tier3 = etree.SubElement(document, 'Folder')
name_t3 = etree.SubElement(folder_tier3, 'name')
name_t3.text = 'Tier 3: Activity Hotspot Monitors'
desc_t3 = etree.SubElement(folder_tier3, 'description')
desc_t3.text = '3 cameras for intensive elephant behavior monitoring'
open_t3 = etree.SubElement(folder_tier3, 'open')
open_t3.text = '1'

# Add camera placemarks
for idx, row in cameras.iterrows():
    rank = int(row['rank'])
    cell_id = row['cell_id']
    lat = row['latitude']
    lon = row['longitude']
    visible = int(row['cells_visible'])
    demand = row['weighted_demand_coverage']
    visits = int(row['elephant_visits_in_range'])
    
    # Determine tier and folder
    if rank <= 4:
        folder = folder_tier1
        style_ref = 'Tier1'
        tier_name = 'ANCHOR'
    elif rank <= 13:
        folder = folder_tier2
        style_ref = 'Tier2'
        tier_name = 'REINFORCEMENT'
    else:
        folder = folder_tier3
        style_ref = 'Tier3'
        tier_name = 'HOTSPOT'
    
    # Create placemark
    placemark = etree.SubElement(folder, 'Placemark')
    
    # Name and description
    pm_name = etree.SubElement(placemark, 'name')
    pm_name.text = f"Camera {rank}: {cell_id} ({tier_name})"
    
    pm_desc = etree.SubElement(placemark, 'description')
    pm_desc.text = f"""
Camera Placement Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rank: {rank} of 16
Cell ID: {cell_id}
Tier: {tier_name}
Coordinates: {lat:.6f}°N, {lon:.6f}°E

Coverage Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cells Visible: {visible}
Demand Coverage: {demand:.2f}
Elephant Visits in Range: {visits}
Coverage Radius: 40 km (360° view)

Deployment Priority:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 (Critical): {rank <= 4}
Phase 2 (Recommended): {5 <= rank <= 13}
Phase 3 (Optional): {rank > 13}
"""
    
    # Style
    style_url = etree.SubElement(placemark, 'styleUrl')
    style_url.text = f'#{style_ref}'
    
    # Point coordinates
    point = etree.SubElement(placemark, 'Point')
    coordinates = etree.SubElement(point, 'coordinates')
    coordinates.text = f'{lon},{lat},0'
    
    # Extended data (for Google Earth)
    ext_data = etree.SubElement(placemark, 'ExtendedData')
    
    data_rank = etree.SubElement(ext_data, 'Data', name='Rank')
    data_rank_val = etree.SubElement(data_rank, 'value')
    data_rank_val.text = str(rank)
    
    data_cell = etree.SubElement(ext_data, 'Data', name='Cell_ID')
    data_cell_val = etree.SubElement(data_cell, 'value')
    data_cell_val.text = cell_id
    
    data_visible = etree.SubElement(ext_data, 'Data', name='Visible_Cells')
    data_visible_val = etree.SubElement(data_visible, 'value')
    data_visible_val.text = str(visible)
    
    data_demand = etree.SubElement(ext_data, 'Data', name='Demand_Coverage')
    data_demand_val = etree.SubElement(data_demand, 'value')
    data_demand_val.text = f'{demand:.2f}'
    
    data_visits = etree.SubElement(ext_data, 'Data', name='Elephant_Visits')
    data_visits_val = etree.SubElement(data_visits, 'value')
    data_visits_val.text = str(visits)
    
    data_tier = etree.SubElement(ext_data, 'Data', name='Tier')
    data_tier_val = etree.SubElement(data_tier, 'value')
    data_tier_val.text = tier_name

# Pretty print and save
kml_str = etree.tostring(kml, encoding='unicode', pretty_print=True)

with open('camera_placements_16_cameras.kml', 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write(kml_str)

print("✓ Generated: camera_placements_16_cameras.kml")
print(f"  - {len(cameras)} camera placemarks")
print(f"  - 3 folder tiers (color-coded)")
print(f"  - Ready for Google Maps upload")
