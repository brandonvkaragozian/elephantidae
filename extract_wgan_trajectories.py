#!/usr/bin/env python3
"""
extract_wgan_trajectories.py
=============================
Extract the 8 top WGAN-GP trajectories from FINAL WALAYAR MAP.kml
and save them as a new KML for use in feature computation.
"""

import xml.etree.ElementTree as ET
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_KML = os.path.join(SCRIPT_DIR, "FINAL WALAYAR MAP.kml")
OUTPUT_KML = os.path.join(SCRIPT_DIR, "walayar_wgan_trajectories.kml")

def main():
    print("Parsing FINAL WALAYAR MAP.kml...")
    tree = ET.parse(INPUT_KML)
    root = tree.getroot()
    
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Find the document and folder
    doc = root.find('kml:Document', ns)
    if doc is None:
        print("Error: Could not find Document")
        return
    
    # Create new KML structure
    new_root = ET.Element('kml')
    new_root.set('xmlns', 'http://www.opengis.net/kml/2.2')
    new_doc = ET.SubElement(new_root, 'Document')
    ET.SubElement(new_doc, 'name').text = 'Walayar WGAN-GP Trajectories'
    
    # Find and copy the 8 top trajectories
    trajectory_count = 0
    for placemark in root.findall('.//kml:Placemark', ns):
        name = placemark.findtext('kml:name', '', ns)
        
        # Check if this is a Top 1-8 trajectory
        if name.startswith('Top ') and 'WGAN-GP' in name:
            print(f"  Found: {name}")
            
            # Copy the entire placemark
            placemark_copy = ET.fromstring(ET.tostring(placemark))
            new_doc.append(placemark_copy)
            trajectory_count += 1
    
    print(f"\nExtracted {trajectory_count} trajectories")
    
    # Write new KML
    tree_new = ET.ElementTree(new_root)
    tree_new.write(OUTPUT_KML, encoding='utf-8', xml_declaration=True)
    print(f"Saved to {OUTPUT_KML}")

if __name__ == '__main__':
    main()
