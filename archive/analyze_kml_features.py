import xml.etree.ElementTree as ET

tree = ET.parse("FINAL WALAYAR MAP.kml")
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# Count all placemarks by type and group
placemarks_by_type = {}
for placemark in root.findall('.//kml:Placemark', ns):
    name = placemark.findtext('kml:name', '', ns)
    has_poly = placemark.find('kml:Polygon', ns) is not None
    has_line = placemark.find('kml:LineString', ns) is not None
    has_point = placemark.find('kml:Point', ns) is not None
    geom_type = 'Polygon' if has_poly else ('LineString' if has_line else ('Point' if has_point else 'Unknown'))
    
    # Extract feature type from name
    feature_type = 'Other'
    if 'Railway' in name or 'railway' in name:
        feature_type = 'Railway'
    elif 'Road' in name or 'road' in name or 'NH' in name or 'Service' in name:
        feature_type = 'Road'
    elif 'Water' in name or 'water' in name or 'Lake' in name or 'River' in name:
        feature_type = 'Water'
    elif 'Settlement' in name or 'Town' in name or 'City' in name or 'Village' in name:
        feature_type = 'Settlement'
    elif 'Crop' in name or 'crop' in name or 'Field' in name or 'field' in name:
        feature_type = 'Crop'
    elif 'Forest' in name or 'forest' in name or 'Section' in name:
        feature_type = 'Forest/Section'
    elif 'Grid' in name or 'grid' in name or name.startswith('R') and 'C' in name:
        feature_type = 'Grid'
    elif 'Trajectory' in name or 'trajectory' in name or 'Path' in name or 'Herd' in name:
        feature_type = 'Trajectory'
    elif 'WGAN' in name:
        feature_type = 'WGAN Trajectory'
    
    if feature_type not in placemarks_by_type:
        placemarks_by_type[feature_type] = {'count': 0, 'types': {}, 'examples': []}
    
    placemarks_by_type[feature_type]['count'] += 1
    if geom_type not in placemarks_by_type[feature_type]['types']:
        placemarks_by_type[feature_type]['types'][geom_type] = 0
    placemarks_by_type[feature_type]['types'][geom_type] += 1
    
    if len(placemarks_by_type[feature_type]['examples']) < 3:
        placemarks_by_type[feature_type]['examples'].append(name)

print("=== FEATURE BREAKDOWN ===\n")
for feature_type in sorted(placemarks_by_type.keys()):
    data = placemarks_by_type[feature_type]
    print(f"{feature_type}: {data['count']} placemarks")
    for geom, count in data['types'].items():
        print(f"  - {geom}: {count}")
    print(f"  Examples: {', '.join(data['examples'][:2])}")
    print()

print(f"\nTotal placemarks: {sum(d['count'] for d in placemarks_by_type.values())}")
