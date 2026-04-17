import xml.etree.ElementTree as ET

tree = ET.parse('FINAL WALAYAR MAP.kml')
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# Find all forest-related placemarks
forest_features = []
for pm in root.findall('.//kml:Placemark', ns):
    name = pm.findtext('kml:name', '', ns)
    if any(keyword in name.lower() for keyword in ['forest', 'section', 'reserve', 'wildlife', 'walayar']):
        forest_features.append(name)

print(f"Forest-related placemarks found: {len(forest_features)}")
for feat in forest_features:
    print(f"  - {feat}")
