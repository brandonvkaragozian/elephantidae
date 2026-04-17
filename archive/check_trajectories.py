import xml.etree.ElementTree as ET

tree = ET.parse("FINAL WALAYAR MAP.kml")
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# Count LineStrings (trajectories)
placemarks_info = []
for placemark in root.findall('.//kml:Placemark', ns):
    name = placemark.findtext('kml:name', '', ns)
    has_line = placemark.find('kml:LineString', ns) is not None
    if has_line:
        line_elem = placemark.find('kml:LineString/kml:coordinates', ns)
        if line_elem is not None and line_elem.text:
            coords_text = line_elem.text.strip()
            num_points = len(coords_text.split())
            placemarks_info.append((name, num_points))

print(f"Found {len(placemarks_info)} LineString trajectories:")
for name, num_points in sorted(placemarks_info):
    print(f"  - {name}: {num_points} points")
