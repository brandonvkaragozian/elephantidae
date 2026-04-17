import xml.etree.ElementTree as ET

tree = ET.parse("FINAL WALAYAR MAP.kml")
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# Count placemarks and show names and types
placemarks_info = []
for placemark in root.findall('.//kml:Placemark', ns):
    name = placemark.findtext('kml:name', '', ns)
    has_poly = placemark.find('kml:Polygon', ns) is not None
    has_line = placemark.find('kml:LineString', ns) is not None
    has_point = placemark.find('kml:Point', ns) is not None
    geom_type = 'Polygon' if has_poly else ('LineString' if has_line else ('Point' if has_point else 'Unknown'))
    placemarks_info.append((name, geom_type))

print(f"Total placemarks: {len(placemarks_info)}")
print("\nFirst 50 placemarks:")
for name, geom_type in placemarks_info[:50]:
    print(f"  {geom_type:12} - {name}")
