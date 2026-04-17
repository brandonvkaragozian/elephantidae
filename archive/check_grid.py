import xml.etree.ElementTree as ET

tree = ET.parse('FINAL WALAYAR MAP.kml')
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# Count grid cells
grid_cells = []
for pm in root.findall('.//kml:Placemark', ns):
    name = pm.findtext('kml:name', '', ns)
    if name and name.startswith('R') and 'C' in name:
        try:
            parts = name.split('C')
            row = int(parts[0][1:])
            col = int(parts[1])
            grid_cells.append((row, col, name))
        except:
            pass

grid_cells.sort()
print(f"Total grid cells in KML: {len(grid_cells)}")
print(f"Row range: R{grid_cells[0][0]:03d} to R{grid_cells[-1][0]:03d}")
print(f"Col range: C{grid_cells[0][1]:03d} to C{grid_cells[-1][1]:03d}")

print(f"\nFirst 5 cells:")
for row, col, name in grid_cells[:5]:
    print(f"  {name}")

print(f"\nLast 5 cells:")
for row, col, name in grid_cells[-5:]:
    print(f"  {name}")

# Check coverage
max_row = max(c[0] for c in grid_cells)
max_col = max(c[1] for c in grid_cells)
print(f"\nGrid coverage: {max_row+1} rows × {max_col+1} columns = {(max_row+1)*(max_col+1)} max cells")
print(f"Actual cells in KML: {len(grid_cells)}")
print(f"Sparsity: {len(grid_cells) / ((max_row+1)*(max_col+1)) * 100:.1f}% of possible cells")
