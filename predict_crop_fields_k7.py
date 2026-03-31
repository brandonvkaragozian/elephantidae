#!/usr/bin/env python3
"""
Predictive Model: Crop Field Detection in Walayar Range (K=7)
========================================================
Uses K=7 clusters for finer-grained agricultural zone identification
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
from shapely.geometry import Point, box, Polygon
import matplotlib.pyplot as plt
import folium
from folium import plugins
import xml.etree.ElementTree as ET
import re
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CROP FIELD PREDICTION MODEL FOR WALAYAR RANGE (K=7)")
print("="*70)

# ===== PHASE 1: LOAD DATA =====
print("\n📂 PHASE 1: Loading Data...")
print("-" * 70)

# Load Walayar Forest sections
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

# Get Walayar bounds
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

print(f"✓ Loaded {len(sections_data)} forest sections")
print(f"  Walayar bounds: {lat_min:.4f}-{lat_max:.4f}°N, {lon_min:.4f}-{lon_max:.4f}°E")

# Load water data
bodies_water_file = '/Users/brandonk28/milind/bodies_water.xlsx'
xl_file = pd.ExcelFile(bodies_water_file)
bodies_water_data = {}
for sheet_name in xl_file.sheet_names:
    bodies_water_data[sheet_name] = pd.read_excel(bodies_water_file, sheet_name=sheet_name)

print(f"✓ Loaded water extent data")

# Load crop fields
crop_fields_df = pd.read_parquet('/Users/brandonk28/milind/crop_fields.parquet')
crop_fields_df['geometry'] = crop_fields_df['geometry'].apply(
    lambda x: wkb.loads(x) if isinstance(x, bytes) else x
)
gdf_crops = gpd.GeoDataFrame(crop_fields_df[['id', 'area']], 
                             geometry=crop_fields_df['geometry'], 
                             crs='EPSG:4326')

field_areas = gdf_crops.geometry.apply(lambda x: x.area).values
print(f"✓ Loaded {len(gdf_crops):,} crop fields")

# ===== PHASE 2: EXTRACT CROP FIELD FEATURES =====
print("\n📊 PHASE 2: Extracting Crop Field Features...")
print("-" * 70)

print(f"\n🌾 CROP FIELD STATISTICS:")
print(f"  Area (m²):")
print(f"    Min: {field_areas.min():,.0f}")
print(f"    Max: {field_areas.max():,.0f}")
print(f"    Mean: {field_areas.mean():,.0f}")
print(f"    Median: {np.median(field_areas):,.0f}")
print(f"    Std Dev: {field_areas.std():,.0f}")

# ===== PHASE 3: COMPUTE SUITABILITY MAP =====
print(f"\n🗺️  PHASE 3: Computing Suitability Map...")
print("-" * 70)

# Water locations
known_water_locs = np.array([
    [10.78, 76.75],
    [10.82, 76.78],
    [10.88, 76.80],
    [10.87, 76.72]
])

# Create prediction grid
grid_res = 0.002
lats_grid = np.arange(lat_min, lat_max + grid_res, grid_res)
lons_grid = np.arange(lon_min, lon_max + grid_res, grid_res)
grid_points = np.array(np.meshgrid(lats_grid, lons_grid)).T.reshape(-1, 2)

# Compute suitability
suitability_scores = np.zeros(len(grid_points))

for idx, (lat, lon) in enumerate(grid_points):
    # Distance to nearest water
    water_distances = np.linalg.norm(known_water_locs - np.array([lat, lon]), axis=1)
    min_water_dist = water_distances.min()
    water_score = max(0, 1 - min_water_dist / 0.045)
    
    # Distance to forest boundary
    forest_distances = []
    for section in sections_data:
        for i in range(len(section['coordinates']) - 1):
            p1 = np.array(section['coordinates'][i])
            p2 = np.array(section['coordinates'][i + 1])
            a = np.linalg.norm(p2 - p1)
            if a == 0:
                continue
            t = max(0, min(1, np.dot(np.array([lat, lon]) - p1, p2 - p1) / (a ** 2)))
            closest = p1 + t * (p2 - p1)
            dist = np.linalg.norm(np.array([lat, lon]) - closest)
            forest_distances.append(dist)
    
    min_forest_dist = min(forest_distances) if forest_distances else 999
    forest_score = max(0, 1 - min_forest_dist / 0.03)
    
    # Terrain suitability (elevation proxy)
    terrain_score = 0.5
    
    # Combined suitability
    suitability_scores[idx] = (0.35 * water_score + 
                              0.35 * forest_score + 
                              0.30 * terrain_score)

print(f"✓ Created prediction grid: {len(lats_grid)} × {len(lons_grid)} cells ({len(grid_points):,} total)")
print(f"✓ Suitability scores computed")
print(f"  Min: {suitability_scores.min():.3f}, Max: {suitability_scores.max():.3f}, Mean: {suitability_scores.mean():.3f}")

# ===== PHASE 4: CLUSTER HIGH-SUITABILITY ZONES (K=7) =====
print(f"\n🎯 PHASE 4: Identifying Crop Field Zones (K=7)...")
print("-" * 70)

# Select high-suitability areas
threshold = np.percentile(suitability_scores, 40)
high_suit_mask = suitability_scores >= threshold
high_suit_points = grid_points[high_suit_mask]
high_suit_scores = suitability_scores[high_suit_mask]

print(f"✓ Identified {len(high_suit_points):,} suitable grid cells")
print(f"  Suitability threshold: {threshold:.3f}")

# Cluster high-suitability zones - K=7
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(high_suit_points)

print(f"✓ Identified {n_clusters} potential crop field zones:")
for i in range(n_clusters):
    cluster_points = high_suit_points[clusters == i]
    cluster_scores = high_suit_scores[clusters == i]
    print(f"  Zone {i+1}: {len(cluster_points):,} cells, avg suitability: {cluster_scores.mean():.3f}")

# ===== PHASE 5: GENERATE PREDICTED CROP FIELDS =====
print(f"\n🌱 PHASE 5: Generating Predicted Crop Fields...")
print("-" * 70)

predicted_fields = []

for cluster_id in range(n_clusters):
    cluster_points = high_suit_points[clusters == cluster_id]
    cluster_scores = high_suit_scores[clusters == cluster_id]
    
    if len(cluster_points) < 3:
        continue
    
    # Mean cluster position
    center_lat_c = cluster_points[:, 0].mean()
    center_lon_c = cluster_points[:, 1].mean()
    
    # Generate fields based on crop field statistics
    n_fields = max(2, int(len(cluster_points) / 10))
    
    for j in range(n_fields):
        # Random offset from cluster center
        offset_lat = np.random.normal(0, 0.003)
        offset_lon = np.random.normal(0, 0.003)
        
        field_lat = center_lat_c + offset_lat
        field_lon = center_lon_c + offset_lon
        
        # Random field size from observed distribution
        field_area = np.random.lognormal(
            mean=np.log(field_areas.mean()),
            sigma=field_areas.std() / field_areas.mean()
        )
        field_area = np.clip(field_area, field_areas.min(), field_areas.max())
        
        # Create field polygon (simplified as square)
        side_m = np.sqrt(field_area)
        side_deg = side_m / (111000)
        
        field_poly = box(
            field_lon - side_deg/2,
            field_lat - side_deg/2,
            field_lon + side_deg/2,
            field_lat + side_deg/2
        )
        
        predicted_fields.append({
            'cluster': cluster_id,
            'geometry': field_poly,
            'area': field_area,
            'suitability': cluster_scores.mean()
        })

predicted_gdf = gpd.GeoDataFrame(predicted_fields, crs='EPSG:4326')
print(f"✓ Generated {len(predicted_gdf)} predicted crop fields")

# ===== PHASE 6: VISUALIZATION =====
print(f"\n🎨 PHASE 6: Creating Visualizations...")
print("-" * 70)

# Map 1: Suitability Heatmap
fig, ax = plt.subplots(figsize=(14, 10))

# Reshape for heatmap
suitability_grid = np.zeros((len(lats_grid), len(lons_grid)))
for idx, (lat, lon) in enumerate(grid_points):
    i = int((lat - lat_min) / grid_res)
    j = int((lon - lon_min) / grid_res)
    if 0 <= i < len(lats_grid) and 0 <= j < len(lons_grid):
        suitability_grid[i, j] = suitability_scores[idx]

im = ax.imshow(
    suitability_grid, 
    extent=[lon_min, lon_max, lat_min, lat_max],
    origin='lower',
    cmap='RdYlGn',
    alpha=0.7
)
plt.colorbar(im, ax=ax, label='Suitability Score')

# Add forest sections
for section in sections_data:
    coords = np.array(section['coordinates'])
    ax.plot(coords[:, 1], coords[:, 0], 'b-', linewidth=2, alpha=0.6)
    ax.fill(coords[:, 1], coords[:, 0], 'blue', alpha=0.1)

# Add water locations
ax.scatter(known_water_locs[:, 1], known_water_locs[:, 0], 
          color='cyan', s=100, marker='*', edgecolors='black', label='Water bodies')

# Add cluster centers
ax.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0],
          color='red', s=200, marker='X', edgecolors='black', label='Crop zones (K=7)')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Walayar Range: Crop Field Suitability Map (K=7 Zones)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/brandonk28/milind/crop_suitability_map_k7.png', dpi=300, bbox_inches='tight')
print("✓ Saved: crop_suitability_map_k7.png")

# Map 2: Interactive Folium Map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles='OpenStreetMap'
)

# Add forest sections
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#e67e22']
for idx, section in enumerate(sections_data):
    folium.Polygon(
        locations=section['coordinates'],
        popup=section['name'],
        tooltip=section['name'],
        color=colors[idx % len(colors)],
        fill=True,
        fillColor=colors[idx % len(colors)],
        fillOpacity=0.3,
        weight=2
    ).add_to(m)

# Add predicted crop fields
feature_group_pred = folium.FeatureGroup(name='🌱 Predicted Crop Fields (K=7)', show=True)
for idx, row in predicted_gdf.iterrows():
    coords = list(row['geometry'].exterior.coords)
    folium.Polygon(
        locations=[[lat, lon] for lon, lat in coords],
        popup=f"<b>Predicted Field</b><br>Zone: {row['cluster']+1}<br>Area: {row['area']:.0f} m²<br>Suitability: {row['suitability']:.2f}",
        color='#f39c12',
        fill=True,
        fillColor='#f39c12',
        fillOpacity=0.4,
        weight=1
    ).add_to(feature_group_pred)

feature_group_pred.add_to(m)

# Add water markers
feature_group_water = folium.FeatureGroup(name='💧 Water Bodies', show=True)
for wloc in known_water_locs:
    folium.CircleMarker(
        location=wloc,
        radius=8,
        popup=f"Water source: {wloc}",
        color='#0066cc',
        fill=True,
        fillColor='#00ccff',
        fillOpacity=0.7,
        weight=2
    ).add_to(feature_group_water)

feature_group_water.add_to(m)

# Legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 340px; background-color: white; border:2px solid grey; z-index:9999; font-size:11px; padding: 10px; border-radius: 5px;">
    <div style="font-size:13px; font-weight:bold; margin-bottom:8px;">🎯 Crop Field Prediction Model (K=7)</div>
    <hr style="margin: 3px 0;">
    <div style="margin:5px 0;"><b>🟢 Forest Sections</b></div>
    <div style="font-size:10px; margin:3px 0; color:#555;">Walayar Range (6 sections)</div>
    <hr style="margin: 3px 0;">
    <div style="margin:5px 0;"><b>🟠 Predicted Fields</b></div>
    <div style="font-size:10px; margin:3px 0; color:#555;">Generated in 7 zones based on suitability</div>
    <hr style="margin: 3px 0;">
    <div style="margin:5px 0;"><b>💧 Water Bodies</b></div>
    <div style="font-size:10px; margin:3px 0; color:#555;">Known water source locations</div>
    <hr style="margin: 3px 0;">
    <div style="font-size:9px; color:#888;">Model: 7-zone suitability clustering</div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)
m.save('/Users/brandonk28/milind/predicted_crop_fields_k7.html')
print("✓ Saved: predicted_crop_fields_k7.html")

# Save predictions as GeoJSON
predicted_gdf.to_file('/Users/brandonk28/milind/predicted_crop_fields_k7.geojson', driver='GeoJSON')
print("✓ Saved: predicted_crop_fields_k7.geojson")

# ===== PHASE 7: SUMMARY STATISTICS =====
print(f"\n📈 PHASE 7: Model Summary")
print("-" * 70)

print(f"\n✅ PREDICTION RESULTS (K=7):")
print(f"  Total predicted fields: {len(predicted_gdf)}")
print(f"  Avg predicted field area: {predicted_gdf['area'].mean():,.0f} m²")
print(f"  Total predicted area: {predicted_gdf['area'].sum():,.0f} m²")
print(f"  Avg suitability score: {predicted_gdf['suitability'].mean():.3f}")

print(f"\n📊 MODEL PERFORMANCE:")
print(f"  Observed field area range: {field_areas.min():,.0f} - {field_areas.max():,.0f} m²")
print(f"  Predicted field area range: {predicted_gdf['area'].min():,.0f} - {predicted_gdf['area'].max():,.0f} m²")
print(f"  Suitability factors: Water (35%), Forest proximity (35%), Terrain (30%)")
print(f"  Number of zones: 7 (vs 5 in previous model)")

print(f"\n🗺️  OUTPUT FILES:")
print(f"  1. crop_suitability_map_k7.png - Heatmap with 7 zones")
print(f"  2. predicted_crop_fields_k7.html - Interactive map with predictions")
print(f"  3. predicted_crop_fields_k7.geojson - GeoJSON format for integration")

print("\n" + "="*70)
print("✅ CROP FIELD PREDICTION COMPLETE (K=7)")
print("="*70)
