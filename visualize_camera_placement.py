#!/usr/bin/env python3
"""
Visualize camera placement network on the Walayar Wildlife Sanctuary grid.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('final_data.csv')
cameras = pd.read_csv('camera_placement_16_cameras.csv')

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# ============================================================================
# Plot 1: Camera Network Map with Visibility Ranges
# ============================================================================

# Plot grid cells by activity
ax1.scatter(df['centroid_lon'], df['centroid_lat'], 
           c=df['visit_count'], s=20, cmap='YlOrRd', alpha=0.6, label='Elephant visits')

# Plot 40km visibility ranges for each camera
for idx, row in cameras.iterrows():
    # 40km radius in degrees (approximate)
    km_per_deg_lon = 111.32 * np.cos(np.radians(row['latitude']))
    km_per_deg_lat = 111.32
    
    radius_lon = (row['coverage_radius_km'] / km_per_deg_lon)
    radius_lat = (row['coverage_radius_km'] / km_per_deg_lat)
    
    # Draw ellipse approximation for visibility
    ellipse = patches.Ellipse((row['longitude'], row['latitude']), 
                             width=2*radius_lon, height=2*radius_lat,
                             fill=False, edgecolor='blue', linestyle='--', 
                             linewidth=0.5, alpha=0.3)
    ax1.add_patch(ellipse)

# Plot cameras
scatter = ax1.scatter(cameras['longitude'], cameras['latitude'], 
                     c=cameras['rank'], s=200, cmap='tab20', 
                     marker='*', edgecolor='black', linewidth=2,
                     label='Cameras', zorder=5)

# Add camera labels
for idx, row in cameras.iterrows():
    ax1.annotate(f"C{row['rank']}", 
                xy=(row['longitude'], row['latitude']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax1.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
ax1.set_title('16 Optimal Camera Locations with 40km Visibility Ranges', 
             fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Add colorbar for elephant activity
cbar1 = plt.colorbar(scatter, ax=ax1, label='Camera Rank (1=best)')

# ============================================================================
# Plot 2: Coverage Analysis
# ============================================================================

# Compute visibility from all cameras
visibility = np.zeros((len(df), len(cameras)))
distances_km = np.zeros((len(df), len(cameras)))

EARTH_RADIUS_KM = 6371

def haversine(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

for i, (cell_lat, cell_lon) in enumerate(zip(df['centroid_lat'], df['centroid_lon'])):
    for j, (cam_lat, cam_lon) in enumerate(zip(cameras['latitude'], cameras['longitude'])):
        dist = haversine(cell_lat, cell_lon, cam_lat, cam_lon)
        distances_km[i, j] = dist
        if dist <= 40:
            visibility[i, j] = 1

# Count coverage per cell
coverage_count = visibility.sum(axis=1)

# Plot coverage map
scatter2 = ax2.scatter(df['centroid_lon'], df['centroid_lat'],
                      c=coverage_count, s=30, cmap='RdYlGn', 
                      alpha=0.7, vmin=0, vmax=16)

# Plot cameras
ax2.scatter(cameras['longitude'], cameras['latitude'],
           marker='*', s=300, color='red', edgecolor='black', 
           linewidth=2, label='Cameras', zorder=5)

ax2.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
ax2.set_title('Grid Cell Coverage (# of cameras with line-of-sight)', 
             fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

cbar2 = plt.colorbar(scatter2, ax=ax2, label='Cameras observing cell')

# Add coverage statistics
stats_text = f"""Coverage Statistics:
━━━━━━━━━━━━━━━━━━━━━━━
Total cells: {len(df):,}
Covered cells: {(coverage_count > 0).sum():,}
Mean cameras/cell: {coverage_count.mean():.2f}
Max coverage: {int(coverage_count.max())} cameras
Min coverage: {int(coverage_count.min())} cameras

Elephant Activity:
━━━━━━━━━━━━━━━━━━━━━━━
Total visits: {df['visit_count'].sum():.0f}
Active cells: {(df['visit_count'] > 0).sum()}
Max visits/cell: {df['visit_count'].max():.0f}"""

ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('camera_placement_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved: camera_placement_visualization.png")

# ============================================================================
# Plot 3: Coverage by Rank
# ============================================================================
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

# Cumulative coverage
cumulative_coverage = np.zeros(len(cameras))
uncovered_cells = set(range(len(df)))

for rank, row in cameras.iterrows():
    camera_idx = rank
    visible_idx = np.where(distances_km[:, camera_idx] <= 40)[0]
    uncovered_cells = uncovered_cells - set(visible_idx)
    cumulative_coverage[rank] = len(df) - len(uncovered_cells)

ax3.plot(cameras['rank'], cumulative_coverage, 'o-', linewidth=2, markersize=8, color='darkblue')
ax3.fill_between(cameras['rank'], 0, cumulative_coverage, alpha=0.3, color='skyblue')
ax3.axhline(y=len(df), color='red', linestyle='--', linewidth=2, label=f'Total cells: {len(df)}')
ax3.set_xlabel('Camera Rank (1=highest priority)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cumulative Cells Covered', fontsize=11, fontweight='bold')
ax3.set_title('Cumulative Coverage by Camera Rank', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim([0, len(df) * 1.05])

# Incremental gain
incremental_gain = np.zeros(len(cameras))
uncovered = len(df)
for rank, row in cameras.iterrows():
    camera_idx = rank
    visible_idx = np.where(distances_km[:, camera_idx] <= 40)[0]
    new_coverage = len([i for i in visible_idx if coverage_count[i] == 0])  # Simplified
    incremental_gain[rank] = visible_idx.size

ax4.bar(cameras['rank'], incremental_gain, color='steelblue', edgecolor='black', linewidth=1)
ax4.set_xlabel('Camera Rank', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cells Visible from Camera', fontsize=11, fontweight='bold')
ax4.set_title('Visibility per Camera', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('camera_placement_coverage_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: camera_placement_coverage_analysis.png")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)

print(f"\nCoverage Statistics:")
print(f"  Total grid cells: {len(df)}")
print(f"  Fully covered cells: {(coverage_count > 0).sum()}")
print(f"  Mean cameras per cell: {coverage_count.mean():.2f}")
print(f"  Median cameras per cell: {np.median(coverage_count):.1f}")

print(f"\nCamera Priority Ranking:")
for idx, row in cameras.iterrows():
    visible = (distances_km[:, idx] <= 40).sum()
    print(f"  Rank {row['rank']:2d}: {row['cell_id']:>8s} - {visible:4d} cells visible")

print(f"\n✓ Visualization complete - 2 PNG files generated")
print("="*70)
