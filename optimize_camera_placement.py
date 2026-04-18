#!/usr/bin/env python3
"""
Camera Placement Optimization
Selects 16 camera locations with 360° views (40km range) to maximize elephant observation coverage.

Approach:
1. Build visibility graph considering distance + occlusion
2. Compute observation potential for each cell as camera location
3. Solve budgeted maximum coverage problem (16 cameras, 40km range)
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("CAMERA PLACEMENT OPTIMIZATION")
print("="*80)

df = pd.read_csv('final_data.csv')
print(f"\n✓ Loaded {len(df)} grid cells")

# Extract coordinates and features
coords = df[['centroid_lat', 'centroid_lon']].values
cell_ids = df['cell_id'].values

# Occlusion factors (0-1 scale: higher = more blocking)
# Forest/water/settlements block visibility more than crops
forest_pct = df['pct_forest'].values / 100.0
water_pct = df['pct_water'].values / 100.0
settlement_pct = df['pct_settlements'].values / 100.0
crops_pct = df['pct_crops'].values / 100.0

# Observation value (activity priority)
visit_count = df['visit_count'].values
trajectory_count = df['unique_trajectory_count'].values
observation_value = (visit_count * 0.7 + trajectory_count * 0.3)

print(f"  Coordinate range: {coords[:, 0].min():.4f}°N to {coords[:, 0].max():.4f}°N")
print(f"                   {coords[:, 1].min():.4f}°E to {coords[:, 1].max():.4f}°E")

# ============================================================================
# 2. BUILD VISIBILITY MODEL
# ============================================================================
print("\n" + "-"*80)
print("COMPUTING VISIBILITY GRAPH")
print("-"*80)

MAX_RANGE_KM = 20
EARTH_RADIUS_KM = 6371

def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km"""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def occlusion_factor(source_cell_idx, target_cell_idx):
    """
    Estimate occlusion between two cells.
    Uses Bresenham-like interpolation along line of sight.
    
    Returns: 0 (clear) to 1 (completely blocked)
    """
    # Simplified linear interpolation of occlusion along line
    n_samples = 10
    lats = np.linspace(coords[source_cell_idx, 0], coords[target_cell_idx, 0], n_samples)
    lons = np.linspace(coords[source_cell_idx, 1], coords[target_cell_idx, 1], n_samples)
    
    # Find nearest cells along line of sight
    occlusion_sum = 0
    for lat, lon in zip(lats[1:-1], lons[1:-1]):  # Skip endpoints
        # Find nearest grid cell
        dists = np.sqrt((coords[:, 0] - lat)**2 + (coords[:, 1] - lon)**2)
        nearest_idx = np.argmin(dists)
        
        # Accumulate occlusion (higher forest/water = more blocking)
        cell_occlusion = (forest_pct[nearest_idx] * 0.8 + 
                         water_pct[nearest_idx] * 0.9 + 
                         settlement_pct[nearest_idx] * 0.7 + 
                         crops_pct[nearest_idx] * 0.3)
        occlusion_sum += cell_occlusion
    
    return min(occlusion_sum / (n_samples - 2), 1.0)

# Compute pairwise distances
print("  Computing pairwise distances...")
n_cells = len(df)
distances_km = np.zeros((n_cells, n_cells))

for i in range(n_cells):
    for j in range(i, n_cells):
        dist = haversine_distance(coords[i, 0], coords[i, 1], 
                                  coords[j, 0], coords[j, 1])
        distances_km[i, j] = dist
        distances_km[j, i] = dist

print(f"  Distance range: {distances_km[distances_km > 0].min():.2f}km to {distances_km[distances_km > 0].max():.2f}km")

# Build visibility matrix (cells visible from each camera location)
print("  Computing occlusion factors...")
visibility = np.zeros((n_cells, n_cells))
visibility_strength = np.zeros((n_cells, n_cells))

for i in range(n_cells):
    for j in range(n_cells):
        if i == j:
            visibility[i, j] = 1.0  # Can see self
            visibility_strength[i, j] = 1.0
        elif distances_km[i, j] <= MAX_RANGE_KM:
            # Within range - compute occlusion
            occlusion = occlusion_factor(i, j)
            vis = 1.0 - occlusion
            
            visibility[i, j] = 1.0 if vis > 0.3 else 0.0  # Binary: can see or not
            visibility_strength[i, j] = vis  # Strength of observation
    
    if (i + 1) % 100 == 0:
        print(f"    Progress: {i+1}/{n_cells} cells processed")

# Statistics
visible_cells_per_camera = visibility.sum(axis=1)
print(f"\n  Visibility stats:")
print(f"    Min cells visible: {visible_cells_per_camera.min():.0f}")
print(f"    Mean cells visible: {visible_cells_per_camera.mean():.1f}")
print(f"    Max cells visible: {visible_cells_per_camera.max():.0f}")

# ============================================================================
# 3. COMPUTE CAMERA VALUE
# ============================================================================
print("\n" + "-"*80)
print("COMPUTING CAMERA PLACEMENT VALUE")
print("-"*80)

# Camera value = sum of observation values of cells it can see
camera_value = np.zeros(n_cells)
for i in range(n_cells):
    # Value = (number of visible cells) * (average observation priority)
    visible_idx = visibility[i] > 0
    camera_value[i] = np.sum(observation_value[visible_idx])

# Normalize
camera_value = camera_value / camera_value.max()

print(f"  Camera value range: {camera_value.min():.3f} to {camera_value.max():.3f}")
print(f"  Mean camera value: {camera_value.mean():.3f}")

# ============================================================================
# 4. GREEDY OPTIMIZATION (Budgeted Maximum Coverage)
# ============================================================================
print("\n" + "-"*80)
print("SOLVING BUDGETED MAXIMUM COVERAGE (GREEDY ALGORITHM)")
print("-"*80)

N_CAMERAS = 16
selected_cameras = []
covered_cells = set()
remaining_cells = set(range(n_cells))

print(f"  Selecting {N_CAMERAS} cameras to maximize coverage...")
print(f"  Initial uncovered cells: {len(remaining_cells)}")

for iteration in range(N_CAMERAS):
    # For each remaining candidate, compute marginal coverage gain
    best_camera = -1
    best_gain = -1
    best_new_covered = set()
    
    for candidate_idx in remaining_cells:
        visible_idx = np.where(visibility[candidate_idx] > 0)[0]
        new_covered = set(visible_idx) - covered_cells
        
        # Marginal gain: new cells covered weighted by observation value
        marginal_gain = np.sum(observation_value[list(new_covered)])
        
        if marginal_gain > best_gain:
            best_gain = marginal_gain
            best_camera = candidate_idx
            best_new_covered = new_covered
    
    # Add best camera
    selected_cameras.append(best_camera)
    covered_cells.update(best_new_covered)
    remaining_cells.discard(best_camera)
    
    coverage_pct = len(covered_cells) / n_cells * 100
    print(f"  Camera {iteration+1:2d}: {cell_ids[best_camera]:>8s} | " +
          f"Covers {len(best_new_covered):4d} new cells | " +
          f"Total: {len(covered_cells):4d} cells ({coverage_pct:5.1f}%)")

print(f"\n  ✓ Selected {len(selected_cameras)} cameras")
print(f"  ✓ Total cells covered: {len(covered_cells)}/{n_cells} ({len(covered_cells)/n_cells*100:.1f}%)")

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\n" + "="*80)
print("CAMERA PLACEMENT RESULTS")
print("="*80)

results = []
for rank, camera_idx in enumerate(selected_cameras, 1):
    cell_id = cell_ids[camera_idx]
    lat, lon = coords[camera_idx]
    n_visible = int(visibility[camera_idx].sum())
    val = camera_value[camera_idx]
    
    # Get top elephant activity in range
    visible_idx = np.where(visibility[camera_idx] > 0)[0]
    max_visits = df.loc[visible_idx, 'visit_count'].max()
    
    results.append({
        'rank': rank,
        'cell_id': cell_id,
        'latitude': lat,
        'longitude': lon,
        'visible_cells': n_visible,
        'camera_value': val,
        'max_visits_in_range': max_visits,
        'coverage_radius_km': MAX_RANGE_KM
    })
    
    print(f"\nCamera {rank}:")
    print(f"  Location: {cell_id} ({lat:.6f}°N, {lon:.6f}°E)")
    print(f"  Visible cells: {n_visible}")
    print(f"  Coverage value: {val:.3f}")
    print(f"  Max elephant visits in range: {max_visits:.0f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('camera_placement_16_cameras.csv', index=False)
print("\n✓ Results saved to camera_placement_16_cameras.csv")

# ============================================================================
# 6. COVERAGE ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("COVERAGE ANALYSIS")
print("-"*80)

# Which cells are NOT covered?
uncovered_cells = set(range(n_cells)) - covered_cells
print(f"\nCovered cells: {len(covered_cells)} ({len(covered_cells)/n_cells*100:.1f}%)")
print(f"Uncovered cells: {len(uncovered_cells)} ({len(uncovered_cells)/n_cells*100:.1f}%)")

if uncovered_cells:
    uncovered_visits = df.iloc[list(uncovered_cells)]['visit_count'].sum()
    total_visits = df['visit_count'].sum()
    print(f"Elephant visits in uncovered cells: {uncovered_visits:.0f}/{total_visits:.0f} ({uncovered_visits/total_visits*100:.1f}%)")

# Coverage redundancy
coverage_count = visibility[selected_cameras].sum(axis=0)
mean_coverage = coverage_count.mean()
print(f"\nMean coverage per cell: {mean_coverage:.2f} cameras")
print(f"Max coverage per cell: {int(coverage_count.max())} cameras")
print(f"Min coverage per cell: {int(coverage_count.min())} cameras")

# Spatial distribution
print("\n" + "-"*80)
print("SELECTED CAMERA SPATIAL DISTRIBUTION")
print("-"*80)

camera_lats = coords[selected_cameras, 0]
camera_lons = coords[selected_cameras, 1]

print(f"\nLatitude range: {camera_lats.min():.6f}°N to {camera_lats.max():.6f}°N")
print(f"Longitude range: {camera_lons.min():.6f}°E to {camera_lons.max():.6f}°E")
print(f"Latitude spread: {camera_lats.max() - camera_lats.min():.6f}°")
print(f"Longitude spread: {camera_lons.max() - camera_lons.min():.6f}°")

print("\n" + "="*80)
print("✓ OPTIMIZATION COMPLETE")
print("="*80)
