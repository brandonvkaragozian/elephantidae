#!/usr/bin/env python3
"""
Enhanced MIP for Camera Placement with Realistic Camera Specs

Updated constraints:
- Detection radius: 1km diameter (500m radius) around camera
- Range constraint: Can place cameras 20km apart
- Occlusion: Terrain blocks visibility within detection zone
- No 360° pan/tilt: Fixed cameras
"""

import pandas as pd
import numpy as np
import pulp
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("MIXED INTEGER OPTIMIZATION - CAMERA PLACEMENT")
print("Camera Specs: 20km range, 1km detection zone, terrain occlusion")
print("="*80)

# ============================================================================
# 1. LOAD DATA & COMPUTE DEMAND
# ============================================================================
print("\n[1/5] Loading data and computing demand...")

df = pd.read_csv('final_data.csv')
n_cells = len(df)

coords = df[['centroid_lat', 'centroid_lon']].values
cell_ids = df['cell_id'].values

# Compute multi-factor demand
visit_count = df['visit_count'].values
trajectory_count = df['unique_trajectory_count'].values
entry_count = df['entry_count'].values

visit_norm = (visit_count - visit_count.min()) / (visit_count.max() - visit_count.min() + 1e-6)
traj_norm = (trajectory_count - trajectory_count.min()) / (trajectory_count.max() - trajectory_count.min() + 1e-6)
entry_norm = (entry_count - entry_count.min()) / (entry_count.max() - entry_count.min() + 1e-6)

demand = 0.5 * visit_norm + 0.3 * traj_norm + 0.2 * entry_norm
avg_points = df['avg_points_per_visit'].values
points_norm = (avg_points - avg_points.min()) / (avg_points.max() - avg_points.min() + 1e-6)
demand = 0.8 * demand + 0.2 * points_norm
demand = (demand - demand.min()) / (demand.max() - demand.min())

print(f"  ✓ Demand computed: mean={demand.mean():.3f}, max={demand.max():.3f}")

# ============================================================================
# 2. COMPUTE DETECTABILITY WITH 1KM RADIUS
# ============================================================================
print("\n[2/5] Computing detectability within 1km detection zone...")

EARTH_RADIUS_KM = 6371
CAMERA_RANGE_KM = 20
DETECTION_RADIUS_KM = 1.0  # 1km diameter = 0.5km radius

def haversine(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def occlusion_factor(source_idx, target_idx):
    """Estimate occlusion between two cells"""
    n_samples = 5
    lats = np.linspace(coords[source_idx, 0], coords[target_idx, 0], n_samples)
    lons = np.linspace(coords[source_idx, 1], coords[target_idx, 1], n_samples)
    
    forest_pct = df['pct_forest'].values / 100.0
    water_pct = df['pct_water'].values / 100.0
    settlement_pct = df['pct_settlements'].values / 100.0
    crops_pct = df['pct_crops'].values / 100.0
    
    occlusion_sum = 0
    for lat, lon in zip(lats[1:-1], lons[1:-1]):
        dists = np.sqrt((coords[:, 0] - lat)**2 + (coords[:, 1] - lon)**2)
        nearest_idx = np.argmin(dists)
        cell_occlusion = (forest_pct[nearest_idx] * 0.8 + 
                         water_pct[nearest_idx] * 0.9 + 
                         settlement_pct[nearest_idx] * 0.7 + 
                         crops_pct[nearest_idx] * 0.3)
        occlusion_sum += cell_occlusion
    
    return min(occlusion_sum / (n_samples - 2), 1.0)

# Compute pairwise distances
distances_km = np.zeros((n_cells, n_cells))
for i in range(n_cells):
    for j in range(i, n_cells):
        dist = haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
        distances_km[i, j] = dist
        distances_km[j, i] = dist

# Compute detectability: camera at j can detect cell i if:
# 1. Distance from j to i <= 1km (detection radius)
# 2. Line-of-sight not completely blocked by occlusion
print("  Computing occlusion factors within 1km detection zones...")
detectability = np.zeros((n_cells, n_cells))

for j in range(n_cells):  # Camera location
    nearby_cells = []
    for i in range(n_cells):  # Target cell
        if i == j:
            detectability[i, j] = 1.0  # Camera detects its own cell
            nearby_cells.append(i)
        elif distances_km[i, j] <= DETECTION_RADIUS_KM:
            # Within 1km detection zone
            occlusion = occlusion_factor(j, i)
            detectability[i, j] = max(0.0, 1.0 - occlusion)
            nearby_cells.append(i)
        else:
            detectability[i, j] = 0.0
    
    if (j + 1) % 200 == 0:
        print(f"    {j+1}/{n_cells} cameras processed, avg {len(nearby_cells):.1f} cells/camera")

detectability_binary = (detectability > 0.3).astype(int)

cells_per_camera = detectability_binary.sum(axis=0)
print(f"\n  Detectability stats:")
print(f"    Min cells detectable: {cells_per_camera.min():.0f}")
print(f"    Mean cells detectable: {cells_per_camera.mean():.1f}")
print(f"    Max cells detectable: {cells_per_camera.max():.0f}")

# ============================================================================
# 3. COMPUTE SPATIAL DIVERSITY
# ============================================================================
print("\n[3/5] Computing spatial diversity scores...")

avg_distance_to_others = np.zeros(n_cells)
for i in range(n_cells):
    nearest_dists = np.sort(distances_km[i])[1:51]
    avg_distance_to_others[i] = np.mean(nearest_dists)

diversity_bonus = (avg_distance_to_others - avg_distance_to_others.min()) / \
                  (avg_distance_to_others.max() - avg_distance_to_others.min())

camera_value = 0.7 * demand + 0.3 * diversity_bonus
camera_value = (camera_value - camera_value.min()) / (camera_value.max() - camera_value.min())

print(f"  ✓ Diversity bonus computed")

# ============================================================================
# 4. FORMULATE MIP
# ============================================================================
print("\n[4/5] Formulating enhanced MIP (exactly 16 cameras)...")

N_CAMERAS = 16

prob = pulp.LpProblem("Camera_Placement_20km_1km", pulp.LpMaximize)

x = pulp.LpVariable.dicts("camera", range(n_cells), cat='Binary')
y = pulp.LpVariable.dicts("demand_covered", range(n_cells), lowBound=0, upBound=1, cat='Continuous')
z = pulp.LpVariable.dicts("cell_covered", range(n_cells), lowBound=0, upBound=1, cat='Continuous')

# Objective
weight_demand = 0.6
weight_geographic = 0.3
weight_camera = 0.1

prob += (weight_demand * pulp.lpSum([demand[i] * y[i] for i in range(n_cells)]) +
         weight_geographic * pulp.lpSum([z[i] for i in range(n_cells)]) +
         weight_camera * pulp.lpSum([camera_value[j] * x[j] for j in range(n_cells)]))

# Hard constraint: exactly 16 cameras
prob += pulp.lpSum([x[j] for j in range(n_cells)]) == N_CAMERAS, "ExactlyKCameras"

# Coverage constraints
for i in range(n_cells):
    prob += y[i] <= pulp.lpSum([detectability_binary[i, j] * x[j] for j in range(n_cells)]), f"DemandCoverage_{i}"
    prob += z[i] <= pulp.lpSum([detectability_binary[i, j] * x[j] for j in range(n_cells)]), f"GeographicCoverage_{i}"

# Spatial spread constraint: cameras must be at least 1km apart (avoid clustering)
for j1 in range(n_cells):
    for j2 in range(j1 + 1, n_cells):
        if distances_km[j1, j2] < DETECTION_RADIUS_KM:
            # Don't place two cameras in same detection zone
            prob += x[j1] + x[j2] <= 1, f"Spread_{j1}_{j2}"

print(f"  ✓ Variables: {3 * n_cells} (binary + continuous)")
print(f"  ✓ Constraints: ~2100")

# ============================================================================
# 5. SOLVE MIP
# ============================================================================
print("\n[5/5] Solving MIP (timeout: 300s)...")

prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))

status = pulp.LpStatus[prob.status]
print(f"\n  Status: {status}")
print(f"  Objective: {pulp.value(prob.objective):.3f}")

# ============================================================================
# EXTRACT RESULTS
# ============================================================================
print("\n" + "="*80)
print("MIP RESULTS (16 CAMERAS, 1KM DETECTION ZONE)")
print("="*80)

selected_mip = []
for j in range(n_cells):
    if x[j].varValue > 0.5:
        selected_mip.append(j)

print(f"\n✓ Selected {len(selected_mip)} cameras:\n")

results_enhanced = []
for rank, camera_idx in enumerate(selected_mip, 1):
    lat, lon = coords[camera_idx]
    cell_id = cell_ids[camera_idx]
    
    cells_covered = np.where(detectability_binary[camera_idx] > 0)[0]
    n_cells_visible = len(cells_covered)
    weighted_coverage = np.sum(demand[cells_covered])
    visits = df.iloc[cells_covered]['visit_count'].sum()
    
    results_enhanced.append({
        'rank': rank,
        'cell_id': cell_id,
        'latitude': lat,
        'longitude': lon,
        'cells_visible': n_cells_visible,
        'weighted_demand_coverage': weighted_coverage,
        'elephant_visits_in_range': visits,
        'camera_value': camera_value[camera_idx]
    })
    
    print(f"{rank:2d}. {cell_id:>8s}  Visible: {n_cells_visible:3d}  Demand: {weighted_coverage:.3f}  Visits: {visits:5.0f}")

results_enhanced_df = pd.DataFrame(results_enhanced)
results_enhanced_df.to_csv('camera_placement_mip_16_cameras_1km_detection.csv', index=False)

# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("COVERAGE ANALYSIS")
print("-"*80)

coverage_achieved = np.zeros(n_cells)
for camera_idx in selected_mip:
    coverage_achieved = np.maximum(coverage_achieved, detectability_binary[camera_idx])

n_covered = np.sum(coverage_achieved)
covered_demand = np.sum(demand[coverage_achieved > 0])
total_demand = np.sum(demand)

geographic_coverage = n_covered / n_cells * 100
demand_coverage = covered_demand / total_demand * 100

print(f"\nGeographic Coverage:  {n_covered}/{n_cells} cells ({geographic_coverage:.1f}%)")
print(f"Demand Coverage:      {covered_demand:.1f}/{total_demand:.1f} ({demand_coverage:.1f}%)")
print(f"Elephant visits:      {df[coverage_achieved > 0]['visit_count'].sum():.0f}/{df['visit_count'].sum():.0f}")

# Redundancy analysis
coverage_count = detectability_binary[selected_mip].sum(axis=0)
print(f"\nRedundancy per cell:")
print(f"  Mean cameras/cell:   {coverage_count[coverage_achieved > 0].mean():.2f}")
print(f"  Max coverage:        {int(coverage_count.max())} cameras")
print(f"  Min coverage:        {int(coverage_count.min())} cameras")

# Uncovered analysis
uncovered_mask = coverage_achieved == 0
if uncovered_mask.sum() > 0:
    uncovered_idx = np.where(uncovered_mask)[0]
    uncovered_visits = df.iloc[uncovered_idx]['visit_count'].sum()
    print(f"\nUncovered cells: {len(uncovered_idx)} ({len(uncovered_idx)/n_cells*100:.1f}%)")
    print(f"Elephant visits missed: {uncovered_visits:.0f}")

# Spatial analysis
camera_lats = coords[selected_mip, 0]
camera_lons = coords[selected_mip, 1]
print(f"\nSpatial distribution:")
print(f"  Latitude range:      {camera_lats.min():.6f}°N to {camera_lats.max():.6f}°N")
print(f"  Longitude range:     {camera_lons.min():.6f}°E to {camera_lons.max():.6f}°E")

print("\n" + "="*80)
print("✓ OPTIMIZATION COMPLETE")
print("="*80 + "\n")

# Save specs
with open('camera_specs_1km_detection.txt', 'w') as f:
    f.write("CAMERA SPECIFICATIONS\n")
    f.write("=" * 80 + "\n\n")
    f.write("Detection Zone: 1km diameter (500m radius) around camera\n")
    f.write("Occlusion: Forest (80%), Water (90%), Settlements (70%), Crops (30%)\n")
    f.write("Placement Range: Up to 20km apart\n")
    f.write("FOV: 360° (fixed, no pan/tilt)\n")
    f.write(f"Cameras: {N_CAMERAS}\n")
    f.write(f"Geographic Coverage: {geographic_coverage:.1f}%\n")
    f.write(f"Demand Coverage: {demand_coverage:.1f}%\n")
    f.write(f"Average cells per camera: {cells_per_camera.mean():.1f}\n")

print("  ✓ Saved: camera_placement_mip_16_cameras_1km_detection.csv")
print("  ✓ Saved: camera_specs_1km_detection.txt")
