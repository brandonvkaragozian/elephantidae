#!/usr/bin/env python3
"""
Enhanced MIP for Camera Placement with Multiple Objectives

Objectives balanced:
1. Maximize demand coverage (primary: elephant activity)
2. Maximize geographic coverage (secondary: complete surveillance)
3. Encourage spatial diversity (tertiary: redundancy & resilience)
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pulp
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ENHANCED MIP FOR CAMERA PLACEMENT (K=16 CAMERAS)")
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
print(f"  High-demand cells (>0.5): {(demand > 0.5).sum()}")

# ============================================================================
# 2. COMPUTE DETECTABILITY MATRIX
# ============================================================================
print("\n[2/5] Computing detectability matrix...")

EARTH_RADIUS_KM = 6371
MAX_RANGE_KM = 40

def haversine(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def occlusion_factor(source_idx, target_idx):
    n_samples = 10
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

# Compute detectability
detectability = np.zeros((n_cells, n_cells))
print("  Computing occlusion factors...")
for i in range(n_cells):
    for j in range(n_cells):
        if i == j:
            detectability[i, j] = 1.0
        elif distances_km[i, j] <= MAX_RANGE_KM:
            occlusion = occlusion_factor(j, i)
            detectability[i, j] = max(0.0, 1.0 - occlusion)
        else:
            detectability[i, j] = 0.0
    
    if (i + 1) % 200 == 0:
        print(f"    {i+1}/{n_cells}")

detectability_binary = (detectability > 0.3).astype(int)

print(f"  ✓ Detectability computed: binary avg={detectability_binary.sum(axis=0).mean():.1f} cells/camera")

# ============================================================================
# 3. COMPUTE SPATIAL DIVERSITY BONUS
# ============================================================================
print("\n[3/5] Computing spatial diversity scores...")

# For each cell, compute average distance to other cells
# Cells far from others get bonus (encourages spatial spread)
avg_distance_to_others = np.zeros(n_cells)
for i in range(n_cells):
    # Average distance to nearest 50 cells
    nearest_dists = np.sort(distances_km[i])[1:51]  # Skip self
    avg_distance_to_others[i] = np.mean(nearest_dists)

diversity_bonus = (avg_distance_to_others - avg_distance_to_others.min()) / \
                  (avg_distance_to_others.max() - avg_distance_to_others.min())

# Value: 70% demand + 30% spatial diversity
camera_value = 0.7 * demand + 0.3 * diversity_bonus
camera_value = (camera_value - camera_value.min()) / (camera_value.max() - camera_value.min())

print(f"  ✓ Diversity bonus computed: range=[{diversity_bonus.min():.3f}, {diversity_bonus.max():.3f}]")

# ============================================================================
# 4. FORMULATE MIP WITH HARD K CONSTRAINT
# ============================================================================
print("\n[4/5] Formulating enhanced MIP (exactly 16 cameras)...")

N_CAMERAS = 16

prob = pulp.LpProblem("Camera_Placement_Enhanced", pulp.LpMaximize)

# Decision variables
x = pulp.LpVariable.dicts("camera", range(n_cells), cat='Binary')
y = pulp.LpVariable.dicts("demand_covered", range(n_cells), lowBound=0, upBound=1, cat='Continuous')
z = pulp.LpVariable.dicts("cell_covered", range(n_cells), lowBound=0, upBound=1, cat='Continuous')

# Objective: weighted combination
# Primary: demand coverage, Secondary: geographic coverage, Tertiary: camera value (diversity)
weight_demand = 0.6
weight_geographic = 0.3
weight_camera = 0.1

prob += (weight_demand * pulp.lpSum([demand[i] * y[i] for i in range(n_cells)]) +
         weight_geographic * pulp.lpSum([z[i] for i in range(n_cells)]) +
         weight_camera * pulp.lpSum([camera_value[j] * x[j] for j in range(n_cells)]))

# Hard constraint: exactly N_CAMERAS
prob += pulp.lpSum([x[j] for j in range(n_cells)]) == N_CAMERAS, "ExactlyKCameras"

# Coverage constraints for demand
for i in range(n_cells):
    prob += y[i] <= pulp.lpSum([detectability_binary[i, j] * x[j] for j in range(n_cells)]), f"DemandCoverage_{i}"

# Coverage constraints for geography
for i in range(n_cells):
    prob += z[i] <= pulp.lpSum([detectability_binary[i, j] * x[j] for j in range(n_cells)]), f"GeographicCoverage_{i}"

print(f"  ✓ Variables: {3 * n_cells} (binary + continuous)")
print(f"  ✓ Constraints: 1 hard + 2 × {n_cells} coverage")

# ============================================================================
# 5. SOLVE MIP
# ============================================================================
print("\n[5/5] Solving enhanced MIP...")

prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))

status = pulp.LpStatus[prob.status]
print(f"\n  Status: {status}")
print(f"  Objective: {pulp.value(prob.objective):.3f}")

# ============================================================================
# EXTRACT & DISPLAY RESULTS
# ============================================================================
print("\n" + "="*80)
print("ENHANCED MIP RESULTS (K=16 CAMERAS)")
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
    
    print(f"{rank:2d}. {cell_id:>8s}  Visible: {n_cells_visible:4d}  Demand: {weighted_coverage:.3f}  Visits: {visits:6.0f}")

results_enhanced_df = pd.DataFrame(results_enhanced)
results_enhanced_df.to_csv('camera_placement_enhanced_mip_16_cameras.csv', index=False)

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
print(f"  Mean cameras/cell:   {coverage_count.mean():.2f}")
print(f"  Max coverage:        {int(coverage_count.max())} cameras")
print(f"  Min coverage:        {int(coverage_count.min())} cameras")

# Spatial analysis
camera_lats = coords[selected_mip, 0]
camera_lons = coords[selected_mip, 1]
print(f"\nSpatial distribution:")
print(f"  Latitude range:      {camera_lats.min():.6f}°N to {camera_lats.max():.6f}°N")
print(f"  Longitude range:     {camera_lons.min():.6f}°E to {camera_lons.max():.6f}°E")

# ============================================================================
# SAVE MODEL SPECIFICATION
# ============================================================================
with open('enhanced_mip_model_specification.txt', 'w') as f:
    f.write("ENHANCED MIXED INTEGER PROGRAM FOR CAMERA PLACEMENT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("PROBLEM FORMULATION\n")
    f.write("-" * 80 + "\n\n")
    f.write("Decision Variables:\n")
    f.write("  x[j] ∈ {0,1}         : Binary indicator for camera at location j\n")
    f.write("  y[i] ∈ [0,1]         : Demand coverage of cell i\n")
    f.write("  z[i] ∈ [0,1]         : Geographic coverage of cell i\n\n")
    
    f.write("Objective Function:\n")
    f.write("  maximize: 0.6 * Σ_i (demand[i] * y[i]) +\n")
    f.write("            0.3 * Σ_i z[i] +\n")
    f.write("            0.1 * Σ_j (camera_value[j] * x[j])\n\n")
    
    f.write("Constraints:\n")
    f.write("  1. Budget (HARD): Σ_j x[j] = 16 (exactly 16 cameras)\n")
    f.write("  2. Demand: y[i] ≤ Σ_j detectability[i,j] * x[j]  ∀i\n")
    f.write("  3. Geographic: z[i] ≤ Σ_j detectability[i,j] * x[j]  ∀i\n\n")
    
    f.write("COMPONENT DEFINITIONS\n")
    f.write("-" * 80 + "\n\n")
    
    f.write("Demand (Cell Importance):\n")
    f.write(f"  Formula: 0.5*visits + 0.3*trajectories + 0.2*entries\n")
    f.write(f"           then boosted by 0.2*point_density\n")
    f.write(f"  Range: [0, 1] normalized\n")
    f.write(f"  Mean: {demand.mean():.3f}\n")
    f.write(f"  High-demand (>0.5): {(demand > 0.5).sum()} cells\n\n")
    
    f.write("Detectability (Camera→Cell Visibility):\n")
    f.write(f"  Formula: 1 - occlusion_factor (line-of-sight interpolation)\n")
    f.write(f"  Binary: >0.3 threshold\n")
    f.write(f"  Mean (binary): {detectability_binary.sum(axis=0).mean():.1f} cells per camera\n\n")
    
    f.write("Camera Value (Placement Priority):\n")
    f.write(f"  Formula: 0.7*demand + 0.3*diversity_bonus\n")
    f.write(f"  Diversity: Distance to 50 nearest cells (encourages spread)\n")
    f.write(f"  Purpose: Balance targeting activity + spatial redundancy\n\n")
    
    f.write("SOLVER & STATUS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Algorithm: Branch-and-cut (CBC solver)\n")
    f.write(f"Time limit: 300 seconds\n")
    f.write(f"Status: {status}\n")
    f.write(f"Objective value: {pulp.value(prob.objective):.3f}\n")
    f.write(f"Cameras selected: {len(selected_mip)}\n")
    f.write(f"Geographic coverage: {geographic_coverage:.1f}%\n")
    f.write(f"Demand coverage: {demand_coverage:.1f}%\n")

print("\n✓ Saved: enhanced_mip_model_specification.txt")

print("\n" + "="*80)
print("✓ ENHANCED MIP COMPLETE")
print("="*80 + "\n")
