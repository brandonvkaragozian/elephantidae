#!/usr/bin/env python3
"""
Mixed Integer Programming (MIP) for Camera Placement Optimization

Problem Formulation:
- Decision variables: x_j ∈ {0,1} for each candidate camera location j
- Maximize: Σ_i (demand_i * Σ_j detectability_{i,j} * x_j)
- Subject to: Σ_j x_j ≤ K (budget constraint)

Where:
- demand_i = importance of monitoring cell i (elephant activity)
- detectability_{i,j} = visibility from camera j to cell i
- K = number of cameras (16)
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import pulp
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("MIXED INTEGER OPTIMIZATION FOR CAMERA PLACEMENT")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading dataset...")
df = pd.read_csv('final_data.csv')
n_cells = len(df)
print(f"  ✓ Loaded {n_cells} grid cells")

coords = df[['centroid_lat', 'centroid_lon']].values
cell_ids = df['cell_id'].values

# ============================================================================
# 2. COMPUTE DEMAND (IMPORTANCE) FOR EACH CELL
# ============================================================================
print("\n[2/6] Computing cell importance/demand...")

# Multi-factor importance based on elephant activity
visit_count = df['visit_count'].values
trajectory_count = df['unique_trajectory_count'].values
entry_count = df['entry_count'].values

# Normalize each component to [0, 1]
visit_norm = (visit_count - visit_count.min()) / (visit_count.max() - visit_count.min() + 1e-6)
traj_norm = (trajectory_count - trajectory_count.min()) / (trajectory_count.max() - trajectory_count.min() + 1e-6)
entry_norm = (entry_count - entry_count.min()) / (entry_count.max() - entry_count.min() + 1e-6)

# Weighted combination: prioritize visit count (frequency)
demand = 0.5 * visit_norm + 0.3 * traj_norm + 0.2 * entry_norm

# Boost demand for cells with high visitor density (avg_points_per_visit)
avg_points = df['avg_points_per_visit'].values
points_norm = (avg_points - avg_points.min()) / (avg_points.max() - avg_points.min() + 1e-6)
demand = 0.8 * demand + 0.2 * points_norm

# Normalize final demand to [0, 1]
demand = (demand - demand.min()) / (demand.max() - demand.min())

print(f"  Demand range: [{demand.min():.3f}, {demand.max():.3f}]")
print(f"  Mean demand: {demand.mean():.3f}")
print(f"  Cells with demand > 0.5: {(demand > 0.5).sum()}")
print(f"  Cells with demand > 0.8: {(demand > 0.8).sum()}")

# ============================================================================
# 3. COMPUTE VISIBILITY/DETECTABILITY MATRIX
# ============================================================================
print("\n[3/6] Computing visibility/detectability matrix...")

EARTH_RADIUS_KM = 6371
MAX_RANGE_KM = 40

def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km"""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def occlusion_factor(source_idx, target_idx):
    """Estimate occlusion between two cells"""
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
print("  Computing pairwise distances...")
distances_km = np.zeros((n_cells, n_cells))
for i in range(n_cells):
    for j in range(i, n_cells):
        dist = haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
        distances_km[i, j] = dist
        distances_km[j, i] = dist

# Compute detectability matrix (n_cells x n_cells)
# detectability[i, j] = probability camera at j can detect elephants at i
print("  Computing occlusion factors...")
detectability = np.zeros((n_cells, n_cells))

for i in range(n_cells):
    for j in range(n_cells):
        if i == j:
            detectability[i, j] = 1.0  # Perfect detection of own cell
        elif distances_km[i, j] <= MAX_RANGE_KM:
            # Within range: compute detectability as function of occlusion
            occlusion = occlusion_factor(j, i)  # Camera at j looking at i
            detectability[i, j] = max(0.0, 1.0 - occlusion)
        else:
            detectability[i, j] = 0.0  # Out of range
    
    if (i + 1) % 100 == 0:
        print(f"    Progress: {i+1}/{n_cells}")

# Binarize detectability (threshold at 0.3)
detectability_binary = (detectability > 0.3).astype(int)

print(f"  Detectability range: [{detectability.min():.3f}, {detectability.max():.3f}]")
print(f"  Mean detectability (where >0): {detectability[detectability > 0].mean():.3f}")
print(f"  Binary coverage (avg cells per camera): {detectability_binary.sum(axis=0).mean():.1f}")

# ============================================================================
# 4. FORMULATE MIP
# ============================================================================
print("\n[4/6] Formulating mixed integer program...")

N_CAMERAS = 16

# Create MIP problem
prob = pulp.LpProblem("Camera_Placement_MIP", pulp.LpMaximize)

# Decision variables: x[j] = 1 if camera placed at j, 0 otherwise
x = pulp.LpVariable.dicts("camera", range(n_cells), cat='Binary')

# Coverage variables: y[i] = coverage level for cell i
# y[i] = min(1, Σ_j detectability[i,j] * x[j])
# We approximate this with linear constraints
y = pulp.LpVariable.dicts("coverage", range(n_cells), lowBound=0, upBound=1, cat='Continuous')

# Objective: maximize weighted coverage
# Maximize Σ_i (demand_i * y_i)
prob += pulp.lpSum([demand[i] * y[i] for i in range(n_cells)])

# ============================================================================
# 5. ADD CONSTRAINTS
# ============================================================================
print("\n[5/6] Adding constraints...")

# Budget constraint: at most N_CAMERAS cameras
prob += pulp.lpSum([x[j] for j in range(n_cells)]) <= N_CAMERAS, "Budget"

# Coverage constraints: y[i] ≤ Σ_j detectability[i,j] * x[j]
# This ensures y[i] reflects whether cell i is covered by any camera
for i in range(n_cells):
    prob += y[i] <= pulp.lpSum([detectability_binary[i, j] * x[j] for j in range(n_cells)]), f"Coverage_{i}"

# Optional: Encourage placement at diverse locations (no adjacent cameras)
# This is optional and can be disabled for pure coverage maximization
# For now, we skip this to focus on coverage

print(f"  Variables: {len(x) + len(y)} (binary + continuous)")
print(f"  Constraints: 1 budget + {n_cells} coverage = {1 + n_cells}")

# ============================================================================
# 6. SOLVE MIP
# ============================================================================
print("\n[6/6] Solving MIP (this may take a minute)...")

# Use CBC solver (default in PuLP, better for large problems)
prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

# Check solution status
status = pulp.LpStatus[prob.status]
print(f"\n  Status: {status}")
print(f"  Objective value: {pulp.value(prob.objective):.3f}")

# ============================================================================
# 7. EXTRACT RESULTS
# ============================================================================
print("\n" + "="*80)
print("MIP OPTIMIZATION RESULTS")
print("="*80)

# Selected cameras
selected_cameras_mip = []
for j in range(n_cells):
    if x[j].varValue > 0.5:  # Binary decision
        selected_cameras_mip.append(j)

print(f"\n✓ Selected {len(selected_cameras_mip)} cameras:")

results_mip = []
for rank, camera_idx in enumerate(selected_cameras_mip, 1):
    lat, lon = coords[camera_idx]
    cell_id = cell_ids[camera_idx]
    
    # Compute coverage metrics
    cells_covered = np.where(detectability_binary[camera_idx] > 0)[0]
    n_cells_visible = len(cells_covered)
    
    # Weighted coverage: demand of cells this camera covers
    weighted_coverage = np.sum(demand[cells_covered])
    
    # Elephant activity in range
    visits_in_range = df.iloc[cells_covered]['visit_count'].sum()
    
    results_mip.append({
        'rank': rank,
        'cell_id': cell_id,
        'latitude': lat,
        'longitude': lon,
        'cells_visible': n_cells_visible,
        'weighted_demand_coverage': weighted_coverage,
        'elephant_visits_in_range': visits_in_range,
        'camera_idx': camera_idx
    })
    
    print(f"  {rank:2d}. {cell_id:>8s} ({lat:.6f}°N, {lon:.6f}°E)")
    print(f"       Cells visible: {n_cells_visible}")
    print(f"       Weighted demand coverage: {weighted_coverage:.3f}")
    print(f"       Elephant visits in range: {visits_in_range:.0f}")

# ============================================================================
# 8. COVERAGE ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("COVERAGE ANALYSIS")
print("-"*80)

# Compute total coverage achieved
coverage_achieved = np.zeros(n_cells)
for camera_idx in selected_cameras_mip:
    coverage_achieved = np.maximum(coverage_achieved, detectability_binary[camera_idx])

n_covered_cells = np.sum(coverage_achieved)
covered_demand = np.sum(demand[coverage_achieved > 0])
total_demand = np.sum(demand)

print(f"\nGeographic Coverage:")
print(f"  Cells covered: {n_covered_cells}/{n_cells} ({n_covered_cells/n_cells*100:.1f}%)")
print(f"  Cells not covered: {n_cells - n_covered_cells}")

print(f"\nDemand Coverage:")
print(f"  Demand covered: {covered_demand:.3f}/{total_demand:.3f} ({covered_demand/total_demand*100:.1f}%)")
print(f"  Objective value: {pulp.value(prob.objective):.3f}")

# Which cells are not covered?
uncovered_mask = coverage_achieved == 0
if uncovered_mask.sum() > 0:
    uncovered_cells_idx = np.where(uncovered_mask)[0]
    uncovered_demand = np.sum(demand[uncovered_cells_idx])
    uncovered_visits = df.iloc[uncovered_cells_idx]['visit_count'].sum()
    
    print(f"\nUncovered High-Demand Cells:")
    print(f"  Count: {len(uncovered_cells_idx)}")
    print(f"  Uncovered demand: {uncovered_demand:.3f}")
    print(f"  Elephant visits missed: {uncovered_visits:.0f}")
    
    # Top uncovered cells
    top_uncovered_idx = uncovered_cells_idx[np.argsort(-demand[uncovered_cells_idx])[:5]]
    print(f"\n  Top 5 uncovered cells by demand:")
    for idx in top_uncovered_idx:
        print(f"    {cell_ids[idx]}: demand={demand[idx]:.3f}, visits={df.iloc[idx]['visit_count']:.0f}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "-"*80)
print("SAVING RESULTS")
print("-"*80)

results_mip_df = pd.DataFrame(results_mip)
results_mip_df.to_csv('camera_placement_mip_16_cameras.csv', index=False)
print(f"  ✓ Saved: camera_placement_mip_16_cameras.csv")

# Save demand and detectability for reference
demand_df = pd.DataFrame({
    'cell_id': cell_ids,
    'demand': demand,
    'visit_count': visit_count,
    'trajectory_count': trajectory_count,
    'entry_count': entry_count
})
demand_df.to_csv('cell_demand_values.csv', index=False)
print(f"  ✓ Saved: cell_demand_values.csv")

# ============================================================================
# 10. COMPARISON WITH GREEDY
# ============================================================================
print("\n" + "-"*80)
print("COMPARISON: MIP vs GREEDY")
print("-"*80)

greedy_df = pd.read_csv('camera_placement_16_cameras.csv')
greedy_cameras_idx = [np.where((df['centroid_lat'] == lat) & (df['centroid_lon'] == lon))[0][0] 
                      for lat, lon in zip(greedy_df['latitude'], greedy_df['longitude'])]

# Compute greedy coverage
greedy_coverage = np.zeros(n_cells)
for camera_idx in greedy_cameras_idx:
    greedy_coverage = np.maximum(greedy_coverage, detectability_binary[camera_idx])

greedy_covered_demand = np.sum(demand[greedy_coverage > 0])

print(f"\nObjective Value:")
print(f"  MIP solution: {covered_demand:.3f}")
print(f"  Greedy solution: {greedy_covered_demand:.3f}")
print(f"  Improvement: {((covered_demand - greedy_covered_demand)/greedy_covered_demand*100):.1f}%")

print(f"\nGeographic Coverage:")
print(f"  MIP: {n_covered_cells}/{n_cells} cells ({n_covered_cells/n_cells*100:.1f}%)")
print(f"  Greedy: {greedy_coverage.sum()}/{n_cells} cells ({greedy_coverage.sum()/n_cells*100:.1f}%)")

print("\n" + "="*80)
print("✓ MIP OPTIMIZATION COMPLETE")
print("="*80 + "\n")

# ============================================================================
# 11. SAVE MIP MODEL SPECIFICATION
# ============================================================================
with open('mip_model_specification.txt', 'w') as f:
    f.write("MIXED INTEGER PROGRAM FOR CAMERA PLACEMENT\n")
    f.write("=" * 80 + "\n\n")
    f.write("PROBLEM FORMULATION\n")
    f.write("-" * 80 + "\n")
    f.write("Decision Variables:\n")
    f.write("  x[j] ∈ {0,1}  : Camera placed at location j\n")
    f.write("  y[i] ∈ [0,1]  : Coverage level of cell i\n\n")
    f.write("Objective Function:\n")
    f.write("  maximize: Σ_i (demand[i] * y[i])\n\n")
    f.write("Constraints:\n")
    f.write("  1. Budget: Σ_j x[j] ≤ 16\n")
    f.write("  2. Coverage: y[i] ≤ Σ_j detectability[i,j] * x[j]  ∀i\n\n")
    f.write("COMPONENTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Demand (Importance):\n")
    f.write(f"  Definition: Weighted combination of elephant activity metrics\n")
    f.write(f"  Components: 50% visit count + 30% trajectory diversity + 20% entry points\n")
    f.write(f"  Range: [0, 1] normalized\n")
    f.write(f"  Mean: {demand.mean():.3f}\n")
    f.write(f"  High-demand cells (>0.8): {(demand > 0.8).sum()}\n\n")
    f.write(f"Detectability (Visibility):\n")
    f.write(f"  Definition: Probability camera at j can detect elephants at i\n")
    f.write(f"  Formula: 1 - occlusion_factor(j → i)\n")
    f.write(f"  Occlusion: Line-of-sight interpolation\n")
    f.write(f"    Forest: 80% blocking\n")
    f.write(f"    Water: 90% blocking\n")
    f.write(f"    Settlements: 70% blocking\n")
    f.write(f"    Crops: 30% blocking\n")
    f.write(f"  Binary threshold: >0.3 (>30% clear line-of-sight)\n")
    f.write(f"  Range: [0, 1] continuous, [0,1] binary\n\n")
    f.write("SOLVER\n")
    f.write("-" * 80 + "\n")
    f.write(f"Algorithm: Branch-and-cut (CBC solver)\n")
    f.write(f"Time limit: 120 seconds\n")
    f.write(f"Status: {status}\n")
    f.write(f"Objective: {pulp.value(prob.objective):.3f}\n")

print("  ✓ Saved: mip_model_specification.txt")
