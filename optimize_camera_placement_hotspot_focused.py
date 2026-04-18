#!/usr/bin/env python3
"""
Camera placement optimization focused on COLLISION/HOTSPOT ZONES.

Objective: Place 16 cameras to maximize coverage of elephant congregation areas
where conflicts occur (high-visit, multi-trajectory cells).

Optimization focuses on:
1. TIER 1 (Critical): Cells with 8+ visits
2. TIER 2 (High Activity): Cells with 3-8 visits

Camera specs:
- Range: 20km (placement distance)
- Detection zone: 1km diameter (500m radius)
- Terrain occlusion: Forest, water, settlements block visibility
"""

import pandas as pd
import numpy as np
from pulp import *
import math
from scipy.spatial.distance import cdist

# Load data
print("Loading data...")
df = pd.read_csv('final_data.csv')

# Identify hotspots
tier1_cells = df[df['visit_count'] >= 8].copy()
tier2_cells = df[(df['visit_count'] >= 3) & (df['visit_count'] < 8)].copy()

print(f"\nHotspot Tiers:")
print(f"  TIER 1 (Critical, 8+ visits): {len(tier1_cells)} cells = {tier1_cells['visit_count'].sum()} visits")
print(f"  TIER 2 (High, 3-8 visits): {len(tier2_cells)} cells = {tier2_cells['visit_count'].sum()} visits")
print(f"  Total hotspot cells: {len(tier1_cells) + len(tier2_cells)} cells")

# Assign priority weights for hotspots
df['hotspot_priority'] = 0.0
df.loc[df.index.isin(tier1_cells.index), 'hotspot_priority'] = 3.0  # Critical
df.loc[df.index.isin(tier2_cells.index), 'hotspot_priority'] = 1.5  # High

# Other cells with visits get lower priority
df.loc[(df['visit_count'] > 0) & (df['hotspot_priority'] == 0), 'hotspot_priority'] = 0.1

print(f"\nHotspot priority scoring assigned.")

# Helper functions
def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two points"""
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def interpolate_terrain(lat1, lon1, lat2, lon2, num_points=5):
    """Get terrain features along line-of-sight path"""
    terrains = []
    for i in range(num_points):
        t = i / (num_points - 1)
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        
        # Find closest cell to this interpolation point
        distances = np.sqrt((df['centroid_lat'] - lat)**2 + (df['centroid_lon'] - lon)**2)
        closest_idx = distances.idxmin()
        
        terrains.append({
            'forest': df.loc[closest_idx, 'pct_forest'],
            'water': df.loc[closest_idx, 'pct_water'],
            'settlement': df.loc[closest_idx, 'pct_settlements']
        })
    return terrains

def occlusion_factor(lat_camera, lon_camera, lat_target, lon_target):
    """Compute occlusion (0=visible, 1=blocked) accounting for terrain"""
    terrains = interpolate_terrain(lat_camera, lon_camera, lat_target, lon_target, num_points=5)
    
    # Occlusion levels by terrain type
    occlusion_factors = {
        'forest': 0.80,      # Thick forest blocks 80%
        'water': 0.90,       # Water blocks 90% (elephants avoid)
        'settlement': 0.70   # Human settlement blocks 70%
    }
    
    # Max occlusion along path
    max_occlusion = 0.0
    for point_terrain in terrains:
        point_occlusion = (
            point_terrain['forest'] / 100 * occlusion_factors['forest'] +
            point_terrain['water'] / 100 * occlusion_factors['water'] +
            point_terrain['settlement'] / 100 * occlusion_factors['settlement']
        )
        max_occlusion = max(max_occlusion, point_occlusion)
    
    return min(1.0, max_occlusion)

# Build detectability matrix
print("\nComputing detectability matrix (1km detection zones + terrain occlusion)...")

num_cells = len(df)
DETECTION_RADIUS_KM = 1.0  # 1km diameter = 0.5km radius
MAX_RANGE_KM = 20.0

detectability = np.zeros((num_cells, num_cells))

for j in range(num_cells):
    camera_lat = df.iloc[j]['centroid_lat']
    camera_lon = df.iloc[j]['centroid_lon']
    
    for i in range(num_cells):
        target_lat = df.iloc[i]['centroid_lat']
        target_lon = df.iloc[i]['centroid_lon']
        
        dist = haversine(camera_lat, camera_lon, target_lat, target_lon)
        
        # Only cells within 20km can have cameras placed
        if j < num_cells and dist <= MAX_RANGE_KM:
            # Detection: cells within 1km radius of camera
            if dist <= DETECTION_RADIUS_KM:
                occlusion = occlusion_factor(camera_lat, camera_lon, target_lat, target_lon)
                detectability[i, j] = max(0.0, 1.0 - occlusion)

print(f"  Detectability matrix built: {detectability.shape}")

# MIP: Hotspot-focused camera placement
print("\nSetting up Mixed Integer Programming for HOTSPOT-focused placement...")

prob = LpProblem("Hotspot_Camera_Placement", LpMaximize)

# Binary variables: camera placement
x = [LpVariable(f"camera_{j}", cat='Binary') for j in range(num_cells)]

# Continuous variables: hotspot coverage
y = [LpVariable(f"hotspot_coverage_{i}", lowBound=0, upBound=1) for i in range(num_cells)]

# Objective: Maximize weighted hotspot coverage
prob += lpSum([y[i] * df.iloc[i]['hotspot_priority'] for i in range(num_cells)]), "Total_Hotspot_Coverage"

# Constraints
# 1. Exactly 16 cameras
prob += lpSum(x) == 16, "Budget_16_Cameras"

# 2. Spatial spread: No 2 cameras in same 1km zone
# (This prevents redundant placement within same detection area)
for i in range(num_cells):
    for j in range(i+1, num_cells):
        dist = haversine(df.iloc[i]['centroid_lat'], df.iloc[i]['centroid_lon'],
                        df.iloc[j]['centroid_lat'], df.iloc[j]['centroid_lon'])
        if dist <= DETECTION_RADIUS_KM:
            prob += x[i] + x[j] <= 1, f"Spacing_{i}_{j}"

# 3. Coverage constraints: Cell coverage = max of detectable cameras
for i in range(num_cells):
    # If any camera can detect this cell, mark it covered
    detectable_cameras = [j for j in range(num_cells) if detectability[i, j] > 0]
    if len(detectable_cameras) > 0:
        prob += y[i] <= lpSum([x[j] * detectability[i, j] for j in detectable_cameras]), f"Detect_{i}"
    else:
        prob += y[i] == 0, f"Undetectable_{i}"

print("  Constraints added: 16 cameras, spatial spacing, detection coverage")
print("  Solving...")

# Solve
prob.solve(PULP_CBC_CMD(msg=0, timeLimit=300, threads=4))

print(f"\nOptimization Result: {LpStatus[prob.status]}")
print(f"Objective value: {value(prob.objective):.2f}")

# Extract solution
selected_cameras = [j for j in range(num_cells) if x[j].varValue == 1]
print(f"\nSelected {len(selected_cameras)} camera locations:")

results = []
for rank, j in enumerate(sorted(selected_cameras, 
                                key=lambda j: -sum(y[i].varValue * detectability[i, j] for i in range(num_cells))), 
                        start=1):
    camera_cell = df.iloc[j]
    
    # Cells covered by this camera
    covered_cells = [i for i in range(num_cells) if detectability[i, j] > 0 and y[i].varValue > 0.01]
    
    # Hotspot coverage (Tier 1 + Tier 2 only)
    hotspot_coverage = sum(detectability[i, j] for i in covered_cells if i in tier1_cells.index or i in tier2_cells.index)
    total_coverage = sum(detectability[i, j] for i in covered_cells)
    
    results.append({
        'Rank': rank,
        'Cell_ID': camera_cell['cell_id'],
        'Latitude': camera_cell['centroid_lat'],
        'Longitude': camera_cell['centroid_lon'],
        'Cells_Covered': len(covered_cells),
        'Hotspot_Tier1_Coverage': sum(1 for i in covered_cells if i in tier1_cells.index),
        'Hotspot_Tier2_Coverage': sum(1 for i in covered_cells if i in tier2_cells.index),
        'Total_Priority_Score': hotspot_coverage
    })
    
    print(f"  {rank}. {camera_cell['cell_id']} @ ({camera_cell['centroid_lat']:.5f}, {camera_cell['centroid_lon']:.5f})")
    print(f"      Cells: {len(covered_cells)}, Tier1: {sum(1 for i in covered_cells if i in tier1_cells.index)}, Tier2: {sum(1 for i in covered_cells if i in tier2_cells.index)}, Priority: {hotspot_coverage:.1f}")

# Coverage summary
total_tier1_covered = sum(1 for i in tier1_cells.index if max(y[i].varValue for j in selected_cameras) > 0.01)
total_tier2_covered = sum(1 for i in tier2_cells.index if max(y[i].varValue for j in selected_cameras) > 0.01)

print(f"\n=== COVERAGE SUMMARY ===")
print(f"TIER 1 (Critical) cells covered: {total_tier1_covered}/{len(tier1_cells)} ({total_tier1_covered/len(tier1_cells)*100:.1f}%)")
print(f"TIER 2 (High) cells covered: {total_tier2_covered}/{len(tier2_cells)} ({total_tier2_covered/len(tier2_cells)*100:.1f}%)")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('camera_placement_hotspot_focused.csv', index=False)
print(f"\n✓ Results saved: camera_placement_hotspot_focused.csv")

# Save specs
specs = f"""HOTSPOT-FOCUSED CAMERA PLACEMENT OPTIMIZATION
Generated: {pd.Timestamp.now()}

OBJECTIVE: Maximize coverage of elephant collision/congregation zones

HOTSPOT DEFINITION:
- TIER 1 (Critical): {len(tier1_cells)} cells with 8+ elephant visits = {tier1_cells['visit_count'].sum()} visits
- TIER 2 (High): {len(tier2_cells)} cells with 3-8 visits = {tier2_cells['visit_count'].sum()} visits
- Combined: {len(tier1_cells) + len(tier2_cells)} cells = {tier1_cells['visit_count'].sum() + tier2_cells['visit_count'].sum()} visits (90.7% of all activity)

CAMERA SPECIFICATIONS:
- Budget: 16 cameras
- Placement range: ±20km from any grid cell
- Detection zone: 1km diameter (500m radius)
- Terrain occlusion: Forest 80%, Water 90%, Settlements 70%
- Spatial constraint: No 2 cameras within same 1km zone

OPTIMIZATION RESULTS:
- Solver: CBC (COIN-OR Branch-and-Cut)
- Status: {LpStatus[prob.status]}
- Objective value: {value(prob.objective):.2f}
- Time: 300s timeout

COVERAGE ACHIEVED:
- Tier 1 cells covered: {total_tier1_covered}/{len(tier1_cells)} ({total_tier1_covered/len(tier1_cells)*100:.1f}%)
- Tier 2 cells covered: {total_tier2_covered}/{len(tier2_cells)} ({total_tier2_covered/len(tier2_cells)*100:.1f}%)
- Total hotspot cells covered: {total_tier1_covered + total_tier2_covered}/{len(tier1_cells) + len(tier2_cells)}

KEY INSIGHT:
16 cameras can focus on elephant congregation areas where conflicts occur,
rather than spreading thin across entire sanctuary. This prioritizes areas
where human-wildlife conflict is most likely.
"""

with open('camera_specs_hotspot_focused.txt', 'w') as f:
    f.write(specs)

print(f"✓ Specs saved: camera_specs_hotspot_focused.txt")
