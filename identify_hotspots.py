#!/usr/bin/env python3
"""
Identify elephant collision/hotspot zones from trajectory data.
Analyzes visit concentration, trajectory intersections, and movement patterns.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Load data
df = pd.read_csv('final_data.csv')

# Sort by visit count to see distribution
print("=" * 80)
print("HOTSPOT ANALYSIS: Elephant Collision Zones")
print("=" * 80)

print("\n1. VISIT COUNT DISTRIBUTION:")
print(f"   Total cells: {len(df)}")
print(f"   Cells with visits: {(df['visit_count'] > 0).sum()}")
print(f"   Max visits in single cell: {df['visit_count'].max()}")
print(f"   Mean visits (non-zero): {df[df['visit_count'] > 0]['visit_count'].mean():.2f}")
print(f"   Median visits (non-zero): {df[df['visit_count'] > 0]['visit_count'].median():.2f}")

# Calculate percentiles
percentiles = [50, 75, 90, 95, 99]
for p in percentiles:
    val = df['visit_count'].quantile(p/100)
    print(f"   {p}th percentile: {val:.0f} visits")

print("\n2. TOP 20 CELLS BY VISIT COUNT (Highest Congregation Zones):")
top_visits = df.nlargest(20, 'visit_count')[['cell_id', 'visit_count', 'unique_trajectory_count', 'crossing_intensity', 'centroid_lat', 'centroid_lon']]
print(top_visits.to_string(index=False))

print("\n3. TOP 20 CELLS BY TRAJECTORY INTERSECTIONS (Multi-Elephant Crossings):")
top_traj = df.nlargest(20, 'unique_trajectory_count')[['cell_id', 'unique_trajectory_count', 'visit_count', 'crossing_intensity', 'centroid_lat', 'centroid_lon']]
print(top_traj.to_string(index=False))

print("\n4. TOP 20 CELLS BY CROSSING INTENSITY (Most Complex Intersections):")
top_crossing = df.nlargest(20, 'crossing_intensity')[['cell_id', 'crossing_intensity', 'visit_count', 'unique_trajectory_count', 'centroid_lat', 'centroid_lon']]
print(top_crossing.to_string(index=False))

# Define hotspot tiers
print("\n5. HOTSPOT TIER DEFINITIONS:")

# Tier 1: CRITICAL - Very high activity zones
tier1_threshold = df['visit_count'].quantile(0.95)
tier1_cells = df[df['visit_count'] >= tier1_threshold]
print(f"\n   TIER 1 (CRITICAL HOTSPOTS): {len(tier1_cells)} cells")
print(f"   Threshold: {tier1_threshold:.0f}+ visits")
print(f"   {tier1_cells['visit_count'].sum():.0f} total visits ({tier1_cells['visit_count'].sum() / df['visit_count'].sum() * 100:.1f}%)")
print(f"   Cells: {', '.join(tier1_cells['cell_id'].values)}")

# Tier 2: HIGH - Significant activity zones  
tier2_threshold = df['visit_count'].quantile(0.85)
tier2_cells = df[(df['visit_count'] >= tier2_threshold) & (df['visit_count'] < tier1_threshold)]
print(f"\n   TIER 2 (HIGH ACTIVITY): {len(tier2_cells)} cells")
print(f"   Threshold: {tier2_threshold:.0f}-{tier1_threshold:.0f} visits")
print(f"   {tier2_cells['visit_count'].sum():.0f} total visits ({tier2_cells['visit_count'].sum() / df['visit_count'].sum() * 100:.1f}%)")
print(f"   Cells: {', '.join(tier2_cells['cell_id'].values)}")

# Tier 3: MEDIUM - Moderate activity zones
tier3_threshold = df[df['visit_count'] > 0]['visit_count'].quantile(0.70)
tier3_cells = df[(df['visit_count'] >= tier3_threshold) & (df['visit_count'] < tier2_threshold)]
print(f"\n   TIER 3 (MEDIUM ACTIVITY): {len(tier3_cells)} cells")
print(f"   Threshold: {tier3_threshold:.0f}-{tier2_threshold:.0f} visits")
print(f"   {tier3_cells['visit_count'].sum():.0f} total visits ({tier3_cells['visit_count'].sum() / df['visit_count'].sum() * 100:.1f}%)")
if len(tier3_cells) <= 20:
    print(f"   Cells: {', '.join(tier3_cells['cell_id'].values)}")
else:
    print(f"   Cells: {', '.join(tier3_cells.nlargest(20, 'visit_count')['cell_id'].values)} + {len(tier3_cells) - 20} more")

print("\n6. RECOMMENDED HOTSPOT FOCUS:")
all_hotspots = pd.concat([tier1_cells, tier2_cells, tier3_cells])
print(f"   Total hotspot cells (Tiers 1-3): {len(all_hotspots)} cells")
print(f"   Total visits in hotspots: {all_hotspots['visit_count'].sum():.0f} visits ({all_hotspots['visit_count'].sum() / df['visit_count'].sum() * 100:.1f}%)")
print(f"   → 16 cameras targeting these zones can cover {all_hotspots['visit_count'].sum() / df['visit_count'].sum() * 100:.1f}% of all activity")

# Save hotspots to file
hotspots_df = pd.concat([tier1_cells, tier2_cells, tier3_cells]).sort_values('visit_count', ascending=False)
hotspots_df.to_csv('elephant_hotspots.csv', index=False)
print(f"\n✓ Hotspot data saved to: elephant_hotspots.csv")

print("\n" + "=" * 80)
