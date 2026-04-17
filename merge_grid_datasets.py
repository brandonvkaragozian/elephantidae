#!/usr/bin/env python3
"""
merge_grid_datasets.py
======================
Merge core and advanced grid feature datasets into a single comprehensive dataset.

Inputs:
  - grid_features_dataset.csv
  - grid_advanced_features_dataset.csv

Output:
  - grid_features_complete.csv (all features for all 1071 grid cells)
"""

import csv
import os
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CORE_FEATURES = os.path.join(SCRIPT_DIR, "grid_features_dataset.csv")
ADVANCED_FEATURES = os.path.join(SCRIPT_DIR, "grid_advanced_features_dataset.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "grid_features_complete.csv")

def load_csv(path):
    """Load CSV into list of dicts keyed by cell_id."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = row.get('cell_id', row.get('R000C000'))
            data[cell_id] = row
    return data

def main():
    print("Loading datasets...")
    
    # Load both datasets
    core_data = load_csv(CORE_FEATURES)
    advanced_data = load_csv(ADVANCED_FEATURES)
    
    print(f"Core features: {len(core_data)} rows")
    print(f"Advanced features: {len(advanced_data)} rows")
    
    # Merge on cell_id
    print("\nMerging on cell_id...")
    all_cell_ids = sorted(set(core_data.keys()) | set(advanced_data.keys()))
    
    merged = []
    for cell_id in all_cell_ids:
        row = OrderedDict()
        row['cell_id'] = cell_id
        
        # Add core features
        if cell_id in core_data:
            for key, val in core_data[cell_id].items():
                if key != 'cell_id':
                    row[key] = val
        
        # Add advanced features
        if cell_id in advanced_data:
            for key, val in advanced_data[cell_id].items():
                if key != 'cell_id':
                    row[key] = val
        
        merged.append(row)
    
    print(f"Merged dataset: {len(merged)} rows")
    
    # Write to CSV
    print(f"\nWriting merged dataset to {OUTPUT_CSV}...")
    if merged:
        fieldnames = list(merged[0].keys())
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)
        
        print(f"✓ Wrote {len(merged)} rows to {OUTPUT_CSV}")
        print(f"\nMerged columns ({len(fieldnames)}):")
        for i, col in enumerate(fieldnames, 1):
            print(f"  {i:2}. {col}")
        
        # Summary statistics
        visit_counts = []
        edge_densities = []
        for row in merged:
            try:
                visit_counts.append(int(row.get('visit_count', 0)))
            except:
                pass
            try:
                edge_densities.append(float(row.get('edge_density', 0)))
            except:
                pass
        
        if visit_counts:
            print(f"\nDataset summary:")
            print(f"  Cells with trajectory visits: {sum(1 for v in visit_counts if v > 0)}")
            print(f"  Max visits in a cell: {max(visit_counts)}")
            print(f"  Mean visits per cell: {sum(visit_counts) / len(visit_counts):.2f}")
        
        if edge_densities:
            print(f"  Mean edge density: {sum(edge_densities) / len(edge_densities):.6f}")

if __name__ == '__main__':
    main()
