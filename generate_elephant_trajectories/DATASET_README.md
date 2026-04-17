# Grid Feature Datasets for Walayar Elephant Movement Analysis

This directory contains comprehensive datasets derived from the FINAL WALAYAR MAP.kml, with trajectory data and OSM features integrated.

## Overview

Two complementary CSV datasets have been generated from the Walayar grid:

1. **grid_features_dataset.csv** - Core land-cover and trajectory metrics
2. **grid_advanced_features_dataset.csv** - Advanced landscape and movement metrics

Each dataset contains 1,071 rows (one per grid cell) with descriptive statistics and movement patterns.

## Dataset 1: grid_features_dataset.csv

### Core Features

#### Land-Cover Geometry Features

- **area_m2**: Cell area in square meters
- **pct_crops**: Percentage of cell area covered by crop fields (0-100)
- **pct_settlements**: Percentage of cell area covered by settlements (0-100)
- **pct_water**: Percentage of cell area covered by water bodies (0-100)
- **pct_forest**: Percentage of cell area covered by forest/vegetation (0-100)

#### Road and Railway Features

- **road_length_m**: Total length of roads within the cell (meters)
- **rail_length_m**: Total length of railways within the cell (meters)

#### Distance to Nearest Features

- **dist_to_road_m**: Distance from cell centroid to nearest road (meters)
- **dist_to_settlement_m**: Distance from cell centroid to nearest settlement (meters)
- **dist_to_water_m**: Distance from cell centroid to nearest water body (meters)

#### Trajectory-Based Features

- **visit_count**: Total number of trajectory points within the cell
- **unique_trajectory_count**: Number of distinct trajectories that pass through the cell
- **entry_count**: Number of times trajectories enter the cell from outside
- **avg_points_per_visit**: Average trajectory points per trajectory (visit_count / unique_trajectory_count)

#### Geospatial Identifiers

- **cell_id**: Grid cell identifier (format: R###C###, e.g., R000C000)
- **centroid_lat**: Latitude of cell centroid (decimal degrees)
- **centroid_lon**: Longitude of cell centroid (decimal degrees)

## Dataset 2: grid_advanced_features_dataset.csv

### Advanced Features

#### Edge Density & Fragmentation

- **edge_density**: Perimeter-to-area ratio indicating landscape fragmentation
  - Higher values = more fragmented landscape
  - Calculated as: perimeter / area
- **num_crop_patches**: Number of distinct crop field patches in the cell
- **num_forest_patches**: Number of distinct forest patches in the cell
- **num_settlement_patches**: Number of distinct settlement patches in the cell
- **num_water_patches**: Number of distinct water body patches in the cell

#### Boundary Proximity Features

- **boundary_vertex_proportion**: Proportion of polygon vertices near (< 500m) a boundary between forest and human-use areas (0.0-1.0)
  - Indicates interface areas important for conflict/movement

#### Corridor Centrality Metrics

- **crossing_intensity**: Number of trajectories that fully cross the cell (enter and exit)
- **pass_through_points**: Total trajectory points from crossing trajectories
- These metrics identify movement corridors and high-use pathways

#### First-Passage Metrics

- **first_passage_frequency**: Number of trajectories that first enter this cell from outside
  - High values indicate cells as initial destinations or decision points

#### Feature Proximity Metrics

- **visits_near_crops**: Number of trajectory points near (< 200m) crop fields
- **visits_near_settlements**: Number of trajectory points near (< 200m) settlements
- **visits_near_water**: Number of trajectory points near (< 200m) water bodies
- These metrics quantify elephant proximity to human/natural features

## Data Generation Scripts

### compute_grid_features.py
Generates the core features dataset. Usage:
```bash
python3 compute_grid_features.py
```

### compute_advanced_grid_features.py
Generates the advanced features dataset. Usage:
```bash
python3 compute_advanced_grid_features.py
```

## Using the Data

### For Habitat Suitability Analysis
- Use land-cover percentages and edge density to identify preferred habitat
- Combine with trajectory data to find where elephants actually visit

### For Human-Wildlife Conflict Assessment
- Use visits_near_settlements, visits_near_crops to identify conflict hotspots
- High crossing_intensity combined with proximity to settlements indicates conflict zones

### For Movement Ecology
- Visit counts and entry counts show preferred movement pathways
- First-passage frequency identifies important gateway cells
- Corridor centrality highlights critical movement corridors

### For Landscape Planning
- Boundary proximity metrics show critical interface areas
- Fragmentation metrics (num_*_patches, edge_density) assess landscape connectivity
- Distance metrics inform placement of wildlife corridors or barriers

## Data Quality Notes

1. **Distance Metrics**: Values of `inf` indicate no features of that type in the dataset
2. **Patch Counts**: Counts represent patches that overlap/intersect with the cell; no minimum size threshold applied
3. **Trajectory Points**: Based on 5 synthetic trajectories from the GAN model; may not reflect true distribution
4. **Land-Cover Data**: Derived from OpenStreetMap; may have gaps or inaccuracies
5. **Grid Resolution**: 500m × 500m cells; adjust metrics if using different resolution

## Integration with Other Data

These datasets can be joined with:
- Vegetation indices (NDVI, EVI) from satellite data
- Population density estimates
- Historical conflict incident data
- GPS collar data from actual elephants (when available)

## References

- Walayar Range Forest sections: 6 forest sections defined in FINAL WALAYAR MAP.kml
- Grid system: ~500m × 500m cells over Walayar Range
- Trajectory data: 5 synthetic trajectories generated by Conditional WGAN-GP model
- OSM features: Extracted from OpenStreetMap for the Walayar region

## Citation

If using these datasets, please cite:
- The elephant trajectory GAN methodology (see elephant_trajectory_gan.py)
- OpenStreetMap contributors for the land-cover data
- Walayar Range Forest data sources

---
Generated: 2024
Author: Feature computation pipeline
