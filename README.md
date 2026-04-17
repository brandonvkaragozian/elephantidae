# Walayar Elephant Movement Dataset

## Overview
Complete geospatial analysis dataset of elephant movement patterns in Walayar Wildlife Sanctuary, Kerala, India.

**Study Area:** 250 km² (25km E-W × 10km N-S)  
**Grid Resolution:** 500m × 500m cells  
**Total Cells:** 1,071  
**Active Cells (with visits):** 275  
**Elephant Visits:** 1,472 total  
**Trajectories:** 8 WGAN-GP trajectories at 286 points each

---

## Final Dataset

### `grid_features_complete.csv`
**Dimensions:** 1,071 rows × 29 columns

Complete grid-cell feature matrix with all computed metrics.

**Column Categories:**

**Spatial (5):**
- `cell_id` - Grid cell identifier (R###C###)
- `area_m2` - Cell area in square meters (~250,075 m²)
- `centroid_lat`, `centroid_lon` - Cell center coordinates
- `avg_points_per_visit` - Mean trajectory points per visit

**Land-Use Coverage (4):**
- `pct_forest` - Forest coverage (822 cells, mean 93.22%)
- `pct_water` - Water coverage (235 cells, mean 15.08%)
- `pct_settlements` - Settlement coverage (56 cells, mean 25.41%)
- `pct_crops` - Crop coverage (12 cells, mean 2.64%)

**Infrastructure (2):**
- `road_length_m` - Road access (326 cells)
- `rail_length_m` - Railway access (97 cells)

**Distance to Features (3):**
- `dist_to_road_m` - Nearest road distance
- `dist_to_water_m` - Nearest water body distance
- `dist_to_settlement_m` - Nearest settlement distance

**Elephant Movement (4):**
- `visit_count` - Number of elephant visits (0-24)
- `entry_count` - Number of trajectory entries
- `unique_trajectory_count` - Distinct trajectories in cell
- `pass_through_points` - Total trajectory points in cell

**Landscape Complexity (4):**
- `edge_density` - Proportion of edges to area
- `boundary_vertex_proportion` - Boundary complexity
- `crossing_intensity` - Trajectory crossing frequency
- `first_passage_frequency` - First visit frequency

**Feature Patch Counts (4):**
- `num_forest_patches` - Forest patches in cell
- `num_water_patches` - Water body patches
- `num_crop_patches` - Crop patches
- `num_settlement_patches` - Settlement patches

**Vicinity Metrics (3):**
- `visits_near_water` - Visits within proximity to water
- `visits_near_crops` - Visits within proximity to crops
- `visits_near_settlements` - Visits within proximity to settlements

---

## Source Data

### `FINAL WALAYAR MAP.kml` (2.9MB)
Main map file containing:
- 1,071 grid cells (R000-R020, C000-C050)
- 6 forest sections
- 143 water bodies
- 30 settlements
- 10 crop fields
- 320 road segments
- 137 railway segments
- 41 trajectory placemarks

### `walayar_wgan_trajectories.kml` (87KB)
Extracted WGAN-GP trajectories:
- 8 high-realism synthetic elephant trajectories
- 286 points per trajectory
- Realism scores: 0.8029 - 0.7824

---

## Processing Scripts

### `extract_kml_features.py`
Extracts feature geometries from FINAL WALAYAR MAP.kml and saves as individual JSON files for each feature type.

**Output:** `kml_*.json` (archived)

### `compute_grid_features.py`
Computes core geospatial and movement metrics:
- Grid cell extraction and validation
- Land-cover percentages (polygon-grid intersection)
- Infrastructure access (roads/railways)
- Distance to features (point-in-polygon)
- Elephant movement statistics

**Output:** `grid_features_dataset.csv` (intermediate, archived)

### `compute_advanced_grid_features.py`
Computes landscape complexity and movement behavior metrics:
- Edge density and boundary complexity
- Fragmentation and patch counts
- First-passage frequency
- Crossing intensity and trajectory diversity

**Output:** `grid_advanced_features_dataset.csv` (intermediate, archived)

### `merge_grid_datasets.py`
Merges core and advanced features on cell_id.

**Output:** `grid_features_complete.csv` (final)

---

## Data Quality Notes

1. **Forest Coverage:** 822/1,071 cells (76.8%) covered with mean 93.22% forest, reflecting the sanctuary's dense forest habitat

2. **Water Bodies:** 235 cells with water coverage (mean 15.08%), capturing scattered lakes, rivers, and water tanks

3. **Elephant Activity:** 275 active cells (25.7%) with visits; 796 zero-visit cells provide habitat avoidance information

4. **Hotspots:** R012C022 shows highest activity (24 visits) in pure forest habitat

5. **Feature Extraction:** Accurate polygon-grid intersection computation using 20×20 grid sampling for reliable area estimates

---

## Grid Dimensions

| Metric | Value |
|--------|-------|
| Rows | 21 (R000 to R020) |
| Columns | 51 (C000 to C050) |
| Total Cells | 1,071 |
| Cell Size | 500m × 500m |
| Study Area Width (E-W) | 25.0 km |
| Study Area Height (N-S) | 10.0 km |
| Total Area | 250.1 km² |
| North Boundary | 10.8414°N |
| South Boundary | 10.7516°N |
| East Boundary | 76.8539°E |
| West Boundary | 76.6253°E |

---

## Usage

Load the dataset in Python:
```python
import pandas as pd
df = pd.read_csv('grid_features_complete.csv')

# Analyze hotspots
hotspots = df[df['visit_count'] > 10].sort_values('visit_count', ascending=False)

# Habitat preference
forest_preference = df[df['visit_count'] > 0]['pct_forest'].mean()
```

Or in R:
```r
df <- read.csv('grid_features_complete.csv')
summary(df$visit_count)
```

---

## Archive

Intermediate files stored in `archive/`:
- Intermediate CSV outputs (merged into final dataset)
- Diagnostic scripts
- Feature JSON files (kml_*.json)
