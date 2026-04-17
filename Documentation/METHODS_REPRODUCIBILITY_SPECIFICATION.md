# Methods Reproducibility Specification
## Comprehensive Gaps & Implementation Details

**Document Purpose:** Identify gaps preventing full reproducibility and specify exact implementation for journal publication.

---

## 1. CRITICAL INCONSISTENCY: Model Architecture

### Current Documentation Describes:
- **Vanilla MLP-based GAN** (scikit-learn)
- Generator: MLPRegressor with layers (50, 128, 256)
- Discriminator: MLPClassifier with layers (50, 128, 64)
- SGD optimizer, learning_rate=0.0001
- Leave-One-Out CV (14 folds, 100 epochs each)

### Conflicting Reference (User mentions "Page-18 design"):
- **Conditional WGAN-GP** (unknown library)
- Grid-based environmental conditioning
- 24-step trajectory windows
- Leave-Two-Elephants-Out CV
- **STATUS: NOT FOUND IN CURRENT CODEBASE**

### Questions to Resolve:
1. **Which is the FINAL implementation?**
   - Was vanilla GAN replaced by WGAN-GP?
   - Or was "page-18" an earlier proposal?
   - Which version matches the 12-trajectory output?

2. **If WGAN-GP was replaced by vanilla GAN, why?**
   - Performance comparison data?
   - Complexity vs. accuracy tradeoffs?

---

## 2. DATA EXTRACTION & PREPROCESSING GAPS

### 2.1 OSM/KML Extraction Pipeline
**Status:** Not formally documented

**What we know from code:**
- OSM data: 4 GeoJSON files (osm_water.geojson, osm_roads.geojson, osm_settlements.geojson, osm_natural.geojson)
- Walayar map: FINAL WALAYAY MAP.kml (hand-traced, 180+ features)

**Missing specifications:**
- [ ] Tool/method used for OSM data fetch (Overpass API? Direct download?)
- [ ] OSM query parameters (geospatial bounds, feature types)
- [ ] KML tracing methodology (manual digitization? Source imagery?)
- [ ] Coordinate system & datum (WGS84? Accuracy?)
- [ ] Data collection date(s)
- [ ] Validation against recent imagery

### 2.2 Placemark Name-to-Class Conversion
**Status:** Code-only, not formally specified

**Current logic (from code lines 123-127):**
```python
elif any(major_road in name for major_road in ['nh', 'railway', 'rail ', 'train', 'highway', 'national highway', 'state highway']):
    cat = 'road'
```

**Missing detail:**
- [ ] Complete classification keyword list (why these keywords?)
- [ ] Case sensitivity treatment
- [ ] Handling of ambiguous names (e.g., "Railway Station Road" - road or railway?)
- [ ] False positive examples (names incorrectly classified?)
- [ ] Validation accuracy of keyword-based classification
- [ ] Alternative classification methods considered/rejected?

### 2.3 Feature Representation
**Status:** Partially documented in code, not in methods

**Current implementation:**
- Polygons → centroid extraction via np.mean(coords)
- Points → direct coordinates
- LineStrings → midpoint via np.mean(coords)

**Missing detail:**
- [ ] Why centroid/midpoint? Justification vs. alternatives?
- [ ] Boundary representation (for settlements, are buffer radii stored in KML?)
- [ ] How settlement radii determined (2.5km soft, 1km hard)?
- [ ] Railway representation: LineString OR point density sampling?
- [ ] Road representation: individual road linestrings OR segmented?

### 2.4 Walayar Boundary
**Status:** Mentioned as 807-vertex polygon, not detailed

**Missing detail:**
- [ ] Boundary source (official park boundary? GPS survey? Map digitization?)
- [ ] Coordinate precision (decimal places?)
- [ ] Validation against satellite imagery
- [ ] Legal/administrative boundary definition

---

## 3. TRAINING DATA SPECIFICATION GAPS

### 3.1 Kruger Elephant Trajectories
**Status:** Specified as "14 elephants, Aug 2007-Aug 2008"

**Missing detail:**
- [ ] Data source (Kruger NP records? Collars by which manufacturer?)
- [ ] GPS fix frequency (30-min sampling assumed, but verify)
- [ ] Individual elephant IDs
- [ ] Collar attachment success/failure timeline
- [ ] Data quality issues (missing data periods? Outliers?)
- [ ] Preprocessing (outlier removal? Interpolation strategy?)

### 3.2 Segment Generation
**Status:** "10-point segments" mentioned

**Missing detail:**
- [ ] Segment generation algorithm (sliding window? Non-overlapping?)
- [ ] 50% overlap claimed - how exactly implemented?
- [ ] Segment filtering (e.g., speed-based outliers?)
- [ ] Total segments generated: stated as ~26,569, but derive as (14 elephants × avg_points_per_elephant / 10 × overlap_factor)

---

## 4. PREPROCESSING & NORMALIZATION GAPS

### 4.1 Per-Trajectory Normalization
**Status:** Documented in code, not in methods

**Implementation:**
```python
MinMaxScaler fit per trajectory individually → [0, 1]
```

**Missing detail:**
- [ ] Why per-trajectory vs. global normalization?
- [ ] Handling of single-point trajectories?
- [ ] Effect on learned patterns (are ranges meaningful?)
- [ ] Denormalization back to Walayar: exact formula & precision

### 4.2 Supplementary Features
**Status:** NOT IMPLEMENTED (but may be needed)

**Missing detail:**
- [ ] Are landscape features (terrain, vegetation) incorporated?
- [ ] Are temporal features (hour, season) encoded?
- [ ] Are social features (herd size) included?

---

## 5. MODEL ARCHITECTURE SPECIFICATION GAPS (if vanilla GAN confirmed)

### 5.1 Generator Architecture
**Specified:**
- Input: [N, 20] Gaussian noise
- Dense(50, ReLU) → Dense(128, ReLU) → Dense(256, ReLU)
- Output: [N, 40] (20 points × 2 coords)

**Missing detail:**
- [ ] Batch normalization? (current: none stated)
- [ ] Dropout regularization? (current: none stated)
- [ ] Alternative architectures tested? (CNN? RNN?)
- [ ] Weight initialization method?

### 5.2 Discriminator Architecture
**Specified:**
- Input: [N, 40] real/fake trajectory
- Dense(50, ReLU) → Dense(128, ReLU) → Dense(64, ReLU)
- Output: [N, 1] binary classification

**Missing detail:**
- [ ] Sigmoid on output? (assumed yes)
- [ ] Flattening step? (input is 1D, but specify)
- [ ] Score normalization?

### 5.3 Training Dynamics
**Status:** Missing altogether

**Missing detail:**
- [ ] Generator vs. Discriminator update ratio (1:1? 5:1?)
- [ ] Loss functions (MSE? BCE? Custom?)
- [ ] Gradient clipping?
- [ ] Early stopping criteria?

---

## 6. CROSS-VALIDATION INCONSISTENCY

### Current Implementation:
- Leave-One-Out CV: 14 folds (one elephant held out per fold)
- 100 epochs per fold
- 1,400 total partial_fit() calls

### Conflicting Reference:
- "Leave-Two-Elephants-Out CV" mentioned by user
- **STATUS: NOT IN CURRENT CODE**

**Questions:**
- [ ] Was L2O attempted? Why abandoned?
- [ ] What metrics would justify L2O over LOO?

---

## 7. HYPERPARAMETER JUSTIFICATION GAPS

| Hyperparameter | Value | Source | Status |
|---|---|---|---|
| Latent dimension | 20 | "Empirical: tested 10/15/20/30" | Missing: Why stop at 30? Results table? |
| Learning rate | 0.0001 | "Conservative rate" | Missing: Ablation study? Prior work? |
| Batch size | 32 | "Standard MLP" | Missing: Tested alternatives? |
| Epochs | 100 | "Convergence on N=14" | Missing: Validation loss plateau evidence? |
| ReLU activation | - | "Standard GAN" | Missing: Why not Leaky ReLU? ELU? |
| Random seed | 42 | "Reproducibility" | ✓ Clear |
| Max generation attempts | 2000 | "Empirical" | Missing: Convergence curve? diminishing returns analysis? |

---

## 8. CONSTRAINT PARAMETER JUSTIFICATION GAPS

### Water (5 km)
**Status:** Cited "Pinter-Wollman et al. 2015"
**Missing detail:**
- [ ] Exact quote/page from paper?
- [ ] Geographic context in paper (Kruger? Africa-wide? Species verification?)
- [ ] Variability across season/habitat?
- [ ] Individual vs. herd behavior?

### Settlement (2.5 km soft, 1 km hard)
**Status:** Cited "Tumenta et al. 2010"
**Missing detail:**
- [ ] How were soft/hard thresholds differentiated?
- [ ] Context: Where in Cameroon? Habitat type?
- [ ] Are buffers from elephant behavior or injury statistics?

### Cropfields (3 km nocturnal, 2 km diurnal)
**Status:** Cited "Goswami et al. 2017"
**Missing detail:**
- [ ] Is this raiding distance or observed approach distance?
- [ ] Crop type specificity?
- [ ] Individual vs. group raiding?

### Roads (1.0 km, reduced from 1.5 km)
**Status:** Cited "Kioko et al. 2006"
**Missing detail:**
- [ ] Is this collision rate radius or avoidance distance?
- [ ] Road types: highways only? Local roads? Trails?
- [ ] Why was 1.5 km reduced to 1.0 km? (Document decision)

---

## 9. CONTEXT-AWARE ROAD CROSSING LOGIC GAPS

### 9.1 Implementation Detail
**Specified:**
- Crossing zone: 0.5 km
- Resource thresholds: 4 km for water, 3 km for crops
- Justified ratio: ≥50% of crossings must be toward resources

**Missing detail:**
- [ ] How were these thresholds chosen? (Literature? Empirical?)
- [ ] Sensitivity analysis: what if thresholds were 0.3 km / 2 km / 75%?
- [ ] Edge cases: elephant at road junction, unclear direction?

### 9.2 Behavioral Validation
**Missing detail:**
- [ ] Does context-aware logic match observed Kruger behavior?
- [ ] What % of Kruger road-crossings are resource-driven vs. other?
- [ ] Are there seasonal patterns in crossing justification?

---

## 10. OUTPUT & VALIDATION GAPS

### 10.1 Trajectory Generation Success Metrics
**Documented:**
- 12 trajectories from 2000 attempts (0.6%)
- 286 points each (~5 days)
- 85-95% Walayar containment

**Missing detail:**
- [ ] Trajectory diversity metrics (pairwise distance?)
- [ ] Spatial autocorrelation (are trajectories visually distinct?)
- [ ] Temporal realism (speed consistency? Plausible pause-move patterns?)
- [ ] Constraint compliance per trajectory (show 12×4 compliance matrix)

### 10.2 Cross-Validation Results
**Documented:**
- Mean discriminator accuracy: 58.8% ± 2.0%
- Best fold: 65.1%

**Missing detail:**
- [ ] Full fold-by-fold results table
- [ ] Per-elephant held-out accuracy (which elephant hardest to model?)
- [ ] Generator loss progression (not shown)
- [ ] Fold-to-fold variance interpretation

### 10.3 External Validation
**Status:** Not implemented

**Missing detail:**
- [ ] Comparison to Markov chain baseline?
- [ ] Comparison to simple random walk?
- [ ] Do synthetic trajectories pass human expert review?
- [ ] Do they predict observed conflict hotspots?

---

## 11. CODE & REPRODUCIBILITY GAPS

### 11.1 Source Code Documentation
**Specified:**
- gan_walayar_multiconstraint.py (available)

**Missing detail:**
- [ ] Function docstrings (parameters, returns, assumptions?)
- [ ] Inline comments for complex logic?
- [ ] Code version control (git history)?
- [ ] Required library versions (sklearn 1.x vs 0.x?)

### 11.2 Computational Requirements
**Missing detail:**
- [ ] RAM required for full training?
- [ ] GPU acceleration (or CPU-only)?
- [ ] Estimated runtime on standard hardware?
- [ ] Reproducibility testing: does code produce identical results on rerun?

---

## 12. GEOGRAPHIC & TEMPORAL CONTEXT GAPS

### 12.1 Spatial Context
**Missing detail:**
- [ ] Why Walayar specifically? (Conservation priority? Data availability?)
- [ ] Similarity to Kruger (habitat, elephant density, land use??)
- [ ] Is Africa→Asia transfer realistic?
- [ ] What assumptions break? (Seasonal patterns? Migration corridors?)

### 12.2 Temporal Context  
**Missing detail:**
- [ ] 26-year gap (2007-2024): is behavior stationarity realistic?
- [ ] Climate change impact on movement patterns?
- [ ] Human infrastructure changes (new roads, settlements)?

---

## 13. ASSUMPTIONS DOCUMENTATION GAPS

### Currently Documented (8 assumptions)
✓ Geographic transfer, Temporal stationarity, Constraint independence, Spatial representation, Distance metric, Latent prior, Segment length, Trajectory expansion

### Missing Assumptions to Document
- [ ] **Generator invertibility:** Does latent space map 1:1 to realistic movements?
- [ ] **Discriminator robustness:** Can it distinguish real vs. generated across all segments?
- [ ] **Constraint orthogonality:** Are water/settlement/crop/road constraints truly independent?
- [ ] **Spatial smoothness:** Is linear interpolation realistic vs. actual movement curvature?
- [ ] **Herd behavior:** Model assumes individual; how do groups change dynamics?

---

## 14. DECISION TRACEABILITY GAPS

**Questions to document:**
- [ ] Why MLP over CNN/RNN for trajectory generation?
- [ ] Why SGD warm_start over standard batch training?
- [ ] Why Leave-One-Out over 5-fold or stratified CV?
- [ ] Why 286-point output (over 50, 100, 500)?
- [ ] Why AND logic constraints (over weighted or soft constraints)?
- [ ] Why 2000 generation attempts (over 500, 5000)?

---

## SUMMARY: Priority Actions for Reproducibility

### TIER 1 (Must Resolve Before Submission):
1. **Clarify vanilla GAN vs. WGAN-GP discrepancy** ← CRITICAL
2. **Document exact placemark classification logic** with examples
3. **Specify constraint parameter sources** (exact papers, sections, quotes)
4. **Provide full cross-validation results table** (14 folds detailed)
5. **Document hyperparameter ablation** (at least latent_dim, learning_rate)

### TIER 2 (Strongly Recommended):
6. **Create reproducibility checklist** for others to validate
7. **Document code version history** and dependencies
8. **Provide baseline comparisons** (random walk, Markov chain)
9. **Sensitivity analysis** for key thresholds (road buffer, resource distances)
10. **Spatial analysis** of generated trajectories (diversity, coverage)

### TIER 3 (Publication Enhancement):
11. **Include per-trajectory compliance matrices**
12. **Discuss failure cases** (why did 99.4% of attempts fail?)
13. **Biological validation** against known Walayar behavior
14. **Limitations & generalization discuss** beyond Walayar

---

## Next Steps

**Questions for Author:**
1. Is the **vanilla MLP-based GAN** the final method?
2. What was the **"page-18 WGAN-GP design"**? (Earlier version? Proposed enhancement? Misunderstanding?)
3. **Hyperparameter ablation data:** Which values were tested empirically?
4. **Constraint parameter derivation:** Which papers directly support each buffer distance? (Exact sections?)
5. **Placeholder naming convention:** Can you provide the complete keyword list used for classification?

**Deliverables to Generate:**
1. Full methods section (2000-2500 words) with all gaps filled
2. Supplementary Table S1: Hyperparameter justification & ablation results
3. Supplementary Table S2: Full cross-validation results (14 folds)
4. Supplementary Table S3: Per-trajectory constraint compliance (12 trajectories × 4 constraints)
5. Reproducibility code with version pinning and minimal example
