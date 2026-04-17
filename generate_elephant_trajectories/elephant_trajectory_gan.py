#!/usr/bin/env python3
"""
elephant_trajectory_gan.py
===========================
Train a Conditional WGAN-GP to learn elephant movement patterns from
Kruger National Park GPS data + OSM environmental features, then generate
synthetic trajectories for the Walayar region (or any target region).

Pipeline:
  1. Build 5 km × 5 km grid over Kruger NP
  2. Extract per-cell features: water, crop, settlement, road, railway fractions
  3. Process GPS trajectories → movement vector segments (24 steps = 12 h)
  4. Train Conditional WGAN-GP with k-fold cross validation
  5. Generate trajectories on target region grid → export KML

Run:
    python elephant_trajectory_gan.py              # train on Kruger data
    python elephant_trajectory_gan.py --generate   # generate Walayar trajectories

Output:
    models/gan_fold_*.pt                           # trained model checkpoints
    generated_walayar_trajectories.kml             # synthetic trajectories
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

# Data paths
ELEPHANT_CSV      = os.path.join(SCRIPT_DIR, "ThermochronTracking Elephants Kruger 2007.csv")
KRUGER_OSM_CACHE  = os.path.join(SCRIPT_DIR, "south_africa_osm_cache.json")
WALAYAR_OSM_FEAT  = os.path.join(ROOT_DIR, "osm_features.json")
WALAYAR_OSM_ROADS = os.path.join(ROOT_DIR, "osm_roads.json")
MODEL_DIR         = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR        = os.path.join(SCRIPT_DIR, "kmls")
OUTPUT_KML        = os.path.join(OUTPUT_DIR, "walayar_synthetic_trajectories.kml")
BACKGROUND_KML    = os.path.join(ROOT_DIR, "Walayar Range Forest Map with Grid.kml")

# Trajectory filtering
TRAJECTORY_START = "2007-08-13"
TRAJECTORY_END   = "2008-08-14"

# Grid config
CELL_SIZE_KM = 1.0         # 1 km cells for fine-grained features
LAT_M        = 111320.0    # metres per degree latitude

# Kruger NP bounding box (matching south_africa_osm_kml.py)
KRUGER_BBOX = dict(min_lat=-25.45, min_lon=31.00, max_lat=-23.90, max_lon=32.07)

# Walayar bounding box (from osm_features.json)
WALAYAR_BBOX = dict(min_lat=10.7498, min_lon=76.6225, max_lat=10.9305, max_lon=76.8539)

# Model hyperparameters
SEQ_LEN      = 24          # steps per segment (24 × 30 min = 12 hours)
LATENT_DIM   = 32          # noise vector dimension
HIDDEN_DIM   = 256         # hidden layer width
N_FEATURES   = 5           # per-cell features: water, crop, settle, road, rail
NEIGHBOR_SIZE = 3           # 3×3 neighborhood
COND_DIM     = NEIGHBOR_SIZE * NEIGHBOR_SIZE * N_FEATURES   # 45
OUTPUT_DIM   = SEQ_LEN * 2                                  # 48

# Training hyperparameters
BATCH_SIZE   = 64
EPOCHS       = 100
LR_G        = 1e-4
LR_D        = 1e-4
N_CRITIC     = 5            # critic updates per generator update
LAMBDA_GP    = 10.0         # gradient penalty weight
K_FOLDS      = 3            # number of cross-validation folds

# Generation config
N_TRAJECTORIES   = 5        # number of synthetic trajectories to generate
TRAJ_SEGMENTS    = 20       # segments per trajectory (20 × 12h = 10 days)
MOVEMENT_SCALE   = 1.0      # physical step-length scaling factor for target region

# Reproducibility
SEED = 42

# ---------------------------------------------------------------------------
# GRID + FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def lon_m_per_deg(lat_deg):
    return LAT_M * math.cos(math.radians(lat_deg))


def create_grid(bbox, cell_km=CELL_SIZE_KM):
    """
    Create a grid of cells over a bounding box.
    Returns:
      grid_info: dict with rows, cols, lat/lon bounds, cell dimensions
      cells: 2D numpy array of shape (rows, cols) — each entry is (lat_min, lon_min, lat_max, lon_max)
    """
    mid_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
    dlat = (cell_km * 1000) / LAT_M                     # degrees latitude per cell
    dlon = (cell_km * 1000) / lon_m_per_deg(mid_lat)    # degrees longitude per cell

    n_rows = int(math.ceil((bbox["max_lat"] - bbox["min_lat"]) / dlat))
    n_cols = int(math.ceil((bbox["max_lon"] - bbox["min_lon"]) / dlon))

    cells = np.zeros((n_rows, n_cols, 4))  # (lat_min, lon_min, lat_max, lon_max)
    for r in range(n_rows):
        for c in range(n_cols):
            lat0 = bbox["min_lat"] + r * dlat
            lon0 = bbox["min_lon"] + c * dlon
            cells[r, c] = [lat0, lon0, lat0 + dlat, lon0 + dlon]

    grid_info = {
        "rows": n_rows, "cols": n_cols,
        "dlat": dlat, "dlon": dlon,
        "mid_lat": mid_lat,
        "bbox": bbox,
    }
    print(f"  Grid: {n_rows} rows × {n_cols} cols = {n_rows * n_cols} cells "
          f"({cell_km} km × {cell_km} km)")
    return grid_info, cells


def _coord_to_cell(lat, lon, grid_info):
    """Map a (lat, lon) point to grid (row, col). Clamps to valid range."""
    r = int((lat - grid_info["bbox"]["min_lat"]) / grid_info["dlat"])
    c = int((lon - grid_info["bbox"]["min_lon"]) / grid_info["dlon"])
    r = max(0, min(r, grid_info["rows"] - 1))
    c = max(0, min(c, grid_info["cols"] - 1))
    return r, c


def _polygon_area_km2(coords, mid_lat):
    """Shoelace area in km² for [(lon, lat), ...] coords."""
    if len(coords) < 3:
        return 0.0
    km_lat = LAT_M / 1000
    km_lon = lon_m_per_deg(mid_lat) / 1000
    pts = [(c[0] * km_lon, c[1] * km_lat) for c in coords]
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _extract_polygon_features(polygons, feature_idx, grid_info, features, mid_lat):
    """
    Fast polygon → grid cell assignment using centroid mapping.
    For each polygon, compute its area and assign to the cell containing
    its centroid. Area fraction = polygon_area / cell_area.
    """
    cell_area_km2 = CELL_SIZE_KM * CELL_SIZE_KM
    count = 0
    for poly in polygons:
        coords = poly if isinstance(poly, list) else poly.get("coords", [])
        if len(coords) < 3:
            continue
        # Centroid
        clat = sum(c[1] for c in coords) / len(coords)
        clon = sum(c[0] for c in coords) / len(coords)
        r, c = _coord_to_cell(clat, clon, grid_info)

        # Estimate area
        area_km2 = _polygon_area_km2(coords, mid_lat)
        frac = min(area_km2 / cell_area_km2, 1.0)
        features[r, c, feature_idx] += frac
        count += 1
    return count


def _extract_line_features(lines, feature_idx, grid_info, features, mid_lat):
    """
    Fast line → grid cell assignment.
    For each line segment, compute its midpoint → cell, and add its
    length (km/km²) to that cell.
    """
    cell_area_km2 = CELL_SIZE_KM * CELL_SIZE_KM
    km_lat = LAT_M / 1000
    km_lon = lon_m_per_deg(mid_lat) / 1000
    count = 0

    for line in lines:
        coords = line if isinstance(line, list) else line.get("coords", [])
        if len(coords) < 2:
            continue
        # Process each segment of the polyline
        for i in range(len(coords) - 1):
            mid_pt_lat = (coords[i][1] + coords[i+1][1]) / 2
            mid_pt_lon = (coords[i][0] + coords[i+1][0]) / 2
            r, c = _coord_to_cell(mid_pt_lat, mid_pt_lon, grid_info)

            # Segment length in km
            dy = (coords[i+1][1] - coords[i][1]) * km_lat
            dx = (coords[i+1][0] - coords[i][0]) * km_lon
            seg_km = math.sqrt(dx*dx + dy*dy)
            features[r, c, feature_idx] += seg_km / cell_area_km2
        count += 1
    return count


def extract_kruger_features(osm_cache_path, grid_info, cells):
    """
    Extract per-cell features from Kruger OSM cache.
    Fast centroid-based approach: O(n_features) instead of O(n_features × n_cells).
    Returns features array of shape (rows, cols, N_FEATURES).
    """
    with open(osm_cache_path) as f:
        osm = json.load(f)

    rows, cols = grid_info["rows"], grid_info["cols"]
    mid_lat = grid_info["mid_lat"]
    features = np.zeros((rows, cols, N_FEATURES))
    # Feature indices: 0=water, 1=crop, 2=settlement, 3=road_density, 4=rail_density

    # Water polygons (only closed ones)
    water_polys = [f for f in osm.get("water", [])
                   if len(f.get("coords", [])) >= 4]
    n = _extract_polygon_features(
        [f["coords"] for f in water_polys], 0, grid_info, features, mid_lat)
    print(f"  Water:       {n} polygon features mapped")

    # Crop polygons
    crop_polys = [f for f in osm.get("crops", [])
                  if len(f.get("coords", [])) >= 4]
    n = _extract_polygon_features(
        [f["coords"] for f in crop_polys], 1, grid_info, features, mid_lat)
    print(f"  Crops:       {n} polygon features mapped")

    # Settlement polygons
    settle_polys = [f for f in osm.get("settlements", [])
                    if len(f.get("coords", [])) >= 4]
    n = _extract_polygon_features(
        [f["coords"] for f in settle_polys], 2, grid_info, features, mid_lat)
    print(f"  Settlements: {n} polygon features mapped")

    # Settlement points: add a small area fraction based on place type
    settle_buf = {"city": 0.15, "town": 0.08, "suburb": 0.05,
                  "village": 0.03, "hamlet": 0.01}
    sp_count = 0
    for sp in osm.get("settlement_points", []):
        r, c = _coord_to_cell(sp["lat"], sp["lon"], grid_info)
        frac = settle_buf.get(sp.get("place_type", ""), 0.02)
        features[r, c, 2] = max(features[r, c, 2], frac)
        sp_count += 1
    print(f"  Settle pts:  {sp_count} points mapped")

    # Roads
    n = _extract_line_features(
        osm.get("roads", []), 3, grid_info, features, mid_lat)
    print(f"  Roads:       {n} line features mapped")

    # Railways
    n = _extract_line_features(
        osm.get("railways", []), 4, grid_info, features, mid_lat)
    print(f"  Railways:    {n} line features mapped")

    features[:, :, :3] = np.clip(features[:, :, :3], 0, 1)
    return features


def extract_walayar_features(feat_path, roads_path, grid_info, cells):
    """
    Extract per-cell features from Walayar OSM data (different format).
    Returns features array of shape (rows, cols, N_FEATURES).
    """
    with open(feat_path) as f:
        osm_feat = json.load(f)
    with open(roads_path) as f:
        osm_roads = json.load(f)

    rows, cols = grid_info["rows"], grid_info["cols"]
    mid_lat = grid_info["mid_lat"]
    features = np.zeros((rows, cols, N_FEATURES))

    # Water polygons
    water_polys = [p["coords"] for p in osm_feat.get("water_polygons", [])
                   if len(p.get("coords", [])) >= 3]
    n = _extract_polygon_features(water_polys, 0, grid_info, features, mid_lat)
    print(f"  Water:       {n} polygon features mapped")

    # Crop polygons
    crop_polys = [p["coords"] for p in osm_feat.get("crop_polygons", [])
                  if len(p.get("coords", [])) >= 3]
    n = _extract_polygon_features(crop_polys, 1, grid_info, features, mid_lat)
    print(f"  Crops:       {n} polygon features mapped")

    # Settlements (point-based with buffer)
    settle_buf = {"city": 0.15, "town": 0.08, "suburb": 0.05,
                  "village": 0.03, "hamlet": 0.01}
    sp_count = 0
    for s in osm_feat.get("settlements", []):
        r, c = _coord_to_cell(s["lat"], s["lon"], grid_info)
        frac = settle_buf.get(s.get("place_type", ""), 0.02)
        features[r, c, 2] = max(features[r, c, 2], frac)
        sp_count += 1
    print(f"  Settle pts:  {sp_count} points mapped")

    # Roads
    n = _extract_line_features(
        osm_roads.get("roads", []), 3, grid_info, features, mid_lat)
    print(f"  Roads:       {n} line features mapped")

    # Railways
    n = _extract_line_features(
        osm_roads.get("railways", []), 4, grid_info, features, mid_lat)
    print(f"  Railways:    {n} line features mapped")

    features[:, :, :3] = np.clip(features[:, :, :3], 0, 1)
    return features


# ---------------------------------------------------------------------------
# TRAJECTORY PROCESSING
# ---------------------------------------------------------------------------

def load_trajectories(csv_file, start_date, end_date):
    """Load all elephant trajectories within date range."""
    tracks = OrderedDict()
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")
            if ts < start_date or ts > end_date:
                continue
            eid = row["individual-local-identifier"]
            try:
                lon = float(row["location-long"])
                lat = float(row["location-lat"])
                tracks.setdefault(eid, []).append((lon, lat, ts))
            except (ValueError, KeyError):
                pass

    print(f"  Loaded {len(tracks)} elephants:")
    for eid, pts in tracks.items():
        print(f"    {eid}: {len(pts)} points")
    return tracks


def trajectories_to_segments(tracks, grid_info, cells, features):
    """
    Convert GPS tracks into training segments.

    Each segment is:
      - movement: (SEQ_LEN, 2) array of (dx_km, dy_km)
      - condition: (NEIGHBOR_SIZE^2 * N_FEATURES,) flattened 3×3 neighborhood features
      - elephant_id: string

    Returns list of (movement, condition, elephant_id).
    """
    mid_lat = grid_info["mid_lat"]
    rows, cols = grid_info["rows"], grid_info["cols"]
    dlat, dlon = grid_info["dlat"], grid_info["dlon"]
    bbox = grid_info["bbox"]

    km_per_dlat = LAT_M / 1000          # km per degree lat
    km_per_dlon = lon_m_per_deg(mid_lat) / 1000  # km per degree lon

    segments = []

    for eid, pts in tracks.items():
        # Sort by timestamp
        pts.sort(key=lambda x: x[2])

        # Compute movement vectors in km
        movements = []
        for i in range(1, len(pts)):
            dx_km = (pts[i][0] - pts[i-1][0]) * km_per_dlon
            dy_km = (pts[i][1] - pts[i-1][1]) * km_per_dlat
            movements.append((dx_km, dy_km, pts[i][0], pts[i][1]))

        # Skip if too short for gaps — detect gaps > 2 hours (4 steps)
        # Create windows avoiding gaps
        # Parse timestamps to detect gaps
        gap_indices = set()
        for i in range(1, len(pts)):
            # Simple gap detection: if step length > 10 km, likely a GPS gap
            if len(movements) > i-1:
                dx, dy = movements[i-1][0], movements[i-1][1]
                if math.sqrt(dx*dx + dy*dy) > 10:
                    gap_indices.add(i-1)

        # Sliding window
        stride = SEQ_LEN // 2   # 50% overlap
        i = 0
        while i + SEQ_LEN <= len(movements):
            # Check for gaps in this window
            has_gap = any(j in gap_indices for j in range(i, i + SEQ_LEN))
            if has_gap:
                i += 1
                continue

            # Movement segment
            seg = np.array([(m[0], m[1]) for m in movements[i:i+SEQ_LEN]])

            # Starting position → grid cell
            start_lon, start_lat = movements[i][2], movements[i][3]
            r = int((start_lat - bbox["min_lat"]) / dlat)
            c = int((start_lon - bbox["min_lon"]) / dlon)
            r = max(0, min(r, rows - 1))
            c = max(0, min(c, cols - 1))

            # Extract 3×3 neighborhood features (with padding)
            pad = NEIGHBOR_SIZE // 2
            neighborhood = np.zeros((NEIGHBOR_SIZE, NEIGHBOR_SIZE, N_FEATURES))
            for dr in range(-pad, pad + 1):
                for dc in range(-pad, pad + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighborhood[dr + pad, dc + pad] = features[nr, nc]

            condition = neighborhood.flatten()
            segments.append((seg, condition, eid))
            i += stride

    print(f"  Created {len(segments)} training segments from {len(tracks)} elephants")
    return segments


# ---------------------------------------------------------------------------
# WGAN-GP MODEL
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Conditional generator: noise + env_features → movement segment."""

    def __init__(self, latent_dim=LATENT_DIM, cond_dim=COND_DIM,
                 hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
                                                                                                                                                                                            super().__init__()
                                                                                                                                                                                            self.net = nn.Sequential(
                                                                                                                                                                                                nn.Linear(latent_dim + cond_dim, hidden_dim),
                                                                                                                                                                                                nn.LayerNorm(hidden_dim),
                                                                                                                                                                                                nn.LeakyReLU(0.2),
                                                                                                                                                                                                nn.Linear(hidden_dim, hidden_dim),
                                                                                                                                                                                                nn.LayerNorm(hidden_dim),
                                                                                                                                                                                                nn.LeakyReLU(0.2),
                                                                                                                                                                                                nn.Linear(hidden_dim, hidden_dim),
                                                                                                                                                                                                nn.LayerNorm(hidden_dim),
                                                                                                                                                                                                nn.LeakyReLU(0.2),
                                                                                                                                                                                                nn.Linear(hidden_dim, output_dim),
                                                                                                                                                                                            )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        out = self.net(x)
        return out.view(-1, SEQ_LEN, 2)


class Critic(nn.Module):
    """Conditional critic: movement segment + env_features → realness score."""

    def __init__(self, input_dim=OUTPUT_DIM, cond_dim=COND_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, cond):
        x_flat = x.view(x.size(0), -1)
        inp = torch.cat([x_flat, cond], dim=1)
        return self.net(inp)


def compute_gradient_penalty(critic, real, fake, cond, device):
    """WGAN-GP gradient penalty."""
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = critic(interpolated, cond)

    gradients = autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, segments, stats=None):
        """
        segments: list of (movement_array, condition_array, elephant_id)
        stats: dict with movement_mean, movement_std, cond_mean, cond_std
               if None, compute from this data.
        """
        self.movements = np.array([s[0] for s in segments], dtype=np.float32)
        self.conditions = np.array([s[1] for s in segments], dtype=np.float32)
        self.elephant_ids = [s[2] for s in segments]

        if stats is None:
            self.stats = {
                "mov_mean": self.movements.mean(axis=(0, 1)),
                "mov_std":  self.movements.std(axis=(0, 1)) + 1e-8,
                "cond_mean": self.conditions.mean(axis=0),
                "cond_std":  self.conditions.std(axis=0) + 1e-8,
            }
        else:
            self.stats = stats

        # Normalize
        self.movements_norm = (self.movements - self.stats["mov_mean"]) / self.stats["mov_std"]
        self.conditions_norm = (self.conditions - self.stats["cond_mean"]) / self.stats["cond_std"]

    def __len__(self):
        return len(self.movements_norm)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.movements_norm[idx]),
            torch.tensor(self.conditions_norm[idx]),
        )


def train_fold(train_segments, val_segments, fold_idx, device, epochs=EPOCHS):
    """Train one fold of the WGAN-GP. Returns trained generator + stats."""

    train_ds = SegmentDataset(train_segments)
    val_ds = SegmentDataset(val_segments, stats=train_ds.stats)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    G = Generator().to(device)
    D = Critic().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

    print(f"\n  --- Fold {fold_idx + 1}/{K_FOLDS} ---")
    print(f"  Train: {len(train_ds)} segments, Val: {len(val_ds)} segments")

    best_w_dist = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        G.train()
        D.train()

        d_losses, g_losses, w_dists = [], [], []

        for batch_idx, (real_mov, cond) in enumerate(train_loader):
            real_mov = real_mov.to(device)
            cond = cond.to(device)
            bs = real_mov.size(0)

            # --- Train Critic ---
            for _ in range(N_CRITIC):
                z = torch.randn(bs, LATENT_DIM, device=device)
                fake_mov = G(z, cond).detach()

                d_real = D(real_mov, cond)
                d_fake = D(fake_mov, cond)
                gp = compute_gradient_penalty(D, real_mov, fake_mov, cond, device)

                d_loss = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp
                w_dist = d_real.mean() - d_fake.mean()

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            d_losses.append(d_loss.item())
            w_dists.append(w_dist.item())

            # --- Train Generator ---
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_mov = G(z, cond)
            g_loss = -D(fake_mov, cond).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_losses.append(g_loss.item())

        # Epoch logging
        avg_d = np.mean(d_losses)
        avg_g = np.mean(g_losses)
        avg_w = np.mean(w_dists)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Validation: compute step-length distribution match
            val_metrics = evaluate_model(G, val_ds, device)
            print(f"  Epoch {epoch+1:>4d}/{epochs} | "
                  f"D: {avg_d:>8.3f} | G: {avg_g:>8.3f} | "
                  f"W-dist: {avg_w:>7.3f} | "
                  f"Val step-len KS: {val_metrics['ks_step']:.3f} | "
                  f"Val angle KS: {val_metrics['ks_angle']:.3f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"gan_fold_{fold_idx}.pt")
    torch.save({
        "generator": G.state_dict(),
        "critic": D.state_dict(),
        "stats": train_ds.stats,
    }, save_path)
    print(f"  Saved → {save_path}")

    return G, train_ds.stats


def evaluate_model(generator, dataset, device):
    """Evaluate generator against real data using distribution similarity."""
    generator.eval()

    with torch.no_grad():
        # Generate same number of samples as dataset
        n = len(dataset)
        # Use random conditions from the dataset
        indices = list(range(n))
        random.shuffle(indices)

        real_steps = []
        fake_steps = []
        real_angles = []
        fake_angles = []

        batch = min(n, 512)
        for start in range(0, n, batch):
            end = min(start + batch, n)
            idx_batch = indices[start:end]
            bs = end - start

            real_mov = torch.stack([dataset[i][0] for i in idx_batch]).to(device)
            cond = torch.stack([dataset[i][1] for i in idx_batch]).to(device)
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_mov = generator(z, cond)

            # Step lengths
            real_sl = torch.sqrt((real_mov ** 2).sum(dim=2)).cpu().numpy().flatten()
            fake_sl = torch.sqrt((fake_mov ** 2).sum(dim=2)).cpu().numpy().flatten()
            real_steps.extend(real_sl)
            fake_steps.extend(fake_sl)

            # Turning angles
            for mov in [real_mov, fake_mov]:
                m = mov.cpu().numpy()
                for b in range(m.shape[0]):
                    for t in range(1, m.shape[1]):
                        dx0, dy0 = m[b, t-1]
                        dx1, dy1 = m[b, t]
                        angle = math.atan2(dy1, dx1) - math.atan2(dy0, dx0)
                        angle = (angle + math.pi) % (2 * math.pi) - math.pi
                        if mov is real_mov:
                            real_angles.append(angle)
                        else:
                            fake_angles.append(angle)

    # KS statistic
    from scipy.stats import ks_2samp
    ks_step = ks_2samp(real_steps[:5000], fake_steps[:5000]).statistic
    ks_angle = ks_2samp(real_angles[:5000], fake_angles[:5000]).statistic

    generator.train()
    return {"ks_step": ks_step, "ks_angle": ks_angle}


def train_kfold(segments, device):
    """K-fold cross validation, splitting by elephant."""
    # Group segments by elephant
    elephant_ids = list(OrderedDict.fromkeys(s[2] for s in segments))
    n_elephants = len(elephant_ids)
    print(f"\n  K-fold CV: {K_FOLDS} folds over {n_elephants} elephants")

    random.seed(SEED)
    random.shuffle(elephant_ids)

    fold_size = max(1, n_elephants // K_FOLDS)
    all_generators = []

    for fold in range(K_FOLDS):
        # Held-out elephants for this fold
        start = fold * fold_size
        end = start + fold_size if fold < K_FOLDS - 1 else n_elephants
        val_ids = set(elephant_ids[start:end])
        train_ids = set(elephant_ids) - val_ids

        train_segs = [s for s in segments if s[2] in train_ids]
        val_segs = [s for s in segments if s[2] in val_ids]

        if len(val_segs) == 0 or len(train_segs) < BATCH_SIZE:
            print(f"  Fold {fold + 1}: skipping (insufficient data)")
            continue

        print(f"\n  Fold {fold+1}: train on {len(train_ids)} elephants "
              f"({', '.join(sorted(train_ids))})")
        print(f"           val on {len(val_ids)} elephants "
              f"({', '.join(sorted(val_ids))})")

        gen, stats = train_fold(train_segs, val_segs, fold, device)
        all_generators.append((gen, stats))

    return all_generators


# ---------------------------------------------------------------------------
# TRAJECTORY GENERATION
# ---------------------------------------------------------------------------

def _get_edge_starts(grid_info, target_features):
    """
    Get starting positions along the edges of the grid.
    Elephants approach FROM the forest periphery and navigate INTO the
    region with crops/settlements. Returns list of (row, col) edge cells
    weighted toward forested (low-settlement) edges.
    """
    rows, cols = grid_info["rows"], grid_info["cols"]
    edge_cells = []

    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                settle_frac = target_features[r, c, 2]
                if settle_frac < 0.1:
                    edge_cells.append((r, c))

    if not edge_cells:
        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    edge_cells.append((r, c))

    return edge_cells


def generate_trajectories(generator, stats, target_features, target_grid_info,
                          target_cells, train_grid_info,
                          n_traj=N_TRAJECTORIES,
                          n_segments=TRAJ_SEGMENTS, device="cpu"):
    """
    Generate synthetic trajectories on a target region grid.
    Elephants start at grid edges (forest) and move through the region.

    Movement vectors from the GAN are scaled by the ratio of target/source
    region sizes so step lengths are appropriate for the target area.
    """
    generator.eval()
    rows, cols = target_grid_info["rows"], target_grid_info["cols"]
    dlat, dlon = target_grid_info["dlat"], target_grid_info["dlon"]
    bbox = target_grid_info["bbox"]
    mid_lat = target_grid_info["mid_lat"]

    km_per_dlat = LAT_M / 1000
    km_per_dlon = lon_m_per_deg(mid_lat) / 1000

    mov_mean = stats["mov_mean"]
    mov_std = stats["mov_std"]
    cond_mean = stats["cond_mean"]
    cond_std = stats["cond_std"]

    # Elephant movement is fundamentally driven by animal kinematics and time interval,
    # not by the geographic extent of the target map. Use training-derived step lengths
    # directly, with a small physical scaling factor for habitat differences if needed.
    scale_factor = MOVEMENT_SCALE
    print(f"  Movement magnitude factor: {scale_factor:.3f}")

    edge_cells = _get_edge_starts(target_grid_info, target_features)
    print(f"  Available edge start cells: {len(edge_cells)}")

    trajectories = []

    for t_idx in range(n_traj):
        start_r, start_c = edge_cells[np.random.randint(len(edge_cells))]

        start_lat = bbox["min_lat"] + (start_r + 0.5) * dlat
        start_lon = bbox["min_lon"] + (start_c + 0.5) * dlon

        path = [(start_lon, start_lat)]
        cur_lat, cur_lon = start_lat, start_lon

        with torch.no_grad():
            for seg_idx in range(n_segments):
                r = int((cur_lat - bbox["min_lat"]) / dlat)
                c = int((cur_lon - bbox["min_lon"]) / dlon)
                r = max(0, min(r, rows - 1))
                c = max(0, min(c, cols - 1))

                # 3×3 neighborhood
                pad = NEIGHBOR_SIZE // 2
                neighborhood = np.zeros((NEIGHBOR_SIZE, NEIGHBOR_SIZE, N_FEATURES))
                for dr in range(-pad, pad + 1):
                    for dc in range(-pad, pad + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighborhood[dr + pad, dc + pad] = target_features[nr, nc]

                cond = neighborhood.flatten()
                cond_norm = (cond - cond_mean) / cond_std
                cond_t = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).to(device)

                z = torch.randn(1, LATENT_DIM, device=device)
                fake_seg = generator(z, cond_t).cpu().numpy()[0]  # (SEQ_LEN, 2)

                # Denormalize and scale to target region
                fake_seg = (fake_seg * mov_std + mov_mean) * scale_factor

                # Convert km movements back to lat/lon
                for dx_km, dy_km in fake_seg:
                    cur_lon += dx_km / km_per_dlon
                    cur_lat += dy_km / km_per_dlat

                    # Soft boundary clamp
                    margin = dlat * 2
                    if cur_lat < bbox["min_lat"] - margin:
                        cur_lat = bbox["min_lat"]
                    if cur_lat > bbox["max_lat"] + margin:
                        cur_lat = bbox["max_lat"]
                    if cur_lon < bbox["min_lon"] - margin:
                        cur_lon = bbox["min_lon"]
                    if cur_lon > bbox["max_lon"] + margin:
                        cur_lon = bbox["max_lon"]

                    path.append((cur_lon, cur_lat))

        trajectories.append(path)
        print(f"  Trajectory {t_idx + 1}: {len(path)} points, "
              f"start edge ({start_lon:.4f}, {start_lat:.4f})")

    return trajectories


# ---------------------------------------------------------------------------
# KML EXPORT
# ---------------------------------------------------------------------------

TRAJ_COLORS = [
    "ff00ff00", "ff0000ff", "ffffff00", "ffff00ff", "ff00ffff",
    "ffff6600", "ff6600ff", "ff00cc66", "ff3399ff", "ffcc33ff",
]


def _copy_background_kml(background_path, parent_doc):
    if not os.path.exists(background_path):
        print(f"  WARNING: Background KML not found: {background_path}")
        return

    try:
        bg_tree = ET.parse(background_path)
        bg_root = bg_tree.getroot()
        bg_doc = bg_root.find('{http://www.opengis.net/kml/2.2}Document')
        if bg_doc is None:
            bg_doc = bg_root.find('Document')
        if bg_doc is None:
            print(f"  WARNING: Could not find Document element in {background_path}")
            return

        for child in list(bg_doc):
            if child.tag in ({'{http://www.opengis.net/kml/2.2}name', '{http://www.opengis.net/kml/2.2}description', 'name', 'description'}):
                continue
            parent_doc.append(copy.deepcopy(child))
        print(f"  Merged background KML from {background_path}")
    except Exception as exc:
        print(f"  WARNING: Failed to merge background KML: {exc}")


def export_trajectories_kml(trajectories, output_path, region_name="Walayar",
                            background_kml_path=None):
    """Export generated trajectories to KML, optionally merging a background map."""
    KML_NS = "http://www.opengis.net/kml/2.2"
    ET.register_namespace("", KML_NS)
    kml = ET.Element(f"{{{KML_NS}}}kml")
    doc = ET.SubElement(kml, f"{{{KML_NS}}}Document")
    ET.SubElement(doc, "name").text = f"Generated Elephant Trajectories — {region_name}"
    ET.SubElement(doc, "description").text = (
        f"Synthetic elephant trajectories generated by Conditional WGAN-GP "
        f"trained on Kruger NP movement data + OSM environmental features. "
        f"{len(trajectories)} trajectories, {TRAJ_SEGMENTS * 12} hours each."
    )

    if background_kml_path is not None:
        _copy_background_kml(background_kml_path, doc)

    # Styles
    for idx, color in enumerate(TRAJ_COLORS):
        s = ET.SubElement(doc, "Style", id=f"traj{idx}")
        ls = ET.SubElement(s, "LineStyle")
        ET.SubElement(ls, "color").text = color
        ET.SubElement(ls, "width").text = "3"

    # Trajectories
    folder = ET.SubElement(doc, "Folder")
    ET.SubElement(folder, "name").text = "Generated Trajectories"

    for idx, traj in enumerate(trajectories):
        pm = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm, "name").text = f"Synthetic Elephant {idx + 1}"
        ET.SubElement(pm, "description").text = (
            f"Generated trajectory #{idx + 1}\n"
            f"Points: {len(traj)}\n"
            f"Duration: {TRAJ_SEGMENTS * 12} hours ({TRAJ_SEGMENTS * 12 / 24:.0f} days)"
        )
        ET.SubElement(pm, "styleUrl").text = f"#traj{idx % len(TRAJ_COLORS)}"
        ls = ET.SubElement(pm, "LineString")
        ET.SubElement(ls, "tessellate").text = "1"
        coords_text = "\n          ".join(
            f"{lon:.7f},{lat:.7f},0" for lon, lat in traj
        )
        ET.SubElement(ls, "coordinates").text = "\n          " + coords_text

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    with open(output_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Elephant Trajectory GAN")
    parser.add_argument("--generate", action="store_true",
                        help="Generate Walayar trajectories using trained model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--n-traj", type=int, default=N_TRAJECTORIES)
    parser.add_argument("--n-segments", type=int, default=TRAJ_SEGMENTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.generate:
        # --- GENERATION MODE ---
        print("=" * 65)
        print("Generating Walayar Elephant Trajectories")
        print("=" * 65)

        # Find best model
        model_files = sorted(
            [f for f in os.listdir(MODEL_DIR) if f.startswith("gan_fold_")],
            key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
            reverse=True,
        )
        if not model_files:
            print("ERROR: No trained models found. Run training first.")
            return

        model_path = os.path.join(MODEL_DIR, model_files[0])
        print(f"\n[1] Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        G = Generator().to(device)
        G.load_state_dict(checkpoint["generator"])
        stats = checkpoint["stats"]

        # Build Walayar grid + features
        print(f"\n[2] Building Walayar grid + extracting features...")
        w_grid_info, w_cells = create_grid(WALAYAR_BBOX)
        w_features = extract_walayar_features(
            WALAYAR_OSM_FEAT, WALAYAR_OSM_ROADS, w_grid_info, w_cells
        )

        print(f"\n  Walayar feature summary per cell (mean):")
        feat_names = ["water", "crop", "settle", "road_density", "rail_density"]
        for i, name in enumerate(feat_names):
            print(f"    {name:15s}: {w_features[:,:,i].mean():.4f} "
                  f"(max {w_features[:,:,i].max():.4f})")

        # Build Kruger grid info for scale factor computation
        print(f"\n  Building Kruger reference grid for scale factor...")
        k_grid_info, _ = create_grid(KRUGER_BBOX)

        # Generate
        print(f"\n[3] Generating {args.n_traj} trajectories "
              f"({args.n_segments} segments × {SEQ_LEN} steps each)...")
        trajectories = generate_trajectories(
            G, stats, w_features, w_grid_info, w_cells,
            train_grid_info=k_grid_info,
            n_traj=args.n_traj, n_segments=args.n_segments, device=device,
        )

        # Export KML
        print(f"\n[4] Exporting KML...")
        export_trajectories_kml(
            trajectories,
            OUTPUT_KML,
            "Walayar",
            background_kml_path=BACKGROUND_KML,
        )

        print(f"\n{'=' * 65}")
        print(f"Generated {len(trajectories)} trajectories → {OUTPUT_KML}")
        print("Open in Google Earth to visualize.")
        print("=" * 65)

    else:
        # --- TRAINING MODE ---
        print("=" * 65)
        print("Training Elephant Trajectory GAN on Kruger NP Data")
        print("=" * 65)

        # 1. Build Kruger grid
        print(f"\n[1] Building Kruger NP grid ({CELL_SIZE_KM} km cells)...")
        k_grid_info, k_cells = create_grid(KRUGER_BBOX)

        # 2. Extract features
        print(f"\n[2] Extracting environmental features from OSM...")
        k_features = extract_kruger_features(KRUGER_OSM_CACHE, k_grid_info, k_cells)

        feat_names = ["water", "crop", "settle", "road_density", "rail_density"]
        print(f"\n  Feature summary per cell (mean):")
        for i, name in enumerate(feat_names):
            print(f"    {name:15s}: {k_features[:,:,i].mean():.4f} "
                  f"(max {k_features[:,:,i].max():.4f})")

        # 3. Load + process trajectories
        print(f"\n[3] Loading elephant trajectories...")
        tracks = load_trajectories(ELEPHANT_CSV, TRAJECTORY_START, TRAJECTORY_END)

        print(f"\n  Building training segments (SEQ_LEN={SEQ_LEN}, "
              f"stride={SEQ_LEN // 2})...")
        segments = trajectories_to_segments(tracks, k_grid_info, k_cells, k_features)

        if len(segments) < BATCH_SIZE:
            print(f"ERROR: Only {len(segments)} segments — need at least {BATCH_SIZE}.")
            return

        # 4. Train with k-fold CV
        print(f"\n[4] Training WGAN-GP ({args.epochs} epochs, "
              f"{K_FOLDS}-fold CV)...")
        generators = train_kfold(segments, device)

        # 5. Summary
        print(f"\n{'=' * 65}")
        print(f"Training complete!")
        print(f"  Models saved to: {MODEL_DIR}/")
        print(f"  Folds trained: {len(generators)}")
        print(f"\nTo generate Walayar trajectories:")
        print(f"  python elephant_trajectory_gan.py --generate")
        print("=" * 65)


if __name__ == "__main__":
    main()
