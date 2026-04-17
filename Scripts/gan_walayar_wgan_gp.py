#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Penalty (WGAN-GP) for Walayar Elephant Trajectories
Addresses vanilla GAN training instability with Wasserstein distance & gradient penalty
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut

print("""
================================================================================
WGAN-GP (Wasserstein GAN with Gradient Penalty) FOR WALAYAR ELEPHANTS
================================================================================
Improvements over vanilla GAN:
  1. Wasserstein distance loss (better convergence)
  2. Gradient penalty (training stability)
  3. Critic (discriminator) clipping removed → smoother gradients
  
Behavioral Constraints from Literature:
  1. Water (Attraction):    ≤5km daily (Pinter-Wollman 2015)
  2. Settlements (Avoidance): >1km hard / 2.5km soft (Tumenta 2010)
  3. Cropfields (Nocturnal): 3km night / 2km day (Goswami 2017)
  4. Roads (Avoidance):     Strategic crossing for resources (Kioko 2006)
================================================================================
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
LATENT_DIM = 20
GENERATOR_LAYERS = (50, 128, 256)
DISCRIMINATOR_LAYERS = (50, 128, 64)
BATCH_SIZE = 16  # Smaller batch for small dataset
N_EPOCHS = 100
LEARNING_RATE_GEN = 0.0001
LEARNING_RATE_CRITIC = 0.0001
LAMBDA_GP = 10  # Gradient penalty coefficient
CRITIC_UPDATES = 5  # Update critic 5 times per generator update
DEVICE = torch.device('cpu')  # Use CPU (no CUDA on macOS)

# Ecological parameters
WATER_DAILY_REQUIREMENT_KM = 5
SETTLEMENT_AVOIDANCE_KM = 2.5
SETTLEMENT_CRITICAL_KM = 1.0
CROPFIELD_NOCTURNAL_ATTRACTION_KM = 3
CROPFIELD_DAYTIME_AVOIDANCE_KM = 2
ROAD_AVOIDANCE_KM = 0.8

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"[CONFIG] Device: {DEVICE}, Random seed: {RANDOM_SEED}\n")

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class Generator(nn.Module):
    """Generator network: noise → trajectory"""
    def __init__(self, latent_dim=20, output_dim=40):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    """Critic network: trajectory → Wasserstein distance score"""
    def __init__(self, input_dim=40):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_kml_trajectories(kml_file):
    """Parse trajectory LineStrings from KML."""
    trajectories = []
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        for placemark in root.findall('.//kml:Placemark', ns):
            linestring = placemark.find('.//kml:LineString', ns)
            if linestring is not None:
                coords_elem = linestring.find('.//kml:coordinates', ns)
                if coords_elem is not None and coords_elem.text:
                    coords = []
                    for coord_str in coords_elem.text.strip().split():
                        parts = coord_str.replace('\n', '').split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                coords.append([lon, lat])
                            except:
                                pass
                    
                    if len(coords) > 50:
                        trajectories.append(np.array(coords, dtype=np.float32))
    except Exception as e:
        print(f"  Error parsing trajectories: {e}")
    
    return trajectories

def extract_features_from_kml(kml_file):
    """Extract environmental features from Walayar map."""
    features = {'water': [], 'settlement': [], 'cropfield': [], 'road': []}
    
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        for placemark in root.findall('.//kml:Placemark', ns):
            name_elem = placemark.find('.//kml:name', ns)
            name = name_elem.text.lower() if name_elem is not None else ""
            
            # Categorize features
            if 'water' in name or 'lake' in name or 'pond' in name or 'malampuzha' in name:
                cat = 'water'
            elif 'settle' in name or 'village' in name or 'colony' in name or 'palakeezh' in name:
                cat = 'settlement'
            elif 'crop' in name or 'field' in name or 'agri' in name:
                cat = 'cropfield'
            elif any(major_road in name for major_road in ['nh', 'railway', 'rail ', 'train', 'highway', 'national highway', 'state highway']):
                cat = 'road'
            else:
                continue
            
            # Extract centroid or representative point
            polygon = placemark.find('.//kml:Polygon', ns)
            point = placemark.find('.//kml:Point', ns)
            linestring = placemark.find('.//kml:LineString', ns)
            
            coords = None
            if polygon is not None:
                outer = polygon.find('.//kml:outerBoundaryIs', ns)
                if outer is not None:
                    ring = outer.find('.//kml:LinearRing', ns)
                    if ring is not None:
                        coords_elem = ring.find('.//kml:coordinates', ns)
                        if coords_elem is not None and coords_elem.text:
                            coords_list = []
                            for coord_str in coords_elem.text.strip().split():
                                parts = coord_str.replace('\n', '').split(',')
                                if len(parts) >= 2:
                                    try:
                                        lon, lat = float(parts[0]), float(parts[1])
                                        coords_list.append([lon, lat])
                                    except:
                                        pass
                            if coords_list:
                                coords = np.mean(coords_list, axis=0)
            
            elif point is not None:
                coords_elem = point.find('.//kml:coordinates', ns)
                if coords_elem is not None and coords_elem.text:
                    parts = coords_elem.text.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            coords = np.array([float(parts[0]), float(parts[1])])
                        except:
                            pass
            
            elif linestring is not None:
                coords_elem = linestring.find('.//kml:coordinates', ns)
                if coords_elem is not None and coords_elem.text:
                    coords_list = []
                    for coord_str in coords_elem.text.strip().split():
                        parts = coord_str.replace('\n', '').split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                coords_list.append([lon, lat])
                            except:
                                pass
                    if coords_list:
                        coords = np.mean(coords_list, axis=0)
            
            if coords is not None:
                features[cat].append(coords)
        
        # Convert to arrays
        for key in features:
            if features[key]:
                features[key] = np.array(features[key])
            else:
                features[key] = np.array([]).reshape(0, 2)
                
    except Exception as e:
        print(f"  Error extracting features: {e}")
    
    return features

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km."""
    R = 6371
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    delta_lat, delta_lon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def is_movement_toward_resource(point, direction_point, resources, resource_threshold_km=3.0):
    """Check if trajectory segment moves toward a resource."""
    if len(resources) == 0:
        return False
    
    distances_from_point = [haversine_distance(point[1], point[0], r[1], r[0]) 
                           for r in resources]
    min_dist = min(distances_from_point) if distances_from_point else float('inf')
    
    if min_dist > resource_threshold_km:
        return False
    
    distances_from_next = [haversine_distance(direction_point[1], direction_point[0], r[1], r[0]) 
                          for r in resources]
    min_dist_next = min(distances_from_next) if distances_from_next else float('inf')
    
    return min_dist_next < min_dist

def evaluate_multi_constraints(traj_walayar, features, time_of_day_fraction=None):
    """Evaluate constraint compliance with context-aware road crossings."""
    constraints = {
        'water_visited': False,
        'settlements_avoided': True,
        'cropfields_appropriate': True,
        'roads_avoided': True,
        'all_met': False
    }
    
    if len(traj_walayar) == 0:
        return constraints
    
    # Constraint 1: Water
    if len(features['water']) > 0:
        checkpoint_interval = max(1, len(traj_walayar) // 5)
        for i in range(0, len(traj_walayar), checkpoint_interval):
            point = traj_walayar[i]
            distances = [haversine_distance(point[1], point[0], w[1], w[0]) for w in features['water']]
            if min(distances) <= WATER_DAILY_REQUIREMENT_KM:
                constraints['water_visited'] = True
                break
    
    # Constraint 2: Settlements
    if len(features['settlement']) > 0:
        distances_to_settlement = [haversine_distance(p[1], p[0], s[1], s[0]) 
                                   for p in traj_walayar for s in features['settlement']]
        if distances_to_settlement and min(distances_to_settlement) < SETTLEMENT_CRITICAL_KM:
            constraints['settlements_avoided'] = False
    
    # Constraint 3: Cropfields
    if len(features['cropfield']) > 0:
        if time_of_day_fraction is not None and (19/24 <= time_of_day_fraction or time_of_day_fraction <= 6/24):
            constraints['cropfields_appropriate'] = True
        else:
            distances_to_crops = [haversine_distance(p[1], p[0], c[1], c[0]) 
                                 for p in traj_walayar for c in features['cropfield']]
            if distances_to_crops and min(distances_to_crops) < CROPFIELD_DAYTIME_AVOIDANCE_KM:
                constraints['cropfields_appropriate'] = False
    
    # Constraint 4: Roads (context-aware)
    if len(features['road']) > 0:
        road_crossing_zone = 0.5
        unjustified_crossings = 0
        total_road_violations = 0
        
        for i in range(len(traj_walayar)):
            point = traj_walayar[i]
            distances_to_road = [haversine_distance(point[1], point[0], r[1], r[0]) 
                                for r in features['road']]
            min_road_dist = min(distances_to_road) if distances_to_road else float('inf')
            
            if min_road_dist < road_crossing_zone:
                total_road_violations += 1
                
                if i < len(traj_walayar) - 1:
                    next_point = traj_walayar[i + 1]
                    toward_water = is_movement_toward_resource(point, next_point, features['water'], 4.0)
                    toward_crops = is_movement_toward_resource(point, next_point, features['cropfield'], 3.0)
                    
                    if not (toward_water or toward_crops):
                        unjustified_crossings += 1
                else:
                    unjustified_crossings += 1
        
        if total_road_violations > 0:
            justified_ratio = (total_road_violations - unjustified_crossings) / total_road_violations
            if justified_ratio < 0.5:
                constraints['roads_avoided'] = False
    
    constraints['all_met'] = (constraints['water_visited'] and 
                             constraints['settlements_avoided'] and 
                             constraints['cropfields_appropriate'] and 
                             constraints['roads_avoided'])
    
    return constraints

def generate_trajectory_kml(trajectories, output_file="gan_walayar_wgan_gp.kml"):
    """Generate KML file from trajectories."""
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>WGAN-GP Generated Trajectories - {}</name>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    for i, traj in enumerate(trajectories, 1):
        coords_str = " ".join([f"{p[0]},{p[1]},0" for p in traj])
        kml += f"""    <Placemark>
      <name>WGAN-GP Trajectory {i}</name>
      <LineString>
        <coordinates>{coords_str}</coordinates>
      </LineString>
    </Placemark>
"""
    
    kml += """  </Document>
</kml>"""
    
    with open(output_file, 'w') as f:
        f.write(kml)

def compute_gradient_penalty(critic, real_data, fake_data):
    """Compute gradient penalty for WGAN-GP training stability."""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=DEVICE)
    
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    critic_output = critic(interpolates)
    
    gradients = grad(
        outputs=critic_output,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_output.size(), device=DEVICE),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("[PHASE 1] Loading ecological data...")

# Parse data
trajectories = parse_kml_trajectories('S. Africa Elephants.kml')
features = extract_features_from_kml('FINAL WALAYAY MAP.kml')

print(f"✓ Water bodies: {len(features['water'])} locations")
print(f"✓ Settlements: {len(features['settlement'])} locations")
print(f"✓ Cropfields: {len(features['cropfield'])} locations")
print(f"✓ Roads/Rails: {len(features['road'])} locations")
print(f"✓ Source trajectories: {len(trajectories)}\n")

# Preprocess: normalize per trajectory
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_trajs = []
for traj in trajectories:
    normalized = scaler.fit_transform(traj)
    # Create 20-point segments (40 dimensions) for training
    for i in range(len(normalized) - 20):
        segment = normalized[i:i+20].flatten()
        if len(segment) == 40:
            normalized_trajs.append(segment)

print(f"✓ Training segments: {len(normalized_trajs)}\n")

# Initialize networks
generator = Generator(LATENT_DIM, 40).to(DEVICE)
critic = Critic(40).to(DEVICE)

optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.9))
optimizer_c = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC, betas=(0.5, 0.9))

print("[PHASE 2] WGAN-GP Training with Leave-One-Out CV...\n")

# Leave-One-Out CV
loo = LeaveOneOut()
fold = 0
critic_accuracies = []
held_out_idx = 0

# Build mapping from trajectory index to segment indices
trajectory_to_segments = {}
seg_idx = 0
for traj_idx, traj in enumerate(trajectories):
    normalized = scaler.fit_transform(traj)
    segment_indices = []
    for i in range(len(normalized) - 20):
        if len(normalized[i:i+20].flatten()) == 40:
            segment_indices.append(seg_idx)
            seg_idx += 1
    trajectory_to_segments[traj_idx] = segment_indices

for train_traj_idx, test_traj_idx in loo.split(trajectories):
    fold += 1
    if fold % 2 == 0:
        continue
    
    # Collect training segments from non-held-out trajectories
    train_segments = []
    for traj_idx in train_traj_idx:
        if traj_idx in trajectory_to_segments:
            for seg_idx in trajectory_to_segments[traj_idx]:
                if seg_idx < len(normalized_trajs):
                    train_segments.append(normalized_trajs[seg_idx])
    
    if len(train_segments) < BATCH_SIZE:
        print(f"[FOLD {fold}/14] Skipping (insufficient training segments)")
        continue
    
    # Train for N_EPOCHS
    for epoch in range(N_EPOCHS):
        # Critic updates
        for _ in range(CRITIC_UPDATES):
            batch_indices = np.random.choice(len(train_segments), BATCH_SIZE, replace=False)
            real_batch = torch.FloatTensor(np.array([train_segments[i] for i in batch_indices])).to(DEVICE)
            
            noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
            fake_batch = generator(noise)
            
            # Wasserstein loss
            critic_real = critic(real_batch)
            critic_fake = critic(fake_batch.detach())
            
            gradient_penalty = compute_gradient_penalty(critic, real_batch, fake_batch.detach())
            critic_loss = -torch.mean(critic_real) + torch.mean(critic_fake) + LAMBDA_GP * gradient_penalty
            
            optimizer_c.zero_grad()
            critic_loss.backward()
            optimizer_c.step()
        
        # Generator update
        noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
        fake_batch = generator(noise)
        critic_fake = critic(fake_batch)
        generator_loss = -torch.mean(critic_fake)
        
        optimizer_g.zero_grad()
        generator_loss.backward()
        optimizer_g.step()
        
        if epoch % 25 == 0:
            critic_acc = torch.mean(critic_real).item() - torch.mean(critic_fake).item()
            print(f"  Epoch {epoch+1:3d}/100 | Wasserstein distance: {critic_acc:.3f}")
    
    critic_accuracies.append(critic_acc)
    print(f"✓ Fold {fold}/14 complete\n")

print(f"✓ WGAN-GP Training Complete")
print(f"  Mean Wasserstein distance: {np.mean(critic_accuracies):.3f} ± {np.std(critic_accuracies):.3f}\n")

# ============================================================================
# GENERATION PHASE
# ============================================================================

print("[PHASE 3] Generating multi-constraint trajectories...\n")

walayar_bounds = (76.6239, 76.8523, 10.7235, 10.8269)  # lon_min, lon_max, lat_min, lat_max
walayar_polygon = []  # Load from KML if needed

generated_trajectories = []
max_attempts = 2000

for attempt in range(max_attempts):
    with torch.no_grad():
        noise = torch.randn(1, LATENT_DIM).to(DEVICE)
        fake_traj_norm = generator(noise).cpu().numpy()[0]
        
        # Reshape to points
        fake_traj = fake_traj_norm.reshape(-1, 2)
        
        # Denormalize to Walayar
        x_start = np.random.uniform(walayar_bounds[0] + 0.01, walayar_bounds[1] - 0.01)
        y_start = np.random.uniform(walayar_bounds[2] + 0.01, walayar_bounds[3] - 0.01)
        
        x_range = walayar_bounds[1] - walayar_bounds[0]
        y_range = walayar_bounds[3] - walayar_bounds[2]
        
        traj_walayar = fake_traj.copy()
        traj_walayar[:, 0] = x_start + fake_traj[:, 0] * x_range
        traj_walayar[:, 1] = y_start + fake_traj[:, 1] * y_range
        
        # Interpolate to 286 points
        from scipy.interpolate import interp1d
        t_old = np.linspace(0, 1, len(traj_walayar))
        t_new = np.linspace(0, 1, 286)
        f_lon = interp1d(t_old, traj_walayar[:, 0], kind='linear')
        f_lat = interp1d(t_old, traj_walayar[:, 1], kind='linear')
        traj_walayar_interp = np.column_stack([f_lon(t_new), f_lat(t_new)])
        
        # Validate constraints
        time_of_day = np.random.uniform(0, 24)
        constraints = evaluate_multi_constraints(traj_walayar_interp, features, time_of_day / 24)
        
        if constraints['all_met']:
            generated_trajectories.append(traj_walayar_interp)
            print(f"✓ Trajectory {len(generated_trajectories)}: 286 pts, all constraints met")
    
    if attempt % 500 == 0 and attempt > 0:
        print(f"  [{attempt}/{max_attempts}] Generated {len(generated_trajectories)} valid trajectories")

print(f"\n✓ Generated {len(generated_trajectories)} multi-constraint trajectories\n")

# Generate KML
generate_trajectory_kml(generated_trajectories, 'gan_walayar_wgan_gp.kml')
print(f"✓ Output: gan_walayar_wgan_gp.kml")

print(f"\n{'='*70}")
print("WGAN-GP TRAINING COMPLETE")
print(f"{'='*70}\n")
