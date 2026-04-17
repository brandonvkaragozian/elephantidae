#!/usr/bin/env python3
"""
WGAN-GP Full Training - Generates multi-constraint trajectories for comparison with vanilla GAN
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import time

print("\n" + "="*75)
print("FULL WGAN-GP TRAINING WITH TRAJECTORY GENERATION")
print("="*75 + "\n")

DEVICE = torch.device('cpu')
RANDOM_SEED = 42
LATENT_DIM = 20
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def parse_kml_trajectories(kml_file):
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
                                coords.append([float(parts[0]), float(parts[1])])
                            except:
                                pass
                    if len(coords) > 50:
                        trajectories.append(np.array(coords, dtype=np.float32))
    except Exception as e:
        print(f"[ERROR] Parsing trajectories: {e}")
    return trajectories

def extract_features_from_kml(kml_file):
    features = {'water': [], 'settlement': [], 'cropfield': [], 'road': []}
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        for placemark in root.findall('.//kml:Placemark', ns):
            name_elem = placemark.find('.//kml:name', ns)
            name = name_elem.text.lower() if name_elem is not None else ""
            
            if 'water' in name or 'lake' in name:
                cat = 'water'
            elif 'settle' in name or 'village' in name:
                cat = 'settlement'
            elif 'crop' in name or 'field' in name:
                cat = 'cropfield'
            elif any(road in name for road in ['nh', 'railway', 'rail', 'train', 'highway']):
                cat = 'road'
            else:
                continue
            
            coords = None
            for elem_type in ['Polygon', 'Point', 'LineString']:
                elem = placemark.find(f'.//kml:{elem_type}', ns)
                if elem is not None:
                    if elem_type == 'Polygon':
                        ring = elem.find('.//kml:LinearRing', ns)
                        if ring:
                            coords_elem = ring.find('.//kml:coordinates', ns)
                    else:
                        coords_elem = elem.find('.//kml:coordinates', ns)
                    
                    if coords_elem is not None and coords_elem.text:
                        coords_list = []
                        for coord_str in coords_elem.text.strip().split():
                            parts = coord_str.replace('\n', '').split(',')
                            if len(parts) >= 2:
                                try:
                                    coords_list.append([float(parts[0]), float(parts[1])])
                                except:
                                    pass
                        if coords_list:
                            coords = np.mean(coords_list, axis=0)
                            break
            
            if coords is not None:
                features[cat].append(coords)
        
        for key in features:
            features[key] = np.array(features[key]) if features[key] else np.array([]).reshape(0, 2)
    except Exception as e:
        print(f"[ERROR] Extracting features: {e}")
    
    return features

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    delta_lat, delta_lon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def is_movement_toward_resource(point, next_point, resources, threshold_km=3.0):
    if len(resources) == 0:
        return False
    dist_from = np.array([haversine_distance(point[1], point[0], r[1], r[0]) for r in resources])
    min_from = dist_from.min() if len(dist_from) > 0 else float('inf')
    if min_from > threshold_km:
        return False
    dist_to = np.array([haversine_distance(next_point[1], next_point[0], r[1], r[0]) for r in resources])
    min_to = dist_to.min() if len(dist_to) > 0 else float('inf')
    return min_to < min_from

def evaluate_constraints(traj, features):
    constraints = {'water': False, 'settlements': True, 'roads': True, 'all_met': False}
    
    # Water visit
    if len(features['water']) > 0:
        for i in range(0, len(traj), max(1, len(traj)//5)):
            dists = np.array([haversine_distance(traj[i, 1], traj[i, 0], w[1], w[0]) for w in features['water']])
            if dists.min() <= 5.0:
                constraints['water'] = True
                break
    
    # Settlement avoidance
    if len(features['settlement']) > 0:
        dists = np.array([haversine_distance(traj[i, 1], traj[i, 0], s[1], s[0]) 
                          for i in range(len(traj)) for s in features['settlement']])
        if len(dists) > 0 and dists.min() < 1.0:
            constraints['settlements'] = False
    
    # Road crossing (context-aware)
    if len(features['road']) > 0:
        violations = 0
        justified = 0
        for i in range(len(traj)):
            dists_road = np.array([haversine_distance(traj[i, 1], traj[i, 0], r[1], r[0]) 
                                   for r in features['road']])
            if dists_road.min() < 0.5:
                violations += 1
                if i < len(traj) - 1:
                    if (is_movement_toward_resource(traj[i], traj[i+1], features['water'], 4.0) or
                        is_movement_toward_resource(traj[i], traj[i+1], features['cropfield'], 3.0)):
                        justified += 1
        
        if violations > 0 and justified / violations < 0.5:
            constraints['roads'] = False
    
    constraints['all_met'] = constraints['water'] and constraints['settlements'] and constraints['roads']
    return constraints

class GeneratorWGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 50), nn.ReLU(),
            nn.Linear(50, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 40)
        )
    def forward(self, z):
        return self.net(z)

class CriticWGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 50), nn.ReLU(),
            nn.Linear(50, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def compute_gradient_penalty(critic, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=DEVICE)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    critic_output = critic(interpolates)
    gradients = grad(outputs=critic_output, inputs=interpolates,
                    grad_outputs=torch.ones(critic_output.size(), device=DEVICE),
                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# ============================================================================
# LOAD DATA & TRAIN
# ============================================================================

print("[1/3] Loading data...")
trajs = parse_kml_trajectories('S. Africa Elephants.kml')
features = extract_features_from_kml('FINAL WALAYAY MAP.kml')
print(f"✓ {len(trajs)} trajectories, {len(features['water'])} water, {len(features['road'])} roads\n")

# Preprocess
scaler = MinMaxScaler()
train_data = []
for traj in trajs:
    norm = scaler.fit_transform(traj)
    for i in range(max(0, len(norm) - 20)):
        seg = norm[i:i+20].flatten()
        if len(seg) == 40:
            train_data.append(seg)

print(f"[2/3] Training WGAN-GP ({len(train_data)} segments)...")
start = time.time()

generator = GeneratorWGAN().to(DEVICE)
critic = CriticWGAN().to(DEVICE)
opt_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_c = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

BATCH_SIZE = 32
EPOCHS = 20
scores = []

for epoch in range(EPOCHS):
    for _ in range(5):  # 5 critic updates
        idx = np.random.choice(len(train_data), BATCH_SIZE, replace=False)
        real = torch.FloatTensor(np.array([train_data[i] for i in idx])).to(DEVICE)
        noise = torch.randn(BATCH_SIZE, 20).to(DEVICE)
        fake = generator(noise)
        
        c_real = critic(real)
        c_fake = critic(fake.detach())
        gp = compute_gradient_penalty(critic, real, fake.detach())
        c_loss = -c_real.mean() + c_fake.mean() + 10.0 * gp
        
        opt_c.zero_grad()
        c_loss.backward()
        opt_c.step()
    
    # Generator
    noise = torch.randn(BATCH_SIZE, 20).to(DEVICE)
    fake = generator(noise)
    g_loss = -critic(fake).mean()
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()
    
    ws_dist = (c_real.mean() - c_fake.mean()).item()
    scores.append(ws_dist)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}/20 | Wasserstein: {ws_dist:7.3f}")

print(f"✓ Training complete: {time.time() - start:.1f}s\n")

# ============================================================================
# GENERATE TRAJECTORIES
# ============================================================================

print("[3/3] Generating multi-constraint trajectories (1000 attempts)...")
bounds = (76.6239, 76.8523, 10.7235, 10.8269)
generated = []

with torch.no_grad():
    for attempt in range(1000):
        for _ in range(5):  # 5 tries per attempt
            noise = torch.randn(1, 20).to(DEVICE)
            fake_norm = generator(noise).cpu().numpy()[0].reshape(-1, 2)
            
            x_start = np.random.uniform(bounds[0] + 0.01, bounds[1] - 0.01)
            y_start = np.random.uniform(bounds[2] + 0.01, bounds[3] - 0.01)
            x_range = bounds[1] - bounds[0]
            y_range = bounds[3] - bounds[2]
            
            traj_walayar = fake_norm.copy()
            traj_walayar[:, 0] = x_start + fake_norm[:, 0] * x_range
            traj_walayar[:, 1] = y_start + fake_norm[:, 1] * y_range
            
            # Interpolate to 286 points
            t_old = np.linspace(0, 1, len(traj_walayar))
            t_new = np.linspace(0, 1, 286)
            f_lon = interp1d(t_old, traj_walayar[:, 0], kind='linear')
            f_lat = interp1d(t_old, traj_walayar[:, 1], kind='linear')
            traj_interp = np.column_stack([f_lon(t_new), f_lat(t_new)])
            
            if evaluate_constraints(traj_interp, features)['all_met']:
                generated.append(traj_interp)
                print(f"  ✓ Trajectory {len(generated)}: 286 pts, constraints met")
                break
        
        if (attempt + 1) % 250 == 0:
            print(f"    [{attempt+1}/1000] Generated {len(generated)} trajectories")

# ============================================================================
# OUTPUT
# ============================================================================

kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>WGAN-GP Generated Trajectories - {}</name>
""".format(time.strftime("%Y-%m-%d %H:%M:%S"))

for i, traj in enumerate(generated, 1):
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

with open('gan_walayar_wgan_gp.kml', 'w') as f:
    f.write(kml)

print(f"\n" + "="*75)
print("WGAN-GP RESULTS")
print("="*75)
print(f"Generated: {len(generated)} multi-constraint trajectories")
print(f"Success rate: {len(generated)/10:.1f}% (1000 attempts = 5000 samples)")
print(f"Output: gan_walayar_wgan_gp.kml")
print(f"Mean Wasserstein distance: {np.mean(scores[-5:]):.3f}")
print("="*75 + "\n")

print("[✓] WGAN-GP TRAINING COMPLETE")
print("[→] Compare with vanilla GAN results:")
print("    - Vanilla GAN: 12 trajectories (gan_walayar_multiconstraint.kml)")
print("    - WGAN-GP:      " + str(len(generated)) + " trajectories (gan_walayar_wgan_gp.kml)")
print("[→] Both outputs ready for field validation\n")
