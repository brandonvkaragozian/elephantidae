#!/usr/bin/env python3
"""
K-Fold Trajectory Realism Scoring
Ranks WGAN-GP generated trajectories by how "realistic" they are
compared to held-out real trajectories
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import json
from datetime import datetime

print("\n" + "="*75)
print("K-FOLD TRAJECTORY REALISM ASSESSMENT")
print("="*75 + "\n")

DEVICE = torch.device('cpu')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================================
# LOAD DATA
# ============================================================================

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
        print(f"Error parsing: {e}")
    return trajectories

def parse_kml_generated(kml_file, name_prefix="WGAN-GP"):
    """Parse generated trajectories from KML"""
    trajectories = []
    names = []
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        for placemark in root.findall('.//kml:Placemark', ns):
            name_elem = placemark.find('.//kml:name', ns)
            name = name_elem.text if name_elem is not None else "Unknown"
            
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
                        names.append(name)
    except Exception as e:
        print(f"Error parsing: {e}")
    return trajectories, names

print("[1/4] Loading data...")

real_trajs = parse_kml_trajectories('S. Africa Elephants.kml')
generated_trajs, traj_names = parse_kml_generated('gan_walayar_wgan_gp.kml')

print(f"✓ Real trajectories: {len(real_trajs)}")
print(f"✓ Generated trajectories: {len(generated_trajs)}\n")

# Preprocess real trajectories
scaler = MinMaxScaler()
real_segments = []
for traj in real_trajs:
    norm = scaler.fit_transform(traj)
    for i in range(max(0, len(norm) - 20)):
        seg = norm[i:i+20].flatten()
        if len(seg) == 40:
            real_segments.append(seg)

print(f"✓ Real segments for training: {len(real_segments)}\n")

# Preprocess generated trajectories
generated_segments = []
for traj in generated_trajs:
    norm = scaler.fit_transform(traj)
    for i in range(max(0, len(norm) - 20)):
        seg = norm[i:i+20].flatten()
        if len(seg) == 40:
            generated_segments.append(seg)

print(f"✓ Generated segments: {len(generated_segments)}")
print(f"  (segments per trajectory avg: {len(generated_segments)/len(generated_trajs):.1f})\n")

# ============================================================================
# K-FOLD SCORING
# ============================================================================

class CriticRealism(nn.Module):
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

print("[2/4] Running 5-fold cross-validation for realism scoring...\n")

K = 5
kfold = KFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
trajectory_scores = {i: [] for i in range(len(generated_trajs))}

fold_num = 0
for train_idx, test_idx in kfold.split(real_trajs):
    fold_num += 1
    
    # Collect training segments (from held-out folds)
    train_segments = []
    for idx in train_idx:
        traj = real_trajs[idx]
        norm = scaler.fit_transform(traj)
        for i in range(max(0, len(norm) - 20)):
            seg = norm[i:i+20].flatten()
            if len(seg) == 40:
                train_segments.append(seg)
    
    # Collect test segments (held-out fold - baseline for realism)
    test_segments = []
    for idx in test_idx:
        traj = real_trajs[idx]
        norm = scaler.fit_transform(traj)
        for i in range(max(0, len(norm) - 20)):
            seg = norm[i:i+20].flatten()
            if len(seg) == 40:
                test_segments.append(seg)
    
    # Train critic on train_segments
    critic = CriticRealism().to(DEVICE)
    optimizer = optim.Adam(critic.parameters(), lr=1e-4)
    
    BATCH_SIZE = 32
    for epoch in range(10):  # Quick training
        idx = np.random.choice(len(train_segments), BATCH_SIZE, replace=False)
        real = torch.FloatTensor(np.array([train_segments[i] for i in idx])).to(DEVICE)
        
        critic_real = critic(real)
        loss = -critic_real.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Score test segments (baseline for realism = high critic score)
    with torch.no_grad():
        test_scores = []
        for i in range(0, len(test_segments), BATCH_SIZE):
            batch = torch.FloatTensor(np.array(test_segments[i:i+BATCH_SIZE])).to(DEVICE)
            scores = critic(batch).cpu().numpy().flatten()
            test_scores.extend(scores)
    
    baseline_realism = np.mean(test_scores) if test_scores else 0.0
    
    # Score each generated trajectory
    with torch.no_grad():
        for traj_idx, traj_segs in enumerate([generated_segments[i:i+50] 
                                              for i in range(0, len(generated_segments), 50)]):
            if traj_idx >= len(generated_trajs):
                break
            
            scores = []
            for seg in traj_segs:
                batch = torch.FloatTensor(np.array([seg])).to(DEVICE)
                score = critic(batch).item()
                scores.append(score)
            
            if scores:
                avg_score = np.mean(scores)
                realism = avg_score / baseline_realism if baseline_realism > 0 else 0.0
                trajectory_scores[traj_idx].append(realism)
    
    print(f"  Fold {fold_num}/5 | Baseline realism: {baseline_realism:.3f}")

print("\n[3/4] Computing final realism scores...\n")

# Average scores across folds
final_scores = {}
for traj_idx in range(len(generated_trajs)):
    if trajectory_scores[traj_idx]:
        avg_realism = np.mean(trajectory_scores[traj_idx])
        final_scores[traj_idx] = {
            'name': traj_names[traj_idx],
            'realism_score': float(avg_realism),
            'fold_scores': [float(s) for s in trajectory_scores[traj_idx]]
        }

# Sort by realism
sorted_trajs = sorted(final_scores.items(), 
                      key=lambda x: x[1]['realism_score'], 
                      reverse=True)

print("TOP 20 MOST REALISTIC TRAJECTORIES:")
print("=" * 75)
print(f"{'Rank':<6} {'Trajectory':<25} {'Realism Score':<15} {'K-Fold Avg':<15}")
print("-" * 75)

for rank, (idx, data) in enumerate(sorted_trajs[:20], 1):
    traj_name = data['name'].replace('WGAN-GP ', '').replace('Trajectory ', 'T')
    scores = data['fold_scores']
    avg_score = data['realism_score']
    std_score = np.std(scores) if len(scores) > 1 else 0
    
    print(f"{rank:<6} {traj_name:<25} {avg_score:>7.4f}±{std_score:.3f}    {np.mean(scores):>7.4f}")

print("\n" + "=" * 75)

# Statistics
realism_values = [s['realism_score'] for s in final_scores.values()]
print(f"\nRealism Score Statistics:")
print(f"  Mean:   {np.mean(realism_values):.4f}")
print(f"  Median: {np.median(realism_values):.4f}")
print(f"  Std:    {np.std(realism_values):.4f}")
print(f"  Min:    {np.min(realism_values):.4f}")
print(f"  Max:    {np.max(realism_values):.4f}")

# Percentile recommendations
top_1_pct = int(len(final_scores) * 0.01)
top_10_pct = int(len(final_scores) * 0.10)
top_25_pct = int(len(final_scores) * 0.25)
top_50_pct = int(len(final_scores) * 0.50)
top_85_pct = int(len(final_scores) * 0.85)

print(f"\nField Deployment Recommendations:")
print(f"  Top 1% ({top_1_pct} trajectories):    realism ≥ {sorted_trajs[top_1_pct-1][1]['realism_score']:.4f}")
print(f"  Top 10% ({top_10_pct} trajectories):   realism ≥ {sorted_trajs[top_10_pct-1][1]['realism_score']:.4f}")
print(f"  Top 25% ({top_25_pct} trajectories):   realism ≥ {sorted_trajs[top_25_pct-1][1]['realism_score']:.4f}")
print(f"  Top 50% ({top_50_pct} trajectories):   realism ≥ {sorted_trajs[top_50_pct-1][1]['realism_score']:.4f}")
print(f"  Top 85% ({top_85_pct} trajectories):   realism ≥ {sorted_trajs[top_85_pct-1][1]['realism_score']:.4f}")

print("\n[4/4] Saving results...")

# Save detailed results
results = {
    'timestamp': datetime.now().isoformat(),
    'method': 'k-fold_cross_validation',
    'k': K,
    'total_trajectories': len(generated_trajs),
    'statistics': {
        'mean_realism': float(np.mean(realism_values)),
        'median_realism': float(np.median(realism_values)),
        'std_realism': float(np.std(realism_values))
    },
    'top_100_realistic': [
        {
            'rank': i+1,
            'trajectory_id': idx,
            'name': data['name'],
            'realism_score': data['realism_score'],
            'fold_consistency': float(np.std(data['fold_scores'])) if len(data['fold_scores']) > 1 else 0.0
        }
        for i, (idx, data) in enumerate(sorted_trajs[:100])
    ]
}

with open('trajectory_realism_scores.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save top 1% trajectories to separate KML
top_1_indices = [idx for idx, _ in sorted_trajs[:top_1_pct]]
kml_1pct = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Top 1% Most Realistic Trajectories ({datetime.now().strftime('%Y-%m-%d')})</name>
    <description>Premium selection: {top_1_pct} trajectories from {len(generated_trajs)} WGAN-GP generated</description>
"""

for rank, idx in enumerate(top_1_indices, 1):
    traj = generated_trajs[idx]
    coords_str = " ".join([f"{p[0]},{p[1]},0" for p in traj])
    realism = final_scores[idx]['realism_score']
    
    kml_1pct += f"""    <Placemark>
      <name>Top {rank} - {traj_names[idx]} (Realism: {realism:.4f})</name>
      <description>Rank {rank}/{top_1_pct} | Realism Score: {realism:.4f}</description>
      <LineString>
        <coordinates>{coords_str}</coordinates>
      </LineString>
    </Placemark>
"""

kml_1pct += """  </Document>
</kml>"""

with open('gan_walayar_wgan_gp_top_1pct.kml', 'w') as f:
    f.write(kml_1pct)

# Save top trajectories to KML
top_n = top_85_pct  # Deploy top 85%
top_indices = [idx for idx, _ in sorted_trajs[:top_n]]

kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Top {top_n} Most Realistic Trajectories ({datetime.now().strftime('%Y-%m-%d')})</name>
    <description>Selected from {len(generated_trajs)} WGAN-GP generated trajectories using 5-fold realism scoring</description>
"""

for rank, idx in enumerate(top_indices, 1):
    traj = generated_trajs[idx]
    coords_str = " ".join([f"{p[0]},{p[1]},0" for p in traj])
    realism = final_scores[idx]['realism_score']
    
    kml += f"""    <Placemark>
      <name>Trajectory {rank} (Realism: {realism:.4f})</name>
      <description>Original: {traj_names[idx]}</description>
      <LineString>
        <coordinates>{coords_str}</coordinates>
      </LineString>
    </Placemark>
"""

kml += """  </Document>
</kml>"""

with open('gan_walayar_wgan_gp_top_realistic.kml', 'w') as f:
    f.write(kml)

print(f"✓ trajectory_realism_scores.json (detailed ranking)")
print(f"✓ gan_walayar_wgan_gp_top_1pct.kml ({top_1_pct} trajectories - premium)")
print(f"✓ gan_walayar_wgan_gp_top_realistic.kml ({top_n} trajectories for deployment)\n")

print("="*75)
print(f"RESULTS: Top 1% = {top_1_pct} trajectories | Top 85% = {top_n} trajectories")
print(f"Exclusion rate (85%): {((len(generated_trajs)-top_n)/len(generated_trajs)*100):.1f}%")
print(f"Quality improvement: Top trajectories are {(sorted_trajs[0][1]['realism_score']/np.mean(realism_values)):.2f}× above average")
print("="*75 + "\n")
