"""
overlay_trajectories.py
=======================
Generate synthetic but literature-grounded Asian elephant trajectories for the
Walayar Range Forest (Kerala/Tamil Nadu border, India) and overlay them onto
the existing KML file.

Environmental feature layers are loaded from  osm_features.json  (produced by
fetch_osm_features.py) which queries the OpenStreetMap Overpass API for water
bodies, settlements, and agricultural land in the bounding box.

Run first:
    python fetch_osm_features.py     (requires network; caches to osm_features.json)
    python overlay_trajectories.py   (uses cached JSON; no network needed)

=============================================================================
MOVEMENT MODEL: Biased Correlated Random Walk (BCRW)
=============================================================================

Step length:  l_t ~ Gamma(α=0.90, scale_m) × speed_factor(T)
  α < 1 → right-skewed (occasional long bouts, mostly short foraging steps)
  Refs: Roever et al. (2014) PLOS ONE 9:e111982; Wall et al. (2006) Oryx 40:169

Heading:  θ_t = bias_angle + δ_t,   δ_t ~ WrappedCauchy(0, ρ)
  ρ ∈ [0,1]: concentration / directional persistence (0=random, 1=straight)
  Wrapped Cauchy preferred for heavier turning-angle tails
  Refs: Codling et al. (2008) J R Soc Interface 5:813; Wall et al. (2006)

Bias angle:  b = w_p·v_persist + w_w·∇̂φ_water − w_s·∇̂φ_settle
  where ∇φ_water = gradient of exponential-decay water attraction potential
        ∇φ_settle = gradient of settlement/crop repulsion potential
  Refs: Benhamou (2006) Ecology 87:1075; Pinter-Wollman et al. (2009) Ecology 90:3075

Temperature speed reduction (dry-season midday):
  factor = max(0.1, 1 − 0.025 × max(0, T − 26))
  Refs: Thaker et al. (2019) Sci Rep 9:17964; Kumar et al. (2010) Curr Sci 99:936

Environmental potentials:
  Water: φ_w(x) = Σ_i exp(−d_i/λ_w),  λ_w = 3 km (Walayar reservoir radius 3–5 km)
    Refs: Chamaille-Jammes et al. (2007) J Anim Ecol 76:789; Varma et al. (2012)
  Settle: φ_s(x) = Σ_i w_i · exp(−d_i/λ_s), weighted by area
    λ_s = 1.5 km day / 0.6 km night
    Refs: Graham et al. (2009) Divers Distrib 15:1081; Ngene et al. (2009)

Diurnal modulation (Baskaran et al. 2010; Sitati et al. 2005; Varma et al. 2012):
  Day 06:00–18:00: avoid settlements strongly; stay in forest interior
  Night 18:00–06:00: reduced avoidance; crop-raiding archetype attracted to fields

Five behavioural archetypes (individual variation: Sukumar 2003; Varma 2012):
  1  Forest Forager      — cow/calf group, deep interior, high persistence
  2  Water Seeker        — dry-season afternoon water visit (Walayar Lake/reservoir)
  3  Agricultural Raider — daytime forest retreat, nocturnal crop-raiding
  4  Range Explorer      — wide-ranging adult bull, low ρ
  5  Riparian Corridor   — follows Walayar River / stream network, high w_water

Key Walayar-specific parameters (Varma et al. 2012, Gajah 37:6–14):
  Daily path: 8–14 km dry season; home ranges 65–312 km²
  Water visits: once daily afternoon, Walayar Lake attractor 3–5 km radius
  Crop raids: October–March, clustered at forest–agriculture boundary

All citations in docstring of fetch_osm_features.py and in-code comments.
"""

import json
import math
import os
import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

KML_NS       = "http://www.opengis.net/kml/2.2"
OSM_JSON     = "osm_features.json"
LAT_M_PER_DEG = 111320.0

WATER_LAMBDA_M        = 3000.0   # water attraction decay [Chamaille-Jammes 2007]
SETTLE_LAMBDA_DAY_M   = 1500.0   # settlement repulsion, daytime [Graham et al. 2009]
SETTLE_LAMBDA_NIGHT_M =  600.0   # reduced repulsion, nighttime [Ngene et al. 2009]
MIN_WATER_AREA_KM2    = 0.002    # only include polygons > 0.002 km² (~2 ha)
MIN_CROP_AREA_KM2     = 0.01     # only include crop polygons > 1 ha

# Temperature model: Walayar dry season (rough approximation)
# peak ~32°C at 13:00, baseline 22°C at night (Kumar et al. 2010)
_HOUR_TEMP = {
    h: 22.0 + 10.0 * math.sin(math.pi * max(0, h - 6) / 14.0)
    if 6 <= h <= 20 else 22.0
    for h in range(24)
}
HEAT_K      = 0.025
HEAT_THRESH = 26.0


# ---------------------------------------------------------------------------
# 1. KML PARSING
# ---------------------------------------------------------------------------

def parse_kml_forest(kml_file):
    """Extract forest section polygons → unified Shapely geometry."""
    tree = ET.parse(kml_file)
    root = tree.getroot()
    polygons = []
    for pm in root.findall(f".//{{{KML_NS}}}Placemark"):
        name_el = pm.find(f"{{{KML_NS}}}name")
        if name_el is None or "Section" not in name_el.text:
            continue
        for outer in pm.findall(f".//{{{KML_NS}}}outerBoundaryIs"):
            coords_el = outer.find(f".//{{{KML_NS}}}coordinates")
            if coords_el is None:
                continue
            pts = []
            for tok in coords_el.text.strip().split():
                parts = tok.split(",")
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
            if len(pts) >= 3:
                polygons.append(Polygon(pts))
    if not polygons:
        raise ValueError("No forest section polygons found in KML.")
    return unary_union(polygons), tree, root


# ---------------------------------------------------------------------------
# 2. LOAD & FILTER OSM FEATURES
# ---------------------------------------------------------------------------

def lon_m_per_deg(lat):
    return LAT_M_PER_DEG * math.cos(math.radians(lat))


def dist_m(p1, p2):
    """Approx flat-Earth distance in metres between (lon,lat) pairs."""
    lo_m = lon_m_per_deg((p1[1] + p2[1]) / 2.0)
    dx = (p2[0] - p1[0]) * lo_m
    dy = (p2[1] - p1[1]) * LAT_M_PER_DEG
    return math.sqrt(dx * dx + dy * dy)


def cluster_points(pts, min_dist_m=500.0):
    """
    Simple greedy spatial clustering: merge points closer than min_dist_m.
    Returns list of (lon, lat, weight) where weight = number of merged points.
    """
    if not pts:
        return []
    clusters = []
    used = [False] * len(pts)
    for i, p in enumerate(pts):
        if used[i]:
            continue
        group = [p]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if not used[j] and dist_m((p[0], p[1]), (pts[j][0], pts[j][1])) < min_dist_m:
                group.append(pts[j])
                used[j] = True
        cx = sum(g[0] for g in group) / len(group)
        cy = sum(g[1] for g in group) / len(group)
        weight = sum(g[2] if len(g) > 2 else 1.0 for g in group)
        clusters.append((cx, cy, weight))
    return clusters


def load_osm_features(forest_geom):
    """
    Load and separate OSM features into THREE distinct source lists.

    Critical ecological distinction (Hoare & Du Toit 1999; Baskaran et al. 2010):
      • dense_settle : towns, cities, suburbs, dense residential areas
                       → ALWAYS strongly repel elephants (day AND night)
                       → λ = 2.5 km  (Graham et al. 2009: avoidance within 2–5 km)
      • crop_sources : farmland, orchards, small hamlets at forest edge
                       → DAY: mild repulsion  (elephants avoid during daylight)
                       → NIGHT: mild attraction (crop-raiding approach to field edge)
                       → λ = 1.2 km  (Sitati et al. 2005: entry distance ~0.8 km from edge)
      • water_sources: all water bodies
                       → Always attractive; λ = 3 km (Walayar Lake 3–5 km draw)
                       → Malampuzha Lake (9.3 km²) and Walayar Lake (2.5 km²) dominate

    Returns (water_sources, dense_settle_sources, crop_sources)
    each a list of (lon, lat, weight) tuples.
    """
    if not os.path.exists(OSM_JSON):
        print(f"  WARNING: {OSM_JSON} not found. Using hardcoded fallback.")
        w = [(76.8530, 10.8485, 3.2), (76.6930, 10.8482, 5.5),  # Walayar, Malampuzha
             (76.7920, 10.8320, 0.8), (76.7750, 10.8250, 0.5)]
        d = [(76.8400, 10.8250, 4.0), (76.7600, 10.8050, 2.0)]
        c = [(76.7850, 10.7950, 1.5), (76.8100, 10.8400, 1.0)]
        return w, d, c

    with open(OSM_JSON) as f:
        data = json.load(f)

    minx, miny, maxx, maxy = forest_geom.bounds
    buf = 0.07   # ~7 km buffer to capture edge settlements and Malampuzha

    def in_zone(lon, lat):
        return (minx - buf <= lon <= maxx + buf and
                miny - buf <= lat <= maxy + buf)

    # ── water ─────────────────────────────────────────────────────────────
    raw_water = []
    for p in data.get("water_polygons", []):
        cx, cy = p["centroid"]
        if not in_zone(cx, cy) or p["area_km2"] < MIN_WATER_AREA_KM2:
            continue
        raw_water.append((cx, cy, math.sqrt(p["area_km2"])))
    for p in data.get("water_points", []):
        if in_zone(p["lon"], p["lat"]):
            raw_water.append((p["lon"], p["lat"], 0.25))

    # ── dense settlements (always repel) ──────────────────────────────────
    DENSE_TYPES = {"city", "town", "suburb", "neighbourhood",
                   "residential", "commercial", "industrial"}
    raw_dense = []
    for s in data.get("settlements", []):
        if not in_zone(s["lon"], s["lat"]):
            continue
        pt = s.get("place_type", "")
        is_dense = s.get("dense", pt in DENSE_TYPES)
        if not is_dense:
            continue
        w = {"city": 8.0, "town": 5.0, "suburb": 3.0,
             "neighbourhood": 2.0, "residential": 2.5,
             "commercial": 3.5, "industrial": 3.0}.get(pt, 2.0)
        raw_dense.append((s["lon"], s["lat"], w))

    # ── crop / agricultural edge (day-repel, night-attract) ───────────────
    raw_crop = []
    # small hamlets and isolated dwellings count as weak crop-zone proxies
    for s in data.get("settlements", []):
        if not in_zone(s["lon"], s["lat"]):
            continue
        pt = s.get("place_type", "")
        if pt in ("hamlet", "isolated_dwelling", "village"):
            raw_crop.append((s["lon"], s["lat"], 0.8))
    for p in data.get("crop_polygons", []):
        cx, cy = p["centroid"]
        if not in_zone(cx, cy) or p["area_km2"] < MIN_CROP_AREA_KM2:
            continue
        raw_crop.append((cx, cy, math.sqrt(p["area_km2"])))

    water_sources  = cluster_points(raw_water, min_dist_m=600.0)
    dense_sources  = cluster_points(raw_dense, min_dist_m=700.0)
    crop_sources   = cluster_points(raw_crop,  min_dist_m=400.0)

    # report the major named water bodies
    named_w = sorted(
        [p for p in data["water_polygons"] if p["name"] and in_zone(*p["centroid"])],
        key=lambda x: x["area_km2"], reverse=True)[:5]
    print(f"  Water attractors (clustered)  : {len(water_sources)}")
    if named_w:
        for w in named_w:
            print(f"    {w['name']:30s} {w['area_km2']:.2f} km²")
    print(f"  Dense settlement repulsors    : {len(dense_sources)}")
    print(f"  Crop/edge sources             : {len(crop_sources)}")

    return water_sources, dense_sources, crop_sources


# ---------------------------------------------------------------------------
# 3. POTENTIAL FIELD
# ---------------------------------------------------------------------------

def potential_gradient(pos, sources, lambda_m):
    """
    Gradient of φ(x) = Σ_i w_i · exp(−d_i/λ) in Cartesian metre coords.
    Returns 2D vector pointing TOWARD sources (direction of attraction).

    Ref: Pinter-Wollman et al. (2009) Ecology 90:3075
    """
    lo_m = lon_m_per_deg(pos[1])
    grad = np.zeros(2)
    for src in sources:
        lon_s, lat_s = src[0], src[1]
        w = src[2] if len(src) > 2 else 1.0
        dx_m = (lon_s - pos[0]) * lo_m
        dy_m = (lat_s - pos[1]) * LAT_M_PER_DEG
        d = math.sqrt(dx_m**2 + dy_m**2)
        if d < 1.0:
            continue
        coeff = w * math.exp(-d / lambda_m) / (lambda_m * d)
        grad[0] += coeff * dx_m
        grad[1] += coeff * dy_m
    return grad


def unit_vec(v):
    n = math.sqrt(v[0]**2 + v[1]**2)
    return v / n if n > 1e-12 else np.zeros(2)


# ---------------------------------------------------------------------------
# 4. WRAPPED CAUCHY SAMPLER
#    Preferred over von Mises for animal turning angles (heavier tails)
#    Refs: Codling et al. (2008); Wall et al. (2006); Signer et al. (2019)
# ---------------------------------------------------------------------------

def wrapped_cauchy_sample(rng, mu, rho):
    """
    WC(mu, rho):  X = mu + 2·arctan(((1+ρ)/(1−ρ)) · tan(πU − π/2))
    U ~ Uniform(0,1).  Ref: Mardia & Jupp (2000) Directional Statistics.
    """
    u = rng.uniform(0.0, 1.0)
    return mu + 2.0 * math.atan(
        ((1.0 + rho) / (1.0 - rho)) * math.tan(math.pi * u - math.pi / 2.0)
    )


# ---------------------------------------------------------------------------
# 5. DECAY SCALES FOR THE THREE POTENTIAL FIELDS
# ---------------------------------------------------------------------------

WATER_LAMBDA_M         = 3000.0   # water attraction [Chamaille-Jammes 2007; Varma 2012]
DENSE_SETTLE_LAMBDA_M  = 2500.0   # dense settlement repulsion [Graham et al. 2009]
CROP_LAMBDA_DAY_M      = 1200.0   # crop-edge mild day repulsion [Sitati et al. 2005]
CROP_LAMBDA_NIGHT_M    =  800.0   # crop-edge night attraction (shorter = sharper pull)

# ---------------------------------------------------------------------------
# 6. FOUR REALISTIC ELEPHANT GROUPS
#
#    Based on: Varma et al. (2012) Gajah 37:6–14 (12 GPS-collared Walayar elephants
#              → ~4–6 distinct family units); Baskaran et al. (2010) Tropical Ecology;
#              Sukumar (2003) The Living Elephants; de Silva et al. (2011).
#
#    Group count rationale for 326 km² forest:
#      Typical Asian elephant family home range: 100–200 km² (Campos-Arceiz 2008;
#      de Silva 2011; Varma 2012 range: 65–312 km²).
#      → 2–4 overlapping family groups + 1–2 solitary bulls fit the landscape.
#      Literature for adjacent Anamalai TR identified 4–6 distinct groups
#      (Baskaran et al. 2010). We model 3 family herds + 1 adult bull = 4 groups.
#
#    Movement weights:
#      w_water      : attraction to water sources (always positive)
#      w_dense      : repulsion from dense settlements (always subtracted; NEVER goes
#                     negative — elephants always avoid cities/towns)
#      w_crop_day   : mild repulsion from crop/field edges during 06:00–18:00
#      w_crop_night : mild attraction to crop/field edges during 18:00–06:00
#                     (crop-raiding; approach to field boundary, NOT town interiors)
#                     Refs: Sitati et al. 2005; Baskaran et al. 2010; Varma et al. 2012
#
#    Key ecological constraint:
#      Dense settlements (Palakkad, suburbs) are repelled with λ=2.5 km regardless
#      of time of day. Only crop fields and small hamlets at the forest–agriculture
#      boundary are approached at night. This matches GPS data showing elephants
#      entering field edges at night but NOT entering towns (Varma et al. 2012;
#      Ngene et al. 2009).
# ---------------------------------------------------------------------------

GROUPS = [
    {
        "name": "Western Family Herd",
        "description": (
            "Resident cow-calf family group using Pudussery North and "
            "Akamalavaram sections (western forest). Regular water visits to "
            "streams and Malampuzha catchment area. Moderate crop-raiding at "
            "southern and western agricultural edges during post-harvest season "
            "(October–March). Dense settlements strongly avoided at all times. "
            "Refs: Varma et al. 2012; Sukumar 2003; Baskaran et al. 2010."
        ),
        "w_persist"    : 0.52,
        "w_water"      : 0.25,
        "w_dense"      : 0.48,   # strong permanent avoidance of towns/suburbs
        "w_crop_day"   : 0.18,   # mild daytime crop-edge avoidance
        "w_crop_night" : 0.22,   # mild nighttime approach to farm edges
        "rho"          : 0.52,
        "step_shape"   : 0.90,
        "step_scale_m" : 390,    # mean ~351 m/hr → ~8.4 km/day
        "start_hour"   : 6,
        "color_kml"    : "ff00cc00",   # green
    },
    {
        "name": "Southern Boundary Herd",
        "description": (
            "Family group ranging along the southern forest–agriculture boundary "
            "(Kottekkad and Walayar sections). Primary crop-raiding group in the "
            "Walayar HEC zone; approaches agricultural edges at dusk, retreats "
            "to forest interior by dawn. Very strong avoidance of Walayar town "
            "and Palakkad area at all times. "
            "Refs: Varma et al. 2012 (peak raids Oct–Mar); "
            "Sitati et al. 2005 (93% of raids 18:30–06:00); "
            "Sukumar & Gadgil 1988."
        ),
        "w_persist"    : 0.44,
        "w_water"      : 0.20,
        "w_dense"      : 0.55,   # highest dense-settle avoidance (near Walayar town)
        "w_crop_day"   : 0.22,   # retreat from field edge during day
        "w_crop_night" : 0.38,   # strong approach to field edge at night (crop-raiding)
        "rho"          : 0.46,
        "step_shape"   : 0.90,
        "step_scale_m" : 480,    # more mobile (raiding bouts at night)
        "start_hour"   : 18,     # begins dusk movement toward field edge
        "color_kml"    : "ff2222ff",   # red
    },
    {
        "name": "Northern Forest Herd",
        "description": (
            "Family group in northern Akathethara and Pudussery South sections. "
            "Least crop-raiding due to deeper forest position; strongly oriented "
            "toward forest interior streams and Malampuzha Lake. High directional "
            "persistence; rarely approaches agricultural boundary. "
            "Refs: Baskaran et al. 2010 (deep-interior foraging groups); "
            "de Silva et al. 2011 (riparian dry-season concentration); "
            "Varma et al. 2012."
        ),
        "w_persist"    : 0.60,
        "w_water"      : 0.30,
        "w_dense"      : 0.45,
        "w_crop_day"   : 0.12,
        "w_crop_night" : 0.12,   # minimal crop raiding — deep forest group
        "rho"          : 0.57,   # high persistence
        "step_shape"   : 0.90,
        "step_scale_m" : 360,    # mean ~324 m/hr → ~7.8 km/day (forest interior)
        "start_hour"   : 5,      # pre-dawn water visit
        "color_kml"    : "ff00aaff",   # orange-blue
    },
    {
        "name": "Adult Bull",
        "description": (
            "Solitary adult male with large home range crossing all forest "
            "sections; uses Palakkad corridor (NH544 crossings 22:00–04:00). "
            "Low directional persistence → wide-ranging exploration. "
            "Stronger water orientation (bulls need more water per day). "
            "Slightly less crop-edge avoidance than family groups (musth bulls "
            "show reduced fear response near human areas). "
            "Refs: Varma et al. 2012 (max home range 312 km²); "
            "Baskaran et al. 1995 (85–350 km² home range); "
            "Baskaran 2013 (NH544 corridor crossings at night)."
        ),
        "w_persist"    : 0.40,
        "w_water"      : 0.35,
        "w_dense"      : 0.35,   # bulls slightly bolder near human areas [Sukumar 2003]
        "w_crop_day"   : 0.12,
        "w_crop_night" : 0.18,
        "rho"          : 0.33,   # low → wide-ranging exploratory
        "step_shape"   : 0.90,
        "step_scale_m" : 640,    # mean ~576 m/hr → ~13.8 km/day [Baskaran 1995]
        "start_hour"   : 4,
        "color_kml"    : "ffcc00ff",   # magenta
    },
]


# ---------------------------------------------------------------------------
# 6. TRAJECTORY GENERATION
# ---------------------------------------------------------------------------

def generate_trajectory(archetype, forest_geom, water_sources, settle_sources,
                         n_steps=240, dt_hours=1.0, rng=None):
    """
    Generate a single elephant trajectory using BCRW with environmental biases.

    Parameters
    ----------
    n_steps = 240  →  10 days at 1-hr intervals
      Matches Varma et al. (2012) GPS-collaring protocol; 10-day windows used
      in step-selection analyses (Thurfjell et al. 2014).

    Returns: trajectory [(lon,lat)], hours [float]
    """
    if rng is None:
        rng = np.random.default_rng()

    # starting position: random inside forest
    minx, miny, maxx, maxy = forest_geom.bounds
    for _ in range(50000):
        lon = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        if forest_geom.contains(Point(lon, lat)):
            break
    else:
        c = forest_geom.centroid
        lon, lat = c.x, c.y

    trajectory = [(lon, lat)]
    hours      = [float(archetype["start_hour"] % 24)]
    theta      = rng.uniform(0, 2 * math.pi)

    for step in range(n_steps - 1):
        hour = (archetype["start_hour"] + step * dt_hours) % 24
        is_day = 6.0 <= hour < 18.0

        # temperature-dependent speed [Thaker 2019; Kumar 2010]
        T = _HOUR_TEMP[int(hour) % 24]
        spd = max(0.1, 1.0 - HEAT_K * max(0.0, T - HEAT_THRESH))

        # day/night settlement weight
        w_settle  = archetype["w_settle_day"] if is_day else archetype["w_settle_night"]
        lambda_s  = SETTLE_LAMBDA_DAY_M      if is_day else SETTLE_LAMBDA_NIGHT_M

        # environmental gradients
        pos = (lon, lat)
        g_w = potential_gradient(pos, water_sources,  WATER_LAMBDA_M)
        g_s = potential_gradient(pos, settle_sources, lambda_s)

        n_w = unit_vec(g_w)   # toward water
        n_s = unit_vec(g_s)   # toward settlements (subtract for repulsion)
        v_p = np.array([math.cos(theta), math.sin(theta)])

        b = (archetype["w_persist"] * v_p
             + archetype["w_water"]  * n_w
             - w_settle              * n_s)   # subtract = repulsion; negative w_settle = attraction

        bias = math.atan2(b[1], b[0]) if math.hypot(b[0], b[1]) > 1e-12 else rng.uniform(0, 2*math.pi)

        # wrapped Cauchy turning angle
        delta = wrapped_cauchy_sample(rng, mu=0.0, rho=archetype["rho"])
        theta_new = (bias + delta) % (2 * math.pi)

        # step length
        step_m = rng.gamma(shape=archetype["step_shape"],
                           scale=archetype["step_scale_m"]) * spd

        lo_m = lon_m_per_deg(lat)
        new_lon = lon + step_m * math.cos(theta_new) / lo_m
        new_lat = lat + step_m * math.sin(theta_new) / LAT_M_PER_DEG

        if forest_geom.contains(Point(new_lon, new_lat)):
            lon, lat, theta = new_lon, new_lat, theta_new
        else:
            # boundary handling: try reverse + 8 random fallbacks
            placed = False
            for attempt in range(9):
                a = (theta + math.pi) % (2 * math.pi) if attempt == 0 \
                    else rng.uniform(0, 2 * math.pi)
                sm = step_m * (0.5 if attempt == 0 else 0.3)
                clon = lon + sm * math.cos(a) / lo_m
                clat = lat + sm * math.sin(a) / LAT_M_PER_DEG
                if forest_geom.contains(Point(clon, clat)):
                    lon, lat, theta = clon, clat, a
                    placed = True
                    break
            if not placed:
                theta = rng.uniform(0, 2 * math.pi)

        trajectory.append((lon, lat))
        hours.append(float(hour))

    return trajectory, hours


# ---------------------------------------------------------------------------
# 7. KML OUTPUT
# ---------------------------------------------------------------------------

# KML colors are AABBGGRR hex strings
# Water:       semi-transparent blue  "9dff8c00"  (alpha=9d, blue fill)
# Crop:        semi-transparent amber "7f00a5ff"
# Settlement:  semi-transparent red   "7f0000ff"

def _poly_style(document, sid, line_color, fill_color, line_width="1"):
    """Add a combined LineStyle + PolyStyle to document."""
    s = ET.Element("Style", {"id": sid})
    ls = ET.SubElement(s, "LineStyle")
    ET.SubElement(ls, "color").text = line_color
    ET.SubElement(ls, "width").text = line_width
    ps = ET.SubElement(s, "PolyStyle")
    ET.SubElement(ps, "color").text = fill_color
    document.insert(0, s)


def _icon_style(document, sid, icon_color, scale="1"):
    """Add a point IconStyle to document."""
    s = ET.Element("Style", {"id": sid})
    ist = ET.SubElement(s, "IconStyle")
    ET.SubElement(ist, "color").text = icon_color
    ET.SubElement(ist, "scale").text = scale
    icon = ET.SubElement(ist, "Icon")
    ET.SubElement(icon, "href").text = \
        "https://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
    document.insert(0, s)


def add_line_style(document, sid, color, width=4):
    s = ET.Element("Style", {"id": sid})
    ls = ET.SubElement(s, "LineStyle")
    ET.SubElement(ls, "color").text = color
    ET.SubElement(ls, "width").text = str(width)
    document.insert(0, s)


def trajectory_to_folder(name, traj, sid, description=""):
    folder = ET.Element("Folder")
    ET.SubElement(folder, "name").text = name

    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = f"{name} path"
    ET.SubElement(pm, "description").text = description
    ET.SubElement(pm, "styleUrl").text = f"#{sid}"
    ls_el = ET.SubElement(pm, "LineString")
    ET.SubElement(ls_el, "tessellate").text = "1"
    ET.SubElement(ls_el, "coordinates").text = " ".join(
        f"{lo:.7f},{la:.7f},0" for lo, la in traj)

    pm_s = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm_s, "name").text = f"▶ Start: {name}"
    ET.SubElement(ET.SubElement(pm_s, "Point"), "coordinates").text = \
        f"{traj[0][0]:.7f},{traj[0][1]:.7f},0"

    pm_e = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm_e, "name").text = f"■ End: {name}"
    ET.SubElement(ET.SubElement(pm_e, "Point"), "coordinates").text = \
        f"{traj[-1][0]:.7f},{traj[-1][1]:.7f},0"

    return folder


def build_osm_kml_folders(document, forest_geom,
                          water_sid="style_water_poly",
                          crop_sid="style_crop_poly",
                          settle_sid="style_settlement_pt"):
    """
    Build three KML Folders from osm_features.json:
      💧 Water Bodies (OSM)      — blue polygon fills
      🌾 Crop Fields (OSM)       — amber polygon outlines
      🏘  Settlements (OSM)       — red point markers

    Filters to features within/near the forest extent.
    Returns a list of Folder elements to append to the KML document.
    """
    if not os.path.exists(OSM_JSON):
        print(f"  WARNING: {OSM_JSON} not found — skipping OSM layers in KML.")
        return []

    with open(OSM_JSON) as f:
        data = json.load(f)

    minx, miny, maxx, maxy = forest_geom.bounds
    PAD = 0.06   # ~6 km buffer — captures edge settlements & water

    def in_zone(lon, lat):
        return (minx - PAD <= lon <= maxx + PAD and
                miny - PAD <= lat <= maxy + PAD)

    def coords_str(coords):
        return " ".join(f"{c[0]:.7f},{c[1]:.7f},0" for c in coords)

    # style IDs are passed in (already registered on the document by caller)

    # ── Water Bodies folder ───────────────────────────────────────────────
    water_folder = ET.Element("Folder")
    ET.SubElement(water_folder, "name").text = "💧 Water Bodies (OSM)"
    ET.SubElement(water_folder, "description").text = (
        "Water bodies (rivers, streams, reservoirs, ponds, canals) sourced "
        "from OpenStreetMap via Overpass API for the Walayar bounding box. "
        "Polygons sized proportionally; only bodies ≥0.002 km² shown. "
        "Key attractor for elephant movement (Chamaille-Jammes et al. 2007; "
        "Varma et al. 2012 — Walayar Lake draws elephants within 3–5 km)."
    )
    ET.SubElement(water_folder, "visibility").text = "1"

    water_count = 0
    for poly in sorted(data["water_polygons"],
                       key=lambda p: p["area_km2"], reverse=True):
        cx, cy = poly["centroid"]
        if not in_zone(cx, cy):
            continue
        if poly["area_km2"] < MIN_WATER_AREA_KM2:
            continue
        coords = poly["coords"]
        if len(coords) < 3:
            continue
        # close ring if not already closed
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        name = poly["name"] or f"Water body ({poly['type']})"
        pm = ET.SubElement(water_folder, "Placemark")
        ET.SubElement(pm, "name").text = name
        ET.SubElement(pm, "description").text = (
            f"Type: {poly['type']}\nArea: {poly['area_km2']:.4f} km²\n"
            f"Source: OpenStreetMap"
        )
        ET.SubElement(pm, "styleUrl").text = f"#{water_sid}"
        pg = ET.SubElement(pm, "Polygon")
        ET.SubElement(pg, "tessellate").text = "1"
        ob = ET.SubElement(pg, "outerBoundaryIs")
        lr = ET.SubElement(ob, "LinearRing")
        ET.SubElement(lr, "coordinates").text = coords_str(coords)
        water_count += 1

    # ── Crop Fields folder ────────────────────────────────────────────────
    crop_folder = ET.Element("Folder")
    ET.SubElement(crop_folder, "name").text = "🌾 Crop Fields (OSM)"
    ET.SubElement(crop_folder, "description").text = (
        "Agricultural land (farmland, orchard, plantation, paddy) from "
        "OpenStreetMap. These zones represent the primary human-elephant "
        "conflict (HEC) pressure areas. Elephants approach these boundaries "
        "at night (Sitati et al. 2005; Baskaran et al. 2010; Varma et al. 2012). "
        "Only fields ≥0.01 km² (1 ha) shown."
    )
    ET.SubElement(crop_folder, "visibility").text = "1"

    crop_count = 0
    for poly in sorted(data["crop_polygons"],
                       key=lambda p: p["area_km2"], reverse=True):
        cx, cy = poly["centroid"]
        if not in_zone(cx, cy):
            continue
        if poly["area_km2"] < MIN_CROP_AREA_KM2:
            continue
        coords = poly["coords"]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        name = poly["name"] or f"{poly['type'].capitalize()} field"
        pm = ET.SubElement(crop_folder, "Placemark")
        ET.SubElement(pm, "name").text = name
        ET.SubElement(pm, "description").text = (
            f"Type: {poly['type']}\nArea: {poly['area_km2']:.4f} km²\n"
            f"Source: OpenStreetMap"
        )
        ET.SubElement(pm, "styleUrl").text = f"#{crop_sid}"
        pg = ET.SubElement(pm, "Polygon")
        ET.SubElement(pg, "tessellate").text = "1"
        ob = ET.SubElement(pg, "outerBoundaryIs")
        lr = ET.SubElement(ob, "LinearRing")
        ET.SubElement(lr, "coordinates").text = coords_str(coords)
        crop_count += 1

    # ── Settlements folder ────────────────────────────────────────────────
    settle_folder = ET.Element("Folder")
    ET.SubElement(settle_folder, "name").text = "🏘 Settlements (OSM)"
    ET.SubElement(settle_folder, "description").text = (
        "Human settlements (cities, towns, villages, hamlets) from "
        "OpenStreetMap. Elephants avoid these areas during daytime (λ=1.5 km) "
        "and approach agricultural zones at night (Hoare & Du Toit 1999; "
        "Graham et al. 2009; Ngene et al. 2009)."
    )
    ET.SubElement(settle_folder, "visibility").text = "1"

    # deduplicate by name+location
    seen = set()
    settle_count = 0
    priority = {"city": 0, "town": 1, "village": 2, "hamlet": 3,
                "suburb": 4, "neighbourhood": 5, "isolated_dwelling": 6}
    for s in sorted(data["settlements"],
                    key=lambda x: priority.get(x.get("place_type", ""), 9)):
        if not in_zone(s["lon"], s["lat"]):
            continue
        key = (round(s["lon"], 4), round(s["lat"], 4))
        if key in seen:
            continue
        seen.add(key)
        name = s["name"] or s["place_type"].capitalize()
        pm = ET.SubElement(settle_folder, "Placemark")
        ET.SubElement(pm, "name").text = name
        ET.SubElement(pm, "description").text = (
            f"Type: {s['place_type']}\nSource: OpenStreetMap"
        )
        ET.SubElement(pm, "styleUrl").text = f"#{settle_sid}"
        pt = ET.SubElement(pm, "Point")
        ET.SubElement(pt, "coordinates").text = f"{s['lon']:.7f},{s['lat']:.7f},0"
        settle_count += 1

    print(f"    Water polygons in KML : {water_count}")
    print(f"    Crop polygons in KML  : {crop_count}")
    print(f"    Settlement points     : {settle_count}")

    return [water_folder, crop_folder, settle_folder]


# ---------------------------------------------------------------------------
# 8. MATPLOTLIB FIGURE — publication-quality preview
# ---------------------------------------------------------------------------

def make_figure(forest_geom, trajectories, archetypes, water_sources,
                settle_sources, output_png="trajectories_preview.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
    except ImportError:
        print("matplotlib not available — skipping PNG preview.")
        return

    minx, miny, maxx, maxy = forest_geom.bounds
    # Display extent: forest + small margin
    PAD = 0.025
    x0, x1 = minx - PAD, maxx + PAD
    y0, y1 = miny - PAD, maxy + PAD

    def in_display(lon, lat, buf=0.04):
        return x0 - buf <= lon <= x1 + buf and y0 - buf <= lat <= y1 + buf

    fig, ax = plt.subplots(figsize=(12, 11))

    # ── forest polygons ──────────────────────────────────────────────────
    polys = list(forest_geom.geoms) if hasattr(forest_geom, "geoms") else [forest_geom]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, color="#c8e6c9", alpha=0.55, zorder=1)
        ax.plot(x, y, color="#2e7d32", linewidth=0.9, zorder=2)

    # ── settlement / crop pressure — filter to display extent, top-N by weight ──
    settle_vis = [s for s in settle_sources if in_display(s[0], s[1])]
    # only show the most significant ones to keep the figure readable
    settle_vis = sorted(settle_vis, key=lambda s: s[2], reverse=True)[:120]
    if settle_vis:
        sl = np.array([[s[0], s[1]] for s in settle_vis])
        sw = np.array([s[2] for s in settle_vis])
        sw_norm = np.clip(sw / sw.max() * 45 + 6, 6, 55)
        ax.scatter(sl[:, 0], sl[:, 1], s=sw_norm, c="#d32f2f",
                   alpha=0.30, zorder=3, linewidths=0)

    # ── water sources — filter to display extent, sized by weight ────────
    water_vis = [w for w in water_sources if in_display(w[0], w[1])]
    water_vis = sorted(water_vis, key=lambda w: w[2], reverse=True)[:120]
    if water_vis:
        wl = np.array([[w[0], w[1]] for w in water_vis])
        ww = np.array([w[2] for w in water_vis])
        ww_norm = np.clip(ww / ww.max() * 70 + 8, 8, 80)
        ax.scatter(wl[:, 0], wl[:, 1], s=ww_norm, c="#1565c0",
                   alpha=0.55, zorder=4, linewidths=0)

    # ── trajectories ─────────────────────────────────────────────────────
    palette = {
        "Forest Forager"     : "#1b5e20",
        "Water Seeker"       : "#0277bd",
        "Agricultural Raider": "#b71c1c",
        "Range Explorer"     : "#880e4f",
        "Riparian Corridor"  : "#006064",
    }
    handles = []
    for traj, arch in zip(trajectories, archetypes):
        col  = palette.get(arch["name"], "black")
        lons = [p[0] for p in traj]
        lats = [p[1] for p in traj]
        ax.plot(lons, lats, color=col, linewidth=1.8, alpha=0.85, zorder=5)
        # start marker
        ax.scatter(lons[0], lats[0], color=col, s=85, zorder=8, marker="D",
                   edgecolors="white", linewidths=0.8)
        # end marker
        ax.scatter(lons[-1], lats[-1], color=col, s=60, zorder=8, marker="s",
                   edgecolors="white", linewidths=0.8)
        handles.append(mlines.Line2D([], [], color=col, linewidth=2.2,
                                     label=arch["name"]))

    # ── legend ────────────────────────────────────────────────────────────
    legend_feat = [
        mpatches.Patch(color="#c8e6c9", alpha=0.7,
                       label="Forest sections (from KML)"),
        mpatches.Patch(color="#1565c0", alpha=0.6,
                       label="Water bodies (OSM; size ∝ area)"),
        mpatches.Patch(color="#d32f2f", alpha=0.45,
                       label="Settlements & crop fields (OSM; size ∝ weight)"),
    ]
    legend_markers = [
        mlines.Line2D([], [], color="grey", linewidth=0, marker="D",
                      markersize=7, markerfacecolor="grey",
                      markeredgecolor="white", label="Trajectory start ◆"),
        mlines.Line2D([], [], color="grey", linewidth=0, marker="s",
                      markersize=7, markerfacecolor="grey",
                      markeredgecolor="white", label="Trajectory end ■"),
    ]
    ax.legend(handles=legend_feat + legend_markers + handles,
              loc="lower left", fontsize=8, framealpha=0.88,
              title="Layers & archetypes", title_fontsize=8.5)

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(
        "Synthetic Asian Elephant Trajectories — Walayar Range Forest\n"
        "BCRW model · Step ~ Gamma(α=0.90) · Heading ~ Wrapped Cauchy(ρ)\n"
        "Water attraction λ=3 km · Settlement repulsion λ=1.5/0.6 km (day/night)\n"
        "OSM-sourced water & agricultural layers · 5 archetypes (Varma et al. 2012)",
        fontsize=9
    )

    # ── scale bar (5 km) ─────────────────────────────────────────────────
    bar_lon0 = minx + 0.015
    bar_lat  = miny - 0.010
    deg_5km  = 5000.0 / lon_m_per_deg(bar_lat)
    ax.annotate("", xy=(bar_lon0 + deg_5km, bar_lat),
                xytext=(bar_lon0, bar_lat),
                arrowprops=dict(arrowstyle="-", color="black", lw=2.5),
                zorder=9)
    ax.text(bar_lon0 + deg_5km / 2, bar_lat + 0.004, "5 km",
            ha="center", va="bottom", fontsize=8, zorder=9)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    print(f"  Figure saved → {output_png}")
    plt.close()


# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------

def main():
    kml_input  = "Walayar Range Forest Map with Grid.kml"
    kml_output = "Walayar_with_Elephant_Trajectories.kml"
    png_output = "trajectories_preview.png"

    print("─" * 65)
    print("Walayar Elephant Trajectory Generator")
    print("BCRW model with OSM environmental layers")
    print("─" * 65)

    print("\n[1] Parsing forest geometry from KML …")
    forest_geom, tree, root = parse_kml_forest(kml_input)
    b = forest_geom.bounds
    print(f"    Forest bounds: lon [{b[0]:.4f}–{b[2]:.4f}]  "
          f"lat [{b[1]:.4f}–{b[3]:.4f}]")

    print("\n[2] Loading OSM environmental features …")
    water_sources, settle_sources = load_osm_features(forest_geom)

    rng = np.random.default_rng(seed=42)

    print("\n[3] Generating trajectories (5 archetypes × 10 days @ 1-hr) …")
    print(f"    {'Archetype':22s}  {'ρ':>5}  {'scale_m':>8}  "
          f"{'mean/hr':>8}  {'total km':>9}  {'km/day':>7}")
    results = []
    for arch in ARCHETYPES:
        traj, hrs = generate_trajectory(arch, forest_geom,
                                        water_sources, settle_sources,
                                        n_steps=240, rng=rng)
        total_km = sum(dist_m(traj[i], traj[i+1]) / 1000.0
                       for i in range(len(traj)-1))
        mean_hr = arch["step_shape"] * arch["step_scale_m"]
        print(f"    {arch['name']:22s}  {arch['rho']:>5.2f}  "
              f"{arch['step_scale_m']:>8}  {mean_hr:>7.0f}m  "
              f"{total_km:>9.1f}  {total_km/10:>7.1f}")
        results.append((traj, hrs, arch))

    print("\n[4] Building output KML …")

    # Build a CLEAN new KML document from scratch to avoid namespace issues.
    # Using the default namespace (no prefix) so Google Earth sees <kml xmlns="...">.
    N = KML_NS
    GX = "http://www.google.com/kml/ext/2.2"
    ET.register_namespace("",     N)
    ET.register_namespace("gx",   GX)
    ET.register_namespace("atom", "http://www.w3.org/2005/Atom")

    def E(tag, attrib=None, text=None, parent=None):
        """Create a KML-namespaced element, optionally appending to parent."""
        el = ET.Element(f"{{{N}}}{tag}", attrib or {})
        if text is not None:
            el.text = str(text)
        if parent is not None:
            parent.append(el)
        return el

    def SE(tag, parent, attrib=None, text=None):
        """SubElement shorthand."""
        el = ET.SubElement(parent, f"{{{N}}}{tag}", attrib or {})
        if text is not None:
            el.text = str(text)
        return el

    kml_root = E("kml")
    doc      = SE("Document", kml_root)
    SE("name", doc, text="Walayar Range Forest — Elephant Trajectories")
    SE("description", doc, text=(
        "Walayar Range Forest, Kerala/Tamil Nadu. "
        "Layers: forest sections (original KML), OSM water bodies, "
        "OSM crop fields, OSM settlements, synthetic elephant trajectories "
        "(BCRW model; Varma et al. 2012; Benhamou 2006)."
    ))

    # ── inline styles ─────────────────────────────────────────────────────
    def poly_style(sid, line_col, fill_col, line_w="1"):
        s = SE("Style", doc, {"id": sid})
        ls = SE("LineStyle", s)
        SE("color", ls, text=line_col)
        SE("width", ls, text=line_w)
        ps = SE("PolyStyle", s)
        SE("color", ps, text=fill_col)

    def line_style(sid, color, width="3"):
        s = SE("Style", doc, {"id": sid})
        ls = SE("LineStyle", s)
        SE("color", ls, text=color)
        SE("width", ls, text=str(width))

    def icon_style(sid, color, scale="0.8"):
        s = SE("Style", doc, {"id": sid})
        ist = SE("IconStyle", s)
        SE("color", ist, text=color)
        SE("scale", ist, text=scale)
        icon = SE("Icon", ist)
        SE("href", icon,
           text="https://maps.google.com/mapfiles/kml/shapes/placemark_circle.png")

    # forest sections
    poly_style("sty_forest",     "ff1a8c1a", "5514cc14", "1.2")
    # water, crop, settlement
    poly_style("sty_water",      "ffb06000", "9dff8c00", "0.8")   # blue fill
    poly_style("sty_crop",       "ff007acc", "4f00aaff", "0.8")   # amber fill
    icon_style("sty_settlement", "ff0000ff", "0.7")               # red dot
    # trajectories
    arch_colors = {
        "Forest Forager"     : ("ff00cc00", "5"),
        "Water Seeker"       : ("ff0088ff", "5"),
        "Agricultural Raider": ("ff2222ff", "5"),
        "Range Explorer"     : ("ffee00cc", "5"),
        "Riparian Corridor"  : ("ff00ffee", "5"),
    }
    for arch in ARCHETYPES:
        col, w = arch_colors.get(arch["name"], ("ffffffff", "4"))
        line_style("sty_" + arch["name"].replace(" ", "_"), col, w)

    # ── Forest Sections (re-extracted from original KML) ─────────────────
    fs_folder = SE("Folder", doc)
    SE("name", fs_folder, text="🌲 Forest Sections")
    SE("visibility", fs_folder, text="1")
    orig_doc = root.find(f".//{{{N}}}Document")
    if orig_doc is not None:
        for pm in orig_doc.findall(f".//{{{N}}}Folder/{{{N}}}Placemark"):
            name_el = pm.find(f"{{{N}}}name")
            if name_el is None or "Section" not in name_el.text:
                continue
            coords_el = pm.find(f".//{{{N}}}coordinates")
            if coords_el is None:
                continue
            new_pm = SE("Placemark", fs_folder)
            SE("name",     new_pm, text=name_el.text)
            SE("styleUrl", new_pm, text="#sty_forest")
            poly = SE("Polygon", new_pm)
            SE("tessellate", poly, text="1")
            ob = SE("outerBoundaryIs", poly)
            lr = SE("LinearRing", ob)
            SE("coordinates", lr, text=coords_el.text.strip())

    # ── OSM layers ────────────────────────────────────────────────────────
    print("    Adding OSM feature layers …")
    osm_folders = build_osm_kml_folders(doc, forest_geom,
                                        water_sid="sty_water",
                                        crop_sid="sty_crop",
                                        settle_sid="sty_settlement")
    for folder in osm_folders:
        doc.append(folder)

    # ── Elephant Trajectories ─────────────────────────────────────────────
    tf = SE("Folder", doc)
    SE("name", tf, text="🐘 Elephant Trajectories (BCRW + OSM)")
    SE("description", tf, text=(
        "Synthetic Asian elephant trajectories — Walayar Range Forest.\n"
        "BCRW model: Step~Gamma(α=0.90), Heading~WrappedCauchy(ρ).\n"
        "Refs: Varma et al. 2012; Benhamou 2006; Codling et al. 2008."
    ))

    for traj, hrs, arch in results:
        sid = "sty_" + arch["name"].replace(" ", "_")
        arch_folder = SE("Folder", tf)
        SE("name", arch_folder, text=arch["name"])
        SE("description", arch_folder, text=arch["description"])

        pm_line = SE("Placemark", arch_folder)
        SE("name",     pm_line, text=f"{arch['name']} path")
        SE("styleUrl", pm_line, text=f"#{sid}")
        ls_el = SE("LineString", pm_line)
        SE("tessellate", ls_el, text="1")
        SE("coordinates", ls_el,
           text=" ".join(f"{lo:.7f},{la:.7f},0" for lo, la in traj))

        pm_s = SE("Placemark", arch_folder)
        SE("name", pm_s, text=f"Start: {arch['name']}")
        SE("coordinates", SE("Point", pm_s),
           text=f"{traj[0][0]:.7f},{traj[0][1]:.7f},0")

        pm_e = SE("Placemark", arch_folder)
        SE("name", pm_e, text=f"End: {arch['name']}")
        SE("coordinates", SE("Point", pm_e),
           text=f"{traj[-1][0]:.7f},{traj[-1][1]:.7f},0")

    # ── Write ─────────────────────────────────────────────────────────────
    out_tree = ET.ElementTree(kml_root)
    ET.indent(out_tree, space="\t")   # pretty-print (Python ≥3.9)
    out_tree.write(kml_output, encoding="utf-8", xml_declaration=True)
    print(f"    Saved → {kml_output}")

    print("\n[5] Generating figure …")
    make_figure(forest_geom, [t for t, _, _ in results], ARCHETYPES,
                water_sources, settle_sources, png_output)

    print("\nDone.")
    print("─" * 65)


if __name__ == "__main__":
    main()
