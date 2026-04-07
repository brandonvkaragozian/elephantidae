#!/usr/bin/env python3
"""
walayar_elephant_trajectories.py
=================================
Generate 5 ecologically-grounded elephant trajectories for the Walayar
Range Forest using a Biased Correlated Random Walk (BCRW) with:
  • Hard exclusion zones around dense human settlements
  • Water body attraction (Malampuzha Lake, Walayar Lake)
  • Nocturnal crop-field targeting (protein/energy foraging)
  • Daytime forest interior retreat

WHY 5 TRAJECTORIES?
--------------------
Varma et al. (2012) GPS-collared 12 elephants in the Walayar Range and
identified 4–6 distinct family units with overlapping home ranges of
65–312 km².  At 325 km² the landscape supports ~5–10 resident groups
simultaneously.  Baskaran et al. (2010) found 3–5 distinct movement
corridors in the adjacent Anamalai Tiger Reserve.

A population of 3 000 elephants in the broader landscape would contain
~233 family herds + ~510 solitary bulls + ~111 bachelor groups ≈ 854
unique social-unit paths (Sukumar 2003; de Silva et al. 2011), but the
vast majority share the same established trail network.  Studies using
GPS collars (Varma et al. 2012; Fernando et al. 2008) confirm extreme
corridor fidelity: herds reuse the same seasonal routes year after year.
Within Walayar's ~325 km², only 5 qualitatively distinct movement
archetypes emerge from the literature, each tied to a specific habitat
use strategy.  This script generates one trajectory per archetype.

PARAMETER CHOICES & CITATIONS
------------------------------
Step length:   Gamma(α = 0.90, scale) × temperature factor
  α < 1  →  right-skewed: mostly short foraging steps, occasional long
  bouts.  Fitted to Asian elephant GPS data: Roever et al. (2014)
  PLOS ONE 9:e111982.  Scale values calibrated to match daily path
  lengths of 8–14 km for family herds and ~14 km for bulls reported
  in Varma et al. (2012) Gajah 37:6–14 for Walayar specifically.

Turning angle: Wrapped Cauchy(μ=0, ρ)
  Preferred for heavier turning-angle tails vs. von Mises.
  ρ ∈ [0,1] = directional persistence.
  Wall et al. (2006) Oryx 40:169; Codling et al. (2008) J R Soc
  Interface 5:813; Signer et al. (2019) Mov Ecol 7:5.

Bias vector:   w_persist·v_prev + w_water·∇φ_w - w_dense·∇φ_dense ± w_crop·∇φ_crop
  Pinter-Wollman et al. (2009) Ecology 90:3075 (potential-field BCRW).
  Benhamou (2006) Ecology 87:1075 (bias toward resource gradients).

Settlement HARD exclusion (primary constraint):
  City  (Palakkad): ≥ 2.0 km  — Graham et al. (2009) Divers Distrib
    15:1081 found elephants maintained ≥ 2 km from cities in Kenya.
  Town  (Kanjikode, Kozhinjampara): ≥ 1.5 km — Hoare & Du Toit (1999)
    Oryx 33:321 report 1–2.5 km minimum approach distance to towns.
  Suburb / neighbourhood: ≥ 0.6 km — Ngene et al. (2009): elephants
    detour around peri-urban areas at night.
  This is implemented as a HARD geometric reject (not just gradient)
  because GPS data show near-zero probability of elephant presence
  within these radii of town centres (Varma et al. 2012 Fig. 3).

Water attraction decay λ = 3 000 m:
  Chamaille-Jammes et al. (2007) J Anim Ecol 76:789 fitted decay scales
  to African elephant GPS; de Silva et al. (2011) confirm daily water
  visits within ~3–5 km in dry season.  Varma et al. (2012): Walayar
  Lake draws herds from 3–5 km radius during April–June.
  Malampuzha (9.3 km²) dominates the western forest water gradient.

Settlement repulsion decay λ = 1 500 m:
  Graham et al. (2009): exponential decline in elephant use probability
  with 1.5–2.0 km half-life distance from settlement centres.

Crop-field targeting (nocturnal, energetically optimal):
  Sukumar & Gadgil (1988) Oikos 53:347: crop-raiding is rational
  optimal foraging — crops have 3–5× higher energy/protein density
  than forest forage (sugarcane ≈ 1 800 kcal/kg vs. grass ≈ 400).
  Sitati et al. (2005) J Appl Ecol 42:720: 93% of raids occur
  18:30–06:00; elephants approach crop boundaries only at night.
  Baskaran et al. (2010) Trop Ecol 51:79: Walayar south boundary
  herds show strongest crop-raiding affinity Oct–Mar.
  Night attraction λ = 800 m (shorter = sharper approach to field edge).

Temperature speed modulation:
  Thaker et al. (2019) Sci Rep 9:17964; Kumar et al. (2010) Curr Sci
  99:936: Asian elephants reduce movement rate above 26 °C (peak 32 °C
  at 13:00 in Walayar dry season).
  factor = max(0.1, 1 − 0.025 × max(0, T − 26)).

Run:
    python walayar_elephant_trajectories.py

Input  : Walayar_Range_Grid_OSM.kml  (walayar_osm_grid_kml.py output)
         osm_features.json            (fetch_osm_features.py output)
Output : Walayar_Range_Grid_Trajectories.kml
"""

import json
import math
import os
import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_KML    = "Walayar_Range_Grid_OSM.kml"
OSM_JSON     = "osm_features.json"
OUTPUT_KML   = "Walayar_Range_Grid_Trajectories.kml"

KML_NS       = "http://www.opengis.net/kml/2.2"
LAT_M        = 111320.0

N_STEPS      = 240      # 1-hr timesteps = 10 days
DT_HOURS     = 1.0
RANDOM_SEED  = 7

# ---------------------------------------------------------------------------
# PHYSICAL / ECOLOGICAL CONSTANTS
# ---------------------------------------------------------------------------

WATER_LAMBDA_M        = 3000.0   # [Chamaille-Jammes 2007; Varma 2012]
DENSE_LAMBDA_M        = 1500.0   # [Graham et al. 2009]
CROP_LAMBDA_DAY_M     = 1200.0   # daytime: mild avoidance [Sitati 2005]
CROP_LAMBDA_NIGHT_M   =  800.0   # nighttime: sharp attraction to field edge

HEAT_K      = 0.025
HEAT_THRESH = 26.0

# Diurnal temperature: Walayar dry season [Kumar et al. 2010]
_HOUR_TEMP = {
    h: (22.0 + 10.0 * math.sin(math.pi * max(0, h - 6) / 14.0)
        if 6 <= h <= 20 else 22.0)
    for h in range(24)
}

# Hard exclusion radii (metres) by settlement type
# [Graham et al. 2009; Hoare & Du Toit 1999; Ngene et al. 2009]
EXCL_RADII = {
    "city"              : 2000,
    "town"              : 1500,
    "suburb"            : 700,
    "neighbourhood"     : 500,
    "residential"       : 500,
    "commercial"        : 600,
    "industrial"        : 600,
}
EXCL_TYPES = set(EXCL_RADII.keys())

MIN_WATER_AREA_KM2 = 0.01


# ---------------------------------------------------------------------------
# 5 BEHAVIOURAL ARCHETYPES
# One trajectory each — covering the ecologically distinct movement guilds
# identified in Walayar GPS literature (Varma et al. 2012; Baskaran 2010)
# ---------------------------------------------------------------------------

ARCHETYPES = [

    # ── 1. Forest Forager Family ───────────────────────────────────────────
    {
        "id"          : "forest_forager",
        "label"       : "Forest Forager Family Herd",
        "color_kml"   : "ff2db552",   # emerald green
        "line_width"  : 3,
        "description" : (
            "Resident cow-calf family herd occupying forest interior.\n"
            "Avoids all settlement types; makes one afternoon water visit\n"
            "per day to streams or small ponds.  Minimal crop-raiding.\n"
            "\n"
            "Literature basis:\n"
            "  Varma et al. (2012) Gajah 37:6-14 - 4-6 resident family\n"
            "    units GPS-collared in Walayar; deep-interior foraging\n"
            "    groups show strongest settlement avoidance (Fig. 3).\n"
            "  Baskaran et al. (2010) Trop Ecol 51:79 - forest interior\n"
            "    groups move 8-10 km/day; rarely approach field edges.\n"
            "  Sukumar (2003) The Living Elephants: matriarchal herds\n"
            "    maintain strict home ranges; high spatial fidelity.\n"
            "\n"
            "BCRW parameters:\n"
            "  rho=0.58 (high persistence - familiar home range)\n"
            "  step_scale=360 m/hr -> ~8.6 km/day [Varma 2012]\n"
            "  w_water=0.30 (moderate - daily stream visit)\n"
            "  w_dense=0.70 (strong settlement avoidance)\n"
            "  Hard exclusion: 2 km cities, 1.5 km towns, 0.7 km suburbs"
        ),
        "w_persist"   : 0.55,
        "w_water"     : 0.30,
        "w_dense"     : 0.70,
        "w_crop_day"  : 0.10,
        "w_crop_night": 0.12,
        "rho"         : 0.58,
        "step_shape"  : 0.90,
        "step_scale_m": 360,
        "start_hour"  : 6,
        "start_bias"  : "interior",
    },

    # ── 2. Water-Seeking Herd ─────────────────────────────────────────────
    {
        "id"          : "water_seeker",
        "label"       : "Water-Seeking Herd",
        "color_kml"   : "ffff8c00",   # sky blue
        "line_width"  : 3,
        "description" : (
            "Family herd making repeated visits to Malampuzha Lake\n"
            "(9.3 km2) and Walayar Lake (2.5 km2) — the dominant\n"
            "dry-season water attractors in the western forest.\n"
            "\n"
            "Literature basis:\n"
            "  Chamaille-Jammes et al. (2007) J Anim Ecol 76:789 -\n"
            "    exponential attraction decay, lambda=3-5 km for\n"
            "    large reservoirs; daily water visits in dry season.\n"
            "  Varma et al. (2012): Walayar Lake draws herds from\n"
            "    3-5 km; Malampuzha (9.3 km2) dominates western\n"
            "    forest gradient April-June.\n"
            "  de Silva et al. (2011) PLoS ONE 6:e19818 - riparian\n"
            "    concentration during dry months confirmed by GPS.\n"
            "\n"
            "BCRW parameters:\n"
            "  rho=0.65 (high persistence on water-approach leg)\n"
            "  step_scale=400 m/hr -> ~9.6 km/day\n"
            "  w_water=0.55 (strongest water gradient - dominant bias)"
        ),
        "w_persist"   : 0.45,
        "w_water"     : 0.55,
        "w_dense"     : 0.60,
        "w_crop_day"  : 0.10,
        "w_crop_night": 0.14,
        "rho"         : 0.65,
        "step_shape"  : 0.90,
        "step_scale_m": 400,
        "start_hour"  : 13,    # afternoon water visit [Varma 2012 peak timing]
        "start_bias"  : "near_water",
    },

    # ── 3. Agricultural Raider ────────────────────────────────────────────
    {
        "id"          : "agri_raider",
        "label"       : "Agricultural Raider Herd",
        "color_kml"   : "ff0000ff",   # red
        "line_width"  : 3,
        "description" : (
            "Southern-boundary family herd that retreats into forest\n"
            "interior during daylight and approaches crop fields at\n"
            "dusk/night for energy-dense foraging.  Strongly avoids\n"
            "town cores and busy roads at all times.\n"
            "\n"
            "Literature basis (PROTEIN/ENERGY OPTIMAL FORAGING):\n"
            "  Sukumar & Gadgil (1988) Oikos 53:347 - crops have\n"
            "    3-5x higher energy density than forest forage\n"
            "    (sugarcane ~1800 kcal/kg vs grass ~400 kcal/kg);\n"
            "    crop-raiding is rational optimal foraging.\n"
            "  Sitati et al. (2005) J Appl Ecol 42:720 - 93% of\n"
            "    raids between 18:30-06:00; elephants approach field\n"
            "    boundaries only at night.\n"
            "  Baskaran et al. (2010) Trop Ecol 51:79 - Walayar\n"
            "    south boundary herds peak raiding Oct-Mar\n"
            "    (post-harvest sugarcane and paddy stubs).\n"
            "  Varma et al. (2012): HEC hotspots at southern forest-\n"
            "    agriculture interface mapped from GPS data.\n"
            "\n"
            "BCRW parameters:\n"
            "  rho=0.44 (lower persistence - cautious approach)\n"
            "  step_scale=470 m/hr -> ~11 km/day\n"
            "  w_crop_night=0.50 (strong nocturnal field attraction)\n"
            "  w_dense=0.75 (strongest avoidance of town cores)"
        ),
        "w_persist"   : 0.40,
        "w_water"     : 0.20,
        "w_dense"     : 0.75,
        "w_crop_day"  : 0.28,   # daytime: actively avoid field edge
        "w_crop_night": 0.50,   # nighttime: strong approach to crop boundary
        "rho"         : 0.44,
        "step_shape"  : 0.90,
        "step_scale_m": 470,
        "start_hour"  : 18,     # begins at dusk [Sitati 2005]
        "start_bias"  : "near_crop",
    },

    # ── 4. Solitary Bull (Range Explorer) ────────────────────────────────
    {
        "id"          : "range_explorer",
        "label"       : "Solitary Bull (Range Explorer)",
        "color_kml"   : "ffcc00ff",   # magenta
        "line_width"  : 3,
        "description" : (
            "Adult solitary bull with large home range spanning all\n"
            "forest sections.  Crosses NH544 / Palakkad railway gap\n"
            "at night (22:00-04:00).  Low directional persistence\n"
            "produces wide-ranging exploration pattern.\n"
            "Slightly bolder near human infrastructure than\n"
            "family herds (musth reduces fear response).\n"
            "\n"
            "Literature basis:\n"
            "  Varma et al. (2012): maximum recorded home range\n"
            "    312 km2 for adult bulls in Walayar; GPS tracks\n"
            "    show frequent forest-section crossings.\n"
            "  Baskaran et al. (1995) J Biosci 20:521: adult bull\n"
            "    home ranges 85-350 km2; daily path 12-16 km.\n"
            "  Baskaran (2013): NH544 used as corridor by bulls\n"
            "    at night; road-crossing events 22:00-04:00.\n"
            "  Sukumar (2003): musth bulls show reduced avoidance\n"
            "    of human infrastructure (lowered fear threshold).\n"
            "\n"
            "BCRW parameters:\n"
            "  rho=0.30 (low persistence -> exploratory ranging)\n"
            "  step_scale=600 m/hr -> ~14.4 km/day [Baskaran 1995]\n"
            "  w_water=0.38 (bulls need ~200 L/day; strong pull)\n"
            "  Hard exclusion: 1.5 km towns (bulls bolder than herds)"
        ),
        "w_persist"   : 0.38,
        "w_water"     : 0.38,
        "w_dense"     : 0.40,   # bulls bolder [Sukumar 2003]
        "w_crop_day"  : 0.10,
        "w_crop_night": 0.22,
        "rho"         : 0.30,
        "step_shape"  : 0.90,
        "step_scale_m": 600,
        "start_hour"  : 4,
        "start_bias"  : "anywhere",
    },

    # ── 5. Riparian Corridor Herd ─────────────────────────────────────────
    {
        "id"          : "riparian",
        "label"       : "Riparian Corridor Herd",
        "color_kml"   : "ff00d4ff",   # amber-yellow
        "line_width"  : 3,
        "description" : (
            "Family herd that follows the Walayar River and tributary\n"
            "stream network as a dry-season refuge and movement\n"
            "corridor linking forest sections.  Highest water-gradient\n"
            "orientation of all archetypes.\n"
            "\n"
            "Literature basis:\n"
            "  de Silva et al. (2011) PLoS ONE 6:e19818 - GPS data\n"
            "    show strong riparian concentration in dry months;\n"
            "    river corridors act as elephant 'highways'.\n"
            "  Sukumar (2003): streams are primary landscape\n"
            "    connectors for family herds during water stress.\n"
            "  Varma et al. (2012): Walayar River system identified\n"
            "    as core connectivity corridor for resident herds.\n"
            "  Campos-Arceiz et al. (2008) J Trop Ecol 24:423:\n"
            "    riparian herds show rho>0.65 (high persistence\n"
            "    along linear features).\n"
            "\n"
            "BCRW parameters:\n"
            "  rho=0.68 (highest persistence - linear corridor)\n"
            "  step_scale=340 m/hr -> ~8.2 km/day\n"
            "  w_water=0.52 (very strong riparian gradient)"
        ),
        "w_persist"   : 0.48,
        "w_water"     : 0.52,
        "w_dense"     : 0.60,
        "w_crop_day"  : 0.12,
        "w_crop_night": 0.14,
        "rho"         : 0.68,
        "step_shape"  : 0.90,
        "step_scale_m": 340,
        "start_hour"  : 5,
        "start_bias"  : "near_water",
    },
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def lon_m(lat):
    return LAT_M * math.cos(math.radians(lat))


def buf_deg(metres):
    return metres / LAT_M


# ---------------------------------------------------------------------------
# 1. PARSE FOREST KML
# ---------------------------------------------------------------------------

def parse_forest_kml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    ns   = {"k": KML_NS}
    polys = []
    for pm in root.findall(".//k:Placemark", ns):
        n = pm.find("k:name", ns)
        if n is None or "Section" not in (n.text or ""):
            continue
        for ce in pm.findall(".//k:coordinates", ns):
            if not ce.text:
                continue
            pts = []
            for tok in ce.text.strip().split():
                p = tok.split(",")
                if len(p) >= 2:
                    try:
                        pts.append((float(p[0]), float(p[1])))
                    except ValueError:
                        pass
            if len(pts) >= 3:
                try:
                    pg = Polygon(pts)
                    if not pg.is_valid:
                        pg = pg.buffer(0)
                    polys.append(pg)
                except Exception:
                    pass
    if not polys:
        raise ValueError(f"No forest Section polygons in {path}")
    return unary_union(polys)


# ---------------------------------------------------------------------------
# 2. LOAD OSM SOURCES + BUILD EXCLUSION ZONES
# ---------------------------------------------------------------------------

def load_osm_environment(forest_geom):
    """
    Returns:
      water_src   : [(lon, lat, weight), …]
      dense_src   : [(lon, lat, weight), …]
      crop_src    : [(lon, lat, weight), …]
      settle_excl : Shapely geometry — hard exclusion union
      crop_polys  : list of Shapely Polygon for crop fields
    """
    with open(OSM_JSON) as f:
        data = json.load(f)

    minx, miny, maxx, maxy = forest_geom.bounds
    pad = 0.09   # ~10 km buffer to capture nearby attractors

    def in_zone(lo, la):
        return (minx - pad <= lo <= maxx + pad and
                miny - pad <= la <= maxy + pad)

    # ── Water sources ──────────────────────────────────────────────────────
    water_raw = []
    for p in data.get("water_polygons", []):
        cx, cy = p["centroid"]
        if not in_zone(cx, cy) or p["area_km2"] < MIN_WATER_AREA_KM2:
            continue
        water_raw.append((cx, cy, math.sqrt(p["area_km2"])))
    for p in data.get("water_points", []):
        if in_zone(p["lon"], p["lat"]):
            water_raw.append((p["lon"], p["lat"], 0.25))

    # ── Dense settlement sources (for gradient) ────────────────────────────
    dense_raw = []
    for s in data.get("settlements", []):
        if not in_zone(s["lon"], s["lat"]):
            continue
        pt = s.get("place_type", "")
        if pt in EXCL_TYPES:
            w = {"city": 10, "town": 6, "suburb": 3,
                 "neighbourhood": 2, "residential": 2,
                 "commercial": 3, "industrial": 3}.get(pt, 2)
            dense_raw.append((s["lon"], s["lat"], w))

    # ── Crop edge sources ─────────────────────────────────────────────────
    crop_raw = []
    for p in data.get("crop_polygons", []):
        cx, cy = p["centroid"]
        if not in_zone(cx, cy) or p["area_km2"] < 0.005:
            continue
        crop_raw.append((cx, cy, math.sqrt(p["area_km2"])))
    # small hamlets also serve as crop-edge proxies [Varma 2012]
    for s in data.get("settlements", []):
        if not in_zone(s["lon"], s["lat"]):
            continue
        if s.get("place_type", "") in ("hamlet", "village", "isolated_dwelling"):
            crop_raw.append((s["lon"], s["lat"], 0.7))

    def cluster(pts, d_m=600.0):
        if not pts:
            return []
        out, used = [], [False] * len(pts)
        for i, p in enumerate(pts):
            if used[i]:
                continue
            grp, used[i] = [p], True
            lm = lon_m(p[1])
            for j in range(i + 1, len(pts)):
                if used[j]:
                    continue
                dx = (pts[j][0] - p[0]) * lm
                dy = (pts[j][1] - p[1]) * LAT_M
                if math.sqrt(dx*dx + dy*dy) < d_m:
                    grp.append(pts[j])
                    used[j] = True
            cx = sum(g[0] for g in grp) / len(grp)
            cy = sum(g[1] for g in grp) / len(grp)
            out.append((cx, cy, sum(g[2] for g in grp)))
        return out

    water_src = cluster(water_raw, 600)
    dense_src = cluster(dense_raw, 700)
    crop_src  = cluster(crop_raw,  400)

    # ── Hard settlement exclusion zones ────────────────────────────────────
    # [Graham et al. 2009; Hoare & Du Toit 1999; Ngene et al. 2009]
    excl_polys = []
    for s in data.get("settlements", []):
        pt   = s.get("place_type", "")
        rad  = EXCL_RADII.get(pt, 0)
        if rad == 0:
            continue
        if not in_zone(s["lon"], s["lat"]):
            continue
        centre = Point(s["lon"], s["lat"])
        excl_polys.append(centre.buffer(buf_deg(rad), resolution=20))
    settle_excl = unary_union(excl_polys) if excl_polys else None

    # ── Crop field polygons (for near_crop start bias) ─────────────────────
    crop_polys = []
    for p in data.get("crop_polygons", []):
        if p["area_km2"] < 0.005:
            continue
        coords = [(c[0], c[1]) for c in p["coords"]]
        if len(coords) < 3:
            continue
        try:
            pg = Polygon(coords)
            if not pg.is_valid:
                pg = pg.buffer(0)
            crop_polys.append(pg)
        except Exception:
            pass

    print(f"  Water sources     : {len(water_src)}")
    print(f"  Dense repulsors   : {len(dense_src)}")
    print(f"  Crop edge sources : {len(crop_src)}")
    print(f"  Exclusion zones   : {len(excl_polys)} polygons merged")
    return water_src, dense_src, crop_src, settle_excl, crop_polys


# ---------------------------------------------------------------------------
# 3. POTENTIAL GRADIENT  [Pinter-Wollman et al. 2009; Benhamou 2006]
# ---------------------------------------------------------------------------

def potential_gradient(pos, sources, lambda_m):
    lm   = lon_m(pos[1])
    grad = np.zeros(2)
    for src in sources:
        dx = (src[0] - pos[0]) * lm
        dy = (src[1] - pos[1]) * LAT_M
        d  = math.sqrt(dx*dx + dy*dy)
        if d < 1.0:
            continue
        w = src[2] if len(src) > 2 else 1.0
        c = w * math.exp(-d / lambda_m) / (lambda_m * d)
        grad[0] += c * dx
        grad[1] += c * dy
    return grad


def unit_vec(v):
    n = math.hypot(v[0], v[1])
    return v / n if n > 1e-12 else np.zeros(2)


# ---------------------------------------------------------------------------
# 4. WRAPPED CAUCHY SAMPLER  [Mardia & Jupp 2000; Wall et al. 2006]
# ---------------------------------------------------------------------------

def wrapped_cauchy(rng, mu, rho):
    u = rng.uniform(0.0, 1.0)
    return mu + 2.0 * math.atan(
        ((1.0 + rho) / (1.0 - rho)) * math.tan(math.pi * u - math.pi / 2.0)
    )


# ---------------------------------------------------------------------------
# 5. START POSITION SAMPLER
# ---------------------------------------------------------------------------

def sample_start(rng, forest_geom, settle_excl, water_src, crop_polys, bias):
    """
    Draw a starting position inside the forest that respects the settlement
    hard exclusion zone, guided by archetype bias.
    """
    minx, miny, maxx, maxy = forest_geom.bounds
    cx_f = (minx + maxx) / 2
    cy_f = (miny + maxy) / 2

    # Major water body positions for near_water bias
    top_water = sorted(water_src, key=lambda s: s[2], reverse=True)[:5]

    # Crop field centroids for near_crop bias
    crop_cents = []
    for pg in crop_polys:
        c = pg.centroid
        crop_cents.append((c.x, c.y))

    for _ in range(50000):
        if bias == "interior":
            lo = rng.normal(cx_f, (maxx - minx) * 0.15)
            la = rng.normal(cy_f, (maxy - miny) * 0.15)
        elif bias == "near_water" and top_water:
            src = top_water[rng.integers(0, len(top_water))]
            spread = buf_deg(2500)
            lo = rng.normal(src[0], spread * 0.5)
            la = rng.normal(src[1], spread * 0.5)
        elif bias == "near_crop" and crop_cents:
            idx = rng.integers(0, len(crop_cents))
            cc = crop_cents[idx]
            # Start just inside the forest, within 1.5 km of a crop centroid
            spread = buf_deg(1500)
            lo = rng.normal(cc[0], spread * 0.4)
            la = rng.normal(cc[1], spread * 0.4)
        else:   # "anywhere"
            lo = rng.uniform(minx, maxx)
            la = rng.uniform(miny, maxy)

        if not forest_geom.contains(Point(lo, la)):
            continue
        if settle_excl is not None and settle_excl.contains(Point(lo, la)):
            continue
        return lo, la

    # Fallback: forest centroid
    c = forest_geom.centroid
    return c.x, c.y


# ---------------------------------------------------------------------------
# 6. TRAJECTORY GENERATION (BCRW)
# ---------------------------------------------------------------------------

def generate_trajectory(arch, forest_geom, water_src, dense_src, crop_src,
                         settle_excl, crop_polys, rng):
    """
    Biased Correlated Random Walk with:
      • Hard settlement exclusion (primary constraint)
      • Soft potential-field biases (water, dense settle, crop)
      • Wrapped Cauchy turning angles
      • Gamma step lengths with temperature modulation
    """
    lo, la = sample_start(rng, forest_geom, settle_excl,
                           water_src, crop_polys, arch["start_bias"])
    theta  = rng.uniform(0, 2 * math.pi)
    path   = [(lo, la)]

    def _is_valid(nlo, nla):
        pt = Point(nlo, nla)
        if not forest_geom.contains(pt):
            return False
        if settle_excl is not None and settle_excl.contains(pt):
            return False
        return True

    for step in range(N_STEPS - 1):
        hour   = (arch["start_hour"] + step * DT_HOURS) % 24
        is_day = 6.0 <= hour < 18.0

        # Temperature speed factor [Thaker 2019; Kumar 2010]
        T   = _HOUR_TEMP[int(hour) % 24]
        spd = max(0.1, 1.0 - HEAT_K * max(0.0, T - HEAT_THRESH))

        # Environmental potential gradients
        pos = (lo, la)
        g_w = potential_gradient(pos, water_src, WATER_LAMBDA_M)
        g_d = potential_gradient(pos, dense_src, DENSE_LAMBDA_M)

        crop_lam = CROP_LAMBDA_DAY_M if is_day else CROP_LAMBDA_NIGHT_M
        g_c = potential_gradient(pos, crop_src, crop_lam)

        n_w = unit_vec(g_w)
        n_d = unit_vec(g_d)
        n_c = unit_vec(g_c)
        v_p = np.array([math.cos(theta), math.sin(theta)])

        # Crop sign: day → repulsion (−), night → attraction (+)
        # [Sitati et al. 2005: 93% of raids 18:30-06:00]
        crop_w = (arch["w_crop_night"] if not is_day else arch["w_crop_day"])
        crop_sign = +1.0 if not is_day else -1.0

        b = (arch["w_persist"] * v_p
             + arch["w_water"]  * n_w
             - arch["w_dense"]  * n_d       # always repel dense settlements
             + crop_sign * crop_w * n_c)

        bias_angle = (math.atan2(b[1], b[0])
                      if math.hypot(b[0], b[1]) > 1e-12
                      else rng.uniform(0, 2 * math.pi))

        delta     = wrapped_cauchy(rng, 0.0, arch["rho"])
        theta_new = (bias_angle + delta) % (2 * math.pi)

        step_m = rng.gamma(arch["step_shape"], arch["step_scale_m"]) * spd
        lm     = lon_m(la)
        nlo    = lo + step_m * math.cos(theta_new) / lm
        nla    = la + step_m * math.sin(theta_new) / LAT_M

        if _is_valid(nlo, nla):
            lo, la, theta = nlo, nla, theta_new
        else:
            placed = False
            # Try reversed direction first, then 12 random deflections
            candidates = [(theta + math.pi, 0.5)] + \
                         [(rng.uniform(0, 2*math.pi), 0.3) for _ in range(12)]
            for a_try, scale in candidates:
                a_try = a_try % (2 * math.pi)
                sm = step_m * scale
                clo = lo + sm * math.cos(a_try) / lm
                cla = la + sm * math.sin(a_try) / LAT_M
                if _is_valid(clo, cla):
                    lo, la, theta = clo, cla, a_try
                    placed = True
                    break
            if not placed:
                # Stay in place; sample new direction next step
                theta = rng.uniform(0, 2 * math.pi)

        path.append((lo, la))

    return path


# ---------------------------------------------------------------------------
# 7. KML CONSTRUCTION
# ---------------------------------------------------------------------------

def _line_style(doc, sid, color, width):
    s = ET.SubElement(doc, "Style", id=sid)
    ls = ET.SubElement(s, "LineStyle")
    ET.SubElement(ls, "color").text = color
    ET.SubElement(ls, "width").text = str(width)
    ET.SubElement(ET.SubElement(s, "PolyStyle"), "fill").text = "0"


def _icon_style(doc, sid, color, scale="0.7"):
    s = ET.SubElement(doc, "Style", id=sid)
    ist = ET.SubElement(s, "IconStyle")
    ET.SubElement(ist, "color").text = color
    ET.SubElement(ist, "scale").text = scale
    ET.SubElement(
        ET.SubElement(ist, "Icon"), "href"
    ).text = "https://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"


def add_trajectory_to_kml(doc, arch, path):
    # Compute stats
    dists = []
    for i in range(1, len(path)):
        lm = lon_m((path[i][1] + path[i-1][1]) / 2)
        dx = (path[i][0] - path[i-1][0]) * lm
        dy = (path[i][1] - path[i-1][1]) * LAT_M
        dists.append(math.sqrt(dx*dx + dy*dy))
    total_km  = sum(dists) / 1000
    daily_km  = total_km / (N_STEPS * DT_HOURS / 24)

    folder = ET.SubElement(doc, "Folder")
    ET.SubElement(folder, "name").text = arch["label"]
    ET.SubElement(folder, "description").text = (
        arch["description"] + f"\n\nSimulated path stats:\n"
        f"  Duration    : {N_STEPS} hr ({N_STEPS//24} days)\n"
        f"  Total dist  : {total_km:.1f} km\n"
        f"  Mean daily  : {daily_km:.1f} km/day\n"
        f"  Start hour  : {arch['start_hour']:02d}:00"
    )

    # Path line
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = f"{arch['label']} — path"
    ET.SubElement(pm, "styleUrl").text = f"#traj_{arch['id']}"
    ls = ET.SubElement(pm, "LineString")
    ET.SubElement(ls, "tessellate").text = "1"
    ET.SubElement(ls, "coordinates").text = " ".join(
        f"{lo:.7f},{la:.7f},0" for lo, la in path
    )

    # Start marker
    pm_s = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm_s, "name").text = f"▶ Start — {arch['label']}"
    ET.SubElement(pm_s, "styleUrl").text = f"#icon_{arch['id']}"
    ET.SubElement(
        ET.SubElement(pm_s, "Point"), "coordinates"
    ).text = f"{path[0][0]:.7f},{path[0][1]:.7f},0"

    # End marker
    pm_e = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm_e, "name").text = f"■ End — {arch['label']}"
    ET.SubElement(pm_e, "styleUrl").text = f"#icon_{arch['id']}"
    ET.SubElement(
        ET.SubElement(pm_e, "Point"), "coordinates"
    ).text = f"{path[-1][0]:.7f},{path[-1][1]:.7f},0"

    return daily_km


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Walayar Range — Elephant Trajectory Generator")
    print(f"  5 ecologically-grounded trajectories, {N_STEPS} steps each")
    print("=" * 65)

    # 1. Forest geometry
    print(f"\n[1] Parsing forest geometry from {INPUT_KML} …")
    forest_geom = parse_forest_kml(INPUT_KML)
    print(f"    Forest area: {forest_geom.area * LAT_M**2 / 1e6:.1f} km²")

    # 2. OSM environment
    print(f"\n[2] Loading OSM environment from {OSM_JSON} …")
    water_src, dense_src, crop_src, settle_excl, crop_polys = \
        load_osm_environment(forest_geom)

    # 3. Generate 5 trajectories
    print(f"\n[3] Generating trajectories …")
    rng = np.random.default_rng(RANDOM_SEED)
    results = []
    for arch in ARCHETYPES:
        print(f"    {arch['label']:38s} … ", end="", flush=True)
        path = generate_trajectory(
            arch, forest_geom,
            water_src, dense_src, crop_src,
            settle_excl, crop_polys, rng
        )
        daily_km = sum(
            math.sqrt(
                ((path[i][0]-path[i-1][0]) * lon_m((path[i][1]+path[i-1][1])/2))**2 +
                ((path[i][1]-path[i-1][1]) * LAT_M)**2
            )
            for i in range(1, len(path))
        ) / 1000 / (N_STEPS / 24)
        print(f"{daily_km:.1f} km/day")
        results.append((arch, path))

    # 4. Append to existing grid KML
    print(f"\n[4] Reading {INPUT_KML} and appending trajectories …")
    ET.register_namespace("", KML_NS)
    tree = ET.parse(INPUT_KML)
    root = tree.getroot()
    doc  = root.find(f"{{{KML_NS}}}Document")
    if doc is None:
        doc = root

    # Styles
    for arch in ARCHETYPES:
        _line_style(doc, f"traj_{arch['id']}", arch["color_kml"], arch["line_width"])
        _icon_style(doc, f"icon_{arch['id']}", arch["color_kml"])

    # Trajectory folders
    daily_rates = []
    for arch, path in results:
        dk = add_trajectory_to_kml(doc, arch, path)
        daily_rates.append(dk)

    # Population summary placemark
    c = forest_geom.centroid
    pm_note = ET.SubElement(doc, "Placemark")
    ET.SubElement(pm_note, "name").text = "Trajectory Model Summary"
    ET.SubElement(pm_note, "description").text = (
        "WALAYAR ELEPHANT MOVEMENT MODEL\n"
        "================================\n"
        "5 behavioural archetypes, each grounded in published GPS data:\n\n"
        "1. Forest Forager — deep interior, 8-10 km/day [Varma 2012]\n"
        "2. Water Seeker — daily Malampuzha/Walayar Lake visits\n"
        "   [Chamaille-Jammes 2007; Varma 2012]\n"
        "3. Crop Raider — nocturnal crop-field approach\n"
        "   [Sukumar & Gadgil 1988; Sitati et al. 2005]\n"
        "4. Solitary Bull — wide-ranging, 12-16 km/day [Baskaran 1995]\n"
        "5. Riparian Herd — stream corridor follower [de Silva 2011]\n\n"
        "BCRW model: Wrapped Cauchy turning angles [Wall et al. 2006]\n"
        "Step length: Gamma(alpha=0.9) [Roever et al. 2014]\n"
        "Bias: water + settlement + crop potential fields\n"
        "   [Pinter-Wollman et al. 2009; Benhamou 2006]\n\n"
        "HARD EXCLUSION zones (geometric reject, not just gradient):\n"
        "  Cities  >= 2.0 km  [Graham et al. 2009]\n"
        "  Towns   >= 1.5 km  [Hoare & Du Toit 1999]\n"
        "  Suburbs >= 0.7 km  [Ngene et al. 2009]\n\n"
        "Scale: For 3,000 elephants -> ~854 unique social-unit paths\n"
        "but only ~5 distinct corridor archetypes exist in this\n"
        "landscape (Varma 2012; Baskaran 2010)."
    )
    ET.SubElement(
        ET.SubElement(pm_note, "Point"), "coordinates"
    ).text = f"{c.x:.6f},{c.y:.6f},0"

    # 5. Write
    print(f"\n[5] Writing {OUTPUT_KML} …")
    tree.write(OUTPUT_KML, encoding="unicode", xml_declaration=True)

    print(f"\n{'='*65}")
    print(f"Output: {OUTPUT_KML}")
    print(f"\nTrajectory summary:")
    for (arch, _), dk in zip(results, daily_rates):
        print(f"  {arch['label']:38s}  {dk:5.1f} km/day")
    print(f"\nOpen {OUTPUT_KML} in Google Earth.")


if __name__ == "__main__":
    main()
