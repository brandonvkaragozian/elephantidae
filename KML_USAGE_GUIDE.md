# Using the Camera Placements KML with Google Maps & Google Earth

## File Information

**File:** `camera_placements_16_cameras.kml`  
**Size:** 26 KB  
**Format:** KML 2.2 (OGC standard)  
**Contents:** 16 camera placements organized in 3 deployment tiers  
**Status:** Ready for upload to Google Maps & Google Earth

---

## How to Open in Google Maps

### Method 1: Direct Upload to My Maps

1. **Open Google Maps**: https://www.google.com/maps
2. **Create or select a map**:
   - Click menu (☰) → "Maps"
   - Click "Create a new map"
   - Name: "Walayar Camera Placement"
3. **Import the KML**:
   - Click "Import layer"
   - Upload `camera_placements_16_cameras.kml`
   - Or drag and drop the file into the map

4. **Visualize**:
   - All 16 cameras appear with camera icons
   - Color-coded by tier (red/green/blue)
   - Click each camera for details

5. **Share**:
   - Click "Share" button
   - Generate shareable link
   - Send to team members

### Method 2: Upload via File

1. **Use Google Drive**:
   - Upload `camera_placements_16_cameras.kml` to Google Drive
   - Right-click → "Open with" → "Google Maps"

2. **Instant preview** with all layers and styling

---

## How to Open in Google Earth

### Desktop Version

1. **Download**: https://www.google.com/earth/versions/
2. **Open Google Earth Pro**
3. **File** → **Open** → Select `camera_placements_16_cameras.kml`
4. **Features**:
   - Full 3D visualization
   - Terrain overlay
   - Better detail than Google Maps

### Web Version

1. **Open Google Earth**: https://earth.google.com/web/
2. **Projects** → **Open** → Upload `camera_placements_16_cameras.kml`
3. **Instant 3D visualization** with satellite imagery

---

## What You'll See

### Color-Coded Camera Tiers

| Tier | Color | Cameras | Purpose |
|------|-------|---------|---------|
| **Tier 1** | 🔴 Red | 1-4 | High-coverage anchors (northern boundary) |
| **Tier 2** | 🟢 Green | 5-13 | Perimeter reinforcement (eastern boundary) |
| **Tier 3** | 🔵 Blue | 14-16 | Activity hotspot monitors (central sanctuary) |

### Camera Details (Click Any Camera)

Each camera placemark includes:

```
Camera Placement Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rank: 1 of 16
Cell ID: R000C000
Tier: ANCHOR
Coordinates: 10.751572°N, 76.625259°E

Coverage Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cells Visible: 1071
Demand Coverage: 99.62
Elephant Visits in Range: 1472
Coverage Radius: 40 km (360° view)

Deployment Priority:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 (Critical): True
Phase 2 (Recommended): False
Phase 3 (Optional): False
```

---

## Spatial Reference

### Sanctuary Boundaries

All cameras are within Walayar Wildlife Sanctuary:
- **Latitude**: 10.7516°N to 10.8414°N
- **Longitude**: 76.6253°E to 76.8539°E
- **Area**: 250 km²
- **Grid**: 1,071 cells (500m × 500m each)

### Camera Distribution

```
┌─────────────────────────────────────────────────────┐
│  WALAYAR WILDLIFE SANCTUARY CAMERA NETWORK          │
│                                                      │
│  Tier 1 (Red): Along northern boundary              │
│    C1, C2, C3, C4                                   │
│                                                      │
│  Tier 2 (Green): Eastern perimeter spread           │
│    C5-C13 distributed for redundancy                │
│                                                      │
│  Tier 3 (Blue): Central hotspot                     │
│    C14, C15, C16 at elephant concentration areas    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Deployment Guidance

### Phase 1 (CRITICAL - 4 cameras)

**Deploy immediately** for 100% coverage:
- **R000C000** ← Start here (covers 1,071 cells alone!)
- **R000C001** (backup)
- **R000C003** (redundancy)
- **R000C007** (perimeter)

**Result**: Complete sanctuary coverage with minimal setup

### Phase 2 (RECOMMENDED - 9 cameras)

**Deploy after Phase 1** for operational redundancy:
- R000C042, R000C046, R000C047, R000C048
- R001C002, R001C007, R001C008, R001C009
- R002C003

**Result**: 11.24x average redundancy, can lose cameras during maintenance

### Phase 3 (OPTIONAL - 3 cameras)

**Deploy for intensive hotspot monitoring**:
- **R012C022** ← Maximum elephant activity (24 visits!)
- R012C023
- R012C026

**Result**: High-resolution behavior tracking at concentration areas

---

## Working with the KML

### Edit in Google My Maps

1. Right-click any camera
2. **Edit** to modify:
   - Name
   - Description
   - Icon color
   - Notes

3. **Delete** to remove cameras
4. **Add** new points (if needed)

### Export Back to KML

1. Click menu (⋮) → **Download**
2. Select **KML** format
3. Use updated file for further analysis

### Combine with Other Layers

- Overlay with elevation data
- Add satellite imagery
- Display administrative boundaries
- Show protected zones
- Mark water sources

---

## Troubleshooting

### Camera icons not showing?
- **Solution**: Refresh the page
- The KML uses Google's camera icon (may take moment to load)

### Colors not displaying?
- **Solution**: Zoom in closer
- Colors are more visible at higher zoom levels
- In Google Earth, colors always visible

### File too large?
- **No issue**: 26 KB is well within limits
- Easily shareable via email/cloud

### Need different styling?
- **Edit source**: Open `camera_placements_16_cameras.kml` in text editor
- Change color codes (ABGR hex format):
  - ff0000ff = Red
  - ff00ff00 = Green
  - ffff0000 = Blue
- Regenerate via Python script: `python3 generate_camera_kml.py`

---

## Technical Details

### KML Structure

```xml
<kml>
  <Document>
    <name>Walayar Wildlife Sanctuary - 16 Camera Placements</name>
    <Folder>Tier 1: High-Coverage Anchors</Folder>
    <Folder>Tier 2: Perimeter Reinforcement</Folder>
    <Folder>Tier 3: Activity Hotspot Monitors</Folder>
    
    <!-- 16 Placemark entries with:
         - Camera icon
         - Coordinates (lat/lon)
         - Detailed description
         - Extended data fields
         - Styling -->
  </Document>
</kml>
```

### Data Fields Per Camera

- **Rank**: Deployment priority (1-16)
- **Cell_ID**: Grid cell identifier (e.g., R000C000)
- **Visible_Cells**: Number of grid cells this camera can observe
- **Demand_Coverage**: Weighted elephant activity coverage
- **Elephant_Visits**: Total visits in observable range
- **Tier**: Deployment phase (ANCHOR, REINFORCEMENT, HOTSPOT)

---

## Generated Files

| File | Purpose |
|------|---------|
| **camera_placements_16_cameras.kml** | ← **MAIN FILE: Use this for Google Maps/Earth** |
| generate_camera_kml.py | Python script to regenerate KML if needed |
| camera_placement_enhanced_mip_16_cameras.csv | Raw camera data (alternative to KML) |

---

## Next Steps

1. **Upload** `camera_placements_16_cameras.kml` to Google Maps
2. **Validate** camera locations on satellite imagery
3. **Plan Phase 1** deployment (4 critical cameras)
4. **Conduct ground survey** to verify line-of-sight
5. **Install equipment** following phase guidance

---

## Questions?

- **What if a camera fails?**: 4 other cameras will cover that location (11.24x redundancy)
- **Can I modify placements?**: Yes, edit in Google My Maps and export updated KML
- **Do I need all 16?**: No - 2 cameras suffice for 100% coverage, 16 provides resilience
- **What's the 40km range?**: Maximum line-of-sight distance for camera visibility

---

*KML Generated: April 17, 2026*  
*Based on Mixed Integer Programming optimization*  
*16 cameras, 3 deployment tiers, 100% coverage*
