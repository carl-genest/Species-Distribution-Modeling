# Project 3: Species Distribution Modeling (SDM)
## Phase 1, Week 1 — Day 1 Execution Guide

**Estimated Time:** 60–90 minutes  
**Core Objective:** Initialize the development environment, establish directory scaffolding, and validate API connectivity to GBIF by pulling and mapping a 1,000-record test probe for *Acer saccharum* (Sugar Maple) across Eastern Canada and the Northeastern United States.

---

## 1. Day 1 Checklist

- [ ] **1. Scoping:** Confirm focal taxon (*Acer saccharum*) and geographic study bounding box.
- [ ] **2. Git & Scaffolding:** Initialize git repository, configure `.gitignore`, and create modular directory tree.
- [ ] **3. Environment Setup:** Create isolated Conda/Mamba environment with geospatial C-bindings (`gdal`, `geos`, `proj`).
- [ ] **4. API Verification Script:** Write and execute `probe_gbif.py` to test taxon key resolution and sample data retrieval.
- [ ] **5. Diagnostic Mapping:** Inspect output GeoParquet file and generated diagnostic scatter plot.

---

## 2. Step-by-Step Instructions

### Step 1: Scaffolding and Version Control

Open your terminal and initialize the repository structure:

```bash
# Create project root directory
mkdir -p sdm-climate-niche
cd sdm-climate-niche

# Initialize version control
git init

# Build directory tree
mkdir -p data/raw data/processed notebooks src/ingestion tests docs

# Create a robust .gitignore to prevent committing large spatial assets
cat << 'EOF' > .gitignore
# Environments
venv/
env/
.conda/

# Data and large artifacts
data/
*.parquet
*.tif
*.tiff
*.nc
*.zip
*.tar.gz

# Python and Jupyter artifacts
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# System files
.DS_Store
Thumbs.db
EOF

git add .gitignore
git commit -m "chore: initial repository scaffolding and gitignore"
```

---

### Step 2: Environment Configuration

To prevent GDAL and PROJ binary incompatibilities, use `conda` or `mamba` to install dependencies from `conda-forge`:

```bash
# Create and activate environment
conda create -n sdm-env python=3.11 -y
conda activate sdm-env

# Install core geospatial and scientific dependencies
conda install -c conda-forge \
    geopandas \
    pygbif \
    rasterio \
    rioxarray \
    pyarrow \
    matplotlib \
    requests \
    ipykernel \
    -y
```

---

### Step 3: Run the Day 1 Ingestion Probe

Create a test script named `probe_gbif.py` in your repository root to confirm GBIF API connectivity, schema parsing, and spatial coordinate plotting:

```python
"""
probe_gbif.py
Validates GBIF API connectivity, taxon key resolution, and georeferenced data retrieval.
"""

from pygbif import occurrences as occ
from pygbif import species
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# 1. Configuration & Parameters
# ---------------------------------------------------------
SPECIES_NAME = "Acer saccharum"
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box: Eastern Canada & NE US Buffer
# Longitude: -82° to -64° | Latitude: 42° to 50°
WKT_POLYGON = "POLYGON((-82 42, -64 42, -64 50, -82 50, -82 42))"

print(f"Resolving GBIF taxon key for: '{SPECIES_NAME}'...")

# ---------------------------------------------------------
# 2. Taxon Resolution
# ---------------------------------------------------------
matched = species.name_backbone(name=SPECIES_NAME, rank="species")
if matched.get("matchType") == "NONE":
    raise ValueError(f"Could not resolve species '{SPECIES_NAME}' on GBIF backbone.")

usage_key = matched["usageKey"]
canonical_name = matched.get("canonicalName", SPECIES_NAME)
print(f"-> Successfully matched: {canonical_name} (Taxon Key: {usage_key})")

# ---------------------------------------------------------
# 3. Sample Ingestion Probe
# ---------------------------------------------------------
print("Fetching initial batch of 1,000 georeferenced occurrences...")
res = occ.search(
    taxonKey=usage_key,
    geometry=WKT_POLYGON,
    hasCoordinate=True,
    limit=1000
)

total_records = res.get("count", 0)
results = res.get("results", [])
print(f"-> Fetched {len(results)} sample records (Total available in bounding box: {total_records:,})")

if not results:
    raise RuntimeError("No records returned. Please verify bounding box coordinates.")

# ---------------------------------------------------------
# 4. DataFrame Construction & GeoData Processing
# ---------------------------------------------------------
parsed_records = []
for r in results:
    parsed_records.append({
        "gbifID": r.get("gbifID"),
        "scientificName": r.get("scientificName"),
        "decimalLatitude": r.get("decimalLatitude"),
        "decimalLongitude": r.get("decimalLongitude"),
        "coordinateUncertaintyInMeters": r.get("coordinateUncertaintyInMeters"),
        "year": r.get("year"),
        "basisOfRecord": r.get("basisOfRecord"),
        "datasetKey": r.get("datasetKey")
    })

df = pd.DataFrame(parsed_records)

# Drop any accidental missing coordinates
df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])

# Build GeoDataFrame with standard WGS84 CRS (EPSG:4326)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude),
    crs="EPSG:4326"
)

# ---------------------------------------------------------
# 5. Serialization
# ---------------------------------------------------------
parquet_out = RAW_DATA_DIR / "probe_occurrences.parquet"
gdf.to_parquet(parquet_out, index=False)
print(f"-> Saved sample dataset to: {parquet_out}")

# ---------------------------------------------------------
# 6. Diagnostic Map Generation
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

# Plot coordinate points
gdf.plot(
    ax=ax,
    markersize=12,
    color="#2E7D32",
    alpha=0.6,
    edgecolor="none"
)

ax.set_title(f"Diagnostic Probe: {canonical_name} (n={len(gdf)})", fontsize=13, weight="bold")
ax.set_xlabel("Longitude (Decimal Degrees)")
ax.set_ylabel("Latitude (Decimal Degrees)")
ax.grid(True, linestyle="--", alpha=0.5)

plot_out = RAW_DATA_DIR / "day1_probe_map.png"
plt.savefig(plot_out, dpi=200, bbox_inches="tight")
plt.close()

print(f"-> Saved diagnostic scatter plot to: {plot_out}")
print("Day 1 verification complete.")
```

Run the script:

```bash
python probe_gbif.py
```

---

## 3. Expected Artifacts & Verification

After running the script, verify that the following files exist in `data/raw/`:

1. **`data/raw/probe_occurrences.parquet`**: A valid GeoParquet table containing 1,000 rows with columns:
   - `gbifID` (int64)
   - `decimalLatitude` (float64)
   - `decimalLongitude` (float64)
   - `coordinateUncertaintyInMeters` (float64, may contain nulls)
   - `year` (float64/int64)
   - `basisOfRecord` (string)
   - `geometry` (Point geometry)
2. **`data/raw/day1_probe_map.png`**: A diagnostic scatter plot confirming that coordinates fall inside the defined geographic bounding box (Longitudes $-82^\circ$ to $-64^\circ$, Latitudes $42^\circ$ to $50^\circ$).

---

## 4. Troubleshooting Common Issues

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `ImportError: libgdal.so` / `DLL load failed` | GDAL installed via pip instead of conda | Recreate environment and install `gdal` exclusively via `conda-forge`. |
| `pygbif.occurrences.search` times out | Slow network or API rate limiting | Re-run script or lower `limit=500` for the initial connectivity probe. |
| `KeyError: 'usageKey'` | Species typo in name backbone resolution | Confirm species name spelling: `"Acer saccharum"`. |

---

## 5. Next Step: Preview of Day 2

Once Day 1 passes verification:
* Configure automated pagination or async download tickets via the **GBIF Occurrence Download API** to ingest the full occurrence pool ($\sim 15{,}000\text{–}30{,}000$ points).
* Implement initial coordinate quality screening: removal of zero coordinates, country centroid markers, records with coordinate uncertainty $> 1{,}000\,\text{m}$, and non-contemporary records.