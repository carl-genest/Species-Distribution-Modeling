"""
probe_gbif.py
Validates GBIF API connectivity, taxon key resolution, and georeferenced data retrieval.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# Ensure environment path and cache settings
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PACKAGES = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_PACKAGES.exists() and str(VENV_PACKAGES) not in sys.path:
    sys.path.append(str(VENV_PACKAGES))

CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR)

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pygbif import occurrences as occ
from pygbif import species

# ---------------------------------------------------------
# 1. Configuration & Parameters
# ---------------------------------------------------------
SPECIES_NAME = "Acer saccharum"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box: Eastern Canada & NE US Buffer
# Longitude: -82° to -64° | Latitude: 42° to 50°
WKT_POLYGON = "POLYGON((-82 42, -64 42, -64 50, -82 50, -82 42))"

print(f"Resolving GBIF taxon key for: '{SPECIES_NAME}'...")

# ---------------------------------------------------------
# 2. Taxon Resolution (compatible with pygbif 0.7.0+ & legacy)
# ---------------------------------------------------------
try:
    matched = species.name_backbone(scientificName=SPECIES_NAME)
except TypeError:
    matched = species.name_backbone(name=SPECIES_NAME)

usage_key = None
canonical_name = SPECIES_NAME

if isinstance(matched, dict):
    if "usage" in matched and isinstance(matched["usage"], dict):
        usage_key = matched["usage"].get("key")
        canonical_name = matched["usage"].get("canonicalName", SPECIES_NAME)
        match_type = matched.get("diagnostics", {}).get("matchType", "")
        if match_type == "NONE":
            usage_key = None
    elif "usageKey" in matched:
        usage_key = matched.get("usageKey")
        canonical_name = matched.get("canonicalName", SPECIES_NAME)
        if matched.get("matchType") == "NONE":
            usage_key = None

# Fallback to legacy backbone if not resolved
if not usage_key:
    matched_legacy = species.name_backbone(scientificName=SPECIES_NAME, checklistKey=None)
    usage = matched_legacy.get("usage", {})
    usage_key = usage.get("key") or matched_legacy.get("usageKey")
    canonical_name = usage.get("canonicalName", canonical_name)

if not usage_key:
    raise ValueError(f"Could not resolve species '{SPECIES_NAME}' on GBIF backbone.")

print(f"-> Successfully matched: {canonical_name} (Taxon Key: {usage_key})")

# ---------------------------------------------------------
# 3. Sample Ingestion Probe (Paginate to 1,000 records)
# ---------------------------------------------------------
TARGET_LIMIT = 1000
PAGE_SIZE = 300
print(f"Fetching initial batch of {TARGET_LIMIT:,} georeferenced occurrences...")

results = []
offset = 0
total_records = 0

while len(results) < TARGET_LIMIT:
    fetch_n = min(PAGE_SIZE, TARGET_LIMIT - len(results))
    res = occ.search(
        taxonKey=usage_key,
        geometry=WKT_POLYGON,
        hasCoordinate=True,
        limit=fetch_n,
        offset=offset,
    )
    total_records = res.get("count", total_records)
    batch = res.get("results", [])
    if not batch:
        break
    results.extend(batch)
    offset += len(batch)
    print(f"   Batch fetched: {len(results)}/{TARGET_LIMIT} (Total available in bounding box: {total_records:,})")

print(f"-> Successfully collected {len(results)} sample records.")

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
        "datasetKey": r.get("datasetKey"),
    })

df = pd.DataFrame(parsed_records)

# Drop any accidental missing coordinates
df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])

# Build GeoDataFrame with standard WGS84 CRS (EPSG:4326)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude),
    crs="EPSG:4326",
)

# ---------------------------------------------------------
# 5. Serialization
# ---------------------------------------------------------
parquet_out = RAW_DATA_DIR / "probe_occurrences.parquet"
try:
    gdf.to_parquet(parquet_out, index=False)
except Exception:
    # Use fastparquet engine with WKB geometry representation
    df_out = pd.DataFrame(gdf)
    df_out["geometry"] = gdf.geometry.to_wkb()
    df_out.to_parquet(parquet_out, engine="fastparquet", index=False)

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
    edgecolor="none",
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
