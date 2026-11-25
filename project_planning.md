# Species Distribution Modeling (SDM) & Bioclimatic Envelope Forecasting

An end-to-end spatial data science pipeline to model the ecological niche and evaluate climate change vulnerability for temperate forest taxa across Eastern Canada and Northeastern North America (*Acer saccharum* / Sugar Maple).

This repository implements bias-corrected species occurrence filtering, target-group background sampling, multicollinearity-pruned bioclimatic and edaphic feature extraction, autocorrelation-informed spatial block cross-validation, comparative machine learning models (MaxEnt/MaxNet vs. down-weighted LightGBM), and multi-GCM ensemble forward projections under CMIP6 climate scenarios (SSP2-4.5 and SSP5-8.5).

## 1. Project Overview & Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │                 RAW DATA INPUTS               │
                  │  - GBIF Async Download (DOI-backed) + TGB     │
                  │  - WorldClim v2.1 Bioclimatic Rasters (1 km)  │
                  │  - Static Edaphic (SoilGrids 2.0) & DEM       │
                  │  - CMIP6 Multi-GCM Projections (2041–2080)    │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │            ETL & SPATIAL CLEANING             │
                  │  - Coordinate validation & ESRI:102001/3978   │
                  │  - Cultivated/centroid/uncertainty filtering  │
                  │  - 1 km grid dedup + 5 km geographic thinning │
                  │  - Target-group background (TGB) sampling     │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │             FEATURE ENGINEERING               │
                  │  - Point-in-raster extraction (climate + soil)│
                  │  - Pairwise Pearson correlation (|r| < 0.75)  │
                  │  - Iterative VIF step-down (VIF < 5.0)        │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │         SPATIAL MODELING & EVALUATION         │
                  │  - Variogram-sized Spatial Block K-Fold CV    │
                  │  - Baseline MaxEnt vs. Weighted LightGBM      │
                  │  - Climate-Only vs. Climate+Soil Ablation     │
                  │  - CBI, OR10, Calibration & AUC-ROC           │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │           PROJECTIONS & DEPLOYMENT            │
                  │  - Multi-GCM CMIP6 Ensemble (Median + IQR)    │
                  │  - MESS / ExDet Non-Analog Climate Masking    │
                  │  - Range Velocity & Core Habitat Shift Metrics│
                  │  - Interactive Streamlit Geospatial Dashboard │
                  └───────────────────────────────────────────────┘
```

### Scientific & Operational Objectives

1. **Prevent Niche Truncation via Dual Spatial Domains:** Calibrate models across the full eastern North American native range buffer (`-96° to -60° E, 34° to 52° N`) so models observe both upper and lower thermal limits, then evaluate and report climate-driven shifts across the focal Eastern Canada / Northeastern US domain (`-82° to -64° E, 42° to 50° N`).
2. **Mitigate Spatial Sampling Bias:** Control for human observer clustering along road networks and urban corridors using 1 km raster-cell deduplication, 5 km geographic distance thinning in an equal-area projection, and **Target-Group Background (TGB)** sampling drawn from co-occurring woody angiosperm genera (`Acer`, `Fagus`, `Betula`, `Quercus`, `Fraxinus`, `Tilia`).
3. **Account for Edaphic Constraints on Northward Expansion:** Pair bioclimatic predictors with static soil and topographic covariates (SoilGrids 2.0 pH, organic carbon, texture; DEM elevation/slope/TWI) and run a **Climate-Only vs. Climate + Edaphic** ablation to quantify how acidic, nutrient-poor Canadian Shield soils constrain northward migration into the Laurentian transition zone.
4. **Prevent Autocorrelation Leakage:** Size spatial CV blocks empirically from the spatial autocorrelation range (semivariogram) of bioclimatic predictors/residuals rather than using arbitrary splits, and nest hyperparameter tuning strictly inside training folds.
5. **Quantify Ensemble Climate Uncertainty & Extrapolation Risk:** Project habitat suitability across a 5-GCM CMIP6 ensemble (`CanESM5`, `IPSL-CM6A-LR`, `MIROC6`, `MPI-ESM1-2-HR`, `MRI-ESM2-0`) for standard WorldClim periods (`2041–2060` and `2061–2080`), reporting ensemble medians, inter-GCM interquartile ranges (IQR), range centroid velocity (`km/decade`), and MESS/ExDet non-analog masks.

## 2. Methodological Specifications & Key Design Decisions

| Component | Specification | Rationale |
| :--- | :--- | :--- |
| **Calibration vs. Focal Domain** | Calibration: `[-96, 34, -60, 52]`; Focal reporting: `[-82, 42, -64, 50]` | Prevents southern thermal niche truncation when projecting warming scenarios. |
| **Projected Equal-Area CRS** | `EPSG:3978` (NAD83 / Canada Atlas Lambert) or `ESRI:102001` | Valid `pyproj` authority codes (`EPSG:102001` is invalid and raises a `CRSError`). Equal-area projection is required for accurate 5 km thinning, block sizing, and habitat area calculations. |
| **GBIF Ingestion** | GBIF Async Occurrence Download API (`occ.download`) for full pulls; `occ.search` only for Day 1 probes | `occ.search` caps pagination at 100,000 records and does not mint a citable GBIF DOI. Async downloads ensure reproducibility and full citation provenance. |
| **Temporal Alignment** | Baseline `1970–2000` (matching WorldClim v2.1), with optional sensitivity run including `2001–2025` | Avoids mismatching recent warming-era occurrences against 1970–2000 climatology. |
| **Presence & Background Weighting** | 10,000–25,000 TGB points; down-weighted background (`sum(w_bg) == sum(w_pres)`) for LightGBM | Eliminates arbitrary class-imbalance distortion in binary log-loss and approximates an inhomogeneous Poisson point process. |
| **Model Specifications** | **MaxEnt:** `elapid` / L1-regularized hinge+quadratic+linear logistic point-process model. **LightGBM:** weighted `binary` log-loss with monotonicity/depth regularization and clamped extrapolation. | Ensures pure-Python reproducibility without JVM/Java MaxEnt dependencies and avoids invalid objectives (e.g., quantile loss) or erratic tree extrapolation. |
| **Primary Evaluation Metrics** | **Continuous Boyce Index (CBI)** (primary), **10% Omission Rate (OR10)**, **Calibration curves**, and **AUC-ROC** (reported with prevalence caveat) | Presence-background data lacks true absences; CBI and OR10 are valid presence-background diagnostics whereas TSS is biased by background extent. |
| **Range Velocity & Thresholding** | Suitability-weighted centroid displacement (`km/decade`) + binarized core habitat area (10th-percentile training presence & max-SSS thresholds), masked where `MESS < 0` | Separates robust core-habitat contraction/expansion from unverified non-analog climate extrapolation. |

## 3. Directory Structure

```
sdm-climate-niche/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated linting, testing, and schema validation
├── config/
│   ├── config.yaml              # Global pipeline parameters (domains, species, thresholds, GCMs)
│   └── logging.yaml             # Structured logging configuration
├── data/
│   ├── raw/                     # Untracked raw downloads (GBIF parquet/zip, WorldClim, SoilGrids, DEM)
│   ├── processed/               # Cleaned/thinned occurrences, TGB matrices, pruned aligned rasters
│   └── outputs/                 # Model artifacts, CV metrics, ensemble rasters, MESS maps, figures
├── docs/
│   ├── methodology.md           # Theoretical grounding (Hutchinsonian niche, TGB, VIF, spatial blocks)
│   └── technical_report.pdf     # 4-page academic synthesis and executive policy brief
├── notebooks/
│   ├── 01_eda_and_thinning.ipynb
│   ├── 02_collinearity_vif.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── gbif_client.py       # GBIF async download & probe client (species + target group)
│   │   └── raster_fetch.py      # WorldClim v2.1, CMIP6 GCMs, SoilGrids 2.0, and DEM staging
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── spatial_clean.py     # Coordinate QA, centroid/cultivated/temporal filtering, CRS reprojection
│   │   ├── thinning.py          # 1 km raster-cell dedup + equal-area geographic distance thinning
│   │   └── background.py        # Target-group background (TGB) sampler & case-weight generator
│   ├── features/
│   │   ├── __init__.py
│   │   ├── raster_align.py      # Reproject, snap, and mask all climate/soil/DEM rasters to master grid
│   │   ├── raster_extract.py    # Point-in-raster extraction via rioxarray/rasterio
│   │   └── collinearity.py      # Pairwise Pearson + iterative VIF step-down filter
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spatial_cv.py        # Semivariogram-informed spatial block k-fold generator
│   │   ├── maxent.py            # Python-native MaxEnt / MaxNet estimator (elapid / L1 point-process)
│   │   ├── lightgbm_sdm.py      # Down-weighted binary LightGBM with nested spatial CV tuning
│   │   ├── train_sdm.py         # CLI entry point for training, ablation, and CV evaluation
│   │   └── project_future.py    # CLI entry point for multi-GCM CMIP6 projections & range velocity
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Continuous Boyce Index (CBI), OR10, AUC-ROC, calibration curves
│   │   └── novelty.py           # MESS / ExDet analysis for non-analog future climates
│   └── visualization/
│       ├── __init__.py
│       └── maps.py              # Publication-ready cartographic, uncertainty, and response plots
├── tests/
│   ├── test_coordinates.py      # Geographic bounding box, authority CRS, and cleaning assertions
│   ├── test_thinning.py         # Equal-area distance constraint & grid deduplication unit tests
│   ├── test_spatial_cv.py       # Spatial block non-overlap and zero-leakage assertions
│   └── test_metrics.py          # CBI, OR10, and MESS calculation sanity checks
├── probe_gbif.py                # Day 1 GBIF API connectivity and diagnostic mapping probe
├── app.py                       # Interactive Streamlit geospatial dashboard
├── environment.yml              # Conda environment definition (C-libraries + Python)
├── pyproject.toml               # Build metadata, dependencies, ruff/black/pytest configs
└── README.md                    # Project documentation
```

## 4. System Requirements & Dependencies

### Hardware Requirements

* **CPU:** 4+ physical cores recommended (x86_64 or Apple Silicon).
* **RAM:** Minimum 8 GB (16 GB recommended for multi-GCM raster stacks and full-domain predictions).
* **Disk Space:** 10–15 GB free disk space (to accommodate WorldClim v2.1, SoilGrids 2.0, and 5-GCM CMIP6 GeoTIFF stacks across 2 scenarios and 2 time periods).

### Software & Environment Specifications

* **Operating System:** Windows 11 (native — no WSL2 required), Linux (Ubuntu 22.04+), or macOS (12+).
* **Base Runtime:** Python 3.11 or 3.12.
* **Spatial Binaries:** Modern PyPI binary wheels (`geopandas`, `pyogrio`, `pyproj`, `shapely`, `rasterio`) bundle GDAL, GEOS, and PROJ directly on Windows (`win_amd64`), macOS, and Linux. Conda/Mamba (`conda-forge`) remains an optional alternative if standalone GDAL CLI tools (`gdalwarp`, `gdalbuildvrt`) are ever needed.

## 5. Installation & Environment Setup

### Option A: Native `.venv` + `pyproject.toml` (Recommended on Windows 11)

```powershell
# 1. Create virtual environment (if not already present)
python -m venv .venv

# 2. Upgrade pip and install the project + dev tools in editable mode
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

> **Windows 11 Note:** Invoke Python CLI tools via `.\.venv\Scripts\python.exe -m <tool>` (e.g., `python -m pip`, `python -m pytest`, `python -m streamlit run app.py`) rather than calling `.venv\Scripts\*.exe` shims directly, as Windows Smart App Control / Application Control may block unsigned script wrappers inside `.venv\Scripts\`.

### Option B: Conda / Mamba (`conda-forge`, Optional)

If you prefer managing C-spatial libraries via `conda-forge`:

```bash
conda create -n sdm-env python=3.12 gdal geos proj -c conda-forge -y
conda activate sdm-env
python -m pip install -e ".[dev]"
```

*(Note: `pygbif>=0.7.0` is required and handled in `probe_gbif.py` and `gbif_client.py`, as `species.name_backbone` updated its parameter and response schema in `0.7.0+`.)*

## 6. Execution Pipeline

The workflow is managed via modular CLI entry points driven by `config/config.yaml`:

### 1. Ingest Occurrences (Focal Taxon + Target Group) & Environmental Rasters

Submit DOI-backed GBIF async downloads for *Acer saccharum* and the woody broadleaf target group across the calibration domain, and stage WorldClim v2.1, SoilGrids 2.0, DEM, and CMIP6 rasters:

```bash
python -m src.ingestion.gbif_client --config config/config.yaml --mode download
python -m src.ingestion.raster_fetch --config config/config.yaml
```

### 2. Preprocess, Clean & Spatially Thin

Filter invalid coordinates, country/province centroids, fossil/cultivated records, and records with coordinate uncertainty $> 1{,}000\,\text{m}$. Reproject to `EPSG:3978`, deduplicate to the 1 km master raster grid, apply 5 km geographic thinning, and sample Target-Group Background (TGB) points with case weights:

```bash
python -m src.preprocessing.spatial_clean --config config/config.yaml
python -m src.preprocessing.thinning --config config/config.yaml --distance-km 5.0
python -m src.preprocessing.background --config config/config.yaml --n-background 20000
```

### 3. Raster Alignment, Feature Extraction & Collinearity Reduction

Align all bioclimatic, soil, and topographic rasters to a single master grid (`EPSG:3978`, 1 km resolution), extract values at presence/background coordinates, and prune variables with pairwise $|r| \ge 0.75$ and $\text{VIF} \ge 5.0$:

```bash
python -m src.features.raster_align --config config/config.yaml
python -m src.features.raster_extract --config config/config.yaml
python -m src.features.collinearity --config config/config.yaml --pearson-thresh 0.75 --vif-thresh 5.0
```

### 4. Train Models, Run Ablation & Spatial Block Cross-Validation

Estimate spatial block size from predictor/residual semivariograms, run 5-fold spatial block CV with nested tuning for MaxEnt and down-weighted LightGBM, and compare **Climate-Only** vs. **Climate + Edaphic** feature sets:

```bash
python -m src.models.train_sdm --config config/config.yaml --models maxent,lightgbm --ablation climate_vs_edaphic
```

### 5. Generate Multi-GCM CMIP6 Future Projections & MESS Novelty Masks

Forward-project suitability across the 5-GCM CMIP6 ensemble for SSP2-4.5 and SSP5-8.5 (`2041–2060` and `2061–2080`), compute ensemble median + inter-GCM IQR uncertainty maps, run MESS/ExDet non-analog detection, and calculate range centroid velocity (`km/decade`) inside the focal Eastern Canada / NE US domain:

```bash
python -m src.models.project_future --config config/config.yaml --scenarios ssp245,ssp585 --periods 2041-2060,2061-2080
```

### 6. Launch Interactive Web Dashboard

Explore suitability surfaces, inter-GCM uncertainty layers, MESS non-analog masks, Climate-Only vs. Climate+Soil comparisons, and partial dependence response curves:

```bash
python -m streamlit run app.py
```

## 7. Testing & Quality Assurance

Run the test suite with coverage reporting:

```bash
# Run unit and integration tests
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Code formatting and static inspection
python -m ruff check src/ tests/
python -m black --check src/ tests/
```

### Critical Automated Assertions

* **CRS & Grid Integrity:** Validates that all occurrence geometries and raster layers share an exact valid authority CRS (`EPSG:3978` or `ESRI:102001`), identical pixel resolution, affine transform, and spatial alignment before extraction or prediction.
* **Zero Spatial Leakage:** Enforces strictly disjoint spatial blocks between train and validation folds in `spatial_cv.py` and verifies that all scalers, encoders, and hyperparameter searches are fit only on training folds.
* **Zero Unmasked NoData:** Validates that extracted feature matrices and prediction tiles contain no unmasked NoData sentinels (`-9999`, `NaN`, or ocean cells) before entering models.
* **Domain & Extrapolation Checks:** Verifies that predicted suitability probabilities lie strictly in `[0, 1]` and that MESS masks (`MESS < 0`) are generated for every GCM/scenario/period combination.

## 8. Data Attribution & Governance

* **Occurrence Data:** Distributed by [GBIF](https://www.gbif.org/) and [Canadensys](https://www.canadensys.net/) under CC-BY 4.0. Async download query metadata, minted GBIF DOI, and dataset citations are automatically serialized to `data/raw/gbif_citation.json`.
* **Baseline Bioclimatology:** WorldClim 2.1 (released under CC-BY 4.0; Fick & Hijmans, 2017; 1970–2000 baseline).
* **Edaphic & Topographic Data:** ISRIC SoilGrids 2.0 (CC-BY 4.0) and Copernicus/SRTM DEM.
* **Climate Projections:** CMIP6 Downscaled Future Climate Data (WorldClim v2.1 CMIP6 multi-GCM ensemble and ClimateData.ca).
* **Vector Basemaps:** Natural Earth and Statistics Canada Open Licence.

## 9. License

This repository is distributed under the terms of the **MIT License**. See `LICENSE` for details.
