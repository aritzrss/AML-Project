# FastF1 – AML Project
**Group 6:** Aritz Ryan San Sebastian · Nico Azcarate · Jon Larrañaga

---

## Project overview
Anomaly detection on Formula 1 telemetry data (all drivers, full 2025 season) using the **FastF1** Python library. The goal is to identify slow/damaged laps from telemetry sensor data, handle the severe class imbalance between normal and anomaly laps, and prepare clean, augmented time-series features for modelling.

---

## Deliverables

| Deliverable | Scope | Notebook |
|---|---|---|
| **Delivery 1** | Imbalanced data + Data augmentation + Data imputation | `deliverable1_fastf1_2025.ipynb` |
| **Delivery 2** | Forecasting / anomaly detection model | `exporatoryda.ipynb` (EDA + 3D tensor prep – modelling stage upcoming) |

`exporatoryda.ipynb` is independent of Delivery 1 — it pulls its own subset from FastF1, defines its own minority class (laps ≥ 5 % slower than the driver's median), runs EDA plots across the four telemetry categories (car telemetry, track/positioning, lap/tire/sector, weather), and builds the 3D tensor that Delivery 2 will train on.

---

## Project tree
```
AML-Project/
├── deliverable1_fastf1_2025.ipynb   # Delivery 1: load → imbalance → augment → impute
├── exporatoryda.ipynb               # Standalone EDA + 3D tensor prep for Delivery 2
├── readme.md
├── cache/                           # FastF1 local cache (auto-populated, gitignored)
├── data/                            # All generated CSVs (gitignored)
│   ├── laps.csv                     # Per-lap data – timing, tyre, position, weather merged
│   ├── weather.csv                  # Raw weather time-series per round
│   ├── results.csv                  # Race results + driver info (grid, points, Q1/Q2/Q3)
│   ├── race_control.csv             # Race control messages (SC, flags, incidents)
│   ├── telemetry.csv                # Raw telemetry (Speed, Throttle, Brake, RPM, DRS, XYZ)
│   ├── telemetry_labelled.csv       # telemetry + Is_Anomaly column
│   ├── telemetry_resampled.csv      # Best resampled training set (Section 2)
│   ├── telemetry_augmented.csv      # Augmented anomaly samples (Section 3)
│   └── telemetry_imputed.csv        # Final imputed dataset with missingness indicators (Section 4)
└── outputs/
    ├── class_imbalance/             # 6 plots + metrics_summary.csv
    ├── data_augmentation/           # 4 plots
    └── imputation/                  # 3 plots + rmse_results.csv
```

---

## How to run

```bash
uv add fastf1 scikit-learn imbalanced-learn matplotlib seaborn scipy nbformat
```

Open `deliverable1_fastf1_2025.ipynb` and run cells top to bottom. Key checkpoints:

| Section | Cells | Purpose | Runtime |
|---|---|---|---|
| **0. Setup** | imports, paths, constants | required | <1 s |
| **1. Data download** | Loop 1 (laps/weather/results/race control) + Loop 2 (telemetry) | first run downloads ~24 races × 20 drivers | 1–3 h first run, seconds on cache hit |
| **Reload saved data** | one cell, after Loop 2 | skip the loops once CSVs exist | <5 s |
| **2. Class imbalance** | Sections after Loop 2 / Anomaly labelling | imbalance handling techniques + evaluation | ~2 min |
| **3. Data augmentation** | Time-series augmentation on the anomaly class | ~1 min |
| **4. Data imputation** | Missing-data imputation methods + evaluation | ~3 min |

The reload cell after Loop 2 lets you skip both download loops on subsequent runs — go straight from Setup to it, then continue with sections 2–4.

---

## Techniques implemented (Delivery 1 – Unit 3 AML)

### Section 2 – Class Imbalance
| Category | Technique |
|---|---|
| Stratification | `train_test_split(stratify=y)` |
| Baseline | `DummyClassifier` (most frequent) |
| Class weights | `balanced`, manual ratio |
| Over-sampling | Random, **SMOTE**, **ADASYN** |
| Under-sampling | Random, **NearMiss v1**, **Tomek Links**, **ENN** |
| Combined | **SMOTE + ENN** |
| Threshold | ROC-based optimal threshold (Youden J) |

**Classifier:** Random Forest (more appropriate than LR for telemetry data).
**Evaluation:** F1 (anomaly class), ROC-AUC, Recall. Best method saved to `data/telemetry_resampled.csv`.

### Section 3 – Data Augmentation
Applied exclusively to the **minority (anomaly) class** in the training set.

| Technique | Description |
|---|---|
| Jitter | Additive Gaussian noise scaled per feature |
| Scaling | Random global amplitude factor per sample |
| Magnitude Warping | Smooth per-feature scaling curve (cubic spline) |
| Time Warping | Smooth time-axis distortion via cumulative warp path |
| Window Slicing | Random sub-sequence crop + resample to original length |

Augmented samples saved to `data/telemetry_augmented.csv`. Before/after F1 comparison included.

### Section 4 – Data Imputation
**Key principle:** imputer always fitted on train set only, then applied to test set.

| Category | Method |
|---|---|
| Univariate | Mean, Median, Mode, Constant |
| Time-Series | LOCF, NOCB, Linear interpolation |
| Multivariate | KNN (k=3), KNN (k=5) |
| Multivariate | MICE – Bayesian Ridge |
| Multivariate | MICE – Random Forest |
| Multivariate | MissForest *(optional, if installed)* |
| Extra | Missingness indicator variables |

Missingness type discussion (MCAR / MAR / MNAR) included in notebook output.
All methods evaluated on 10% synthetic MCAR masking using **RMSE**.
Final clean dataset saved to `data/telemetry_imputed.csv`.

---

## Memory & runtime notes
- Loop 1 and Loop 2 are split so the kernel only holds one race session at a time (~100 MB) instead of the whole season (~2 GB).
- Each round writes to CSV immediately; a crash mid-loop loses at most one round.
- Anomaly labelling reads telemetry in 500 k-row chunks to avoid loading the full ~2 GB file.
- Class-imbalance section sub-samples to 200 k stratified rows by default to keep RAM under control on a 32 GB machine.

---

## Anomaly labelling criterion
A lap is labelled as anomalous (`Is_Anomaly = 1`) if:
- Its `LapTime > mean + 1.5 × std` within that driver's race, **OR**
- `TrackStatus ≠ "1"` (safety car, VSC, red flag), **OR**
- `Deleted == True` (track-limits violation)

All telemetry rows belonging to anomalous laps inherit the label.
