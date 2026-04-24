# FastF1 – AML Project
**Group 6:** Aritz Ryan San Sebastian · Nico Azcarate · Jon Larrañaga

---

## Project overview
Anomaly detection on Formula 1 telemetry data (all drivers, full 2024 season) using the **FastF1** Python library. The goal is to identify slow/damaged laps from telemetry sensor data, handle the severe class imbalance between normal and anomaly laps, and prepare clean, augmented time-series features for modelling.

---

## Deliverables

| Deliverable | Scope | Scripts |
|---|---|---|
| **Delivery 1** | Imbalanced data + Data augmentation + Data imputation | `01` → `04` |
| **Delivery 2** | Forecasting / anomaly detection model | `05+` |

---

## File structure
```
fastf1_project/
├── 01_load_data.py              # Download & cache FastF1 2024 (all drivers) → data/
├── 02_class_imbalance.py        # Imbalanced data techniques + evaluation
├── 03_data_augmentation.py      # Time-series data augmentation (anomaly class)
├── 04_data_imputation.py        # Missing data imputation (all methods)
data/
├── laps.csv                     # Per-lap data (all drivers, all rounds)
├── telemetry.csv                # Raw telemetry
├── telemetry_labelled.csv       # + Is_Anomaly column
├── telemetry_resampled.csv      # Best resampled training set (from script 02)
├── telemetry_augmented.csv      # Augmented anomaly samples (from script 03)
├── telemetry_imputed.csv        # Final imputed dataset with missingness indicators
outputs/
├── class_imbalance/             # Plots + metrics_summary.csv
├── data_augmentation/           # Plots
└── imputation/                  # Plots + rmse_results.csv
```

---

## How to run (in order)
```bash
pip install fastf1 scikit-learn imbalanced-learn matplotlib seaborn scipy

python 01_load_data.py           # ~1–3 h first run (caches 24 races × 20 drivers)
python 02_class_imbalance.py
python 03_data_augmentation.py
python 04_data_imputation.py
```

---

## Techniques implemented (Delivery 1 – Unit 3 AML)

### Script 02 – Class Imbalance
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

### Script 03 – Data Augmentation
Applied exclusively to the **minority (anomaly) class** in the training set.

| Technique | Description |
|---|---|
| Jitter | Additive Gaussian noise scaled per feature |
| Scaling | Random global amplitude factor per sample |
| Magnitude Warping | Smooth per-feature scaling curve (cubic spline) |
| Time Warping | Smooth time-axis distortion via cumulative warp path |
| Window Slicing | Random sub-sequence crop + resample to original length |

Augmented samples saved to `data/telemetry_augmented.csv`. Before/after F1 comparison included.

### Script 04 – Data Imputation
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

Missingness type discussion (MCAR / MAR / MNAR) included in script output.  
All methods evaluated on 10% synthetic MCAR masking using **RMSE**.  
Final clean dataset saved to `data/telemetry_imputed.csv`.

---

## Anomaly labelling criterion
A lap is labelled as anomalous (`Is_Anomaly = 1`) if:
- Its `LapTime > mean + 1.5 × std` within that driver's race, **OR**
- `TrackStatus ≠ "1"` (safety car, VSC, red flag)

All telemetry rows belonging to anomalous laps inherit the label.