# FastF1 – AML Project (Delivery 2)
**Group 6:** Aritz Ryan San Sebastian · Nico Azcarate · Jon Larrañaga

---

## Project overview
Anomaly detection on Formula 1 telemetry data (Hamilton, Silverstone 2020) using the **FastF1** Python library. The goal is to identify slow/damaged laps from telemetry sensor data, handle the severe class imbalance between normal and anomaly laps, and prepare clean time-series features for modelling.

---

## File structure
```
fastf1_project/
├── 01_load_data.py          # Download & cache FastF1 data → data/
├── 02_class_imbalance.py    # Imbalanced data techniques
├── 03_time_series_analysis.py  # TS visualisations & feature engineering
├── 04_data_imputation.py    # Missing data imputation
data/
├── laps.csv                 # Per-lap data
├── telemetry.csv            # Raw telemetry
├── telemetry_labelled.csv   # + Is_Anomaly column
├── telemetry_features.csv   # + rolling/lag features
├── telemetry_imputed.csv    # Final clean dataset
outputs/
├── class_imbalance/         # Imbalance plots
├── time_series/             # TS plots & PCA
└── imputation/              # Imputation RMSE comparison
```

---

## How to run (in order)
```bash
pip install fastf1 scikit-learn imbalanced-learn matplotlib seaborn

python 01_load_data.py        # ~2–5 min (downloads cache)
python 02_class_imbalance.py
python 03_time_series_analysis.py
python 04_data_imputation.py
```

---

## Techniques implemented (Unit 3 – AML)

### Script 02 – Class Imbalance
| Category | Technique |
|---|---|
| Stratification | `train_test_split(stratify=y)` |
| Class weights | `balanced`, manual ratio |
| Over-sampling | Random over-sampling, **SMOTE**, **ADASYN** |
| Under-sampling | Random, **NearMiss v1**, **Tomek Links**, **ENN** |
| Combined | **SMOTE + ENN** |
| Threshold | ROC-based optimal threshold (Youden J) |

### Script 03 – Time Series
- Speed profile: Normal vs Anomaly lap
- Multi-channel telemetry (speed, throttle, brake, RPM, gear)
- Lap-time trend across the race
- Sensor correlation matrix
- PCA latent space (class separation)
- Rolling-window features (mean, std) and lag features

### Script 04 – Data Imputation
| Category | Method |
|---|---|
| Univariate | Mean, Median, Mode, Constant |
| Time-Series | LOCF, NOCB, Linear interpolation |
| Multivariate | KNN Imputer (k=3, k=5) |
| Multivariate | MICE – Bayesian Ridge (default) |
| Multivariate | MICE – Random Forest |
| Extra | Missingness indicator variable |

All methods are evaluated on synthetic missing values (10% masking) using **RMSE**.