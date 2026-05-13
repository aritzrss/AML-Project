# FastF1 – AML Project
**Group 6:** Aritz Ryan San Sebastian · Nico Azcarate · Jon Larrañaga

---

## Project overview
Anomaly detection and performance forecasting on Formula 1 telemetry and lap data (all drivers, full 2025 season) using the **FastF1** Python library.

The project is split into two parts:

1. **Data preparation pipeline** — handle the severe class imbalance between normal and anomaly laps, augment the minority class, and impute missing telemetry values.
2. **Performance forecasting** — predict race lap times, tyre degradation, finishing positions, qualifying times, and run live race simulations with safety-car probability and compound recommendations.

---

## Deliverables

| Deliverable | Scope | Notebook |
|---|---|---|
| **Delivery 1** | Imbalanced data + Data augmentation + Data imputation | `imbalanced_data_fastf1.ipynb` |
| **Delivery 2** | Forecasting / race simulation / strategy recommender | `forecasting_fastf1.ipynb` |
| **Exploratory** | EDA + 3D tensor prep (independent subset from FastF1) | `exporatoryda.ipynb` |

`forecasting_fastf1.ipynb` consumes the CSVs produced by `imbalanced_data_fastf1.ipynb` (it has a fail-fast guard if `data/laps.csv` is missing). `exporatoryda.ipynb` is fully independent.

---

## Project tree
```
AML-Project/
├── imbalanced_data_fastf1.ipynb   # Delivery 1: load → imbalance → augment → impute
├── forecasting_fastf1.ipynb       # Delivery 2: forecasting + race simulator + strategy
├── exporatoryda.ipynb             # Standalone EDA + 3D tensor prep
├── readme.md
├── .gitignore
├── cache/                         # FastF1 local cache (auto-populated, gitignored)
├── data/                          # All generated CSVs (gitignored)
│   ├── laps.csv                   # Per-lap data – timing, tyre, position, weather merged
│   ├── weather.csv                # Raw weather time-series per round
│   ├── results.csv                # Race results + driver info (grid, points, Q1/Q2/Q3)
│   ├── race_control.csv           # Race control messages (SC, flags, incidents)
│   ├── telemetry.csv              # Raw telemetry (Speed, Throttle, Brake, RPM, DRS, XYZ)
│   ├── telemetry_labelled.csv     # telemetry + Is_Anomaly column
│   ├── telemetry_resampled.csv    # Best resampled training set (Delivery 1 §2)
│   ├── telemetry_augmented.csv    # Augmented anomaly samples (Delivery 1 §3)
│   ├── telemetry_imputed.csv      # Final imputed dataset with missingness flags (Delivery 1 §4)
│   ├── aux_laps.csv               # Practice + Qualifying + Sprint laps (Delivery 2 §2)
│   └── sprint_results.csv         # Sprint race results (Delivery 2 §2)
└── outputs/
    ├── class_imbalance/           # 6 plots + metrics_summary.csv  (Delivery 1 §2)
    ├── data_augmentation/         # 4 plots                          (Delivery 1 §3)
    ├── imputation/                # 3 plots + rmse_results.csv       (Delivery 1 §4)
    └── forecasting/               # Forecasting plots + metrics CSVs (Delivery 2 §§4–12)
        ├── race_laptime_metrics_extended.csv
        ├── race_laptime_forecasts.png
        ├── degradation_metrics.csv
        ├── degradation_forecasts.png
        ├── degradation_heatmap.png
        ├── race_pace_ranking_round{R}.csv
        ├── sector_decomposition.png
        ├── lgb_feature_importance.png
        ├── strategy_monte_carlo.png
        ├── position_prediction_scatter.png
        ├── position_feature_importance_v2.png
        ├── features_full.csv
        ├── q_time_prediction.png
        ├── race_simulator_round{R}.png
        ├── sc_compound_recommendation_round{R}_lap{L}.png
        └── pit_window_heatmap_round{R}.png
```

---

## How to run

```bash
uv add fastf1 scikit-learn imbalanced-learn matplotlib seaborn scipy nbformat
# Delivery 2 additionally needs:
uv add statsmodels pmdarima lightgbm chronos-forecasting torch
```

### Delivery 1 — `imbalanced_data_fastf1.ipynb`
Open and run cells top to bottom.

| Section | Cells | Purpose | Runtime |
|---|---|---|---|
| **0. Setup** | imports, paths, constants | required | <1 s |
| **1. Data download** | Loop 1 (laps/weather/results/race control) + Loop 2 (telemetry) | first run downloads ~24 races × 20 drivers | 1–3 h first run, seconds on cache hit |
| **Reload saved data** | one cell, after Loop 2 | skip the loops once CSVs exist | <5 s |
| **2. Class imbalance** | Random Forest comparison across techniques | ~2 min |
| **3. Data augmentation** | Time-series augmentation on the anomaly class | ~1 min |
| **4. Data imputation** | Missing-data imputation methods + evaluation | ~3 min |

The reload cell after Loop 2 lets you skip both download loops on subsequent runs — go straight from Setup to it, then continue with sections 2–4.

### Delivery 2 — `forecasting_fastf1.ipynb`
Requires `data/laps.csv` to exist (produced by Delivery 1). Has a fail-fast guard with clear instructions if it's missing.

| Section | Purpose | Runtime |
|---|---|---|
| **0. Setup** | imports, library availability checks | <1 s |
| **1. Data Loading** | reload race laps + apply fuel correction | <5 s |
| **2. Supplementary data download** | FP1/FP2/FP3/Q/Sprint via FastF1 (one-off, skips if `data/aux_laps.csv` exists) | 15–25 min first run |
| **3. Feature engineering** | practice pace, quali gap-to-pole, sprint pace per (Round, Driver) | <5 s |
| **4. Race lap-time forecasting** | Naive → ARIMA → SARIMA → SARIMAX → auto-ARIMA → Chronos comparison | ~2 min |
| **5. Tyre degradation forecasting** | Per-stint lap-time forecast on the longest available stint | <30 s |
| **6. Domain-specific F1 forecasting** | Fuel-aware per-compound deg regression, race-pace ranking, LightGBM, Monte Carlo strategy, position prediction | ~3 min |
| **7. Improved position prediction** | LightGBM with practice + quali + sprint features | <30 s |
| **8. Qualifying-time prediction** | Gap-to-pole regression on FP3 quali-sim laps | <30 s |
| **9. Race simulation with SC probability** | Logistic-regression SC model + animated race playback (HTML5 player) | ~1 min |
| **10. Forward race-outcome simulator** | Monte Carlo: 1000 simulations of finishing positions per driver | ~10 s |
| **11. Compound recommendation under SC** | Per-driver compound recommendation when SC deploys | <5 s |
| **12. Race playback with live pit calls** | Lap-by-lap pit-window + recommended compound, animated | ~30 s |

---

## Techniques implemented

### Delivery 1 — Imbalanced data, augmentation, imputation

#### Section 2 – Class Imbalance
| Category | Technique |
|---|---|
| Stratification | `train_test_split(stratify=y)` |
| Baseline | `DummyClassifier` (most frequent) |
| Class weights | `balanced`, manual ratio |
| Over-sampling | Random, **SMOTE**, **ADASYN** |
| Under-sampling | Random, **NearMiss v1**, **Tomek Links**, **ENN** |
| Combined | **SMOTE + ENN** |
| Threshold | ROC-based optimal threshold (Youden J) |

**Classifier:** Random Forest. **Evaluation:** F1 (anomaly class), ROC-AUC, Recall. Best method saved to `data/telemetry_resampled.csv`.

#### Section 3 – Data Augmentation
Applied exclusively to the **minority (anomaly) class** in the training set.

| Technique | Description |
|---|---|
| Jitter | Additive Gaussian noise scaled per feature |
| Scaling | Random global amplitude factor per sample |
| Magnitude Warping | Smooth per-feature scaling curve (cubic spline) |
| Time Warping | Smooth time-axis distortion via cumulative warp path |
| Window Slicing | Random sub-sequence crop + resample to original length |

Augmented samples saved to `data/telemetry_augmented.csv`. Before/after F1 comparison included.

#### Section 4 – Data Imputation
**Key principle:** imputer always fitted on train set only, then applied to test set.

| Category | Method |
|---|---|
| Univariate | Mean, Median, Mode, Constant |
| Time-Series | LOCF, NOCB, Linear interpolation |
| Multivariate | KNN (k=3), KNN (k=5) |
| Multivariate | MICE – Bayesian Ridge |
| Multivariate | MICE – Random Forest |
| Extra | Missingness indicator variables |

Missingness type discussion (MCAR / MAR / MNAR) included in notebook output. All methods evaluated on 10% synthetic MCAR masking using **RMSE**. Final clean dataset saved to `data/telemetry_imputed.csv`.

### Delivery 2 — Forecasting, simulation, strategy

#### Section 4 – Classical race lap-time forecasting
Subject: longest clean driver-race series in the data (auto-picked, overridable). Train on first 70%, test on last 30%.

| Method | Slide deck reference |
|---|---|
| Naive (last value / mean / drift) | §22-25 |
| ARIMA(p,d,q) with manual order + AIC search | §98 |
| SARIMA(p,d,q)(P,D,Q,m) using median stint length as period | §104 |
| SARIMAX with TyreLife, Stint, Compound, weather as exogenous | §117 |
| auto-ARIMA (pmdarima grid search) | §122 |
| Chronos-bolt-small (foundation model, zero-shot) | §130 |

Plus diagnostics: STL decomposition, ADF stationarity test, ACF/PACF, Q-Q residual plot, Ljung-Box test.

#### Section 5 – Tyre degradation forecasting
Per-stint forecast on the longest available stint in the dataset. Naive + ARIMA + SARIMAX(TyreLife) + Chronos.

#### Section 6 – Domain-specific F1 forecasting (what real F1 teams use)
| Sub-section | Purpose | Real-F1 use |
|---|---|---|
| 6.1 Per-compound degradation regression | Quadratic Ridge fit per (Round, Compound) on race + practice long runs | Strategy planning |
| 6.2 Race-pace ranking | Clean-air, fuel-corrected median pace per driver + practice cross-check | Pirelli / Sky strategy reports |
| 6.3 Sector-time decomposition | ARIMA per-sector to find where models fail | Engineer pace analysis |
| 6.4 LightGBM with engineered features | Gradient-boosted regression on lag features + tyre + weather + practice + quali aux | Competing private team models |
| 6.5 Strategy Monte Carlo | 1-stop vs 2-stop simulator with fitted degradation curves | Pre-race strategy meetings |
| 6.6 Final-position prediction | Gradient boost on GridPos + MedianPace + BestQ | Bookmaker / broadcaster models |

#### Section 7 – Improved position prediction
LightGBM with the full aux-feature set (practice race pace, quali-sim, qualifying gap-to-pole, sprint position) — improvement vs Section 6.6 baseline.

#### Section 8 – Qualifying-time prediction
Predict Q-time from FP3 short runs and practice features. Gap-to-pole target (track-independent), median-of-top-3 fastest soft laps as a robust practice quali-sim feature.

#### Section 9 – Race simulation with safety-car probability
Logistic-regression SC model fit on per-(Round, Lap) features: lap number, position spread, lap-time spread (proximity proxy), mean tyre age, number of active cars. Outputs P(SC) per lap, then animated race playback rendered as an HTML5 player.

#### Section 10 – Forward race-outcome simulator (Monte Carlo)
1000 simulated races: per-driver pace distributions + per-compound degradation + SC sampling + pit-stop mechanics → distribution of finishing positions per driver. Calibration metric: how often actual position falls within predicted P25-P75 band.

#### Section 11 – Compound recommendation under SC
For each driver at a chosen SC trigger lap, compute the expected remaining race time on each candidate compound (SOFT/MEDIUM/HARD) accounting for compound base-pace deltas + degradation curves + the F1 two-compound regulation rule. Includes a calibration check against what teams actually did.

#### Section 12 – Race playback with live pit calls
Lap-by-lap version of Section 11 wrapped into the Section 9 playback animation. Right-side panel shows the live recommendation list per driver; pit-window annotations (`P→S/M/H`) appear next to each driver's marker on laps where pitting is faster than staying out. Pit-loss adapts to SC status (12s under SC, 22s green-flag).

---

## Memory & runtime notes
- Loops 1 and 2 (Delivery 1) are split so the kernel only holds one race session at a time (~100 MB) instead of the whole season (~2 GB).
- Each round writes to CSV immediately; a crash mid-loop loses at most one round.
- Anomaly labelling reads telemetry in 500 k-row chunks to avoid loading the full ~2 GB file.
- Class-imbalance section sub-samples to 200 k stratified rows by default to keep RAM under control on a 32 GB machine.
- Delivery 2 Section 2 (supplementary download) downloads laps + weather only (no telemetry) for FP1/FP2/FP3/Q/Sprint sessions — total ~25 MB on disk.
- Animations (Sections 9, 12) render inline as HTML5 players using `to_jshtml()` — no ffmpeg dependency. Switch to `%matplotlib qt` at the top of the cell for true OS-level popup windows.

---

## Anomaly labelling criterion
A lap is labelled as anomalous (`Is_Anomaly = 1`) if:
- Its `LapTime > mean + 1.5 × std` within that driver's race, **OR**
- `TrackStatus ≠ "1"` (safety car, VSC, red flag), **OR**
- `Deleted == True` (track-limits violation)

All telemetry rows belonging to anomalous laps inherit the label.

---

## Methods bibliography
Classical time-series methods follow Peixeiro, M. (2022). *Time series forecasting in python.* Simon and Schuster (Unit 4 AML slide deck). Foundation-model baseline uses **Chronos-Bolt** (Amazon, 2025). Class-imbalance techniques use `imbalanced-learn`. Gradient boosting via **LightGBM** with sklearn `GradientBoostingRegressor` fallback. Strategy Monte Carlo design follows the architecture of professional F1 strategy engines, simplified — limitations documented in Section 10.8.
