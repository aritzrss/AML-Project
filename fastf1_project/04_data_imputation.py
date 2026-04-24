"""
04_data_imputation.py
---------------------
Data imputation for F1 telemetry.
Techniques from Unit 3 – AML:
  Univariate  : Mean, Median, Mode, Constant
  Time-series : LOCF, NOCB, Linear interpolation
  Multivariate: KNN (k=3, k=5), MICE-BayesianRidge, MICE-RF, MissForest
  Extra       : Missingness indicator variable

Key principle (from slides):
  Imputer is ALWAYS fitted on train set only, then applied to test set.

Reads:  data/telemetry_labelled.csv
Writes: data/telemetry_imputed.csv
        outputs/imputation/  (plots + RMSE CSV)
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute       import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble     import RandomForestRegressor
from sklearn.metrics      import mean_squared_error
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "../data"
OUTPUT_DIR   = "../outputs/imputation"
FEATURE_COLS = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]
MISSING_RATE = 0.10
RANDOM_STATE = 42
rng          = np.random.default_rng(RANDOM_STATE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("04 – DATA IMPUTATION")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(f"{DATA_DIR}/telemetry_labelled.csv")
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
df_num = df[feature_cols].copy()
print(f"\nShape: {df_num.shape}  |  Features: {feature_cols}")

# Assess real missingness
real_missing = df_num.isna().sum()
print(f"\nReal missing values per column:\n{real_missing.to_string()}")
missing_pct = df_num.isna().mean() * 100
print(f"\nMissing %:\n{missing_pct.round(2).to_string()}")

# Missingness type discussion
print("\n[Missingness type analysis]")
print("  FastF1 telemetry NaNs arise from interpolation gaps and sensor dropouts.")
print("  This is consistent with MCAR (random sensor gaps) or MAR (speed-dependent")
print("  DRS readings). We proceed assuming MAR and apply imputation accordingly.")

# ── Train / test split (imputer fitted on train only!) ───────────────────────
df_complete = df_num.dropna()
print(f"\nComplete rows for evaluation: {len(df_complete):,}")

X_tr_raw, X_te_raw = train_test_split(
    df_complete.values.astype(float),
    test_size=0.20, random_state=RANDOM_STATE
)

# Introduce synthetic MCAR missingness (10%) on train and test separately
def mask_mcar(X, rate=MISSING_RATE):
    m = rng.random(X.shape) < rate
    Xm = X.copy()
    Xm[m] = np.nan
    return Xm, m

X_tr_miss, mask_tr = mask_mcar(X_tr_raw)
X_te_miss, mask_te = mask_mcar(X_te_raw)

print(f"\nSynthetic {MISSING_RATE*100:.0f}% MCAR missingness applied.")
print(f"  Train masked: {mask_tr.sum():,}  |  Test masked: {mask_te.sum():,}")

# ── Time-series helpers (no fit/transform — state-free) ───────────────────────
def locf(X):
    df_ = pd.DataFrame(X)
    return df_.ffill().bfill().values

def nocb(X):
    df_ = pd.DataFrame(X)
    return df_.bfill().ffill().values

def linear_interp(X):
    df_ = pd.DataFrame(X)
    return df_.interpolate(method="linear", limit_direction="both").values

# ── Imputers dict ─────────────────────────────────────────────────────────────
sklearn_imputers = {
    "Mean"              : SimpleImputer(strategy="mean"),
    "Median"            : SimpleImputer(strategy="median"),
    "Mode"              : SimpleImputer(strategy="most_frequent"),
    "Constant (0)"      : SimpleImputer(strategy="constant", fill_value=0),
    "KNN (k=3)"         : KNNImputer(n_neighbors=3),
    "KNN (k=5)"         : KNNImputer(n_neighbors=5),
    "MICE – BayesRidge" : IterativeImputer(estimator=BayesianRidge(),
                                           max_iter=10, random_state=RANDOM_STATE),
    "MICE – RF"         : IterativeImputer(
                              estimator=RandomForestRegressor(
                                  n_estimators=30, random_state=RANDOM_STATE,
                                  n_jobs=-1),
                              max_iter=5, random_state=RANDOM_STATE),
}

ts_imputers = {
    "LOCF"          : locf,
    "NOCB"          : nocb,
    "Linear Interp" : linear_interp,
}

# ── Evaluate: fit on train, transform on test ─────────────────────────────────
print("\n[Evaluating imputers — fitted on train, evaluated on test] ...")
rmse_results = {}

for name, imp in sklearn_imputers.items():
    try:
        imp.fit(X_tr_miss)
        X_te_imp = imp.transform(X_te_miss)
        # RMSE only on masked positions in test
        rmse = float(np.sqrt(mean_squared_error(
            X_te_raw[mask_te], X_te_imp[mask_te]
        )))
        rmse_results[name] = rmse
        print(f"    {name:<25s}  RMSE = {rmse:.4f}")
    except Exception as e:
        print(f"    {name:<25s}  FAILED: {e}")

for name, fn in ts_imputers.items():
    try:
        # TS methods are state-free: applied directly to test
        X_te_imp = fn(X_te_miss)
        rmse = float(np.sqrt(mean_squared_error(
            X_te_raw[mask_te], X_te_imp[mask_te]
        )))
        rmse_results[name] = rmse
        print(f"    {name:<25s}  RMSE = {rmse:.4f}")
    except Exception as e:
        print(f"    {name:<25s}  FAILED: {e}")

# Optional: MissForest (if installed)
try:
    from missforest import MissForest
    mf = MissForest()
    X_tr_mf = pd.DataFrame(X_tr_miss, columns=feature_cols)
    X_te_mf = pd.DataFrame(X_te_miss, columns=feature_cols)
    X_tr_mf_imp = mf.fit_transform(X_tr_mf)
    X_te_mf_imp = mf.transform(X_te_mf).values
    rmse_mf = float(np.sqrt(mean_squared_error(
        X_te_raw[mask_te], X_te_mf_imp[mask_te]
    )))
    rmse_results["MissForest"] = rmse_mf
    print(f"    {'MissForest':<25s}  RMSE = {rmse_mf:.4f}")
except ImportError:
    print("    MissForest not installed — skipping (pip install missforest)")
except Exception as e:
    print(f"    MissForest FAILED: {e}")

# ── Best method ───────────────────────────────────────────────────────────────
best_method = min(rmse_results, key=rmse_results.get)
print(f"\n[Best method] {best_method}  RMSE={rmse_results[best_method]:.4f}")

# Save RMSE table
rmse_df = pd.Series(rmse_results, name="RMSE").sort_values()
rmse_df.to_csv(f"{OUTPUT_DIR}/rmse_results.csv", header=True)
print(rmse_df.to_string())

# ── Apply best imputer to full dataset ───────────────────────────────────────
print(f"\nApplying '{best_method}' to full dataset ...")
df_full_num = df_num.values.astype(float)

if best_method in sklearn_imputers:
    best_imp = sklearn_imputers[best_method]
    best_imp.fit(df_full_num)
    df_imputed_vals = best_imp.transform(df_full_num)
elif best_method == "LOCF":
    df_imputed_vals = locf(df_full_num)
elif best_method == "NOCB":
    df_imputed_vals = nocb(df_full_num)
elif best_method == "Linear Interp":
    df_imputed_vals = linear_interp(df_full_num)
else:
    df_imputed_vals = linear_interp(df_full_num)  # fallback

df_imputed = pd.DataFrame(df_imputed_vals, columns=feature_cols)

# ── Missingness indicator variables ──────────────────────────────────────────
for col in feature_cols:
    df_imputed[f"{col}_missing"] = df_num[col].isna().astype(int)

# Merge back non-numeric columns
non_numeric = [c for c in df.columns if c not in feature_cols]
df_final = pd.concat([
    df[non_numeric].reset_index(drop=True),
    df_imputed.reset_index(drop=True)
], axis=1)

df_final.to_csv(f"{DATA_DIR}/telemetry_imputed.csv", index=False)
print(f"Saved {DATA_DIR}/telemetry_imputed.csv  {df_final.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots ...")

# 1. RMSE comparison
fig, ax = plt.subplots(figsize=(11, 5))
names_sorted = list(rmse_df.index)
vals_sorted  = list(rmse_df.values)
colors = ["#E45756" if n == best_method else "#4C78A8" for n in names_sorted]
bars = ax.bar(names_sorted, vals_sorted, color=colors)
ax.set_ylabel("RMSE")
ax.set_title(f"Imputation RMSE on 10% masked test data  (red = best)", fontsize=13)
ax.set_xticklabels(names_sorted, rotation=35, ha="right", fontsize=8)
for bar, val in zip(bars, vals_sorted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/01_rmse_comparison.png", dpi=150)
plt.close(fig)

# 2. Missingness heatmap (first 500 rows)
sample_miss = df_num.iloc[:500].isna().astype(int)
if sample_miss.values.any():
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(sample_miss.T, aspect="auto", cmap="Reds",
                   interpolation="none")
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols)
    ax.set_xlabel("Sample index (first 500 rows)")
    ax.set_title("Missingness heatmap (red = missing)")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_missingness_heatmap.png", dpi=150)
    plt.close(fig)
else:
    print("    No real missing values in first 500 rows — skipping heatmap.")

# 3. Speed: original vs imputed (300 rows)
n_plot = min(300, len(df_num))
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(df_num["Speed"].iloc[:n_plot].values, color="#4C78A8", lw=0.8)
axes[0].set_title("Speed – original (NaN visible as gaps)")
axes[0].set_ylabel("Speed (km/h)")
axes[1].plot(df_imputed["Speed"].iloc[:n_plot].values, color="#E45756", lw=0.8)
axes[1].set_title(f"Speed – after '{best_method}' imputation")
axes[1].set_ylabel("Speed (km/h)"); axes[1].set_xlabel("Sample index")
fig.suptitle("Data Imputation – Speed Channel", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_speed_before_after.png", dpi=150)
plt.close(fig)

# 4. Distribution shift per feature
fig, axes = plt.subplots(1, len(feature_cols),
                         figsize=(3.5*len(feature_cols), 4))
if len(feature_cols) == 1:
    axes = [axes]
for j, feat in enumerate(feature_cols):
    orig_vals = df_num[feat].dropna()
    imp_vals  = df_imputed[feat]
    axes[j].hist(orig_vals, bins=40, alpha=0.6, density=True,
                 label="Original", color="#4C78A8")
    axes[j].hist(imp_vals,  bins=40, alpha=0.5, density=True,
                 label="Imputed",  color="#E45756")
    axes[j].set_title(feat, fontsize=9)
    axes[j].set_xlabel("Value")
    if j == 0:
        axes[j].set_ylabel("Density")
axes[0].legend(fontsize=8)
fig.suptitle(f"Feature Distributions – Original vs Imputed ({best_method})",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_distribution_shift.png", dpi=150)
plt.close(fig)

print(f"\n✅  04_data_imputation.py complete. Plots → {OUTPUT_DIR}/")