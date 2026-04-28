"""
03_data_augmentation.py
-----------------------
Data augmentation for F1 telemetry time-series (anomaly class).
Techniques from Unit 3 – AML (Data Augmentation section):
  - Jitter             (additive Gaussian noise)
  - Magnitude warping  (smooth random scaling per channel)
  - Time warping       (smooth random time-axis distortion)
  - Window slicing     (random sub-sequence crop + resize)
  - Scaling            (global amplitude scaling)

All augmentations are applied ONLY to the minority (anomaly) class
in the training set — never to the test set.

Reads:  data/telemetry_labelled.csv
Writes: data/telemetry_augmented.csv
        outputs/data_augmentation/  (plots)
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import f1_score, roc_auc_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "../data"
OUTPUT_DIR   = "../outputs/data_augmentation"
FEATURE_COLS = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]
RANDOM_STATE = 42
rng          = np.random.default_rng(RANDOM_STATE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("03 – DATA AUGMENTATION")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(f"{DATA_DIR}/telemetry_labelled.csv")
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
df = df.dropna(subset=feature_cols + ["Is_Anomaly"]).reset_index(drop=True)

X = df[feature_cols].values.astype(float)
y = df["Is_Anomaly"].values.astype(int)

print(f"\nSamples: {len(y):,}  |  Normal={np.sum(y==0):,}  |  Anomaly={np.sum(y==1):,}")

# ── Train / test split ────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# Work only with anomaly samples in train
X_min = X_tr[y_tr == 1]
n_min = len(X_min)
print(f"Anomaly samples in train: {n_min}")

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS
# Each takes (X: np.ndarray [n_samples, n_features]) and returns augmented copy
# ══════════════════════════════════════════════════════════════════════════════

def jitter(X, sigma=0.05):
    """Add Gaussian noise scaled to each feature's std."""
    stds  = X.std(axis=0, keepdims=True) + 1e-8
    noise = rng.standard_normal(X.shape) * sigma * stds
    return X + noise


def scaling(X, sigma=0.15):
    """Multiply each sample by a random scalar drawn per sample."""
    factors = rng.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1))
    return X * factors


def magnitude_warping(X, sigma=0.2, n_knots=4):
    """
    Per-feature smooth random scaling curve (cubic-spline style via interp).
    """
    from scipy.interpolate import CubicSpline
    n_samples, n_features = X.shape
    result = X.copy()
    knot_x = np.linspace(0, n_samples - 1, n_knots)
    for f in range(n_features):
        knot_y = rng.normal(loc=1.0, scale=sigma, size=n_knots)
        cs     = CubicSpline(knot_x, knot_y)
        curve  = cs(np.arange(n_samples))
        result[:, f] *= curve
    return result


def time_warping(X, sigma=0.2, n_knots=4):
    """
    Smooth random time-axis distortion via cumulative warp path.
    """
    from scipy.interpolate import CubicSpline
    n_samples, n_features = X.shape
    knot_x = np.linspace(0, n_samples - 1, n_knots)
    knot_y = rng.normal(loc=1.0, scale=sigma, size=n_knots)
    cs     = CubicSpline(knot_x, knot_y)
    warp   = cs(np.arange(n_samples))
    # Build new time indices via cumulative warp
    warp_cumsum = np.cumsum(np.abs(warp))
    warp_cumsum = (warp_cumsum / warp_cumsum[-1]) * (n_samples - 1)
    result = np.zeros_like(X)
    for f in range(n_features):
        result[:, f] = np.interp(np.arange(n_samples), warp_cumsum, X[:, f])
    return result


def window_slicing(X, ratio=0.9):
    """
    Randomly crop a contiguous sub-window and resize back to original length.
    """
    n_samples, n_features = X.shape
    win_len   = max(2, int(n_samples * ratio))
    start     = rng.integers(0, n_samples - win_len + 1)
    sliced    = X[start : start + win_len, :]
    result    = np.zeros_like(X)
    old_idx   = np.linspace(0, win_len - 1, win_len)
    new_idx   = np.linspace(0, win_len - 1, n_samples)
    for f in range(n_features):
        result[:, f] = np.interp(new_idx, old_idx, sliced[:, f])
    return result


# ── Apply augmentations ───────────────────────────────────────────────────────
augmenters = {
    "Jitter"           : jitter,
    "Scaling"          : scaling,
    "Magnitude Warping": magnitude_warping,
    "Time Warping"     : time_warping,
    "Window Slicing"   : window_slicing,
}

print("\nApplying augmentations to anomaly class ...")
aug_samples = {}
for name, fn in augmenters.items():
    try:
        X_aug = fn(X_min.copy())
        aug_samples[name] = X_aug
        print(f"    [{name:<20s}]  generated {X_aug.shape[0]} new samples")
    except Exception as e:
        print(f"    [{name}] FAILED: {e}")

# ── Build augmented training set ──────────────────────────────────────────────
# Stack original train + all augmented anomaly copies
X_aug_all = np.vstack([X_min] + list(aug_samples.values()))
y_aug_all = np.ones(len(X_aug_all), dtype=int)

X_train_aug = np.vstack([X_tr, X_aug_all])
y_train_aug = np.concatenate([y_tr, y_aug_all])

n_normal_tr  = int(np.sum(y_train_aug == 0))
n_anomaly_tr = int(np.sum(y_train_aug == 1))
print(f"\nAugmented train set: {len(y_train_aug):,}  "
      f"(Normal={n_normal_tr:,}  Anomaly={n_anomaly_tr:,})")

# ── Evaluate: before vs after augmentation ───────────────────────────────────
print("\nEvaluating effect of augmentation ...")
scaler = StandardScaler()

# Before
X_tr_sc  = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)
clf_base = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)
clf_base.fit(X_tr_sc, y_tr)
pred_base = clf_base.predict(X_te_sc)
f1_base   = f1_score(y_te, pred_base, pos_label=1, zero_division=0)
try:
    auc_base = roc_auc_score(y_te, clf_base.predict_proba(X_te_sc)[:,1])
except Exception:
    auc_base = 0.5
print(f"    Before augmentation  →  F1={f1_base:.4f}  AUC={auc_base:.4f}")

# After
scaler2     = StandardScaler()
X_tr_aug_sc = scaler2.fit_transform(X_train_aug)
X_te_aug_sc = scaler2.transform(X_te)
clf_aug  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                   n_jobs=-1)
clf_aug.fit(X_tr_aug_sc, y_train_aug)
pred_aug = clf_aug.predict(X_te_aug_sc)
f1_aug   = f1_score(y_te, pred_aug, pos_label=1, zero_division=0)
try:
    auc_aug = roc_auc_score(y_te, clf_aug.predict_proba(X_te_aug_sc)[:,1])
except Exception:
    auc_aug = 0.5
print(f"    After  augmentation  →  F1={f1_aug:.4f}  AUC={auc_aug:.4f}")
print(f"\n    Classification report (after augmentation):")
print(classification_report(y_te, pred_aug,
                             target_names=["Normal","Anomaly"],
                             zero_division=0))

# ── Save augmented dataset ────────────────────────────────────────────────────
aug_df = pd.DataFrame(X_aug_all, columns=feature_cols)
aug_df["Is_Anomaly"] = 1
aug_df["Source"]     = np.repeat(
    ["original"] + list(aug_samples.keys()),
    [n_min] + [len(v) for v in aug_samples.values()]
)
aug_df.to_csv(f"{DATA_DIR}/telemetry_augmented.csv", index=False)
print(f"\nSaved {DATA_DIR}/telemetry_augmented.csv  ({len(aug_df):,} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots ...")

# 1. Compare original vs augmented for Speed channel
fig, axes = plt.subplots(len(aug_samples), 2, figsize=(14, 3.5*len(aug_samples)))
if len(aug_samples) == 1:
    axes = axes[np.newaxis, :]

speed_idx = feature_cols.index("Speed") if "Speed" in feature_cols else 0
n_show    = min(2000, len(X_min))   # cap for plot readability

for i, (name, X_aug) in enumerate(aug_samples.items()):
    axes[i, 0].plot(X_min[:n_show, speed_idx], color="#4C78A8", lw=0.8)
    axes[i, 0].set_title("Original – Speed (anomaly samples)")
    axes[i, 0].set_ylabel("km/h")

    axes[i, 1].plot(X_aug[:n_show, speed_idx], color="#E45756", lw=0.8)
    axes[i, 1].set_title(f"{name} – Speed (augmented)")

fig.suptitle("Data Augmentation – Anomaly Lap (Speed channel)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/01_augmentation_comparison.png", dpi=150)
plt.close(fig)

# 2. Feature distributions: original anomaly vs each augmentation
fig, axes = plt.subplots(1, len(feature_cols),
                         figsize=(3.5 * len(feature_cols), 4))
if len(feature_cols) == 1:
    axes = [axes]

for j, feat in enumerate(feature_cols):
    axes[j].hist(X_min[:, j], bins=40, alpha=0.6,
                 label="Original", color="#4C78A8", density=True)
    for name, X_aug in aug_samples.items():
        if X_aug.ndim == 2 and X_aug.shape[1] == len(feature_cols):
            axes[j].hist(X_aug[:, j], bins=40, alpha=0.4,
                         label=name, density=True)
    axes[j].set_title(feat, fontsize=9)
    axes[j].set_xlabel("Value")
    if j == 0:
        axes[j].set_ylabel("Density")

axes[0].legend(fontsize=7)
fig.suptitle("Feature Distributions – Original vs Augmented (Anomaly class)",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_feature_distributions.png", dpi=150)
plt.close(fig)

# 3. F1 / AUC before vs after bar chart
fig, ax = plt.subplots(figsize=(5, 4))
labels = ["Before\nAugmentation", "After\nAugmentation"]
f1s  = [f1_base,  f1_aug]
aucs = [auc_base, auc_aug]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, f1s,  w, label="F1 (anomaly)", color=["#4C78A8","#E45756"])
ax.bar(x + w/2, aucs, w, label="ROC-AUC",      color=["#72B7B2","#54A24B"])
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1); ax.legend()
ax.set_title("Impact of Data Augmentation")
for xi, (f, a) in enumerate(zip(f1s, aucs)):
    ax.text(xi - w/2, f + 0.02, f"{f:.3f}", ha="center", fontsize=9)
    ax.text(xi + w/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_before_after_augmentation.png", dpi=150)
plt.close(fig)

# 4. Class balance before / after
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (title, n0, n1) in zip(axes, [
    ("Before augmentation", int(np.sum(y_tr==0)), int(np.sum(y_tr==1))),
    ("After augmentation",  n_normal_tr,           n_anomaly_tr),
]):
    ax.bar(["Normal","Anomaly"], [n0, n1], color=["#4C78A8","#E45756"])
    ax.set_title(title)
    ax.set_ylabel("Samples")
    for i, v in enumerate([n0, n1]):
        ax.text(i, v * 1.01, f"{v:,}", ha="center", fontsize=9)
fig.suptitle("Class Balance Before / After Augmentation", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_class_balance.png", dpi=150)
plt.close(fig)

print(f"\n✅  03_data_augmentation.py complete. Plots → {OUTPUT_DIR}/")