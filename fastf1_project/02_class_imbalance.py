"""
02_class_imbalance.py
---------------------
Handles class imbalance in F1 telemetry anomaly detection.
Techniques from Unit 3 – AML:
  - Stratification
  - Class weights (balanced, manual)
  - Over-sampling: Random, SMOTE, ADASYN
  - Under-sampling: Random, NearMiss v1, Tomek Links, ENN
  - Combined: SMOTE+ENN
  - Prediction threshold tuning (Youden J)

Memory-efficient design:
  - Only the 6 feature columns + label are loaded from CSV (usecols + float32).
  - A stratified sub-sample of MAX_ROWS rows is drawn before any modelling;
    this preserves the class ratio while cutting RAM by ~99 %.
  - ROC curves are stored at a fixed ROC_POINTS resolution so large fpr/tpr
    arrays are never kept in memory across iterations.
  - Each resampler's output is deleted with gc.collect() before the next one.
  - RandomForest uses RF_TREES trees (sufficient for technique comparison).

Reads:  data/telemetry_labelled.csv
Writes: data/telemetry_resampled.csv
        outputs/class_imbalance/  (plots + metrics_summary.csv)
"""

import os, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.dummy           import DummyClassifier
from sklearn.metrics         import (
    classification_report, f1_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay,
)
from imblearn.over_sampling  import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import (RandomUnderSampler, NearMiss,
                                     TomekLinks, EditedNearestNeighbours)
from imblearn.combine        import SMOTEENN

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "../data"
OUTPUT_DIR   = "../outputs/class_imbalance"
FEATURE_COLS = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]
RANDOM_STATE = 42
MAX_ROWS     = 200_000   # stratified sub-sample cap; set None to use full dataset
ROC_POINTS   = 200       # fixed FPR grid size for stored ROC curves
RF_TREES     = 50        # trees per RandomForest; fewer = less RAM
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional live memory reporter (install psutil for visibility)
try:
    import psutil as _ps
    def _mem() -> str:
        return f"  [{_ps.Process().memory_info().rss / 1e9:.1f} GB RSS]"
except ImportError:
    def _mem() -> str:
        return ""

# ── Load (only needed columns, float32) ──────────────────────────────────────
print("=" * 65)
print("02 – CLASS IMBALANCE")
print("=" * 65)

col_dtypes = {c: "float32" for c in FEATURE_COLS} | {"Is_Anomaly": "int8"}
df = pd.read_csv(
    f"{DATA_DIR}/telemetry_labelled.csv",
    usecols=FEATURE_COLS + ["Is_Anomaly"],
    dtype=col_dtypes,
)
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
df = df.dropna(subset=feature_cols + ["Is_Anomaly"]).reset_index(drop=True)
print(f"\nLoaded {len(df):,} rows{_mem()}")

# ── Stratified sub-sample ─────────────────────────────────────────────────────
if MAX_ROWS is not None and len(df) > MAX_ROWS:
    df, _ = train_test_split(
        df, train_size=MAX_ROWS,
        stratify=df["Is_Anomaly"], random_state=RANDOM_STATE,
    )
    df = df.reset_index(drop=True)
    print(f"Sub-sampled to {len(df):,} rows (stratified, class ratio preserved){_mem()}")

X = df[feature_cols].values   # already float32 from col_dtypes
y = df["Is_Anomaly"].values.astype(np.int8)
del df
gc.collect()

n_normal  = int(np.sum(y == 0))
n_anomaly = int(np.sum(y == 1))
ratio     = n_normal / max(n_anomaly, 1)
print(f"\nSamples : {len(y):,}  |  Normal={n_normal:,}  |  Anomaly={n_anomaly:,}")
print(f"Imbalance ratio: {ratio:.1f}:1")

# ── 1. Stratified split ───────────────────────────────────────────────────────
print("\n[1] Stratified split 80/20 ...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
)
del X
gc.collect()

scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
X_te_sc = scaler.transform(X_te).astype(np.float32)
del X_tr, X_te
gc.collect()
print(f"    Train: {len(y_tr):,}  |  Test: {len(y_te):,}{_mem()}")

# ── Helper ────────────────────────────────────────────────────────────────────
ROC_X = np.linspace(0, 1, ROC_POINTS)   # shared fixed FPR grid

def evaluate(X_train, y_train, X_test, y_test,
             class_weight=None, threshold: float = 0.5, label: str = ""):
    """
    Fits a RandomForest, computes metrics, returns raw probabilities and a
    fixed-resolution interpolated TPR array (avoids storing millions-point
    fpr/tpr arrays for every method).
    """
    clf = RandomForestClassifier(
        n_estimators=RF_TREES, class_weight=class_weight,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    del clf   # free the forest immediately

    pred   = (proba >= threshold).astype(np.int8)
    f1     = f1_score(y_test, pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = 0.5
    recall = float(np.sum((pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1))

    fpr, tpr, _ = roc_curve(y_test, proba)
    tpr_interp  = np.interp(ROC_X, fpr, tpr)   # compress to fixed grid
    del fpr, tpr, pred

    if label:
        print(f"    [{label:<28s}]  F1={f1:.4f}  AUC={auc:.4f}  Recall={recall:.4f}")
    return proba, tpr_interp, auc, f1, recall

# ── Baseline ──────────────────────────────────────────────────────────────────
dummy      = DummyClassifier(strategy="most_frequent").fit(X_tr_sc, y_tr)
dummy_pred = dummy.predict(X_te_sc)
print(f"\n[Baseline] DummyClassifier "
      f"F1={f1_score(y_te, dummy_pred, pos_label=1, zero_division=0):.4f}")
del dummy, dummy_pred
gc.collect()

# ── 2. Class weights ──────────────────────────────────────────────────────────
print("\n[2] Class weights ...")
results = {}

proba_bal, tpr_interp, auc_b, f1_b, rec_b = evaluate(
    X_tr_sc, y_tr, X_te_sc, y_te, class_weight="balanced", label="balanced",
)
results["Balanced weights"] = dict(f1=f1_b, auc=auc_b, recall=rec_b, tpr=tpr_interp)
gc.collect()

_, tpr_interp, auc_m, f1_m, rec_m = evaluate(
    X_tr_sc, y_tr, X_te_sc, y_te,
    class_weight={0: 1.0, 1: float(ratio)}, label=f"manual {ratio:.0f}:1",
)
results["Manual weights"] = dict(f1=f1_m, auc=auc_m, recall=rec_m, tpr=tpr_interp)
gc.collect()

# ── 3. Resampling ─────────────────────────────────────────────────────────────
print("\n[3] Resampling ...")
k = min(5, n_anomaly - 1) if n_anomaly > 1 else 1
samplers = {
    "RandomOverSampler" : RandomOverSampler(random_state=RANDOM_STATE),
    "SMOTE"             : SMOTE(random_state=RANDOM_STATE, k_neighbors=k),
    "ADASYN"            : ADASYN(random_state=RANDOM_STATE),
    "RandomUnderSampler": RandomUnderSampler(random_state=RANDOM_STATE),
    "NearMiss v1"       : NearMiss(version=1),
    "Tomek Links"       : TomekLinks(),
    "ENN"               : EditedNearestNeighbours(),
    "SMOTE+ENN"         : SMOTEENN(random_state=RANDOM_STATE),
}

best_f1, best_name   = -1.0, "Balanced weights"
best_Xres, best_yres = X_tr_sc.copy(), y_tr.copy()

for name, sampler in samplers.items():
    try:
        X_res, y_res = sampler.fit_resample(X_tr_sc, y_tr)
        X_res = X_res.astype(np.float32)
        n0 = int(np.sum(y_res == 0))
        n1 = int(np.sum(y_res == 1))
        _, tpr_interp, auc_s, f1_s, rec_s = evaluate(
            X_res, y_res, X_te_sc, y_te, label=name,
        )
        results[name] = dict(f1=f1_s, auc=auc_s, recall=rec_s, tpr=tpr_interp,
                             n_0=n0, n_1=n1)
        if f1_s > best_f1:
            best_f1, best_name = f1_s, name
            del best_Xres, best_yres
            best_Xres, best_yres = X_res.copy(), y_res.copy()
        del X_res, y_res
        gc.collect()
    except Exception as e:
        print(f"    [{name}] FAILED: {e}")

# ── 4. Prediction threshold (Youden J) ───────────────────────────────────────
print("\n[4] Optimal threshold via Youden J ...")
fpr_c, tpr_c, thresholds = roc_curve(y_te, proba_bal)
best_idx = int(np.argmax(tpr_c - fpr_c))
best_thr = float(thresholds[best_idx])
pred_opt = (proba_bal >= best_thr).astype(int)
del proba_bal
gc.collect()

print(f"    Optimal threshold = {best_thr:.4f}")
print(classification_report(y_te, pred_opt,
                             target_names=["Normal", "Anomaly"],
                             zero_division=0))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"[Best method] {best_name}  F1={best_f1:.4f}")
metrics_df = pd.DataFrame({
    k: {m: v for m, v in d.items() if m in ["f1", "auc", "recall"]}
    for k, d in results.items()
}).T.round(4)
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics_summary.csv")
print(metrics_df.to_string())

# ── Save resampled training set ───────────────────────────────────────────────
resampled_df = pd.DataFrame(best_Xres, columns=feature_cols)
resampled_df["Is_Anomaly"] = best_yres
resampled_df.to_csv(f"{DATA_DIR}/telemetry_resampled.csv", index=False)
print(f"\nSaved {DATA_DIR}/telemetry_resampled.csv  ({len(resampled_df):,} rows)")
del best_Xres, best_yres, resampled_df
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots ...")
names = list(results.keys())

# 1. Class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(["Normal", "Anomaly"], [n_normal, n_anomaly],
            color=["#4C78A8", "#E45756"])
axes[0].set_title("Original class distribution")
axes[0].set_ylabel("Samples")
for i, v in enumerate([n_normal, n_anomaly]):
    axes[0].text(i, v * 1.01, f"{v:,}", ha="center", fontsize=9)

sampler_names = [k for k in names if "n_0" in results[k]]
x = np.arange(len(sampler_names)); w = 0.4
axes[1].bar(x - w/2, [results[k]["n_0"] for k in sampler_names], w,
            label="Normal",  color="#4C78A8")
axes[1].bar(x + w/2, [results[k]["n_1"] for k in sampler_names], w,
            label="Anomaly", color="#E45756")
axes[1].set_xticks(x)
axes[1].set_xticklabels(sampler_names, rotation=30, ha="right", fontsize=8)
axes[1].legend(); axes[1].set_title("After resampling")
fig.suptitle("Class Imbalance – FastF1 2025", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=150)
plt.close(fig)

# 2. F1 / AUC
fig, ax = plt.subplots(figsize=(12, 5))
f1s  = [results[k]["f1"]  for k in names]
aucs = [results[k]["auc"] for k in names]
x = np.arange(len(names)); w = 0.35
ax.bar(x - w/2, f1s, w,
       color=["#E45756" if n == best_name else "#4C78A8" for n in names],
       label="F1 (anomaly)")
ax.bar(x + w/2, aucs, w, color="#72B7B2", alpha=0.85, label="ROC-AUC")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.1); ax.legend()
ax.set_title("F1 and AUC by technique  (red = best F1)")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_f1_auc_comparison.png", dpi=150)
plt.close(fig)

# 3. ROC curves  (tpr stored on fixed ROC_X FPR grid — no huge arrays)
fig, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    ax.plot(ROC_X, res["tpr"],
            lw=2.5 if name == best_name else 1.0,
            label=f"{name} ({res['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC curves – all techniques")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_roc_curves.png", dpi=150)
plt.close(fig)

# 4. Youden J
# sklearn roc_curve may return len(thresholds) == len(fpr) or len(fpr)-1;
# slicing to len(thresholds) handles both cases safely.
n_thr = len(thresholds)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, tpr_c[:n_thr],               label="TPR",      color="green")
ax.plot(thresholds, fpr_c[:n_thr],               label="FPR",      color="red")
ax.plot(thresholds, tpr_c[:n_thr] - fpr_c[:n_thr], label="Youden J", color="#4C78A8", lw=2)
ax.axvline(best_thr, color="black", linestyle="--",
           label=f"Optimal = {best_thr:.3f}")
ax.set_xlabel("Threshold"); ax.set_xlim(0, 1)
ax.set_title("Youden J – optimal threshold")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_youden_threshold.png", dpi=150)
plt.close(fig)

# 5. Confusion matrix
fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay.from_predictions(
    y_te, pred_opt, display_labels=["Normal", "Anomaly"],
    cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix (thr={best_thr:.3f})", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/05_confusion_matrix.png", dpi=150)
plt.close(fig)

# 6. Recall
fig, ax = plt.subplots(figsize=(12, 4))
recalls = [results[k]["recall"] for k in names]
ax.bar(names, recalls,
       color=["#E45756" if n == best_name else "#F58518" for n in names])
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.1); ax.set_ylabel("Recall (anomaly)")
ax.set_title("Anomaly Recall by technique")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/06_recall_comparison.png", dpi=150)
plt.close(fig)

print(f"\n✅  02_class_imbalance.py complete. Plots → {OUTPUT_DIR}/")
