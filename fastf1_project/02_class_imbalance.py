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

Reads:  data/telemetry_labelled.csv
Writes: data/telemetry_resampled.csv
        outputs/class_imbalance/  (plots + metrics_summary.csv)
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
from sklearn.dummy           import DummyClassifier
from sklearn.metrics         import (
    classification_report, f1_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling  import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import (RandomUnderSampler, NearMiss,
                                     TomekLinks, EditedNearestNeighbours)
from imblearn.combine        import SMOTEENN

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "../data"
OUTPUT_DIR    = "../outputs/class_imbalance"
FEATURE_COLS  = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]
RANDOM_STATE  = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("02 – CLASS IMBALANCE")
print("=" * 65)

df = pd.read_csv(f"{DATA_DIR}/telemetry_labelled.csv")
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
df = df.dropna(subset=feature_cols + ["Is_Anomaly"]).reset_index(drop=True)

X = df[feature_cols].values.astype(float)
y = df["Is_Anomaly"].values.astype(int)

n_normal  = int(np.sum(y == 0))
n_anomaly = int(np.sum(y == 1))
ratio     = n_normal / max(n_anomaly, 1)

print(f"\nSamples : {len(y):,}  |  Normal={n_normal:,}  |  Anomaly={n_anomaly:,}")
print(f"Imbalance ratio: {ratio:.1f}:1")

# ── 1. Stratified split ───────────────────────────────────────────────────────
print("\n[1] Stratified split 80/20 ...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)
print(f"    Train: {len(y_tr):,}  |  Test: {len(y_te):,}")

# ── Baseline ──────────────────────────────────────────────────────────────────
dummy = DummyClassifier(strategy="most_frequent").fit(X_tr_sc, y_tr)
dummy_pred = dummy.predict(X_te_sc)
print(f"\n[Baseline] DummyClassifier F1={f1_score(y_te, dummy_pred, pos_label=1, zero_division=0):.4f}")

# ── Helper ────────────────────────────────────────────────────────────────────
def evaluate(X_train, y_train, X_test, y_test,
             class_weight=None, threshold=0.5, label=""):
    clf = RandomForestClassifier(
        n_estimators=100, class_weight=class_weight,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)
    f1  = f1_score(y_test, pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = 0.5
    recall = np.sum((pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
    if label:
        print(f"    [{label:<28s}]  F1={f1:.4f}  AUC={auc:.4f}  Recall={recall:.4f}")
    fpr, tpr, _ = roc_curve(y_test, proba)
    return clf, pred, proba, fpr, tpr, auc, f1, recall

# ── 2. Class weights ──────────────────────────────────────────────────────────
print("\n[2] Class weights ...")
results = {}

_, _, proba_bal, fpr_bal, tpr_bal, auc_b, f1_b, rec_b = evaluate(
    X_tr_sc, y_tr, X_te_sc, y_te, class_weight="balanced", label="balanced")
results["Balanced weights"] = dict(f1=f1_b, auc=auc_b, recall=rec_b,
                                    fpr=fpr_bal, tpr=tpr_bal)

_, _, _, fpr_m, tpr_m, auc_m, f1_m, rec_m = evaluate(
    X_tr_sc, y_tr, X_te_sc, y_te,
    class_weight={0: 1.0, 1: float(ratio)}, label=f"manual {ratio:.0f}:1")
results["Manual weights"] = dict(f1=f1_m, auc=auc_m, recall=rec_m,
                                  fpr=fpr_m, tpr=tpr_m)

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

best_f1, best_name = -1, "Balanced weights"
best_Xres, best_yres = X_tr_sc.copy(), y_tr.copy()

for name, sampler in samplers.items():
    try:
        X_res, y_res = sampler.fit_resample(X_tr_sc, y_tr)
        _, _, _, fpr_s, tpr_s, auc_s, f1_s, rec_s = evaluate(
            X_res, y_res, X_te_sc, y_te, label=name)
        results[name] = dict(f1=f1_s, auc=auc_s, recall=rec_s,
                             fpr=fpr_s, tpr=tpr_s,
                             n_0=int(np.sum(y_res==0)),
                             n_1=int(np.sum(y_res==1)))
        if f1_s > best_f1:
            best_f1, best_name = f1_s, name
            best_Xres, best_yres = X_res, y_res
    except Exception as e:
        print(f"    [{name}] FAILED: {e}")

# ── 4. Prediction threshold (Youden J) ───────────────────────────────────────
print("\n[4] Optimal threshold via Youden J ...")
fpr_c, tpr_c, thresholds = roc_curve(y_te, proba_bal)
best_idx = int(np.argmax(tpr_c - fpr_c))
best_thr = float(thresholds[best_idx])
pred_opt = (proba_bal >= best_thr).astype(int)
print(f"    Optimal threshold = {best_thr:.4f}")
print(classification_report(y_te, pred_opt,
                             target_names=["Normal", "Anomaly"],
                             zero_division=0))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"[Best method] {best_name}  F1={best_f1:.4f}")
metrics_df = pd.DataFrame({
    k: {m: v for m, v in d.items() if m in ["f1","auc","recall"]}
    for k, d in results.items()
}).T.round(4)
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics_summary.csv")
print(metrics_df.to_string())

# ── Save resampled training set ───────────────────────────────────────────────
resampled_df = pd.DataFrame(best_Xres, columns=feature_cols)
resampled_df["Is_Anomaly"] = best_yres
resampled_df.to_csv(f"{DATA_DIR}/telemetry_resampled.csv", index=False)
print(f"\nSaved {DATA_DIR}/telemetry_resampled.csv  ({len(resampled_df):,} rows)")

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
    axes[0].text(i, v*1.01, f"{v:,}", ha="center", fontsize=9)

sampler_names = [k for k in names if "n_0" in results[k]]
x = np.arange(len(sampler_names)); w = 0.4
axes[1].bar(x - w/2, [results[k]["n_0"] for k in sampler_names], w,
            label="Normal",  color="#4C78A8")
axes[1].bar(x + w/2, [results[k]["n_1"] for k in sampler_names], w,
            label="Anomaly", color="#E45756")
axes[1].set_xticks(x)
axes[1].set_xticklabels(sampler_names, rotation=30, ha="right", fontsize=8)
axes[1].legend(); axes[1].set_title("After resampling")
fig.suptitle("Class Imbalance – FastF1 2024", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=150)
plt.close(fig)

# 2. F1 / AUC
fig, ax = plt.subplots(figsize=(12, 5))
f1s  = [results[k]["f1"]  for k in names]
aucs = [results[k]["auc"] for k in names]
x = np.arange(len(names)); w = 0.35
ax.bar(x-w/2, f1s,  w, color=["#E45756" if n==best_name else "#4C78A8" for n in names],
       label="F1 (anomaly)")
ax.bar(x+w/2, aucs, w, color="#72B7B2", alpha=0.85, label="ROC-AUC")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.1); ax.legend()
ax.set_title("F1 and AUC by technique  (red = best F1)")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_f1_auc_comparison.png", dpi=150)
plt.close(fig)

# 3. ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    ax.plot(res["fpr"], res["tpr"], lw=2.5 if name==best_name else 1.0,
            label=f"{name} ({res['auc']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=0.8)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC curves – all techniques")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_roc_curves.png", dpi=150)
plt.close(fig)

# 4. Youden J
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, tpr_c[:-1], label="TPR", color="green")
ax.plot(thresholds, fpr_c[:-1], label="FPR", color="red")
ax.plot(thresholds, tpr_c[:-1]-fpr_c[:-1], label="Youden J",
        color="#4C78A8", lw=2)
ax.axvline(best_thr, color="black", linestyle="--",
           label=f"Optimal = {best_thr:.3f}")
ax.set_xlabel("Threshold"); ax.set_xlim(0,1)
ax.set_title("Youden J – optimal threshold")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_youden_threshold.png", dpi=150)
plt.close(fig)

# 5. Confusion matrix
fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay.from_predictions(
    y_te, pred_opt, display_labels=["Normal","Anomaly"],
    cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix (thr={best_thr:.3f})", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/05_confusion_matrix.png", dpi=150)
plt.close(fig)

# 6. Recall
fig, ax = plt.subplots(figsize=(12, 4))
recalls = [results[k]["recall"] for k in names]
ax.bar(names, recalls,
       color=["#E45756" if n==best_name else "#F58518" for n in names])
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.1); ax.set_ylabel("Recall (anomaly)")
ax.set_title("Anomaly Recall by technique")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/06_recall_comparison.png", dpi=150)
plt.close(fig)

print(f"\n✅  02_class_imbalance.py complete. Plots → {OUTPUT_DIR}/")