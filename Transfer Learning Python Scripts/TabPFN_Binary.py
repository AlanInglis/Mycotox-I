# tabpfn_binary_preds_presplit_v2.py
# -----------------------------------------------------
# Train a TabPFN classifier per binary toxin using PRE-SPLIT CSVs.
# For each toxin column in Y_bin_*, fits on train and predicts on test.
# Saves per-toxin CSVs with True_Label, Pred_Prob, and row_id (if present).
# Also saves a structured metrics CSV for all toxins for easy R loading.
# -----------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tabpfn import TabPFNClassifier
import torch
import csv

# ---------- config ----------
SEED = 42
toxins = [
    "Trichothence_producer_bin",
    "F_langsethiae_bin",
    "F_poae_bin",
    "DON_bin",
    "D3G_bin",
    "Nivalenol_bin",
    "3-AC-DON_bin",
    "15-AC-DON_bin",
    "T-2_toxin_bin",
    "HT-2_toxin_bin",
    "T2G_bin",
    "Neos_bin",
    "ENN_A1_bin",
    "ENN_A_bin",
    "ENN_B_bin",
    "ENN_B1_bin",
    "BEAU_bin",
    "ZEN_bin",
    "Apicidin_bin",
    "STER_bin",
    "DAS_bin",
    "Quest_bin",
    "AOH_bin",
    "AME_bin",
    "MON_bin",
    "Ergocristine_bin",
    "EGT_bin"
]

base_dir   = "~/data"
outpt_test_dir = "~/new test"
output_dir = os.path.join(outpt_test_dir, "tabpfn_binary_predictions_v2")
os.makedirs(output_dir, exist_ok=True)

def set_seeds(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_to_numeric_01(arr):
    s = pd.Series(arr).astype(str).str.strip().str.lower()
    mapping = {"0":0,"1":1,"false":0,"true":1,"neg":0,"pos":1,"no":0,"yes":1}
    s = s.map(lambda x: mapping.get(x, x))
    s = pd.to_numeric(s, errors="coerce")
    return s.values.astype(float)

def get_numeric_X(df):
    return df.select_dtypes(include=[np.number]).copy()

set_seeds()

X_train_df = pd.read_csv(os.path.join(base_dir, "X_train.csv"))
X_test_df  = pd.read_csv(os.path.join(base_dir, "X_test.csv"))
Ytr_df     = pd.read_csv(os.path.join(base_dir, "Y_bin_train.csv"))
Yte_df     = pd.read_csv(os.path.join(base_dir, "Y_bin_test.csv"))

test_ids = X_test_df["row_id"].copy() if "row_id" in X_test_df.columns else None

X_train = get_numeric_X(X_train_df).values
X_test  = get_numeric_X(X_test_df).values

if np.isnan(X_train).any() or np.isnan(X_test).any():
    raise ValueError("Predictor matrices contain NaNs; TabPFN requires finite inputs.")

# --- Prepare metrics collector ---
metrics_rows = []

for tox in toxins:
    if tox not in Ytr_df.columns or tox not in Yte_df.columns:
        print(f"Skipping {tox}: column not found in Y_bin_* CSVs")
        metrics_rows.append({
            "Type": "binary",
            "Variable": tox,
            "F1_or_Acc": "",
            "AUC": "",
        })
        continue

    y_tr_raw = Ytr_df[tox].values
    y_te_raw = Yte_df[tox].values

    y_tr = safe_to_numeric_01(y_tr_raw)
    y_te = safe_to_numeric_01(y_te_raw)

    tr_mask = ~np.isnan(y_tr)
    te_mask = ~np.isnan(y_te)

    X_tr, y_tr = X_train[tr_mask], y_tr[tr_mask].astype(int)
    X_te, y_te = X_test[te_mask],  y_te[te_mask].astype(int)
    ids_te = test_ids[te_mask] if test_ids is not None else None

    counts_tr = Counter(y_tr.tolist())
    if len(counts_tr) < 2 or min(counts_tr.values()) < 2:
        print(f"Skipping {tox}: training set class issue {dict(counts_tr)}")
        metrics_rows.append({
            "Type": "binary",
            "Variable": tox,
            "F1_or_Acc": "",
            "AUC": "",
        })
        continue

    counts_te = Counter(y_te.tolist())
    if len(counts_te) < 2:
        print(f"Skipping {tox}: test set single-class, cannot compute AUC/F1")
        metrics_ok = False
    else:
        metrics_ok = True

    clf = TabPFNClassifier(device="cpu")
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)
    pred_prob = proba[:, 1] if proba.shape[1] > 1 else np.full_like(y_te, np.nan, dtype=float)

    try:
        if metrics_ok:
            y_pred = (pred_prob >= 0.5).astype(int)
            f1 = float(f1_score(y_te, y_pred))
            auc = float(roc_auc_score(y_te, pred_prob))
            f1_or_acc = f"{f1:.6f}"
        else:
            y_pred = (pred_prob >= 0.5).astype(int)
            acc = float(accuracy_score(y_te, y_pred))
            auc = ""
            f1_or_acc = f"{acc:.6f}"
    except Exception:
        auc = ""
        f1_or_acc = ""

    # Save predictions
    out_csv  = os.path.join(output_dir, f"{tox}_tabpfn_preds.csv")
    out_df = pd.DataFrame({
        "True_Label": y_te,
        "Pred_Prob":  pred_prob
    })
    if ids_te is not None:
        out_df.insert(0, "row_id", ids_te.values)
    out_df.to_csv(out_csv, index=False)

    metrics_rows.append({
        "Type": "binary",
        "Variable": tox,
        "F1_or_Acc": f1_or_acc,
        "AUC": f"{auc:.6f}" if auc != "" else ""
    })

    msg = f"Saved → {out_csv}"
    extras = []
    if f1_or_acc != "": extras.append(f"F1_or_Acc={f1_or_acc}")
    if auc != "": extras.append(f"AUC={auc}")
    if extras: msg += "  | " + " ".join(extras)
    print(msg)

metrics_path = os.path.join(output_dir, "tabpfn_binary_metrics_structured.csv")
with open(metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Type", "Variable", "F1_or_Acc", "AUC"])
    writer.writeheader()
    writer.writerows(metrics_rows)

print("\n✓ Saved all toxin predictions and metrics in:")
print(" -", output_dir)
print(" -", metrics_path)