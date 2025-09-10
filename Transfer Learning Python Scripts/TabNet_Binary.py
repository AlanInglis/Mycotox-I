# tabnet_binary_presplit_filtered_v2.py
# -----------------------------------------------------
# Train TabNetClassifier per binary toxin using PRE-SPLIT CSVs.
# Drops NaN labels from train and test *for fitting/metrics*
# Predicts on the FULL test set, but saves metrics on observed subset
# Produces a single metrics CSV for easy loading in R.
# -----------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
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

base_dir   = "/Users/alaninglis/Desktop/Transfer Learning models/TabNET"  # project root
outpt_test_dir = "~/new test"
data_dir   = os.path.join(base_dir, "data")
output_dir = os.path.join(outpt_test_dir, "tabnet_binary_filtered_predictions_v2")
os.makedirs(output_dir, exist_ok=True)

# ---------- reproducibility ----------
def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds()

def get_numeric_X(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()

def to_01_float(arr) -> np.ndarray:
    s = pd.Series(arr).astype(str).str.strip().str.lower()
    mapping = {"0":0.0,"1":1.0,"false":0.0,"true":1.0,"neg":0.0,"pos":1.0,"no":0.0,"yes":1.0}
    s = s.map(lambda x: mapping.get(x, x))
    s = pd.to_numeric(s, errors="coerce")
    return s.values.astype(np.float32)

X_train_df = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
X_test_df  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
Ytr_df     = pd.read_csv(os.path.join(data_dir, "Y_bin_train.csv"))
Yte_df     = pd.read_csv(os.path.join(data_dir, "Y_bin_test.csv"))

test_ids = X_test_df["row_id"].copy() if "row_id" in X_test_df.columns else None

X_train_full = get_numeric_X(X_train_df).values.astype(np.float32)
X_test_full  = get_numeric_X(X_test_df).values.astype(np.float32)

if np.isnan(X_train_full).any() or np.isnan(X_test_full).any():
    raise ValueError("Predictor matrices contain NaNs; TabNet requires finite inputs.")

# --- Prepare metrics collector ---
metrics_rows = []

for tox in toxins:
    print(f"--- Processing {tox} ---")

    if tox not in Ytr_df.columns or tox not in Yte_df.columns:
        print(f"Skipping {tox}: column not found in Y_bin_* CSVs")
        metrics_rows.append({
            "Type": "binary",
            "Variable": tox,
            "F1_or_Acc": "",
            "AUC": "",
        })
        continue

    y_tr_full = to_01_float(Ytr_df[tox].values)
    y_te_full = to_01_float(Yte_df[tox].values)

    tr_mask = ~np.isnan(y_tr_full)
    te_mask = ~np.isnan(y_te_full)

    X_tr = X_train_full[tr_mask]
    y_tr = y_tr_full[tr_mask].astype(int).ravel()

    X_te_obs = X_test_full[te_mask]
    y_te_obs = y_te_full[te_mask].astype(int).ravel()

    counts_tr = Counter(y_tr.tolist())
    if len(counts_tr) < 2 or min(counts_tr.values()) < 2:
        print(f"Skipping {tox}: training set class issue after filtering NaNs: {dict(counts_tr)}")
        metrics_rows.append({
            "Type": "binary",
            "Variable": tox,
            "F1_or_Acc": "",
            "AUC": "",
        })
        continue

    clf = TabNetClassifier(
        seed=SEED,
        verbose=0,
        device_name="cpu"
    )

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te_obs, y_te_obs)],
        eval_metric=['auc'],
        max_epochs=200,
        patience=20,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0
    )

    preds_prob_full = clf.predict_proba(X_test_full)[:, 1]
    preds_prob_obs = preds_prob_full[te_mask]

    try:
        counts_te = Counter(y_te_obs.tolist())
        if len(counts_te) > 1:
            y_pred = (preds_prob_obs >= 0.5).astype(int)
            f1 = float(f1_score(y_te_obs, y_pred))
            auc = float(roc_auc_score(y_te_obs, preds_prob_obs))
            f1_or_acc = f"{f1:.6f}"
        else:
            y_pred = (preds_prob_obs >= 0.5).astype(int)
            acc = float(accuracy_score(y_te_obs, y_pred))
            auc = ""
            f1_or_acc = f"{acc:.6f}"
    except Exception:
        auc = ""
        f1_or_acc = ""

    # Save predictions for observed subset
    out_csv = os.path.join(output_dir, f"{tox}_tabnet_preds.csv")
    out_df = pd.DataFrame({
        "True_Label": y_te_obs,
        "Pred_Prob":  preds_prob_obs
    })
    if test_ids is not None:
        out_df.insert(0, "row_id", test_ids.values[te_mask])
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

# --- Save metrics as a single CSV for easy R loading ---
metrics_path = os.path.join(output_dir, "tabnet_binary_metrics_structured.csv")
with open(metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Type", "Variable", "F1_or_Acc", "AUC"])
    writer.writeheader()
    writer.writerows(metrics_rows)

print("\n✓ Saved all toxin predictions and metrics in:")
print(" -", output_dir)
print(" -", metrics_path)