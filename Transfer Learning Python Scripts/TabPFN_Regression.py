# tabpfn_regression_preds_presplit_v2.py
# -----------------------------------------------------
# Train a TabPFN regressor per continuous toxin using PRE-SPLIT CSVs.
# For each toxin column in Y_cont_*, fits on train and predicts on test.
# Saves per-toxin CSVs with row_id (if present), True, Pred.
# Also saves a structured metrics CSV for all toxins for easy R loading.
# -----------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
import csv

# ---- config ----
SEED = 42
toxins = [
    "Trichothence_producer",
    "F_langsethiae",
    "F_poae",
    "DON",
    "D3G",
    "Nivalenol",
    "3-AC-DON",
    "15-AC-DON",
    "T-2_toxin",
    "HT-2_toxin",
    "T2G",
    "Neos",
    "ENN_A1",
    "ENN_A",
    "ENN_B",
    "ENN_B1",
    "BEAU",
    "ZEN",
    "Apicidin",
    "STER",
    "DAS",
    "Quest",
    "AOH",
    "AME",
    "MON",
    "Ergocristine",
    "EGT"
]

base_dir   = "~/data"
outpt_test_dir = "~/new test"
output_dir = os.path.join(outpt_test_dir, "tabpfn_cont_predictions_v2")
os.makedirs(output_dir, exist_ok=True)

# ---- seeding ----
def set_seeds(seed=SEED):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seeds()

# ---- metrics helpers (version-agnostic RMSE) ----
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # RMSE
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        from sklearn.metrics import mean_squared_error
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # R2
    from sklearn.metrics import r2_score
    r2  = float(r2_score(y_true, y_pred))
    return rmse, r2

def get_numeric_X(df):
    return df.select_dtypes(include=[np.number]).copy()

X_train_df = pd.read_csv(os.path.join(base_dir, "X_train.csv"))
X_test_df  = pd.read_csv(os.path.join(base_dir, "X_test.csv"))
Ytr_df     = pd.read_csv(os.path.join(base_dir, "Y_cont_train.csv"))
Yte_df     = pd.read_csv(os.path.join(base_dir, "Y_cont_test.csv"))

test_ids = X_test_df["row_id"].copy() if "row_id" in X_test_df.columns else None

X_train = get_numeric_X(X_train_df).values.astype(np.float32)
X_test  = get_numeric_X(X_test_df).values.astype(np.float32)

if np.isnan(X_train).any() or np.isnan(X_test).any():
    raise ValueError("Predictor matrices contain NaNs; TabPFN requires finite inputs.")

# --- Prepare metrics collector ---
metrics_rows = []

# ---- per-toxin loop ----
for tox in toxins:
    if tox not in Ytr_df.columns or tox not in Yte_df.columns:
        print(f"Skipping {tox}: column not found in Y_cont_* CSVs")
        metrics_rows.append({
            "Type": "continuous",
            "Variable": tox,
            "RMSE": "",
            "R2": ""
        })
        continue

    y_tr = pd.to_numeric(Ytr_df[tox], errors="coerce").values.astype(np.float32)
    y_te = pd.to_numeric(Yte_df[tox], errors="coerce").values.astype(np.float32)

    tr_mask = ~np.isnan(y_tr)
    te_mask = ~np.isnan(y_te)
    if tr_mask.sum() < 10:
        print(f"Skipping {tox}: <10 usable training rows after NaN filtering")
        metrics_rows.append({
            "Type": "continuous",
            "Variable": tox,
            "RMSE": "",
            "R2": ""
        })
        continue

    X_tr, y_tr = X_train[tr_mask], y_tr[tr_mask]
    X_te, y_te = X_test[te_mask],  y_te[te_mask]
    ids_te = test_ids[te_mask] if test_ids is not None else None

    # Fit and predict
    reg = TabPFNRegressor(device="cpu")
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)

    # Metrics
    rmse, r2 = compute_metrics(y_te, preds)
    metrics_rows.append({
        "Type": "continuous",
        "Variable": tox,
        "RMSE": f"{rmse:.6f}",
        "R2": f"{r2:.6f}"
    })

    # Save predictions
    out_csv  = os.path.join(output_dir, f"{tox}_tabpfn_preds.csv")
    out_df = pd.DataFrame({"True": y_te, "Pred": preds})
    if ids_te is not None:
        out_df.insert(0, "row_id", ids_te.values)
    out_df.to_csv(out_csv, index=False)

    print(f"Saved → {out_csv}  | RMSE={rmse:.3f} R2={r2:.3f}")

# ---- Save metrics as a single structured CSV ----
metrics_path = os.path.join(output_dir, "tabpfn_regression_metrics_structured.csv")
with open(metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Type", "Variable", "RMSE", "R2"])
    writer.writeheader()
    writer.writerows(metrics_rows)

print("\n✓ Saved all toxin predictions and metrics in:")
print(" -", output_dir)
print(" -", metrics_path)