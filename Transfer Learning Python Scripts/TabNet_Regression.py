# tabnet_regression_presplit_filtered_v2.py
# -----------------------------------------------------
# Train TabNetRegressor per continuous toxin using PRE-SPLIT CSVs.
# Drops NaN targets from train and test *for fitting/metrics*
# Predicts on the FULL test set; evaluates only on observed subset
# Saves all toxin metrics in a single structured CSV for R loading.
# -----------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import csv

# ---------- config ----------
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


base_dir   = "~/TabNET"
data_dir   = os.path.join(base_dir, "data")
outpt_test_dir = "~/new test"
output_dir = os.path.join(outpt_test_dir, "tabnet_regression_filtered_predictions_v2")
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

def to_float(arr) -> np.ndarray:
    return pd.to_numeric(pd.Series(arr), errors="coerce").values.astype(np.float32)

def rmse_version_agnostic(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        from sklearn.metrics import mean_squared_error
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

X_train_df = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
X_test_df  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
Ytr_df     = pd.read_csv(os.path.join(data_dir, "Y_cont_train.csv"))
Yte_df     = pd.read_csv(os.path.join(data_dir, "Y_cont_test.csv"))

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
        print(f"Skipping {tox}: column not found in Y_cont_* CSVs")
        metrics_rows.append({
            "Type": "continuous",
            "Variable": tox,
            "RMSE": "",
            "R2": ""
        })
        continue

    y_tr_full = to_float(Ytr_df[tox].values)
    y_te_full = to_float(Yte_df[tox].values)

    tr_mask = ~np.isnan(y_tr_full)
    te_mask = ~np.isnan(y_te_full)

    if tr_mask.sum() < 10:
        print(f"Skipping {tox}: <10 usable training rows after NaN filtering")
        metrics_rows.append({
            "Type": "continuous",
            "Variable": tox,
            "RMSE": "",
            "R2": ""
        })
        continue

    X_tr = X_train_full[tr_mask]
    y_tr = y_tr_full[tr_mask].reshape(-1, 1)

    X_te_obs = X_test_full[te_mask]
    y_te_obs = y_te_full[te_mask].reshape(-1, 1)
    y_te_obs_1d = y_te_full[te_mask].ravel()

    reg = TabNetRegressor(
        seed=SEED,
        verbose=0,
        device_name="cpu"
    )

    reg.fit(
        X_tr, y_tr,
        eval_set=[(X_te_obs, y_te_obs)],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0
    )

    preds_full = reg.predict(X_test_full).ravel()
    preds_obs  = preds_full[te_mask]

    # Metrics
    if te_mask.any():
        rmse = rmse_version_agnostic(y_te_obs_1d, preds_obs)
        r2   = float(r2_score(y_te_obs_1d, preds_obs))
    else:
        rmse = ""
        r2 = ""

    # Save predictions for observed subset
    out_csv  = os.path.join(output_dir, f"{tox}_tabnet_preds.csv")
    out_df = pd.DataFrame({
        "True": y_te_obs_1d,
        "Pred": preds_obs
    })
    if test_ids is not None:
        out_df.insert(0, "row_id", test_ids.values[te_mask])
    out_df.to_csv(out_csv, index=False)

    metrics_rows.append({
        "Type": "continuous",
        "Variable": tox,
        "RMSE": f"{rmse:.6f}" if rmse != "" else "",
        "R2": f"{r2:.6f}" if r2 != "" else ""
    })

    print(f"Saved → {out_csv}  | RMSE={rmse if rmse != '' else 'NA'} R2={r2 if r2 != '' else 'NA'}")

# --- Save metrics as a single CSV for easy R loading ---
metrics_path = os.path.join(output_dir, "tabnet_regression_metrics_structured.csv")
with open(metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Type", "Variable", "RMSE", "R2"])
    writer.writeheader()
    writer.writerows(metrics_rows)

print("\n✓ Saved all toxin predictions and metrics in:")
print(" -", output_dir)
print(" -", metrics_path)