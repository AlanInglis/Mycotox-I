# ftt_joint_multitask_presplit_align_alias_v2.py
# -----------------------------------------------------
# FT-Transformer joint model (multi-task: regression + multi-label binary)
# Uses your PRE-SPLIT CSVs and is robust to column *naming* differences.
#
# It will reorder target columns to the expected CONT/BIN lists and
# resolve common aliases like:
#   X3.AC.DON   -> 3-AC-DON / 3_AC_DON / X3_AC_DON
#   X15.AC.DON  -> 15-AC-DON / 15_AC_DON / X15_AC_DON
#   T.2_toxin   -> T-2_toxin / T2_toxin / T_2_toxin
#   HT.2_toxin  -> HT-2_toxin / HT2_toxin / HT_2_toxin
# If any expected column is truly missing after aliasing, it errors clearly.
#
# v2 Changes:
# - Regression head now uses a ReLU activation to ensure non-negative outputs.
# - Classification head now uses a Sigmoid activation to output probabilities directly.
# - Loss function for classification changed to binary_cross_entropy.
# - Final metrics are saved as a structured CSV with columns:
#     Type, Variable, RMSE, R2, F1_or_Acc, AUC
# -----------------------------------------------------

import os, platform, csv
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import rtdl_revisiting_models as rtdl

# Allow PyTorch to use full Apple GPU memory (MPS)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ---------- config ----------
SEED = 42
BASE = "/Users/alaninglis/Desktop/Transfer Learning models/FT Transformer/data"
outpt_test_dir = "/Users/alaninglis/Desktop/Transfer Learning models/FT Transformer/ftt_results"
OUT  = os.path.join(outpt_test_dir, "ftt_joint_predictions_v2"); os.makedirs(OUT, exist_ok=True)

CONT = [
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
BIN  = [f"{t}_bin" for t in CONT]
N_CONT, N_BIN = len(CONT), len(BIN)

BS = 128
MAX_E = 60
PATIENCE = 8
LR = 3e-3
D_BLOCK = 192
N_BLOCKS = 2

# ---------- reproducibility ----------
def set_seeds(seed=SEED):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seeds()

# ---------- masked losses ----------
def masked_mse(pred, tgt, msk):
    diff = (pred - tgt).pow(2) * msk.float()
    denom = msk.sum().clamp(min=1)
    return diff.sum() / denom

def masked_bce(probs, tgt, msk):
    bce = nn.functional.binary_cross_entropy(probs, tgt, reduction="none")
    denom = msk.sum().clamp(min=1)
    return (bce * msk.float()).sum() / denom

# ---------- helpers ----------
def get_numeric_X(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()

def to_float_matrix(df: pd.DataFrame) -> np.ndarray:
    # Coerce every column to numeric; non-numeric -> NaN
    return pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in df.columns}).values.astype(np.float32)

def make_loader(X, Yc, Yb, Mc, Mb, bs, shuffle):
    num_workers = 0 if platform.system() == "Darwin" else max(os.cpu_count() or 1, 1)
    ds = TensorDataset(X, Yc, Yb, Mc, Mb)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers)

def reorder_with_aliases(df: pd.DataFrame, expected: list, alias_map: dict, label: str) -> pd.DataFrame:
    resolved_cols = []
    missing = []
    extras = [c for c in df.columns if c not in expected and all(c not in alias_map.get(e, []) for e in expected)]
    for name in expected:
        candidates = [name] + alias_map.get(name, [])
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            missing.append(name)
        else:
            resolved_cols.append(found)
    if extras:
        print(f"[Info] {label}: dropping unexpected columns: {extras}")
    if missing:
        present = list(df.columns)
        raise ValueError(
            f"{label}: missing expected columns after alias matching: {missing}\n"
            f"Present columns: {present}\n"
            f"Tip: extend alias_map or fix the CSV headers."
        )
    return df.loc[:, resolved_cols]

# Alias maps for common naming variants
alias_cont = {
    "X3.AC.DON":  ["3-AC-DON", "3_AC_DON", "X3_AC_DON"],
    "X15.AC.DON": ["15-AC-DON", "15_AC_DON", "X15_AC_DON"],
    "T.2_toxin":  ["T-2_toxin", "T2_toxin", "T_2_toxin"],
    "HT.2_toxin": ["HT-2_toxin", "HT2_toxin", "HT_2_toxin"],
}
# Build binary alias map automatically
alias_bin = {f"{k}_bin": [f"{v}_bin" for v in vals] for k, vals in alias_cont.items()}

# ---------- load pre-split data ----------
X_train_df = pd.read_csv(os.path.join(BASE, "x_train.csv"))
X_test_df  = pd.read_csv(os.path.join(BASE, "x_test.csv"))

Yc_tr_df   = pd.read_csv(os.path.join(BASE, "y_cont_train.csv"))
Yc_te_df   = pd.read_csv(os.path.join(BASE, "y_cont_test.csv"))
Yb_tr_df   = pd.read_csv(os.path.join(BASE, "y_bin_train.csv"))
Yb_te_df   = pd.read_csv(os.path.join(BASE, "y_bin_test.csv"))

# Reorder target columns to match CONT/BIN using aliases
Yc_tr_df = reorder_with_aliases(Yc_tr_df, CONT, alias_cont, "y_cont_train.csv")
Yc_te_df = reorder_with_aliases(Yc_te_df, CONT, alias_cont, "y_cont_test.csv")
Yb_tr_df = reorder_with_aliases(Yb_tr_df, BIN,  alias_bin,  "y_bin_train.csv")

# Features (numeric only); standardise by TRAIN stats
X_train_np = get_numeric_X(X_train_df).values.astype(np.float32)
X_test_np  = get_numeric_X(X_test_df).values.astype(np.float32)
if np.isnan(X_train_np).any() or np.isnan(X_test_np).any():
    raise ValueError("Predictor matrices contain NaNs; model requires finite inputs.")

mu = X_train_np.mean(0, keepdims=True)
sd = X_train_np.std(0, keepdims=True); sd[sd == 0] = 1.0
X_train_np = (X_train_np - mu) / sd
X_test_np  = (X_test_np  - mu) / sd

# Targets + masks; keep masks for missing labels
Yc_tr_np = to_float_matrix(Yc_tr_df)
Yc_te_np = to_float_matrix(Yc_te_df)
Yb_tr_np = to_float_matrix(Yb_tr_df)
Yb_te_np = to_float_matrix(Yb_te_df)

Mc_tr_np = ~np.isnan(Yc_tr_np)
Mb_tr_np = ~np.isnan(Yb_tr_np)
Mc_te_np = ~np.isnan(Yc_te_np)
Mb_te_np = ~np.isnan(Yb_te_np)

Yc_tr_np = np.nan_to_num(Yc_tr_np, nan=0.0)
Yc_te_np = np.nan_to_num(Yc_te_np, nan=0.0)
Yb_tr_np = np.nan_to_num(Yb_tr_np, nan=0.0)
Yb_te_np = np.nan_to_num(Yb_te_np, nan=0.0)

# Torch tensors
to_f32 = lambda a: torch.tensor(a, dtype=torch.float32)
X_train = to_f32(X_train_np)
X_test  = to_f32(X_test_np)
Yc_tr   = to_f32(Yc_tr_np)
Yc_te   = to_f32(Yc_te_np)
Yb_tr   = to_f32(Yb_tr_np)
Yb_te   = to_f32(Yb_te_np)
Mc_tr   = torch.tensor(Mc_tr_np, dtype=torch.bool)
Mb_tr   = torch.tensor(Mb_tr_np, dtype=torch.bool)
Mc_te   = torch.tensor(Mc_te_np, dtype=torch.bool)
Mb_te   = torch.tensor(Mb_te_np, dtype=torch.bool)

# Internal validation split (10% of TRAIN) for early stopping
n = X_train.shape[0]
rng = np.random.default_rng(SEED)
perm = torch.tensor(rng.permutation(n))
cut = int(n * 0.9)
tr_idx, va_idx = perm[:cut], perm[cut:]
take = lambda T, idx: T[idx]

X_tr, X_va   = take(X_train, tr_idx), take(X_train, va_idx)
Yc_tr_i, Yc_va = take(Yc_tr, tr_idx), take(Yc_tr, va_idx)
Yb_tr_i, Yb_va = take(Yb_tr, tr_idx), take(Yb_tr, va_idx)
Mc_tr_i, Mc_va = take(Mc_tr, tr_idx), take(Mc_tr, va_idx)
Mb_tr_i, Mb_va = take(Mb_tr, tr_idx), take(Mb_tr, va_idx)

train_dl = make_loader(X_tr, Yc_tr_i, Yb_tr_i, Mc_tr_i, Mb_tr_i, BS, True)
val_dl   = make_loader(X_va, Yc_va,   Yb_va,   Mc_va,   Mb_va,   BS, False)

# Device & AMP
if torch.cuda.is_available():
    DEV, AMP, AMP_DTYPE = "cuda", True, torch.float16
    scaler = torch.cuda.amp.GradScaler()
elif torch.backends.mps.is_available():
    DEV, AMP, scaler, AMP_DTYPE = "mps", False, None, None
else:
    DEV, AMP, scaler, AMP_DTYPE = "cpu", False, None, None

# Model: FT-Transformer backbone + two heads
kw = rtdl.FTTransformer.get_default_kwargs(); kw.update(n_blocks=N_BLOCKS, d_block=D_BLOCK)
backbone = rtdl.FTTransformer(
    n_cont_features=X_train.shape[1],
    cat_cardinalities=[],
    d_out=None,
    **kw,
).to(DEV)
d_model = kw["d_block"]

# Regression head with ReLU to ensure non-negative outputs
head_reg = nn.Sequential(
    nn.Linear(d_model, N_CONT),
    nn.ReLU()
).to(DEV)

# Binary classification head with Sigmoid to output probabilities directly
head_bin = nn.Sequential(
    nn.Linear(d_model, N_BIN),
    nn.Sigmoid()
).to(DEV)

if DEV == "cuda":
    backbone = torch.compile(backbone, mode="reduce-overhead")

opt = torch.optim.AdamW(
    list(backbone.parameters()) + list(head_reg.parameters()) + list(head_bin.parameters()),
    lr=LR
)

# Training loop with early stopping
best, wait = float("inf"), 0
for ep in range(1, MAX_E + 1):
    backbone.train(); head_reg.train(); head_bin.train()
    for xb, yc, yb, mc, mb in train_dl:
        xb, yc, yb, mc, mb = (t.to(DEV) for t in (xb, yc, yb, mc, mb))
        opt.zero_grad()
        if AMP:
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                tok = backbone(xb, None)
                loss = masked_mse(head_reg(tok), yc, mc) + masked_bce(head_bin(tok), yb, mb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            tok = backbone(xb, None)
            loss = masked_mse(head_reg(tok), yc, mc) + masked_bce(head_bin(tok), yb, mb)
            loss.backward(); opt.step()
        if DEV == "mps": torch.mps.empty_cache()

    backbone.eval(); head_reg.eval(); head_bin.eval()
    val = 0.0
    with torch.no_grad():
        for xb, yc, yb, mc, mb in val_dl:
            xb, yc, yb, mc, mb = (t.to(DEV) for t in (xb, yc, yb, mc, mb))
            if AMP:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    tok = backbone(xb, None)
                    val += masked_mse(head_reg(tok), yc, mc).item() + masked_bce(head_bin(tok), yb, mb).item()
            else:
                tok = backbone(xb, None)
                val += masked_mse(head_reg(tok), yc, mc).item() + masked_bce(head_bin(tok), yb, mb).item()
    val /= max(len(val_dl), 1)
    print(f"Epoch {ep:02d}  val_loss = {val:.5f}")

    if val < best - 1e-6:
        best, wait = val, 0
        torch.save(
            {"backbone": backbone.state_dict(),
             "reg": head_reg.state_dict(),
             "bin": head_bin.state_dict()},
            os.path.join(OUT, "best_joint.pt")
        )
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping."); break

# Load best and predict on TEST
ckpt = torch.load(os.path.join(OUT, "best_joint.pt"), map_location=DEV)
backbone.load_state_dict(ckpt["backbone"])
head_reg.load_state_dict(ckpt["reg"])
head_bin.load_state_dict(ckpt["bin"])
backbone.eval(); head_reg.eval(); head_bin.eval()

with torch.no_grad():
    tok_te = backbone(torch.tensor(X_test_np, dtype=torch.float32, device=DEV), None)
    preds_cont = head_reg(tok_te).cpu().numpy() # (N_test, N_CONT)
    probs_bin  = head_bin(tok_te).cpu().numpy()  # (N_test, N_BIN)

# Save combined predictions (test set)
cont_cols_true = [f"{t}_true" for t in CONT]
cont_cols_pred = [f"{t}_pred" for t in CONT]
bin_cols_true  = [f"{t}_true" for t in BIN]
bin_cols_prob  = [f"{t}_prob" for t in BIN]

cont_df = pd.DataFrame(np.hstack([np.where(np.isfinite(Yc_te_np), Yc_te_np, np.nan), preds_cont]),
                       columns=cont_cols_true + cont_cols_pred)
bin_df  = pd.DataFrame(np.hstack([np.where(np.isfinite(Yb_te_np), Yb_te_np, np.nan), probs_bin]),
                       columns=bin_cols_true + bin_cols_prob)

cont_df.to_csv(os.path.join(OUT, "test_predictions_continuous.csv"), index=False)
bin_df.to_csv(os.path.join(OUT, "test_predictions_binary.csv"), index=False)

# Metrics on observed test rows (Structured CSV output)
from sklearn.metrics import r2_score, roc_auc_score, f1_score, accuracy_score

def rmse_version_agnostic(y_true, y_pred):
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        from sklearn.metrics import mean_squared_error
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

metrics_rows = []

# Continuous variables
for j, name in enumerate(CONT):
    mask = Mc_te_np[:, j]
    if mask.sum() == 0:
        metrics_rows.append({
            "Type": "continuous",
            "Variable": name,
            "RMSE": "",
            "R2": "",
            "F1_or_Acc": "",
            "AUC": ""
        })
        continue
    y = Yc_te_np[mask, j]
    p = preds_cont[mask, j]
    rmse = rmse_version_agnostic(y, p)
    r2   = float(r2_score(y, p))
    metrics_rows.append({
        "Type": "continuous",
        "Variable": name,
        "RMSE": f"{rmse:.6f}",
        "R2": f"{r2:.6f}",
        "F1_or_Acc": "",
        "AUC": ""
    })

# Binary variables
for j, name in enumerate(BIN):
    mask = Mb_te_np[:, j]
    vals = Yb_te_np[mask, j]
    if mask.sum() == 0 or len(np.unique(vals)) < 2:
        metrics_rows.append({
            "Type": "binary",
            "Variable": name,
            "RMSE": "",
            "R2": "",
            "F1_or_Acc": "",
            "AUC": ""
        })
        continue
    y = vals
    p = probs_bin[mask, j]
    # F1 if both classes present, else accuracy; threshold at 0.5
    if len(np.unique(y)) == 2:
        y_pred = (p >= 0.5).astype(int)
        f1 = float(f1_score(y, y_pred))
        auc = float(roc_auc_score(y, p))
        f1_or_acc = f"{f1:.6f}"
    else:
        # Only one class present, use accuracy
        y_pred = (p >= 0.5).astype(int)
        acc = float(accuracy_score(y, y_pred))
        auc = ""
        f1_or_acc = f"{acc:.6f}"
    metrics_rows.append({
        "Type": "binary",
        "Variable": name,
        "RMSE": "",
        "R2": "",
        "F1_or_Acc": f1_or_acc,
        "AUC": f"{auc:.6f}" if auc != "" else ""
    })

# Write to structured CSV
metrics_path = os.path.join(OUT, "test_metrics_structured.csv")
with open(metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Type", "Variable", "RMSE", "R2", "F1_or_Acc", "AUC"])
    writer.writeheader()
    writer.writerows(metrics_rows)

print("✓ Saved:")
print(" -", os.path.join(OUT, "test_predictions_continuous.csv"))
print(" -", os.path.join(OUT, "test_predictions_binary.csv"))
print(" -", metrics_path)