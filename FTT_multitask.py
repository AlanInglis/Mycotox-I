# ftt_multitask_classifier.py – multi-task FT-Transformer for *binary* toxins
# macOS 14 + M-series (MPS, AMP off) • CUDA (AMP on)
# torch ≥ 2.3 • rtdl-revisiting-models 0.0.2

# ── let PyTorch use the full Apple GPU quota ──────────────────────────
import os, platform
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import rtdl_revisiting_models as rtdl


# ───────────────────────── helpers ────────────────────────────────────
def masked_bce(logits: torch.Tensor, tgt: torch.Tensor, msk: torch.Tensor) -> torch.Tensor:
    """Binary-cross-entropy, masking out missing labels."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, tgt, reduction="none"
    )
    return (bce * msk.float()).sum() / (msk.sum() + 1e-7)


def make_loader(X, Y, M, bs, shuffle):
    nw = 0 if platform.system() == "Darwin" else os.cpu_count()
    return DataLoader(TensorDataset(X, Y, M), bs, shuffle, num_workers=nw)


# ───────────────────────── main ───────────────────────────────────────
def main() -> None:
    BASE = ".../FT_Transformer"
    DATA = os.path.join(BASE, "data", "binary_data")        # <-- classification labels
    OUT  = os.path.join(BASE, "ftt_multiclass_predictions")
    os.makedirs(OUT, exist_ok=True)

    TOX = [
        "Trichothence_producer","F_langsethiae","F_poae","DON","D3G","Nivalenol",
        "X3.AC.DON","X15.AC.DON","T.2_toxin","HT.2_toxin","T2G","Neos","ENN_A1",
        "ENN_A","ENN_B","ENN_B1","BEAU","ZEN","Apicidin","STER","DAS","Quest",
        "AOH","AME","MON","Ergocristine","EGT"
    ]
    N_TGT = len(TOX)

    # ── predictors ────────────────────────────────────────────────────
    X = pd.read_csv(os.path.join(DATA, "predictor_data.csv")).values.astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-7)

    # ── labels & mask ────────────────────────────────────────────────
    Y_list, mask_list = [], []
    for t in TOX:
        y = pd.read_csv(os.path.join(DATA, f"{t}_bin.csv")).values.flatten().astype(np.float32)
        mask_list.append(~np.isnan(y))
        y[np.isnan(y)] = 0.0
        Y_list.append(y)
    Y   = np.stack(Y_list, axis=1)
    msk = np.stack(mask_list, axis=1)

    # ── split ────────────────────────────────────────────────────────
    X_tr,X_va,Y_tr,Y_va,M_tr,M_va = train_test_split(
        X, Y, msk, test_size=.2, random_state=42
    )

    to_f32 = lambda a: torch.tensor(a, dtype=torch.float32)
    X_tr,X_va,Y_tr,Y_va = map(to_f32,(X_tr,X_va,Y_tr,Y_va))
    M_tr,M_va = map(lambda m: torch.tensor(m, dtype=torch.bool),(M_tr,M_va))

    BATCH = 128
    train_dl = make_loader(X_tr,Y_tr,M_tr,BATCH,True)
    val_dl   = make_loader(X_va,Y_va,M_va,BATCH,False)

    # ── device & AMP ────────────────────────────────────────────────
    if torch.cuda.is_available():
        DEV, USE_AMP, AMP_DTYPE = "cuda", True, torch.float16
        scaler = torch.amp.GradScaler(DEV, enabled=True)
    elif torch.backends.mps.is_available():
        DEV, USE_AMP, scaler = "mps", False, None          # AMP off on MPS
        AMP_DTYPE = None
    else:
        DEV, USE_AMP, scaler = "cpu", False, None
        AMP_DTYPE = None

    # ── model ────────────────────────────────────────────────────────
    kw = rtdl.FTTransformer.get_default_kwargs()
    kw.update(n_blocks=2, d_block=192)

    ft = rtdl.FTTransformer(
        n_cont_features=X.shape[1],
        cat_cardinalities=[],
        d_out=N_TGT,                         # one logit per toxin
        **kw
    ).to(DEV)

    if DEV == "cuda":
        ft = torch.compile(ft, mode="reduce-overhead")

    opt = ft.make_default_optimizer()
    for g in opt.param_groups: g["lr"] = 3e-3

    # ── training loop ────────────────────────────────────────────────
    best, wait, PAT, MAX_EPOCH = float("inf"), 0, 8, 60
    for ep in range(1, MAX_EPOCH+1):
        ft.train()
        for xb,yb,mb in train_dl:
            xb,yb,mb = xb.to(DEV),yb.to(DEV),mb.to(DEV)
            opt.zero_grad()
            if USE_AMP:
                with torch.autocast(device_type=DEV,dtype=AMP_DTYPE):
                    loss = masked_bce(ft(xb,None), yb, mb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = masked_bce(ft(xb,None), yb, mb)
                loss.backward(); opt.step()
            if DEV=="mps": torch.mps.empty_cache()

        # validation
        ft.eval(); val=0.0
        with torch.no_grad():
            for xb,yb,mb in val_dl:
                xb,yb,mb = xb.to(DEV),yb.to(DEV),mb.to(DEV)
                if USE_AMP:
                    with torch.autocast(device_type=DEV,dtype=AMP_DTYPE):
                        val += masked_bce(ft(xb,None), yb, mb).item()
                else:
                    val += masked_bce(ft(xb,None), yb, mb).item()
        val /= len(val_dl)
        print(f"Epoch {ep:02d}  val_loss = {val:.4f}")
        if val<best:
            torch.save(ft.state_dict(), os.path.join(OUT,"best.pt")); best,wait=val,0
        else:
            wait+=1
            if wait==PAT: print("Early stopping."); break

    # ── generate validation probabilities & save ────────────────────
    ft.load_state_dict(torch.load(os.path.join(OUT,"best.pt"), map_location=DEV))
    ft.eval()
    with torch.no_grad():
        probs = torch.sigmoid(ft(X_va.to(DEV), None)).detach().cpu().numpy()

    cols = [f"{t}_true" for t in TOX] + [f"{t}_prob" for t in TOX]
    pd.DataFrame(np.hstack([Y_va.numpy(), probs]), columns=cols) \
        .to_csv(os.path.join(OUT, "validation_probs.csv"), index=False)

    print("✓ multi-task classification completed.")


if __name__ == "__main__":
    main()