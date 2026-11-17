#!/usr/bin/env python3
"""
Map readability prediction via 5-fold CV.
Three conditions: visual embeddings (PE), hand-crafted features, and their fusion.
"""

import os
import random
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import open_clip

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("Data")
XLS_PATH    = DATA_DIR / "Indata.xls"
TRAIN_DIR   = DATA_DIR / "maps_used_for_training"
ASSESS_DIR  = DATA_DIR / "maps_to_be_assessed"
EMBED_CACHE = DATA_DIR / "pe_embeddings.pt"
MODEL_DIR   = DATA_DIR / "models"

MODEL_ID     = "hf-hub:timm/PE-Core-bigG-14-448"
BATCH_SIZE   = 4        # conservative for 4070 Ti 16 GB
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FOLDS      = 5
N_REPEATS    = 20   # repeated k-fold: total folds = N_FOLDS × N_REPEATS
N_EPOCHS     = 300
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 40

VISUAL_DIM   = 1280
PROJ_DIM     = 128
HIDDEN_DIM   = 256

# Approximate substrings to locate feature columns in the spreadsheet
FEATURE_KEY_SUBSTRINGS = [
    "total length of line",
    "number of points",
    "number of object types",
    "less than 0.2 mm",   # targets exactly the 0.2 mm proximity threshold
]
TARGET_SUBSTRING = r"Alla - medel - I"
REGION_COL       = "Region"


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Column lookup (case-insensitive substring match) ─────────────────────────
def find_col(df: pd.DataFrame, substring: str) -> str:
    """Return the first column name whose lower-case form contains `substring`."""
    substring = substring.lower()
    matches = [c for c in df.columns if substring in str(c).lower()]
    if not matches:
        raise KeyError(
            f"No column containing '{substring}' found.\n"
            f"Available columns: {list(df.columns)}"
        )
    if len(matches) > 1:
        print(f"  [warn] Multiple columns match '{substring}': {matches} — using first")
    return matches[0]


# ── Models ────────────────────────────────────────────────────────────────────
class VisualOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VISUAL_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, vis, feat=None):
        return self.net(vis).squeeze(-1)


class FeaturesOnlyModel(nn.Module):
    def __init__(self, n_feat: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, vis=None, feat=None):
        return self.net(feat).squeeze(-1)


class FusionModel(nn.Module):
    """
    Each modality is first projected to PROJ_DIM, then concatenated
    before the shared hidden layer. This prevents the 1280-dim visual
    signal from drowning the 4 hand-crafted features in a raw concat.
    """
    def __init__(self, n_feat: int = 4):
        super().__init__()
        self.vis_proj  = nn.Sequential(nn.Linear(VISUAL_DIM, PROJ_DIM), nn.ReLU())
        self.feat_proj = nn.Sequential(nn.Linear(n_feat,     PROJ_DIM), nn.ReLU())
        self.head      = nn.Sequential(
            nn.Linear(PROJ_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, vis, feat):
        fused = torch.cat([self.vis_proj(vis), self.feat_proj(feat)], dim=-1)
        return self.head(fused).squeeze(-1)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    # "AllData_transpose" already has rows = map patches, columns = measures.
    # sheet_name=0 was a smaller summary sheet (~172 rows) — wrong sheet.
    df = pd.read_excel(XLS_PATH, sheet_name="AllData_transpose")

    # Locate target and feature columns
    target_col   = find_col(df, TARGET_SUBSTRING)
    feature_cols = [find_col(df, s) for s in FEATURE_KEY_SUBSTRINGS]

    print(f"\nTarget column  : {target_col}")
    print(f"Feature columns: {feature_cols}")

    # Build case-insensitive stem → Path map across both image folders
    img_map: dict[str, Path] = {}
    for folder in [TRAIN_DIR, ASSESS_DIR]:
        for p in folder.iterdir():
            if p.suffix.lower() == ".png":
                img_map[p.stem.lower()] = p

    regions = df[REGION_COL].astype(str).str.strip()
    matched = regions.str.lower().isin(img_map)

    n_miss = (~matched).sum()
    if n_miss:
        print(f"\n[warn] {n_miss} rows have no matching image file and will be skipped:")
        print(f"  {regions[~matched].tolist()}")

    df        = df[matched].reset_index(drop=True)
    img_paths = [img_map[r.lower()] for r in df[REGION_COL].astype(str).str.strip()]

    targets  = np.array(df[target_col].values,   dtype=np.float32)
    features = np.array(df[feature_cols].values, dtype=np.float32)

    print(f"\nDataset: {len(df)} samples")
    print(f"Target  — min={targets.min():.3f}  max={targets.max():.3f}  "
          f"mean={targets.mean():.3f}  std={targets.std():.3f}")

    return img_paths, targets, features


# ── Embedding extraction ──────────────────────────────────────────────────────
def extract_embeddings(img_paths: list[Path]) -> torch.Tensor:
    if EMBED_CACHE.exists():
        cached = torch.load(EMBED_CACHE, map_location="cpu")
        if cached.shape[0] == len(img_paths):
            print(f"\nLoaded cached PE embeddings from {EMBED_CACHE}  "
                  f"shape={tuple(cached.shape)}")
            return cached
        print(f"\n[warn] Cache size {cached.shape[0]} != {len(img_paths)} — re-extracting")

    print(f"\nExtracting PE embeddings on {str(DEVICE)}  (batch_size={BATCH_SIZE}) ...")
    pe_model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    pe_model = pe_model.to(DEVICE).eval()

    all_embeddings = []
    for start in range(0, len(img_paths), BATCH_SIZE):
        batch = img_paths[start : start + BATCH_SIZE]
        imgs  = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in batch
        ]).to(DEVICE)
        autocast_ctx = (torch.amp.autocast("cuda")
                        if DEVICE.type == "cuda"
                        else torch.amp.autocast("cpu", enabled=False))
        with torch.no_grad(), autocast_ctx:
            emb = pe_model.encode_image(imgs, normalize=True)
        all_embeddings.append(emb.cpu().float())
        done = min(start + BATCH_SIZE, len(img_paths))
        print(f"  {done}/{len(img_paths)}", end="\r")
    print()

    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings, EMBED_CACHE)
    print(f"Embeddings saved → {EMBED_CACHE}  shape={tuple(embeddings.shape)}")

    del pe_model
    torch.cuda.empty_cache()
    return embeddings


# ── Training ──────────────────────────────────────────────────────────────────
def train_fold(
    model: nn.Module,
    vis_tr: torch.Tensor, feat_tr: torch.Tensor, y_tr: torch.Tensor,
    vis_va: torch.Tensor, feat_va: torch.Tensor, y_va: torch.Tensor,
) -> nn.Module:
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion  = nn.MSELoss()
    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        criterion(model(vis_tr, feat_tr), y_tr).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(vis_va, feat_va), y_va).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(
    model: nn.Module,
    vis: torch.Tensor, feat: torch.Tensor, y_np: np.ndarray,
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        preds = model(vis, feat).cpu().numpy()
    mae  = float(np.array(mean_absolute_error(y_np, preds)))
    rmse = float(np.sqrt(np.array(mean_squared_error(y_np, preds))))
    r2   = float(r2_score(y_np, preds))
    return mae, rmse, r2


# ── Cross-validation ──────────────────────────────────────────────────────────
def run_cv(
    embeddings: torch.Tensor,
    targets:    np.ndarray,
    features:   np.ndarray,
) -> tuple[dict[str, list], dict[str, np.ndarray]]:
    n       = len(embeddings)
    n_feat  = features.shape[1]
    results: dict[str, list]     = {"Visual": [], "Features": [], "Fusion": []}
    # Accumulate per-sample predictions across all repeats for the CSV output
    pred_sum: dict[str, np.ndarray] = {k: np.zeros(n, dtype=np.float64)
                                        for k in results}
    pred_cnt: np.ndarray            = np.zeros(n, dtype=np.int32)
    total   = N_REPEATS * N_FOLDS

    print(f"\nFeature dim: {n_feat}  |  Visual dim: {VISUAL_DIM}")
    print(f"Repeated {N_REPEATS}× {N_FOLDS}-fold CV  ({total} total folds)\n")

    conditions = [
        ("Visual",   lambda: VisualOnlyModel()),
        ("Features", lambda: FeaturesOnlyModel(n_feat)),
        ("Fusion",   lambda: FusionModel(n_feat)),
    ]

    fold_global = 0
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(embeddings)):
            fold_global += 1
            print(f"── Rep {repeat + 1}/{N_REPEATS}  Fold {fold + 1}/{N_FOLDS}"
                  f"  [{fold_global}/{total}]"
                  f"  (train={len(tr_idx)}, val={len(va_idx)}) ──")

            vis_tr = embeddings[tr_idx].to(DEVICE)
            vis_va = embeddings[va_idx].to(DEVICE)

            scaler = StandardScaler()
            scaler.fit(features[tr_idx])
            feat_tr = torch.from_numpy(scaler.transform(features[tr_idx]).astype(np.float32)).to(DEVICE)
            feat_va = torch.from_numpy(scaler.transform(features[va_idx]).astype(np.float32)).to(DEVICE)

            y_tr    = torch.from_numpy(targets[tr_idx]).to(DEVICE)
            y_va    = torch.from_numpy(targets[va_idx]).to(DEVICE)
            y_va_np = targets[va_idx]

            fold_preds: dict[str, np.ndarray] = {}
            for cond_idx, (name, build_model) in enumerate(conditions):
                set_seeds(fold_global * len(conditions) + cond_idx)
                model = build_model().to(DEVICE)
                model = train_fold(model, vis_tr, feat_tr, y_tr, vis_va, feat_va, y_va)
                mae, rmse, r2 = evaluate(model, vis_va, feat_va, y_va_np)
                results[name].append((mae, rmse, r2))
                print(f"  {name:<10s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
                # Collect raw predictions for this fold
                model.eval()
                with torch.no_grad():
                    fold_preds[name] = model(vis_va, feat_va).cpu().numpy()

            # Accumulate (every sample appears in test exactly N_REPEATS times)
            for name in pred_sum:
                pred_sum[name][va_idx] += fold_preds[name]
            pred_cnt[va_idx] += 1

    # Average predictions across all repeats each sample was in test
    mean_preds = {k: pred_sum[k] / pred_cnt for k in pred_sum}
    return results, mean_preds


# ── Reporting ─────────────────────────────────────────────────────────────────
def report(results: dict[str, list]) -> None:
    print("\n" + "=" * 62)
    print(f"{'Condition':<12}  {'MAE (mean±std)':>18}  {'RMSE (mean±std)':>18}  {'R² (mean±std)':>16}")
    print("-" * 62)
    for name, folds in results.items():
        arr   = np.array(folds)         # (N_FOLDS, 3)
        m, s  = arr.mean(0), arr.std(0)
        print(f"{name:<12}  "
              f"{m[0]:.4f} ± {s[0]:.4f}   "
              f"{m[1]:.4f} ± {s[1]:.4f}   "
              f"{m[2]:.4f} ± {s[2]:.4f}")
    print("=" * 62)


# ── Per-map prediction CSV ────────────────────────────────────────────────────
def save_predictions_csv(
    img_paths:  list[Path],
    targets:    np.ndarray,
    mean_preds: dict[str, np.ndarray],
) -> None:
    regions = [p.stem for p in img_paths]
    df = pd.DataFrame({"Region": regions, "True_Score": targets.round(4)})
    for name, preds in mean_preds.items():
        df[f"Pred_{name}"]  = preds.round(4)
        df[f"Error_{name}"] = (targets - preds).round(4)
    out = DATA_DIR / "per_map_predictions.csv"
    df.to_csv(out, index=False)
    print(f"\nPer-map predictions saved → {out}")
    print(df[["Region", "True_Score"] +
              [f"Pred_{n}" for n in mean_preds]].to_string(index=False))


# ── Final model training (all data, fixed seed, for deployment) ───────────────
def train_final_models(
    embeddings: torch.Tensor,
    targets:    np.ndarray,
    features:   np.ndarray,
) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    n_feat = features.shape[1]

    set_seeds(0)

    scaler = StandardScaler()
    scaler.fit(features)
    vis_all  = embeddings.to(DEVICE)
    feat_all = torch.from_numpy(scaler.transform(features).astype(np.float32)).to(DEVICE)
    y_all    = torch.from_numpy(targets).to(DEVICE)
    criterion = nn.MSELoss()

    conditions = [
        ("visual",   VisualOnlyModel()),
        ("features", FeaturesOnlyModel(n_feat)),
        ("fusion",   FusionModel(n_feat)),
    ]

    print("\nTraining final models on all data...")
    for name, model in conditions:
        set_seeds(0)
        model = model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        model.train()
        for _ in range(N_EPOCHS):
            optimizer.zero_grad()
            criterion(model(vis_all, feat_all), y_all).backward()
            optimizer.step()
        model.eval()
        torch.save(model.state_dict(), MODEL_DIR / f"model_{name}.pt")
        print(f"  Saved → {MODEL_DIR / f'model_{name}.pt'}")

    with open(MODEL_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    torch.save({
        "n_feat":     n_feat,
        "visual_dim": VISUAL_DIM,
        "proj_dim":   PROJ_DIM,
        "hidden_dim": HIDDEN_DIM,
    }, MODEL_DIR / "metadata.pt")

    print(f"  Saved scaler + metadata → {MODEL_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    set_seeds(42)
    print(f"Device: {DEVICE!s}")
    img_paths, targets, features = load_data()
    embeddings = extract_embeddings(img_paths)
    results, mean_preds = run_cv(embeddings, targets, features)
    report(results)
    save_predictions_csv(img_paths, targets, mean_preds)
    train_final_models(embeddings, targets, features)


if __name__ == "__main__":
    main()
