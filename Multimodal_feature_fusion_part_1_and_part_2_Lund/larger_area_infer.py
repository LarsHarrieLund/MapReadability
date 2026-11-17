#!/usr/bin/env python3
"""
Sliding-window MULTIMODAL readability inference on a large GeoTIFF.

Three readability models are trained here on ALL available assessment-map patches
(Data/maps_to_be_assessed/, EXCLUDING the 16 maps in Data/maps_used_for_training/),
then applied to every 256x256 non-overlapping window of the raster:

  V     visual only  -- PE-Core-bigG-14-448 embedding (1280-d)
  T     text only    -- Qwen3-Embedding-8B embedding (4096-d) of a
                        Qwen3.5-27B readability description of the window
  V+T   fusion       -- visual + text

None of these three conditions uses the hand-crafted vector features, so NO
features CSV is required: every input can be computed from the raster alone.

Training reads the patch embeddings straight from the cached artifacts
(Data/pe_embeddings.pt, Data/text_embeddings.pt, Data/qwen_descriptions.json), so no
foundation model is reloaded for the patch side. Outputs are date-tagged
(mm.RESULTS_DATE) so earlier inference results are preserved untouched.

The pipeline is cache-driven so an HPC job can be restarted without redoing
finished work, and so predictors can later be re-run locally without a GPU.
Per-map caches (keyed by the TIF stem + window grid) live next to the outputs:

  Stage A  PE encoder    window image -> visual embedding   -> <stem>_pe_windows.pt
  Stage B  Qwen3.5-27B   window image -> readability text    -> <stem>_qwen_descriptions.json
  Stage C  Qwen3-Embed   text -> text embedding (4096)       -> <stem>_text_embeddings.pt
  Stage D  Fusion MLPs   embeddings -> per-window scores      -> <stem>_readability_*.tif

Proven components (Py3.14 shim, prompts, model IDs, dimensions, the loss/optimiser
constants, and the MultiModalModel architecture itself) are imported from
map_readability_multimodal so the patch models are trained exactly as in that study
and the window descriptions use the identical prompt.

Outputs per model (saved to Data/inference_output/, date-tagged with mm.RESULTS_DATE):
  <stem>_readability_<model>_<date>.tif      -- float32 GeoTIFF, one pixel per window
  <stem>_readability_<model>_<date>_viz.tif  -- colour overlay + legend

Usage on HPC (all inputs derived from the raster; no CSV needed):
  python larger_area_infer.py                       # full run on TIF_PATH
  python larger_area_infer.py --tif OTHER.tif       # different raster
  python larger_area_infer.py --smoke 3             # generate 3 window texts, then exit
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless HPC node: no display, write files only
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image

# Importing the multimodal module FIRST applies its Python-3.14 compatibility
# shim (neutralise torch.compile + block `import fla`) before any transformers
# import, and gives us the exact prompts / IDs / dims / model class the saved
# models_mm/ weights were trained with. All heavy ML imports inside it are lazy,
# so this import is cheap and does not pull in transformers/open_clip here.
import map_readability_multimodal as mm

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:
    sys.exit("rasterio not found -- install it with:  pip install rasterio")

# ── Config ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
TIF_PATH    = ROOT / "Data" / "Larger maps" / "Lund" / "617_38_05.tif"
MODEL_DIR   = ROOT / "Data" / "models_mm"          # multimodal models (V / T / V+T)
OUT_DIR     = ROOT / "Data" / "inference_output"

WINDOW_SIZE = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCORE_MIN   = 1.5   # colourmap lower anchor (hard to read)
SCORE_MAX   = 3.5   # colourmap upper anchor (easy to read)
CMAP        = "RdYlGn"

# The three conditions to run:  (display name, saved-file tag, model kwargs)
INFER_CONDITIONS = [
    ("V",   "V",  dict(use_vis=True,  use_text=False, use_feat=False)),
    ("T",   "T",  dict(use_vis=False, use_text=True,  use_feat=False)),
    ("V+T", "VT", dict(use_vis=True,  use_text=True,  use_feat=False)),
]


# ── TIF loading + window extraction ─────────────────────────────────────────────
def load_tif(tif_path: Path):
    with rasterio.open(tif_path) as src:
        profile       = src.profile.copy()
        geo_transform = src.transform
        data          = src.read()               # (bands, H, W)

    if data.shape[0] == 1:
        rgb = np.concatenate([data, data, data], axis=0)
    else:
        rgb = data[:3]

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        for c in range(3):
            lo, hi = rgb[c].min(), rgb[c].max()
            if hi > lo:
                rgb[c] = (rgb[c] - lo) / (hi - lo) * 255.0
        rgb = rgb.astype(np.uint8)

    print(f"TIF loaded: {rgb.shape[2]}x{rgb.shape[1]} px, {rgb.shape[0]} bands -> RGB uint8")
    return rgb, profile, geo_transform


def extract_windows(rgb_chw: np.ndarray):
    """Return (list[PIL.Image] in raster order, n_rows, n_cols)."""
    _, H, W = rgb_chw.shape
    n_rows  = H // WINDOW_SIZE
    n_cols  = W // WINDOW_SIZE
    print(f"Grid: {n_cols} cols x {n_rows} rows = {n_rows * n_cols} windows")

    windows = []
    for r in range(n_rows):
        for c in range(n_cols):
            patch = rgb_chw[:,
                            r * WINDOW_SIZE:(r + 1) * WINDOW_SIZE,
                            c * WINDOW_SIZE:(c + 1) * WINDOW_SIZE]
            windows.append(Image.fromarray(patch.transpose(1, 2, 0)))
    return windows, n_rows, n_cols


# ── Cache fingerprints (so per-map caches self-invalidate on change) ────────────
def _grid_tag(n_rows: int, n_cols: int) -> list[int]:
    return [n_rows, n_cols, WINDOW_SIZE]


def _text_fingerprint(descs_in_order: list[str]) -> str:
    """Depends on the embedder, the output dim, and every window description
    (in raster order). Any change forces re-embedding."""
    h = hashlib.sha256()
    h.update(mm.QWEN_EMBED_ID.encode("utf-8"))
    h.update(str(mm.TEXT_DIM).encode("utf-8"))
    for t in descs_in_order:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


# ── Stage A: PE-Core-bigG-14-448 window embeddings ──────────────────────────────
def extract_pe_embeddings(windows, n_rows, n_cols, cache_path: Path) -> torch.Tensor:
    grid = _grid_tag(n_rows, n_cols)
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu")
        if isinstance(blob, dict) and blob.get("grid") == grid \
                and blob["emb"].shape[0] == len(windows):
            print(f"\n[Stage A] Loaded cached PE window embeddings -> {cache_path.name} "
                  f"{tuple(blob['emb'].shape)}")
            return blob["emb"]
        print(f"\n[Stage A] Grid changed (or old-format cache) -> re-extracting PE.")

    import open_clip
    print(f"\n[Stage A] Extracting PE embeddings with {mm.PE_MODEL_ID} "
          f"({len(windows)} windows) ...")
    pe_model, _, preprocess = open_clip.create_model_and_transforms(mm.PE_MODEL_ID)
    pe_model = pe_model.to(DEVICE).eval()

    autocast_ctx = (torch.amp.autocast("cuda") if DEVICE.type == "cuda"
                    else torch.amp.autocast("cpu", enabled=False))
    out = []
    bs = mm.PE_BATCH_SIZE
    for start in range(0, len(windows), bs):
        batch = torch.stack([preprocess(im) for im in windows[start:start + bs]]).to(DEVICE)
        with torch.no_grad(), autocast_ctx:
            emb = pe_model.encode_image(batch, normalize=True)
        out.append(emb.cpu().float())
        print(f"  {min(start + bs, len(windows))}/{len(windows)}", end="\r")
    print()
    embeddings = torch.cat(out, 0)
    torch.save({"grid": grid, "emb": embeddings}, cache_path)
    print(f"[Stage A] PE embeddings saved -> {cache_path.name} {tuple(embeddings.shape)}")
    del pe_model
    mm.free_gpu()
    return embeddings


# ── Stage B: Qwen3.5-27B readability descriptions (one per window) ──────────────
def generate_descriptions(windows, n_rows, n_cols, cache_path: Path,
                          max_new: int | None = None) -> list[str]:
    """Return a list of descriptions in raster order. Cache is a JSON keyed by
    window index; it carries the prompt hash (identical to the training script)
    and the grid, so a prompt change or a different map regenerates cleanly."""
    fp   = mm._desc_fingerprint()
    grid = _grid_tag(n_rows, n_cols)
    cache: dict[str, str] = {}
    if cache_path.exists():
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(blob, dict) and blob.get("prompt_hash") == fp and blob.get("grid") == grid:
            cache = blob.get("descriptions", {})
        else:
            print("\n[Stage B] Prompt/model/token-budget or grid changed (or old-format "
                  "cache) -> ALL window descriptions will be regenerated.")

    todo = [i for i in range(len(windows)) if str(i) not in cache]
    if not todo:
        print(f"\n[Stage B] All {len(windows)} window descriptions cached -> {cache_path.name}")
        return [cache[str(i)] for i in range(len(windows))]
    if max_new is not None:                     # smoke test: only the first few
        todo = todo[:max_new]
        print(f"\n[Stage B] SMOKE MODE: generating only {len(todo)} description(s).")

    print(f"\n[Stage B] Generating {len(todo)} window descriptions with {mm.QWEN_VL_ID} "
          f"({len(cache)} cached) ...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    processor = AutoProcessor.from_pretrained(mm.QWEN_VL_ID, trust_remote_code=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            mm.QWEN_VL_ID, torch_dtype="auto", device_map="auto",
            trust_remote_code=True,
        ).eval()
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            f"Could not load {mm.QWEN_VL_ID} as an image-text-to-text model: {e}\n"
            "Upgrade: pip install -U transformers accelerate qwen-vl-utils"
        ) from e

    # Fail fast on CPU/disk offload. The 27B (~54GB bf16) does not fit on one 48GB
    # L40S; device_map="auto" then offloads layers to CPU/meta, which is ~100x
    # slower AND makes the meta-device layers emit broken 1-word output. Abort now
    # with guidance instead of wasting hours -- request 2 GPUs.
    offloaded = sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()
                        if str(v) in ("cpu", "disk", "meta")})
    if offloaded:
        raise RuntimeError(
            f"{mm.QWEN_VL_ID} was partially offloaded to {offloaded} because it did not "
            "fit on the visible GPU(s). Generation would be extremely slow and produce "
            "garbage. Request 2 GPUs: sbatch --gres=gpu:2 with CUDA_VISIBLE_DEVICES=0,1 "
            "(2x L40S = 96GB fits the ~54GB model with no CPU offload).")

    if mm._disable_fla:
        print("=" * 78)
        print("  [INFO] fla disabled (Python 3.14) -> using the pure-torch fallback.")
        print("  Correct but slower; this is expected on this environment.")
        print("=" * 78, flush=True)

    def _save() -> None:
        cache_path.write_text(
            json.dumps({"prompt_hash": fp, "grid": grid, "descriptions": cache},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    import re
    t_start = time.time()
    for n, idx in enumerate(todo, 1):
        image = windows[idx].convert("RGB")
        messages = [
            {"role": "system", "content": mm.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": mm.USER_PROMPT},
            ]},
        ]
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = [image], None
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=mm.GEN_MAX_NEW_TOKENS,
                                        do_sample=False, use_cache=True)   # greedy = reproducible
        dt = time.time() - t0

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)[0]
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[-1]
        desc = re.sub(r"<\|.*?\|>", "", raw)
        desc = desc.replace("<think>", "").replace("</think>", "").strip()
        cache[str(idx)] = desc
        if len(desc.split()) < 20:              # expected ~160-220 words; short => something broke
            print(f"  [WARN] window {idx} produced only {len(desc.split())} word(s): "
                  f"{desc!r} -- check for GPU offload / truncation.", flush=True)

        _save()                                 # save after EVERY window
        avg = (time.time() - t_start) / n
        eta_min = avg * (len(todo) - n) / 60.0
        print(f"  [{n}/{len(todo)}] window {idx}  {dt:.1f}s  "
              f"({len(desc.split())} words)  avg={avg:.1f}s/img  ETA={eta_min:.0f} min",
              flush=True)

    print(f"[Stage B] Descriptions saved -> {cache_path.name}")
    del model, processor
    mm.free_gpu()

    if max_new is not None:                     # smoke: cache may be incomplete
        return [cache.get(str(i), "") for i in range(len(windows))]
    return [cache[str(i)] for i in range(len(windows))]


# ── Stage C: Qwen3-Embedding-8B text embeddings ─────────────────────────────────
def embed_descriptions(descs: list[str], cache_path: Path) -> torch.Tensor:
    fp = _text_fingerprint(descs)
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu")
        if isinstance(blob, dict) and blob.get("hash") == fp \
                and blob["emb"].shape == (len(descs), mm.TEXT_DIM):
            print(f"\n[Stage C] Loaded cached text embeddings -> {cache_path.name} "
                  f"{tuple(blob['emb'].shape)}")
            return blob["emb"]
        print("\n[Stage C] Descriptions/embedder/dim changed (or old-format cache) "
              "-> re-embedding.")

    print(f"\n[Stage C] Embedding {len(descs)} window descriptions with "
          f"{mm.QWEN_EMBED_ID} (dim={mm.TEXT_DIM}) ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        mm.QWEN_EMBED_ID,
        model_kwargs={"torch_dtype": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
        truncate_dim=mm.TEXT_DIM,
        device=str(DEVICE),
    )
    emb = model.encode(descs, batch_size=mm.EMBED_BATCH_SIZE, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    embeddings = torch.from_numpy(emb.astype(np.float32))
    torch.save({"hash": fp, "emb": embeddings}, cache_path)
    print(f"[Stage C] Text embeddings saved -> {cache_path.name} {tuple(embeddings.shape)}")
    del model
    mm.free_gpu()
    return embeddings


# ── Model training (on the assessment-map patches) ───────────────────────────────
def train_models():
    """Train the V / T / V+T readability models on ALL available assessment-map
    patches -- the maps in Data/maps_to_be_assessed/, EXCLUDING the 16 maps in
    Data/maps_used_for_training/ -- then use them for window inference below.

    Every patch embedding is read from the cached artifacts
    (Data/pe_embeddings.pt, Data/text_embeddings.pt, Data/qwen_descriptions.json),
    which cover the full 346-patch set. We load over that full set so the caches stay
    valid (no foundation model is reloaded), then slice down to the assessment maps.
    """
    region_id, img_paths, targets, features = mm.load_data()        # full matched set
    descriptions = mm.generate_descriptions(region_id, img_paths)   # cache hit -> no VLM
    text_emb     = mm.embed_descriptions(region_id, descriptions)   # cache hit -> no 8B
    pe_emb       = mm.extract_pe_embeddings(img_paths)              # cache hit -> no PE

    assess_mask = np.array([p.parent.name == mm.ASSESS_DIR.name for p in img_paths], dtype=bool)
    n_keep, n_excl = int(assess_mask.sum()), int((~assess_mask).sum())
    print(f"\n[train] Training on {n_keep} assessment-map patches "
          f"(excluded {n_excl} training-map patches).")
    sel      = torch.from_numpy(assess_mask)
    vis_all  = pe_emb[sel].to(DEVICE)
    text_all = text_emb[sel].to(DEVICE)
    y_all    = torch.from_numpy(targets[assess_mask]).to(DEVICE)
    n_feat   = features.shape[1]

    crit = torch.nn.MSELoss()
    models = {}
    for name, tag, cfg in INFER_CONDITIONS:
        mm.set_seeds(0)
        model = mm.MultiModalModel(n_feat=n_feat, **cfg).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=mm.LR, weight_decay=mm.WEIGHT_DECAY)
        model.train()
        for _ in range(mm.N_EPOCHS):                 # train on all data (no early stop)
            opt.zero_grad()
            crit(model(vis=vis_all, text=text_all), y_all).backward()
            opt.step()
        models[name] = model.eval()
        print(f"  Trained {name} model.")
    print(f"\nTrained multimodal models: {list(models.keys())}")
    return models


# ── Stage D: inference ──────────────────────────────────────────────────────────
def run_inference(models, vis_all, text_all, n_rows, n_cols):
    score_maps = {}
    for name, tag, cfg in INFER_CONDITIONS:
        model = models[name]
        print(f"  Inferring {name} ...")
        with torch.no_grad():
            if cfg["use_vis"] and cfg["use_text"]:
                preds = model(vis=vis_all, text=text_all)
            elif cfg["use_vis"]:
                preds = model(vis=vis_all)
            else:
                preds = model(text=text_all)
        score_maps[name] = preds.cpu().numpy().reshape(n_rows, n_cols)
    return score_maps


# ── Visualisation & saving ──────────────────────────────────────────────────────
def save_outputs(score_maps, rgb_chw, profile, geo_transform, n_rows, n_cols, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(CMAP)
    norm = mcolors.Normalize(vmin=SCORE_MIN, vmax=SCORE_MAX)

    used_h = n_rows * WINDOW_SIZE
    used_w = n_cols * WINDOW_SIZE
    bg     = rgb_chw.transpose(1, 2, 0)[:used_h, :used_w]

    for name, score_grid in score_maps.items():
        tag = f"{name.replace('+', '').lower()}_{mm.RESULTS_DATE}"

        score_profile = profile.copy()
        score_profile.update(
            dtype=rasterio.float32, count=1, height=n_rows, width=n_cols,
            transform=Affine(
                geo_transform.a * WINDOW_SIZE, geo_transform.b, geo_transform.c,
                geo_transform.d, geo_transform.e * WINDOW_SIZE, geo_transform.f,
            ),
        )
        tif_out = OUT_DIR / f"{stem}_readability_{tag}.tif"
        with rasterio.open(tif_out, "w", **score_profile) as dst:
            dst.write(score_grid.astype(np.float32)[np.newaxis])
        print(f"  Score GeoTIFF  -> {tif_out.name}")

        overlay_scores = np.repeat(np.repeat(score_grid, WINDOW_SIZE, 0), WINDOW_SIZE, 1)
        overlay_rgba         = cmap(norm(overlay_scores))
        overlay_rgba[..., 3] = 0.5

        fig, (ax_map, ax_cb) = plt.subplots(
            1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [20, 1]})
        ax_map.imshow(bg)
        ax_map.imshow(overlay_rgba)
        ax_map.set_title(f"Map Readability -- {name} model  ({n_cols}x{n_rows} windows)",
                         fontsize=13, pad=10)
        ax_map.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.set_label("Predicted readability score", fontsize=10, labelpad=8)
        cb.set_ticks([1.5, 2.0, 2.5, 3.0, 3.5])
        cb.set_ticklabels(["1.5\n(very hard)", "2.0", "2.5\n(threshold)", "3.0", "3.5\n(very easy)"])

        plt.tight_layout()
        viz_out = OUT_DIR / f"{stem}_readability_{tag}_viz.tif"
        plt.savefig(viz_out, dpi=150, format="tiff", bbox_inches="tight")
        plt.close()
        print(f"  Visualisation  -> {viz_out.name}")


# ── Entry point ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Sliding-window multimodal (V/T/V+T) readability inference")
    ap.add_argument("--tif", type=Path, default=TIF_PATH, help="GeoTIFF to run on")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="Generate at most N window descriptions, then exit (quick GPU "
                         "test of the Qwen3.5 path before the full run).")
    args = ap.parse_args()

    tif_path = args.tif
    stem     = tif_path.stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc_cache = OUT_DIR / f"{stem}_qwen_descriptions.json"
    text_cache = OUT_DIR / f"{stem}_text_embeddings.pt"
    pe_cache   = OUT_DIR / f"{stem}_pe_windows.pt"

    mm.set_seeds(42)
    print(f"Device : {DEVICE}")
    print(f"TIF    : {tif_path}")

    print("\nLoading TIF...")
    rgb_chw, profile, geo_transform = load_tif(tif_path)
    windows, n_rows, n_cols = extract_windows(rgb_chw)

    # Stage B first (loads the 27B), so the smoke test exits before anything else.
    descs = generate_descriptions(windows, n_rows, n_cols, desc_cache,
                                  max_new=(args.smoke or None))
    if args.smoke:
        print(f"\n[smoke] Generated up to {args.smoke} window description(s); exiting "
              f"before embedding/inference. Inspect {desc_cache.name}.")
        return

    text_emb = embed_descriptions(descs, text_cache)                    # Stage C
    pe_emb   = extract_pe_embeddings(windows, n_rows, n_cols, pe_cache)  # Stage A

    vis_all  = pe_emb.to(DEVICE)
    text_all = text_emb.to(DEVICE)

    print("\nTraining models on assessment-map patches...")
    models = train_models()

    print("\nRunning inference...")
    score_maps = run_inference(models, vis_all, text_all, n_rows, n_cols)  # Stage D

    print("\nSaving outputs...")
    save_outputs(score_maps, rgb_chw, profile, geo_transform, n_rows, n_cols, stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
