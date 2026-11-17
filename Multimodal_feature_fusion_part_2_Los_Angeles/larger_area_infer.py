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

The final models' step count is chosen by a held-out epoch search rather than being
hard-coded (see EPOCH_RECIPE): the search splits pick the number of steps, then the
models are refit on all 330 patches.

Outputs per model (saved under Data/inference_output/<map>/, tagged with RUN_TAG):
  <stem>_readability_<model>_<tag>.tif      -- float32 GeoTIFF, one pixel per window
  <stem>_readability_<model>_<tag>_viz.tif  -- colour overlay + legend

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
OUT_DIR     = ROOT / "Data" / "inference_output"

WINDOW_SIZE = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Output layout ---------------------------------------------------------------
# Each raster gets its own subtree so inference_output/ stays readable as maps are
# added. Friendly folder names for the maps we know; anything else falls back to the
# file stem, so a new raster organises itself without touching this file.
MAP_DIR_ALIASES = {
    "617_38_05":              "lund",        # Lantmateriet topographic sheet, Lund
    "generated_map_enhanced": "synthetic",   # AI-generated map, no georeferencing
}


def map_dirs(stem: str) -> dict[str, Path]:
    """Return (and create) this map's output folders.

    'cache' holds the expensive foundation-model artifacts -- the 361 Qwen captions cost
    ~7 h on a single GPU -- so it is deliberately a sibling of the cheap outputs rather
    than mixed in with them.
    """
    root = OUT_DIR / MAP_DIR_ALIASES.get(stem, stem)
    d = {"root": root, "cache": root / "cache", "grids": root / "grids",
         "overlays": root / "overlays", "tables": root / "tables"}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d

# -- Running the identical stack on one consumer GPU -----------------------------
# The study ran on 2x L40S (96 GB), where the 27B (~54 GB bf16) is fully resident. On a
# single 16 GB card the SAME weights still run, with most layers placed on the CPU and
# a slice resident on the GPU; nothing is quantised and no model is substituted.
#
# Speed then comes from BATCHING. A forward pass reads all ~54 GB of weights regardless
# of batch size, so N windows in one pass cost barely more than one -- the difference
# between ~170 s and ~21 s per window. Greedy decoding with correct left-padding is
# mathematically the same as the unbatched decode the L40S run used; verify with
# --verify-batching before trusting a long run.
GEN_BATCH_SIZE   = 8       # windows per model.generate call  (--gen-batch)
VLM_GPU_GIB      = 11.0    # GiB of VLM weights to keep resident; the rest goes to CPU.
                           #   Leave headroom on a 15.9 GiB card for KV cache, the
                           #   vision tower's activations and the desktop compositor.
VLM_ALLOW_CPU    = True    # allow CPU placement. 'meta'/'disk' stay fatal -- those are
                           #   what actually produced the broken 1-word output on HPC.
EMBED_DEVICE     = "cpu"   # Qwen3-Embedding-8B is ~16 GB bf16 and does NOT fit in
                           #   15.9 GiB. On CPU it is the same weights at full
                           #   precision; 361 short texts is ~20 min. "auto" -> DEVICE.

SCORE_MIN   = 1.5   # colourmap lower anchor (hard to read)
SCORE_MAX   = 3.5   # colourmap upper anchor (easy to read)
CMAP        = "RdYlGn"

# -- Final-fit recipe for the inference models ----------------------------------
# The study's CV (map_readability_multimodal.run_cv) early-stops on an inner split, so
# its epoch count is CHOSEN, not fixed. The old code here instead trained a hard-coded
# mm.N_EPOCHS (=300) steps on all 330 patches -- a different estimator from the one the
# reported R2 describes.
#
# Recipe "D": use held-out splits ONLY to estimate the step count, then discard those
# models and refit on 100% of the 330 patches for that many steps.
#   1. EPOCH_SEARCH_SPLITS independent EPOCH_VAL_FRAC validation splits of the 330;
#   2. on each, train with early stopping (patience mm.PATIENCE) and record the epoch
#      at which validation MSE was minimal;
#   3. average those epochs -> E*;
#   4. refit from scratch on ALL 330 patches for exactly E* steps.
# Averaging over several splits matters: a single split's argmin is very noisy at this
# sample size (measured spread for T: 18 to 300 steps). And because the refit uses every
# patch, no map is permanently sacrificed to validation.
#
# "fix300" reproduces the previous behaviour (a flat mm.N_EPOCHS steps, no search).
EPOCH_RECIPE        = "fix300"  # "fix300" | "D"   (--recipe).  fix300 measured better:
                              #   paired 50-fold dR2 = -0.023 (V), -0.020 (T), -0.029 (V+T)
                              #   for D, i.e. early stopping HURTS -- 300 full-batch steps
                              #   still underfits, so there is nothing to stop early.
EPOCH_VAL_FRAC      = 0.20    # fraction of the 330 held out in each search split
EPOCH_SEARCH_SPLITS = 3       # independent search splits to average over

# Output tag, kept SEPARATE from mm.RESULTS_DATE so adopting recipe D does not overwrite
# the earlier fix300 GeoTIFFs -- both sets sit side by side in Data/inference_output/.
RUN_TAG = mm.RESULTS_DATE     # --tag  (default matches the study run tag)

# The three conditions to run:  (display name, saved-file tag, model kwargs)
ALL_CONDITIONS = [
    ("V",   "V",  dict(use_vis=True,  use_text=False, use_feat=False)),
    ("T",   "T",  dict(use_vis=False, use_text=True,  use_feat=False)),
    ("V+T", "VT", dict(use_vis=True,  use_text=True,  use_feat=False)),
]
# Narrowed by --conditions. Running "V" alone needs no VLM and no text embedder, so a
# real readability grid can be produced in minutes while the 27B captions run.
INFER_CONDITIONS = list(ALL_CONDITIONS)


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
    # Placement. On a multi-GPU node everything is resident and max_memory is not needed.
    # On one small card we pin VLM_GPU_GIB to the GPU and let the remainder compute on
    # the CPU -- same bf16 weights, no quantisation, just slower.
    load_kw = dict(torch_dtype="auto", device_map="auto", trust_remote_code=True)
    if VLM_ALLOW_CPU and DEVICE.type == "cuda":
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total_gib < 40:                       # single consumer card
            load_kw["max_memory"] = {0: f"{VLM_GPU_GIB:.1f}GiB", "cpu": "110GiB"}
            print(f"  [placement] {total_gib:.1f} GiB GPU -> pinning {VLM_GPU_GIB} GiB of "
                  f"weights on GPU, remainder computes on CPU (bf16, not quantised).")
    try:
        model = AutoModelForImageTextToText.from_pretrained(mm.QWEN_VL_ID, **load_kw).eval()
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            f"Could not load {mm.QWEN_VL_ID} as an image-text-to-text model: {e}\n"
            "Upgrade: pip install -U transformers accelerate qwen-vl-utils"
        ) from e

    # 'meta' and 'disk' are the placements that actually break generation: meta-device
    # layers hold no weights and emit garbage 1-word output (seen on HPC when the 27B
    # did not fit one 48 GB L40S). Those stay fatal. Plain 'cpu' placement COMPUTES
    # correctly -- it is merely slow -- and is how this model runs on a 16 GB card, so
    # it is permitted when VLM_ALLOW_CPU is set.
    placements = {str(v) for v in getattr(model, "hf_device_map", {}).values()}
    fatal = sorted(p for p in placements if p in ("disk", "meta"))
    if fatal:
        raise RuntimeError(
            f"{mm.QWEN_VL_ID} has layers on {fatal}, which hold no real weights and "
            "produce broken output. Free VRAM/RAM, lower VLM_GPU_GIB, or run on a node "
            "with more GPU memory (2x L40S = 96 GB fits the ~54 GB model resident).")
    if "cpu" in placements:
        n_cpu = sum(1 for v in model.hf_device_map.values() if str(v) == "cpu")
        n_tot = len(model.hf_device_map)
        if not VLM_ALLOW_CPU:
            raise RuntimeError(
                f"{mm.QWEN_VL_ID} was partially offloaded to CPU ({n_cpu}/{n_tot} modules) "
                "and VLM_ALLOW_CPU is False. Set it True to accept the slowdown, or use a "
                "bigger GPU.")
        print(f"  [placement] {n_cpu}/{n_tot} modules on CPU -- correct but slow. "
              f"Batching ({GEN_BATCH_SIZE}/pass) is what makes this tractable.", flush=True)

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

    def _clean(raw: str) -> str:
        """Identical post-processing to the unbatched path: drop any thinking trace and
        every special token, then strip."""
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[-1]
        out = re.sub(r"<\|.*?\|>", "", raw)
        return out.replace("<think>", "").replace("</think>", "").strip()

    def _prompt_for(idx: int):
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
            imgs, vids = process_vision_info(messages)
        else:
            imgs, vids = [image], None
        return text, imgs, vids

    def _generate(indices: list[int]) -> list[str]:
        """Caption a batch of windows in ONE forward pass per token. Left-padding keeps
        each sequence's last token at the end, so greedy decoding matches the unbatched
        result; the pad positions are masked out by attention_mask."""
        texts, all_imgs, all_vids = [], [], []
        for i in indices:
            t, imgs, vids = _prompt_for(i)
            texts.append(t)
            if imgs:
                all_imgs.extend(imgs)
            if vids:
                all_vids.extend(vids)
        prev_side = getattr(processor.tokenizer, "padding_side", None)
        processor.tokenizer.padding_side = "left"          # required for batched decode
        try:
            inputs = processor(text=texts, images=all_imgs or None,
                               videos=all_vids or None, padding=True,
                               return_tensors="pt").to(model.device)
        finally:
            if prev_side is not None:
                processor.tokenizer.padding_side = prev_side
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=mm.GEN_MAX_NEW_TOKENS,
                                       do_sample=False, use_cache=True)   # greedy
        n_in = inputs.input_ids.shape[1]                   # left-padded: uniform width
        trimmed = generated[:, n_in:]
        return [_clean(r) for r in processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)]

    def _generate_resilient(chunk: list[int]) -> list[str]:
        """Generate a batch, halving it on CUDA OOM rather than losing an overnight run.
        Peak memory scales with batch size (KV cache + linear-attention state + the
        vision tower's activations), so a smaller batch almost always gets through."""
        try:
            return _generate(chunk)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            half = max(1, len(chunk) // 2)
            print(f"  [OOM] batch of {len(chunk)} did not fit -- retrying in chunks of "
                  f"{half}. Lower --gen-batch (or VLM_GPU_GIB) for the rest of the run.",
                  flush=True)
            mm.free_gpu()
            out = []
            for i in range(0, len(chunk), half):
                out.extend(_generate_resilient(chunk[i:i + half]))
            return out

    t_start, done = time.time(), 0
    bs = max(1, GEN_BATCH_SIZE)
    for start in range(0, len(todo), bs):
        chunk = todo[start:start + bs]
        t0 = time.time()
        descs_out = _generate_resilient(chunk)
        dt = time.time() - t0

        for idx, desc in zip(chunk, descs_out):
            cache[str(idx)] = desc
            if len(desc.split()) < 20:      # expected ~160-220 words; short => broken
                print(f"  [WARN] window {idx} produced only {len(desc.split())} word(s): "
                      f"{desc!r} -- check placement (meta/disk) or truncation.", flush=True)
        _save()                             # save after EVERY batch, so a crash resumes
        done += len(chunk)

        words = [len(d.split()) for d in descs_out]
        avg = (time.time() - t_start) / done
        eta_h = avg * (len(todo) - done) / 3600.0
        print(f"  [{done}/{len(todo)}] windows {chunk[0]}-{chunk[-1]}  {dt:.1f}s "
              f"({dt/len(chunk):.1f}s/img)  words={min(words)}-{max(words)}  "
              f"avg={avg:.1f}s/img  ETA={eta_h:.1f} h", flush=True)

    print(f"[Stage B] Descriptions saved -> {cache_path.name}")
    del model, processor
    mm.free_gpu()

    if max_new is not None:                     # smoke: cache may be incomplete
        return [cache.get(str(i), "") for i in range(len(windows))]
    return [cache[str(i)] for i in range(len(windows))]


# -- Batching equivalence check -------------------------------------------------
def verify_batching(windows, n_rows, n_cols, n: int) -> None:
    """Caption the first n windows batched and unbatched and diff the text.

    Batched greedy decoding with left-padding SHOULD be identical to the unbatched
    decode the study used. This proves it on the actual model before a multi-hour run
    commits to the fast path. Writes to a throwaway cache so real caches are untouched.
    """
    import tempfile
    tmp = Path(tempfile.gettempdir())
    idx = list(range(n))
    print(f"\n[verify] Captioning windows {idx} unbatched (batch=1) ...")
    global GEN_BATCH_SIZE
    keep = GEN_BATCH_SIZE
    GEN_BATCH_SIZE = 1
    one = generate_descriptions(windows, n_rows, n_cols, tmp / "verify_b1.json", max_new=n)
    GEN_BATCH_SIZE = max(2, keep)
    print(f"\n[verify] Captioning the same windows batched (batch={GEN_BATCH_SIZE}) ...")
    many = generate_descriptions(windows, n_rows, n_cols, tmp / "verify_bN.json", max_new=n)
    GEN_BATCH_SIZE = keep

    print("\n" + "=" * 78)
    same = 0
    for i in idx:
        a, b = one[i], many[i]
        if a == b:
            same += 1
            print(f"  window {i}: IDENTICAL ({len(a.split())} words)")
        else:
            pre = len([1 for x, y in zip(a, b) if x == y])
            print(f"  window {i}: DIFFERS -- {len(a.split())} vs {len(b.split())} words, "
                  f"first divergence at char ~{pre}")
            print(f"    batch=1 : {a[:200]!r}")
            print(f"    batch=N : {b[:200]!r}")
    print(f"\n[verify] {same}/{n} identical.")
    print("  -> all identical: the batched fast path is safe for the full run."
          if same == n else
          "  -> DIVERGENCE: padding/masking is affecting the decode. Use --gen-batch 1 "
          "(slow but matches the study exactly), or investigate before trusting a long run.")
    print("=" * 78)


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
    emb_device = str(DEVICE) if EMBED_DEVICE == "auto" else EMBED_DEVICE
    print(f"[Stage C] embedder device: {emb_device}"
          + ("  (~16 GB bf16 does not fit a 16 GB card; same weights, full precision, "
             "just slower)" if emb_device == "cpu" else ""))
    model = SentenceTransformer(
        mm.QWEN_EMBED_ID,
        model_kwargs={"torch_dtype": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
        truncate_dim=mm.TEXT_DIM,
        device=emb_device,
    )
    emb = model.encode(descs, batch_size=mm.EMBED_BATCH_SIZE, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    embeddings = torch.from_numpy(emb.astype(np.float32))
    torch.save({"hash": fp, "emb": embeddings}, cache_path)
    print(f"[Stage C] Text embeddings saved -> {cache_path.name} {tuple(embeddings.shape)}")
    del model
    mm.free_gpu()
    return embeddings


# -- Model training (on the assessment-map patches) -----------------------------
def _search_best_epoch(vis, text, y, cfg, seed: int, split_seed: int) -> int:
    """Train on (1 - EPOCH_VAL_FRAC) of the patches, early-stopping on the rest, and
    return the epoch at which validation MSE was lowest. The model itself is thrown
    away -- only the epoch number is kept (see EPOCH_RECIPE)."""
    n = len(y)
    perm = np.random.RandomState(split_seed).permutation(n)
    n_va = max(1, int(round(EPOCH_VAL_FRAC * n)))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]

    mm.set_seeds(seed)
    model = mm.MultiModalModel(n_feat=4, **cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=mm.LR, weight_decay=mm.WEIGHT_DECAY)
    crit = torch.nn.MSELoss()
    xt, yt = dict(vis=vis[tr_idx], text=text[tr_idx]), y[tr_idx]
    xv, yv = dict(vis=vis[va_idx], text=text[va_idx]), y[va_idx]

    best, best_epoch, no_imp = float("inf"), 0, 0
    for epoch in range(1, mm.N_EPOCHS + 1):
        model.train(); opt.zero_grad(); crit(model(**xt), yt).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(**xv), yv).item()
        if v < best:
            best, best_epoch, no_imp = v, epoch, 0
        else:
            no_imp += 1
            if no_imp >= mm.PATIENCE:
                break
    del model
    return best_epoch


def train_models():
    """Train the V / T / V+T readability models on ALL available assessment-map
    patches -- the maps in Data/maps_to_be_assessed/, EXCLUDING the 16 maps in
    Data/maps_used_for_training/ -- then use them for window inference below.

    The number of training steps is chosen by EPOCH_RECIPE rather than hard-coded, so
    the deployed model matches the early-stopped estimator the study's CV measured.

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

    if EPOCH_RECIPE == "D":
        print(f"[train] Epoch recipe: D  ({EPOCH_SEARCH_SPLITS} x {EPOCH_VAL_FRAC:.0%} "
              f"search splits, patience {mm.PATIENCE}, max {mm.N_EPOCHS}), "
              f"then refit on all {n_keep}")
    else:
        print(f"[train] Epoch recipe: fix300  (flat {mm.N_EPOCHS} steps on all {n_keep})")

    crit = torch.nn.MSELoss()
    models, chosen_epochs = {}, {}
    for ci, (name, tag, cfg) in enumerate(INFER_CONDITIONS):
        if EPOCH_RECIPE == "D":
            found = [_search_best_epoch(vis_all, text_all, y_all, cfg,
                                        seed=ci, split_seed=7000 + 101 * ci + k)
                     for k in range(EPOCH_SEARCH_SPLITS)]
            n_epochs = max(1, int(round(float(np.mean(found)))))
            print(f"  {name:<4s} epoch search {found} -> E* = {n_epochs}")
        else:
            n_epochs = mm.N_EPOCHS

        mm.set_seeds(0)                              # refit on 100% of the patches
        model = mm.MultiModalModel(n_feat=n_feat, **cfg).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=mm.LR, weight_decay=mm.WEIGHT_DECAY)
        fwd = {k: v for k, v in (("vis", vis_all), ("text", text_all))
               if cfg["use_vis" if k == "vis" else "use_text"]}
        model.train()
        for _ in range(n_epochs):
            opt.zero_grad()
            crit(model(**fwd), y_all).backward()
            opt.step()
        models[name] = model.eval()
        chosen_epochs[name] = n_epochs
        print(f"  Trained {name} model on all {n_keep} patches for {n_epochs} steps.")
    print(f"\nTrained multimodal models: {list(models.keys())}  epochs={chosen_epochs}")
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
    dirs = map_dirs(stem)
    cmap = plt.get_cmap(CMAP)
    norm = mcolors.Normalize(vmin=SCORE_MIN, vmax=SCORE_MAX)

    used_h = n_rows * WINDOW_SIZE
    used_w = n_cols * WINDOW_SIZE
    bg     = rgb_chw.transpose(1, 2, 0)[:used_h, :used_w]

    for name, score_grid in score_maps.items():
        tag = f"{name.replace('+', '').lower()}_{RUN_TAG}"

        # Always write GTiff: the source may be a plain PNG, and the PNG driver refuses
        # Float32 ("Only eight bit (Byte) and sixteen bit (UInt16) bands supported").
        # Drop any source-driver-specific keys for the same reason.
        score_profile = {k: v for k, v in profile.items()
                         if k not in ("driver", "dtype", "count", "height", "width",
                                      "transform", "compress", "interleave", "tiled",
                                      "blockxsize", "blockysize", "photometric")}
        score_profile.update(
            driver="GTiff", dtype=rasterio.float32, count=1,
            height=n_rows, width=n_cols,
            transform=Affine(
                geo_transform.a * WINDOW_SIZE, geo_transform.b, geo_transform.c,
                geo_transform.d, geo_transform.e * WINDOW_SIZE, geo_transform.f,
            ),
        )
        tif_out = dirs["grids"] / f"{stem}_readability_{tag}.tif"
        with rasterio.open(tif_out, "w", **score_profile) as dst:
            dst.write(score_grid.astype(np.float32)[np.newaxis])
        georef = "georeferenced" if not geo_transform.is_identity else \
                 "PIXEL coordinates (source carries no geotransform)"
        print(f"  Score GeoTIFF  -> {tif_out.name}   [{georef}]")

        # Plain-array copies, so a non-georeferenced grid is not trapped in a GeoTIFF.
        npy_out = dirs["grids"] / f"{stem}_readability_{tag}.npy"
        csv_out = dirs["grids"] / f"{stem}_readability_{tag}.csv"
        np.save(npy_out, score_grid.astype(np.float32))
        np.savetxt(csv_out, score_grid, fmt="%.4f", delimiter=",")
        print(f"  Grid arrays    -> {npy_out.name}, {csv_out.name}")

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
        viz_out = dirs["overlays"] / f"{stem}_readability_{tag}_viz.tif"
        plt.savefig(viz_out, dpi=150, format="tiff", bbox_inches="tight")
        plt.close()
        print(f"  Visualisation  -> {viz_out.name}")


# ── Entry point ──────────────────────────────────────────────────────────────────
def main():
    global EPOCH_RECIPE, RUN_TAG, GEN_BATCH_SIZE, EMBED_DEVICE, INFER_CONDITIONS
    ap = argparse.ArgumentParser(description="Sliding-window multimodal (V/T/V+T) readability inference")
    ap.add_argument("--tif", type=Path, default=TIF_PATH, help="GeoTIFF to run on")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="Generate at most N window descriptions, then exit (quick GPU "
                         "test of the Qwen3.5 path before the full run).")
    ap.add_argument("--recipe", choices=("D", "fix300"), default=EPOCH_RECIPE,
                    help="How the final models' step count is chosen: 'D' averages the "
                         "early-stopping optimum over several held-out splits then refits "
                         "on all patches; 'fix300' is the old flat mm.N_EPOCHS.")
    ap.add_argument("--tag", default=RUN_TAG,
                    help="Suffix for the output GeoTIFFs, so runs do not overwrite each "
                         "other (default: %(default)s).")
    ap.add_argument("--gen-batch", type=int, default=GEN_BATCH_SIZE, metavar="N",
                    help="Windows per model.generate call (default: %(default)s). One "
                         "weight pass serves the whole batch, so this is the main speed "
                         "lever when the VLM is partly on the CPU.")
    ap.add_argument("--embed-device", default=EMBED_DEVICE, choices=("cpu", "cuda", "auto"),
                    help="Where Qwen3-Embedding-8B runs (default: %(default)s).")
    ap.add_argument("--conditions", default="V,T,V+T",
                    help="Comma-separated subset of V,T,V+T to run (default: %(default)s). "
                         "'V' alone skips the VLM and the text embedder entirely.")
    ap.add_argument("--verify-batching", type=int, default=0, metavar="N",
                    help="Caption the first N windows BOTH batched and unbatched, diff "
                         "the text, and exit. Run this once before a long batched run.")
    args = ap.parse_args()
    EPOCH_RECIPE, RUN_TAG = args.recipe, args.tag
    GEN_BATCH_SIZE, EMBED_DEVICE = args.gen_batch, args.embed_device

    want = [c.strip().upper().replace("VT", "V+T") if c.strip().upper() != "V" else "V"
            for c in args.conditions.split(",") if c.strip()]
    INFER_CONDITIONS = [c for c in ALL_CONDITIONS if c[0].upper() in
                        {w if w != "V+T" else "V+T" for w in want}]
    unknown = {w for w in want} - {c[0].upper() for c in ALL_CONDITIONS}
    if unknown:
        sys.exit(f"Unknown condition(s) {sorted(unknown)}; choose from "
                 f"{[c[0] for c in ALL_CONDITIONS]}")
    if not INFER_CONDITIONS:
        sys.exit("--conditions selected nothing to run.")
    needs_text = any(cfg["use_text"] for _, _, cfg in INFER_CONDITIONS)
    print(f"Conditions: {[c[0] for c in INFER_CONDITIONS]}"
          + ("" if needs_text else "   (no text modality -> VLM and embedder skipped)"))

    tif_path = args.tif
    stem     = tif_path.stem
    dirs     = map_dirs(stem)
    desc_cache = dirs["cache"] / f"{stem}_qwen_descriptions.json"
    text_cache = dirs["cache"] / f"{stem}_text_embeddings.pt"
    pe_cache   = dirs["cache"] / f"{stem}_pe_windows.pt"
    print(f"Outputs: {dirs['root']}")

    mm.set_seeds(42)
    print(f"Device : {DEVICE}")
    print(f"TIF    : {tif_path}")

    print("\nLoading TIF...")
    rgb_chw, profile, geo_transform = load_tif(tif_path)
    windows, n_rows, n_cols = extract_windows(rgb_chw)

    if args.verify_batching:
        verify_batching(windows, n_rows, n_cols, args.verify_batching)
        return

    if needs_text:
        # Stage B first (loads the 27B), so the smoke test exits before anything else.
        descs = generate_descriptions(windows, n_rows, n_cols, desc_cache,
                                      max_new=(args.smoke or None))
        if args.smoke:
            print(f"\n[smoke] Generated up to {args.smoke} window description(s); exiting "
                  f"before embedding/inference. Inspect {desc_cache.name}.")
            return
        text_all = embed_descriptions(descs, text_cache).to(DEVICE)      # Stage C
    else:
        print("\n[Stage B/C] Skipped -- no selected condition uses the text modality.")
        text_all = None

    pe_emb  = extract_pe_embeddings(windows, n_rows, n_cols, pe_cache)   # Stage A
    vis_all = pe_emb.to(DEVICE)

    print("\nTraining models on assessment-map patches...")
    models = train_models()

    print("\nRunning inference...")
    score_maps = run_inference(models, vis_all, text_all, n_rows, n_cols)  # Stage D

    print("\nSaving outputs...")
    save_outputs(score_maps, rgb_chw, profile, geo_transform, n_rows, n_cols, stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
