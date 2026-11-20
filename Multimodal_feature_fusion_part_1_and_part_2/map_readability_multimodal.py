#!/usr/bin/env python3
"""
Multimodal map readability prediction.

Three modalities, fused, evaluated with 20x5 repeated k-fold CV:
  V  — visual embeddings from PE-Core-bigG-14-448 (frozen, open_clip)
  F  — 4 hand-crafted features (Harrie et al. 2015)
  T  — text embeddings of a Qwen3.5-27B readability description (Qwen3-Embedding-8B)

The pipeline runs in cache-driven stages so an HPC job can be restarted without
recomputing finished work:

  Stage 1  Qwen3.5-27B  : map image -> readability description   -> Data/qwen_descriptions.json
  Stage 2  Qwen3-Embed  : description -> text embedding (MRL 1024) -> Data/text_embeddings.pt
  Stage 3  PE encoder   : map image -> visual embedding (1280)     -> Data/pe_embeddings.pt
  Stage 4  Fusion CV    : ablation over V / F / T and all fusions  -> Data/per_map_predictions_mm.csv
  Stage 5  Final models : train on all data, save for inference    -> Data/models_mm/

Each heavy model is loaded, used, then freed before the next stage, so the
27B VLM, the 8B embedder, and PE never sit in memory simultaneously.

Dependencies:
  torch>=2.1  transformers>=4.49  accelerate  qwen-vl-utils  sentence-transformers
  open_clip_torch  scikit-learn  pandas  Pillow  openpyxl/xlrd
"""

import os
import re
import sys
import json
import random
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Python 3.14 compatibility shim ──────────────────────────────────────────────
# Qwen3.5's modeling code uses `fla` (flash-linear-attention) for its Gated-DeltaNet
# kernels when fla is importable. On Python 3.14 those kernels CANNOT run: fla applies
# @torch.compile at import (unsupported on 3.14), and even with that neutralised,
# Triton's kernel compiler crashes (it references the removed `ast.Num`). The model
# does, however, have a correct pure-PyTorch fallback that runs when fla is absent.
#
# So on Python 3.14 we (a) neutralise torch.compile and (b) make `import fla` fail,
# which forces Qwen3.5 onto the working torch path. Override with MAPREAD_DISABLE_FLA
# = "0" (force-enable) or "1" (force-disable); default = auto on Py>=3.14.
_disable_fla = os.environ.get("MAPREAD_DISABLE_FLA")
_disable_fla = (sys.version_info >= (3, 14)) if _disable_fla is None else (_disable_fla == "1")

if sys.version_info >= (3, 14):
    def _noop_compile(model=None, *args, **kwargs):
        if callable(model):
            return model                  # used as @torch.compile
        return lambda fn: fn              # used as @torch.compile(...)
    torch.compile = _noop_compile

# Only worth blocking if fla is actually installed. When it is absent, transformers'
# own is_flash_linear_attention_available() already returns False and the pure-torch
# path is selected -- and installing the blocker would then be actively harmful, because
# raising from a MetaPathFinder.find_spec makes that availability PROBE raise instead of
# returning False, which breaks importing the qwen3_5 model class altogether
# (transformers 5.x: "Could not import module 'Qwen3_5ForConditionalGeneration'").
if _disable_fla:
    import importlib.util
    try:
        if importlib.util.find_spec("fla") is None:
            _disable_fla = False
    except (ImportError, ValueError):
        _disable_fla = False

if _disable_fla:
    import importlib.abc

    class _BlockFla(importlib.abc.MetaPathFinder):
        """Make `import fla` (and submodules) raise, so Qwen3.5 uses its torch path."""
        def find_spec(self, name, path=None, target=None):
            if name == "fla" or name.startswith("fla."):
                raise ModuleNotFoundError(
                    f"No module named '{name}' (disabled to force the torch fallback "
                    "on Python 3.14; set MAPREAD_DISABLE_FLA=0 to allow fla)")
            return None                   # defer to the normal finders for everything else

    sys.meta_path.insert(0, _BlockFla())

from PIL import Image
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths (resolved relative to this file, so cwd on HPC does not matter) ───────
ROOT        = Path(__file__).resolve().parent
DATA_DIR    = ROOT / "Data"
XLS_PATH    = DATA_DIR / "Indata.xls"
TRAIN_DIR   = DATA_DIR / "maps_used_for_training"
ASSESS_DIR  = DATA_DIR / "maps_to_be_assessed"

PE_CACHE    = DATA_DIR / "pe_embeddings.pt"        # reused if present (346 x 1280)
DESC_CACHE  = DATA_DIR / "qwen_descriptions.json"  # {region: description}
TEXT_CACHE  = DATA_DIR / "text_embeddings.pt"      # 346 x TEXT_DIM

# ── This run's outputs ───────────────────────────────────────────────────────────
# Modelling is now restricted to the maps in `maps_to_be_assessed/` (the 16 maps in
# `maps_used_for_training/` are excluded). To keep the existing 346-patch caches valid
# (so NO foundation model is reloaded), embeddings/descriptions are still loaded over
# the FULL matched set and then sliced down to the assessment maps in main().
# Outputs carry a date tag so the earlier full-dataset results are preserved untouched.
RESULTS_DATE = "20260715"
MODEL_DIR   = DATA_DIR / f"models_mm_{RESULTS_DATE}"   # assess-only final models
PRED_CSV    = DATA_DIR / f"per_map_predictions_mm_{RESULTS_DATE}.csv"

# ── Model IDs ──────────────────────────────────────────────────────────────────
PE_MODEL_ID     = "hf-hub:timm/PE-Core-bigG-14-448"
QWEN_VL_ID      = "Qwen/Qwen3.5-27B"
QWEN_EMBED_ID   = "Qwen/Qwen3-Embedding-8B"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dimensions ──────────────────────────────────────────────────────────────────
VISUAL_DIM = 1280
TEXT_DIM   = 4096          # Full native Qwen3-Embedding dimension (no MRL truncation).
                           # Was 1024 (MRL-truncated) as a regularised choice for 346
                           # samples; 4096 tests whether truncation lost any text signal.
PROJ_DIM   = 128           # each modality is linearly projected to this before concat
HIDDEN_DIM = 256           # width of the single post-fusion hidden layer
DROPOUT    = 0.3           # regularisation for the small (346-sample) dataset

# ── Generation / encoding knobs ─────────────────────────────────────────────────
GEN_MAX_NEW_TOKENS = 512   # enough for a full description; thinking is disabled below
EMBED_BATCH_SIZE   = 8
PE_BATCH_SIZE      = 4

# ── CV / training ────────────────────────────────────────────────────────────────
N_FOLDS, N_REPEATS = 5, 20
N_EPOCHS, LR, WEIGHT_DECAY, PATIENCE = 300, 1e-3, 1e-4, 40
VAL_FRAC = 0.15            # inner split of each train fold used for early stopping
                          # (the outer test fold is NEVER seen during training)

# ── Spreadsheet column lookup ────────────────────────────────────────────────────
FEATURE_KEY_SUBSTRINGS = [
    "total length of line",
    "number of points",
    "number of object types",
    "less than 0.2 mm",
]
TARGET_SUBSTRING = "Alla - medel - I"
REGION_COL       = "Region"

# ── Qwen readability prompt (condensed from the cartographer's domain brief) ────
SYSTEM_PROMPT = (
    "You are a cartographic expert assessing the READABILITY of a small "
    "topographic map patch produced by the Swedish mapping authority "
    "(Lantmateriet) at scale 1:10,000 or 1:50,000. Readability is how easily a "
    "human can comprehend the map's content. Judge it along four established "
    "aspects:\n"
    "1. Amount of information / clutter (the dominant factor): density of point "
    "symbols, number and total length of lines, number of distinct object types; "
    "too many objects makes them hard to distinguish.\n"
    "2. Spatial distribution: whether objects are spread evenly or concentrated; "
    "dense clusters are hard to read even when the overall count is low.\n"
    "3. Object complexity: how intricate features are (sinuous or angular lines, "
    "fragmented polygons, high line connectivity).\n"
    "4. Graphical resolution: whether objects are distinguishable given colour "
    "contrast and symbol separation; low contrast lowers readability even with "
    "few objects.\n"
    "Describe only what you can actually see. Be concise and concrete. "
    "Output the description text ONLY — do not restate the task, do not narrate "
    "your reasoning, and do not use phrases like 'The user wants' or 'Drafting "
    "the response'."
)

USER_PROMPT = (
    "Describe this map patch in about 160-220 words.\n"
    "Start with 2-3 sentences of general description: the setting and land-use "
    "type (e.g. dense urban, residential blocks, rural, forest, water), the "
    "dominant features, and the overall visual character.\n"
    "Then assess its readability, commenting on each of the four aspects (amount "
    "of information, spatial distribution, object complexity, graphical "
    "resolution) with concrete visual evidence.\n"
    "End with a one-sentence overall legibility verdict (easy / moderate / hard "
    "to read). Do not invent a numeric score."
)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def free_gpu() -> None:
    """Release cached GPU memory. Call AFTER `del`-ing the model in the caller,
    so the weights are actually freed before the allocator returns blocks."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Column lookup ────────────────────────────────────────────────────────────
def find_col(df: pd.DataFrame, substring: str) -> str:
    substring = substring.lower()
    matches = [c for c in df.columns if substring in str(c).lower()]
    if not matches:
        raise KeyError(f"No column containing '{substring}'. Columns: {list(df.columns)}")
    if len(matches) > 1:
        print(f"  [warn] Multiple columns match '{substring}': {matches} - using first")
    return matches[0]


# ── Data loading (regions, targets, features, image paths) ──────────────────────
def load_data():
    df = pd.read_excel(XLS_PATH, sheet_name="AllData_transpose")

    target_col   = find_col(df, TARGET_SUBSTRING)
    feature_cols = [find_col(df, s) for s in FEATURE_KEY_SUBSTRINGS]
    print(f"\nTarget column  : {target_col}")
    print(f"Feature columns: {feature_cols}")

    img_map: dict[str, Path] = {}
    for folder in (TRAIN_DIR, ASSESS_DIR):
        for p in folder.iterdir():
            if p.suffix.lower() == ".png":
                img_map[p.stem.lower()] = p

    regions = df[REGION_COL].astype(str).str.strip()
    matched = regions.str.lower().isin(img_map)
    if (~matched).any():
        print(f"\n[warn] {int((~matched).sum())} rows have no image and are skipped: "
              f"{regions[~matched].tolist()}")

    df        = df[matched].reset_index(drop=True)
    region_id = [str(r).strip() for r in df[REGION_COL]]
    img_paths = [img_map[r.lower()] for r in region_id]
    targets   = np.asarray(df[target_col].values,   dtype=np.float32)
    features  = np.asarray(df[feature_cols].values, dtype=np.float32)

    print(f"\nDataset: {len(df)} samples")
    print(f"Target  - min={targets.min():.3f} max={targets.max():.3f} "
          f"mean={targets.mean():.3f} std={targets.std():.3f}")
    return region_id, img_paths, targets, features


# ── Cache fingerprints (so caches self-invalidate when inputs change) ───────────
def _desc_fingerprint() -> str:
    """Identifies the description-generation config. If the prompt, model, or token
    budget changes, the hash changes and ALL descriptions are regenerated."""
    h = hashlib.sha256()
    for part in (QWEN_VL_ID, str(GEN_MAX_NEW_TOKENS), SYSTEM_PROMPT, USER_PROMPT):
        h.update(part.encode("utf-8"))
    return h.hexdigest()[:16]


def _text_fingerprint(region_id: list[str], descriptions: dict[str, str]) -> str:
    """Identifies the text embeddings: depends on the embedder, the MRL dim, and the
    exact description text of every map (in order). Any change forces re-embedding."""
    h = hashlib.sha256()
    h.update(QWEN_EMBED_ID.encode("utf-8"))
    h.update(str(TEXT_DIM).encode("utf-8"))
    for r in region_id:
        h.update(descriptions[r].encode("utf-8"))
    return h.hexdigest()[:16]


# ── Stage 1: Qwen3.5-27B readability descriptions ───────────────────────────────
def generate_descriptions(region_id: list[str], img_paths: list[Path],
                          max_new: int | None = None) -> dict[str, str]:
    fp = _desc_fingerprint()
    cache: dict[str, str] = {}
    if DESC_CACHE.exists():
        blob = json.loads(DESC_CACHE.read_text(encoding="utf-8"))
        if isinstance(blob, dict) and blob.get("prompt_hash") == fp:
            cache = blob.get("descriptions", {})
        else:
            print("\n[Stage 1] Prompt/model/token-budget changed (or old-format cache) "
                  "-> ALL descriptions will be regenerated.")
    todo = [(r, p) for r, p in zip(region_id, img_paths) if r not in cache]
    if not todo:
        print(f"\n[Stage 1] All {len(region_id)} descriptions cached -> {DESC_CACHE}")
        return cache
    if max_new is not None:               # smoke test: only generate the first few
        todo = todo[:max_new]
        print(f"\n[Stage 1] SMOKE MODE: generating only {len(todo)} description(s).")

    print(f"\n[Stage 1] Generating {len(todo)} descriptions with {QWEN_VL_ID} "
          f"({len(cache)} cached) ...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None  # fall back to passing PIL images directly

    processor = AutoProcessor.from_pretrained(QWEN_VL_ID, trust_remote_code=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            QWEN_VL_ID,
            torch_dtype="auto",
            device_map="auto",            # shards across the 3 L40S
            trust_remote_code=True,
        ).eval()
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            f"Could not load {QWEN_VL_ID} as an image-text-to-text model: {e}\n"
            "This usually means `transformers` is too old to know this VLM "
            "architecture. Upgrade: pip install -U transformers accelerate qwen-vl-utils"
        ) from e

    # Report the generation path. On Python 3.14 we deliberately disable fla (see top
    # of file), so the model runs the pure-torch Gated-DeltaNet path: correct, but
    # slower. find_spec() raises for a blocked module, so guard it.
    import importlib.util

    def _spec_missing(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is None
        except ModuleNotFoundError:
            return True       # blocked (e.g. fla on Py3.14) counts as unavailable

    if _disable_fla:
        print("=" * 78)
        print("  [INFO] fla disabled (Python 3.14) -> using the pure-torch fallback.")
        print("  Correct but slower; this is expected on this environment.")
        print("=" * 78, flush=True)
    else:
        _missing = [m for m in ("fla", "causal_conv1d") if _spec_missing(m)]
        if _missing:
            print("=" * 78)
            print(f"  [PERF WARNING] Missing fast kernels: {_missing}")
            print("  Generation will use the SLOW torch fallback (can be 10-100x slower).")
            print("  Install:  pip install flash-linear-attention causal-conv1d")
            print("=" * 78, flush=True)

    import time
    t_start = time.time()
    for i, (region, path) in enumerate(todo, 1):
        image = Image.open(path).convert("RGB")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": USER_PROMPT},
            ]},
        ]
        # Qwen3.5 is a reasoning model; disable the think trace so the output is
        # the description itself, not the chain-of-thought (falls back gracefully
        # if this build's template doesn't accept the kwarg).
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
            generated = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS,
                                        do_sample=False, use_cache=True)   # greedy = reproducible
        dt = time.time() - t0

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        # Decode WITHOUT dropping special tokens first, so a </think> delimiter
        # survives long enough to split on; then clean the residual markup.
        raw = processor.batch_decode(trimmed, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)[0]
        if "</think>" in raw:                       # keep only the post-think answer
            raw = raw.rsplit("</think>", 1)[-1]
        desc = re.sub(r"<\|.*?\|>", "", raw)        # remove <|im_end|> etc.
        desc = desc.replace("<think>", "").replace("</think>", "").strip()
        cache[region] = desc

        # Save after EVERY image (small JSON) so a killed job loses nothing.
        # Schema carries the prompt hash so a later prompt change invalidates it.
        DESC_CACHE.write_text(
            json.dumps({"prompt_hash": fp, "descriptions": cache},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        avg = (time.time() - t_start) / i
        eta_min = avg * (len(todo) - i) / 60.0
        print(f"  [{i}/{len(todo)}] {region}  {dt:.1f}s  "
              f"({len(desc.split())} words)  avg={avg:.1f}s/img  ETA={eta_min:.0f} min",
              flush=True)

    print(f"[Stage 1] Descriptions saved -> {DESC_CACHE}")
    del model, processor                 # release the 27B before the next stage
    free_gpu()
    return cache


# ── Stage 2: Qwen3-Embedding-8B text embeddings ─────────────────────────────────
def embed_descriptions(region_id: list[str], descriptions: dict[str, str]) -> torch.Tensor:
    fp = _text_fingerprint(region_id, descriptions)
    if TEXT_CACHE.exists():
        blob = torch.load(TEXT_CACHE, map_location="cpu")
        if (isinstance(blob, dict) and blob.get("hash") == fp
                and blob["emb"].shape == (len(region_id), TEXT_DIM)):
            print(f"\n[Stage 2] Loaded cached text embeddings -> {TEXT_CACHE} "
                  f"{tuple(blob['emb'].shape)}")
            return blob["emb"]
        print("\n[Stage 2] Descriptions/embedder/dim changed (or old-format cache) "
              "-> re-embedding.")

    print(f"\n[Stage 2] Embedding descriptions with {QWEN_EMBED_ID} "
          f"(MRL dim={TEXT_DIM}) ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        QWEN_EMBED_ID,
        model_kwargs={"torch_dtype": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
        truncate_dim=TEXT_DIM,            # Matryoshka truncation
        device=str(DEVICE),
    )
    # Documents are encoded WITHOUT an instruction prompt (per model card).
    texts = [descriptions[r] for r in region_id]
    emb = model.encode(texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    embeddings = torch.from_numpy(emb.astype(np.float32))
    torch.save({"hash": fp, "emb": embeddings}, TEXT_CACHE)
    print(f"[Stage 2] Text embeddings saved -> {TEXT_CACHE} {tuple(embeddings.shape)}")
    del model
    free_gpu()
    return embeddings


# ── Stage 3: PE-Core-bigG-14-448 visual embeddings ──────────────────────────────
def extract_pe_embeddings(img_paths: list[Path]) -> torch.Tensor:
    if PE_CACHE.exists():
        cached = torch.load(PE_CACHE, map_location="cpu")
        if cached.shape[0] == len(img_paths):
            print(f"\n[Stage 3] Loaded cached PE embeddings -> {PE_CACHE} "
                  f"{tuple(cached.shape)}")
            return cached
        print(f"\n[Stage 3] Cache size {cached.shape[0]} != {len(img_paths)} - re-extracting")

    import open_clip
    print(f"\n[Stage 3] Extracting PE embeddings with {PE_MODEL_ID} ...")
    pe_model, _, preprocess = open_clip.create_model_and_transforms(PE_MODEL_ID)
    pe_model = pe_model.to(DEVICE).eval()

    autocast_ctx = (torch.amp.autocast("cuda") if DEVICE.type == "cuda"
                    else torch.amp.autocast("cpu", enabled=False))
    out = []
    for start in range(0, len(img_paths), PE_BATCH_SIZE):
        batch = img_paths[start:start + PE_BATCH_SIZE]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(DEVICE)
        with torch.no_grad(), autocast_ctx:
            emb = pe_model.encode_image(imgs, normalize=True)
        out.append(emb.cpu().float())
        print(f"  {min(start + PE_BATCH_SIZE, len(img_paths))}/{len(img_paths)}", end="\r")
    print()
    embeddings = torch.cat(out, 0)
    torch.save(embeddings, PE_CACHE)
    print(f"[Stage 3] PE embeddings saved -> {PE_CACHE} {tuple(embeddings.shape)}")
    del pe_model
    free_gpu()
    return embeddings


# ── Multimodal model ─────────────────────────────────────────────────────────
# Each active modality is LINEARLY projected to PROJ_DIM (a low-rank bottleneck that
# both balances the very different modality widths — 4096 text vs 4 features — and
# regularises the high-dim embeddings). The projections are concatenated and passed
# through a SINGLE post-fusion hidden layer with dropout. One hidden non-linearity
# after fusion (instead of the previous two ReLU stack) — a better fit for 346 samples.
class MultiModalModel(nn.Module):
    def __init__(self, use_vis: bool, use_text: bool, use_feat: bool, n_feat: int):
        super().__init__()
        self.use_vis, self.use_text, self.use_feat = use_vis, use_text, use_feat
        n_active = 0
        if use_vis:
            self.vis_proj  = nn.Linear(VISUAL_DIM, PROJ_DIM); n_active += 1
        if use_text:
            self.text_proj = nn.Linear(TEXT_DIM, PROJ_DIM);   n_active += 1
        if use_feat:
            self.feat_proj = nn.Linear(n_feat, PROJ_DIM);     n_active += 1
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(PROJ_DIM * n_active, HIDDEN_DIM), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, vis=None, text=None, feat=None):
        parts = []
        if self.use_vis:  parts.append(self.vis_proj(vis))
        if self.use_text: parts.append(self.text_proj(text))
        if self.use_feat: parts.append(self.feat_proj(feat))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


CONDITIONS = [
    ("V",     dict(use_vis=True,  use_text=False, use_feat=False)),
    ("F",     dict(use_vis=False, use_text=False, use_feat=True)),
    ("T",     dict(use_vis=False, use_text=True,  use_feat=False)),
    ("V+F",   dict(use_vis=True,  use_text=False, use_feat=True)),
    ("V+T",   dict(use_vis=True,  use_text=True,  use_feat=False)),
    ("F+T",   dict(use_vis=False, use_text=True,  use_feat=True)),
    ("V+F+T", dict(use_vis=True,  use_text=True,  use_feat=True)),
]


# ── Training / evaluation ────────────────────────────────────────────────────
def train_fold(model, tr, va):
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()
    best, best_state, no_imp = float("inf"), None, 0
    for _ in range(N_EPOCHS):
        model.train(); opt.zero_grad()
        crit(model(**tr["x"]), tr["y"]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(**va["x"]), va["y"]).item()
        if v < best:
            best, best_state, no_imp = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model, x, y_np):
    model.eval()
    with torch.no_grad():
        preds = model(**x).cpu().numpy()
    return (float(mean_absolute_error(y_np, preds)),
            float(np.sqrt(mean_squared_error(y_np, preds))),
            float(r2_score(y_np, preds)),
            preds)


def slice_inputs(vis, text, feat, idx):
    return {"vis": vis[idx], "text": text[idx], "feat": feat[idx]}


# ── Stage 4: repeated k-fold CV across all ablation conditions ──────────────────
def run_cv(vis_all, text_all, feat_np, targets):
    n, n_feat = len(targets), feat_np.shape[1]
    results  = {name: [] for name, _ in CONDITIONS}
    pred_sum = {name: np.zeros(n, dtype=np.float64) for name, _ in CONDITIONS}
    pred_cnt = np.zeros(n, dtype=np.int32)
    total    = N_REPEATS * N_FOLDS
    print(f"\n[Stage 4] Repeated {N_REPEATS}x{N_FOLDS}-fold CV  ({total} folds)\n")

    fold_global = 0
    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n))):
            fold_global += 1
            print(f"-- Rep {repeat+1}/{N_REPEATS} Fold {fold+1}/{N_FOLDS} "
                  f"[{fold_global}/{total}] --")

            # Scaler is fit on the outer-train fold only (never sees the test fold).
            scaler = StandardScaler().fit(feat_np[tr_idx])
            feat_t = torch.from_numpy(scaler.transform(feat_np).astype(np.float32)).to(DEVICE)

            # Inner split of the train fold for early stopping. The outer TEST fold
            # (va_idx) is never used for model selection — only for the final metric.
            rng     = np.random.RandomState(1000 + fold_global)
            perm    = rng.permutation(tr_idx)
            n_in_va = max(1, int(round(VAL_FRAC * len(perm))))
            in_va_idx, in_tr_idx = perm[:n_in_va], perm[n_in_va:]

            y_in_tr = torch.from_numpy(targets[in_tr_idx]).to(DEVICE)
            y_in_va = torch.from_numpy(targets[in_va_idx]).to(DEVICE)

            for ci, (name, cfg) in enumerate(CONDITIONS):
                set_seeds(fold_global * len(CONDITIONS) + ci)
                model = MultiModalModel(n_feat=n_feat, **cfg).to(DEVICE)
                tr = {"x": slice_inputs(vis_all, text_all, feat_t, in_tr_idx), "y": y_in_tr}
                va = {"x": slice_inputs(vis_all, text_all, feat_t, in_va_idx), "y": y_in_va}
                model = train_fold(model, tr, va)                       # early-stop on inner val
                test_x = slice_inputs(vis_all, text_all, feat_t, va_idx)
                mae, rmse, r2, preds = evaluate(model, test_x, targets[va_idx])
                results[name].append((mae, rmse, r2))
                pred_sum[name][va_idx] += preds
                print(f"  {name:<6s} MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
            pred_cnt[va_idx] += 1

    mean_preds = {k: pred_sum[k] / pred_cnt for k in pred_sum}
    return results, mean_preds


def report(results):
    print("\n" + "=" * 70)
    print(f"{'Condition':<8} {'MAE (mean±std)':>18} {'RMSE (mean±std)':>18} {'R2 (mean±std)':>16}")
    print("-" * 70)
    for name, folds in results.items():
        arr = np.asarray(folds); m, s = arr.mean(0), arr.std(0)
        print(f"{name:<8} {m[0]:.4f} ± {s[0]:.4f}   {m[1]:.4f} ± {s[1]:.4f}   {m[2]:.4f} ± {s[2]:.4f}")
    print("=" * 70)


def save_predictions_csv(region_id, targets, mean_preds):
    df = pd.DataFrame({"Region": region_id, "True_Score": targets.round(4)})
    for name, preds in mean_preds.items():
        col = name.replace("+", "")
        df[f"Pred_{col}"]  = preds.round(4)
        df[f"Error_{col}"] = (targets - preds).round(4)
    df.to_csv(PRED_CSV, index=False)
    print(f"\nPer-map predictions saved -> {PRED_CSV}")


# ── Stage 5: final models trained on all data (for inference) ────────────────────
def train_final_models(vis_all, text_all, feat_np, targets):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    n_feat = feat_np.shape[1]
    set_seeds(0)
    scaler = StandardScaler().fit(feat_np)
    feat_t = torch.from_numpy(scaler.transform(feat_np).astype(np.float32)).to(DEVICE)
    y_all  = torch.from_numpy(targets).to(DEVICE)
    crit   = nn.MSELoss()

    print("\n[Stage 5] Training final models on all data ...")
    for name, cfg in CONDITIONS:
        set_seeds(0)
        model = MultiModalModel(n_feat=n_feat, **cfg).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        x = slice_inputs(vis_all, text_all, feat_t, slice(None))
        model.train()
        for _ in range(N_EPOCHS):
            opt.zero_grad(); crit(model(**x), y_all).backward(); opt.step()
        model.eval()
        tag = name.replace("+", "")
        torch.save(model.state_dict(), MODEL_DIR / f"model_{tag}.pt")
        print(f"  Saved -> model_{tag}.pt")

    with open(MODEL_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    torch.save({"n_feat": n_feat, "visual_dim": VISUAL_DIM, "text_dim": TEXT_DIM,
                "proj_dim": PROJ_DIM, "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
                "conditions": [c for c, _ in CONDITIONS]},
               MODEL_DIR / "metadata.pt")
    print(f"  Saved scaler + metadata -> {MODEL_DIR}")


# ── Entry point ──────────────────────────────────────────────────────────────
def report_cache_status() -> None:
    """Show which artifacts are cached. If all three are present, the run needs
    no GPU and no foundation-model libraries — only the small MLP CV (CPU is fine)."""
    items = [("Qwen descriptions", DESC_CACHE), ("Text embeddings", TEXT_CACHE),
             ("PE embeddings", PE_CACHE)]
    print("\nCache status:")
    all_present = True
    for label, path in items:
        present = path.exists()
        all_present &= present
        print(f"  [{'x' if present else ' '}] {label:<18} {path.name}")
    if all_present:
        print("  -> all cached: GPU and the transformers/open_clip/sentence-transformers "
              "libraries are NOT needed for this run.")
    else:
        print("  -> some caches missing: the corresponding stage will load its "
              "foundation model (GPU strongly recommended).")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Multimodal map readability pipeline")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="Generate at most N new descriptions, then exit (quick GPU test "
                         "of the Qwen3.5 path before the full run). The N descriptions are "
                         "saved to the normal cache, so they count toward the real run.")
    args = ap.parse_args()

    set_seeds(42)
    print(f"Device: {DEVICE}")
    report_cache_status()
    region_id, img_paths, targets, features = load_data()

    descriptions = generate_descriptions(
        region_id, img_paths, max_new=(args.smoke or None))      # Stage 1
    if args.smoke:
        print(f"\n[smoke] Generated up to {args.smoke} description(s); "
              "exiting before embedding/CV. Inspect Data/qwen_descriptions.json.")
        return

    text_emb     = embed_descriptions(region_id, descriptions)   # Stage 2
    pe_emb       = extract_pe_embeddings(img_paths)              # Stage 3

    # ── Restrict modelling to maps_to_be_assessed/ (exclude maps_used_for_training/) ──
    # Descriptions + embeddings above were loaded over the FULL matched set so the
    # existing 346-patch caches stay valid and NO foundation model was reloaded. Slice
    # them down to the assessment maps here for CV + final training.
    assess_mask = np.array([p.parent.name == ASSESS_DIR.name for p in img_paths], dtype=bool)
    n_keep, n_excl = int(assess_mask.sum()), int((~assess_mask).sum())
    print(f"\n[subset] Modelling on {n_keep} maps from '{ASSESS_DIR.name}' "
          f"(excluded {n_excl} from '{TRAIN_DIR.name}').")
    sel        = torch.from_numpy(assess_mask)
    region_id  = [r for r, m in zip(region_id, assess_mask) if m]
    targets    = targets[assess_mask]
    features   = features[assess_mask]
    pe_emb     = pe_emb[sel]
    text_emb   = text_emb[sel]
    print(f"[subset] Target - min={targets.min():.3f} max={targets.max():.3f} "
          f"mean={targets.mean():.3f} std={targets.std():.3f}")

    vis_all  = pe_emb.to(DEVICE)
    text_all = text_emb.to(DEVICE)

    results, mean_preds = run_cv(vis_all, text_all, features, targets)   # Stage 4
    report(results)
    save_predictions_csv(region_id, targets, mean_preds)
    train_final_models(vis_all, text_all, features, targets)            # Stage 5


if __name__ == "__main__":
    main()
