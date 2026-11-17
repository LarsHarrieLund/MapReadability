#!/bin/bash
#SBATCH --job-name=map_larger_area
#SBATCH --time=4:00:00           # Qwen3.5 captions every 256x256 window + V/T/V+T inference.
                                 #   Scales with the number of windows -> raise if the raster
                                 #   is large. Caches (Data/inference_output/<stem>_*) make a
                                 #   re-queued job resume, so a timeout is not fatal.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2             # 27B (~54GB bf16) does NOT fit on one 48GB L40S -> it gets
                                 #   offloaded to CPU (meta device), which is ~100x slower AND
                                 #   produces broken 1-word output. 2x L40S (96GB) fits fully.
#SBATCH --output=map_larger_area_log
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=W.Huang@leeds.ac.uk

# Load modules / activate conda environment
module load miniforge
conda activate fm_3.13      # Python 3.13: fla's Triton kernels compile -> fast path (no 3.14 ast.Num bug)

# --- Redirect Hugging Face cache to scratch (27B + 8B weights are large) ---
export HF_HOME="/mnt/scratch/mcjj164/cache/huggingface"
mkdir -p "$HF_HOME"

# Let device_map="auto" see both GPUs so the 27B shards across them (no CPU offload)
export CUDA_VISIBLE_DEVICES=0,1
# Avoid CPU oversubscription from tokenizers/threads
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Adjust this to where you place the project on scratch ---
PROJECT_DIR="/mnt/scratch/mcjj164/Projects/MapReadability"
cd "$PROJECT_DIR"

# Paths inside the script are resolved relative to the .py file, so cwd is safe.
# Descriptions + PE + text embeddings are cached under Data/inference_output/, so a
# re-run after this finishes (or after a timeout) only does the unfinished windows.
#
# TIP: before the full run, validate the Qwen path cheaply with:
#   python "$PROJECT_DIR/larger_area_infer.py" --smoke 3
# Point at a different raster with:  --tif "Data/Larger maps/.../<file>.tif"
python "$PROJECT_DIR/larger_area_infer.py"
