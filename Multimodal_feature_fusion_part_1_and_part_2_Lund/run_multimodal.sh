#!/bin/bash
#SBATCH --job-name=map_readability_mm
#SBATCH --time=1:00:00          # VLM captioning + 20x5 CV across 7 conditions
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             # 2x L40S is enough: 27B (~54GB bf16) shards to ~27GB/card
#SBATCH --output=map_readability_mm_log
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

# Let device_map="auto" see both GPUs
export CUDA_VISIBLE_DEVICES=0,1
# Avoid CPU oversubscription from tokenizers/threads
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Adjust this to where you place the project on scratch ---
PROJECT_DIR="/mnt/scratch/mcjj164/Projects/MapReadability"
cd "$PROJECT_DIR"

# Paths inside the script are resolved relative to the .py file, so cwd is safe.
python "$PROJECT_DIR/map_readability_multimodal.py"
