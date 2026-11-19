<br>
<br>
<br>

# Multimodal feature fusion parts 1-2 - Lund
This folder contains the Python scripts and bash-scripts used to perform multimodal feature fusion for the study area of Lund.

<br>

- larger_area_infer.py: Applies the vision, text and fusion (V+T) models, trained on the 330 map samples, to every 256 x 256 pixel cell of a large map and writes the readability values as raster grids (used for the Lund map).
- map_readability_multimodal.py: Generates the MLLM descriptions and the vision and text embeddings for the map samples, and trains and evaluates the fusion models for all combinations of V, T and M with 20 x 5 cross-validation.
- run_larger_area.sh:  Job script for running larger_area_infer.py on an HPC cluster (Slurm) with two GPUs.
- run_multimodal.sh: Job script for running map_readability_multimodal.py on an HPC cluster (Slurm) with two GPUs.
