This folder contains the Python-scripts and BASH-scripts used to execute multimodal feature fusion map readability for the case study of Los Angeles.

A short description of every file is presented below:

larger_area_infer.py: Applies the vision, text and fusion (V+T) models, trained on the 330 map samples, to every 256 x 256 pixel cell of a large map and writes the readability values as raster grids.
map_readability_multimodal.py: Generates the MLLM descriptions and the vision and text embeddings for the map samples, and trains and evaluates the fusion models for all combinations of V, T and M with 20 x 5 cross-validation.
run_larger_area.sh: Job script for running larger_area_infer.py on an HPC cluster (Slurm) with two GPUs.
run_multimodal.sh: Job script for running map_readability_multimodal.py on an HPC cluster (Slurm) with two GPUs.

Instructions: Run map_readability_multimodal.py first (part 1). Then run larger_area_infer.py once per larger map. The Lund map is the default; for Los Angeles, use --tif “Data/Larger maps/generated_map_enhanced.png”.
