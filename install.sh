#!/bin/bash

# Create and activate conda environment with Python 3.11
conda create -n verl python=3.12 -y
conda activate verl

# Install flash-attention
pip install torch torchvision torchaudio
pip install -e ".[test,gpu,sglang]"
pip install flash-attn --no-build-isolation
pip install wandb
# conda install sqlite -y

# # Install moatless from tree-search directory
# conda install --channel conda-forge pygraphviz -y
# pip install -e /share/edc/home/antonis/_swe-planner/moatless-tree-search 
# pip install -e /share/edc/home/antonis/_swe-planner/moatless-tree-search/moatless-testbeds


