#!/bin/bash

# Create and activate conda environment with Python 3.11
conda create -n verl python=3.12 -y
conda activate verl

# Install flash-attention
pip install uv
uv pip install "sglang[all]==0.4.3.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
uv pip install --no-deps torchdata==0.11.0
uv pip install --no-deps tensordict==0.5.0
uv pip install --no-deps codetiming==1.4.0
uv pip install --no-deps hydra-core==1.3.2
uv pip install --no-deps omegaconf==2.3.0  # needed by hydra-core

# Finally install VERL without dependencies
uv pip install -e . --no-deps
# uv pip install -e . --no-deps
# uv pip install codetiming hydra-core tensordict wandb --no-deps
# uv pip install --no-deps wandb
# conda install sqlite -y

# # Install moatless from tree-search directory
# conda install --channel conda-forge pygraphviz -y
# pip install -e /share/edc/home/antonis/_swe-planner/moatless-tree-search 
# pip install -e /share/edc/home/antonis/_swe-planner/moatless-tree-search/moatless-testbeds


