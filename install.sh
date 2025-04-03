#!/bin/bash

# Create and activate conda environment with Python 3.11
conda create -n verl python=3.12 -y
conda activate verl

# Install flash-attention
# uv pip install "sglang[all]==0.4.3.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
pip install uv
uv pip install -e .
uv pip install -e "/share/edc/home/antonis/swe-gym-setup/verl/sglang-fork/python[dev]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
uv pip install --no-deps torchdata==0.11.0
uv pip install --no-deps tensordict==0.5.0
uv pip install --no-deps codetiming==1.4.0
uv pip install --no-deps hydra-core==1.3.2
uv pip install --no-deps omegaconf==2.3.0
uv pip install flash-attn --no-build-isolation
uv pip install torch_memory_saver

# Install moatless-tree-search
conda install --channel conda-forge pygraphviz -y
uv pip install --no-cache -e "/share/edc/home/antonis/_swe-planner/moatless-tree-search"    # build swe-search-2 from source
uv pip install --no-cache -e "/share/edc/home/antonis/_swe-planner/moatless-tree-search/moatless-testbeds"    # build from source

# Finally install VERL without dependencies
# uv pip install -e . --no-deps
# uv pip install codetiming hydra-core tensordict wandb --no-deps
# uv pip install --no-deps wandb
# conda install sqlite -y

# # Install moatless from tree-search directory
# conda install --channel conda-forge pygraphviz -y
# pip install -e /share/edc/home/antonis/_swe-planner/moatless-tree-search 
# pip install -e /share/edc/home/antonis/_swe-planner/moatless-tree-search/moatless-testbeds


