#!/bin/bash

# Create and activate conda environment with Python 3.11
conda create -n verl python=3.12 -y
conda activate verl

# Install moatless from tree-search directory
conda install --channel conda-forge pygraphviz
pip install -e /share/edc/home/antonis/_swe-planner/moatless-tree-search
pip install -e /share/edc/home/antonis/_swe-planner/moatless-tree-search/moatless-testbeds

# Install flash-attention
conda install sqlite -y
pip install flash-attn --no-build-isolation
pip install -e .

