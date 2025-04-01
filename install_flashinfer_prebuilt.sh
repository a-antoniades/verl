#!/bin/bash

# Check if the conda environment exists
if conda info --envs | grep -q "^verl "; then
    echo "Activating verl environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate verl
else
    echo "Error: verl environment not found"
    exit 1
fi

# Install flashinfer with prebuilt kernels
pip install --upgrade git+https://github.com/flashinfer-ai/flashinfer.git#subdirectory=python

# Or alternatively, try to use a specific version
# pip install flashinfer==0.0.4

echo "Installed flashinfer with prebuilt kernels (if available for your GPU architecture)"
echo "If this doesn't work, please run your training script with disable_cuda_graph=True" 