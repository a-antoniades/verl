#!/bin/bash
# Test script for moatless vLLM adapter

set -e  # Exit on error

# Configuration
MOATLESS_PATH="/share/edc/home/antonis/_swe-planner/moatless-tree-search"
MODEL="/tmp/qwen-model"

# Create symbolic link
ln -sf /share/edc/home/antonis/weights/huggingface/models--Qwen--Qwen2.5-0.5B /tmp/qwen-model

# Set environment variables
export VERL_VLLM_ADAPTER="1"
export CUSTOM_LLM_API_BASE="http://localhost:8000/v1"
export CUSTOM_LLM_API_KEY="not-needed"
export HUGGINGFACE_API_KEY="not-needed"
export HUGGINGFACE_API_BASE="http://localhost:8000/v1"
export VERL_LOG_LEVEL=DEBUG
export PYTHONPATH="${MOATLESS_PATH}:${PYTHONPATH}"

# Create a simple Python test script
cat > test_adapter.py << 'EOF'
import os
import sys
from verl.workers.rollout.moatless_vllm_rollout import MoatlessVLLMRollout
import torch
from transformers import AutoTokenizer, AutoConfig

# Load tokenizer and config
model_path = "/tmp/qwen-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_hf_config = AutoConfig.from_pretrained(model_path)

# Create a dummy actor module
class DummyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        return None

# Create config
from omegaconf import OmegaConf
config = OmegaConf.create({
    "tensor_model_parallel_size": 1,
    "enforce_eager": True,
    "free_cache_engine": False,
    "gpu_memory_utilization": 0.9,
    "disable_log_stats": True,
    "enable_chunked_prefill": False,
    "prompt_length": 512,
    "response_length": 2048,
    "load_format": "auto",
    "n": 1,
    "dtype": "auto"
})

# Initialize the rollout
print("Initializing MoatlessVLLMRollout...")
rollout = MoatlessVLLMRollout(
    actor_module=DummyActor(),
    config=config,
    tokenizer=tokenizer,
    model_hf_config=model_hf_config,
    moatless_path=os.environ.get("MOATLESS_PATH", "."),
    model=model_path
)

# Test the adapter
print("Testing adapter with a simple completion...")
import litellm

try:
    response = litellm.completion(
        model="openai/Qwen2.5-0.5B",
        prompt="Hello, world!",
        max_tokens=10
    )
    print("Response:", response)
    print("Test successful!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

# Run the test
python test_adapter.py 