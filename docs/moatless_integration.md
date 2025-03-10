# Moatless-Tree-Search Integration with VERL

This document explains how to use the Moatless-Tree-Search integration with VERL for high-quality code generation.

## Overview

The integration consists of two main components:

1. **Data Generation Pipeline**: Uses Moatless-Tree-Search to generate high-quality trajectories with logprobs
2. **Mock Rollout System**: Loads pre-computed trajectories during VERL training

This approach allows VERL to learn from high-quality code solutions generated by Moatless-Tree-Search.

## Setup

### Prerequisites

- VERL installed and configured
- Moatless-Tree-Search installed and configured
- A VLLM server running for Moatless-Tree-Search

### Environment Variables

Make sure to set the following environment variables:

```bash
export CUSTOM_LLM_API_BASE="http://localhost:8000/v1"
export CUSTOM_LLM_API_KEY="not-needed"
export PYTHONPATH="/path/to/moatless-tree-search:$PYTHONPATH"
```

## Step 1: Generate Data with Moatless-Tree-Search

Use the `MoatlessDataGenerator` to generate high-quality trajectories:

```bash
python -m verl.workers.rollout.moatless_generator \
  --moatless_path "/path/to/moatless-tree-search" \
  --output_dir "/path/to/output/directory" \
  --instances "django__django-11179" "astropy__astropy-14365" \
  --model "openai/Qwen/Qwen2.5-Coder-32B-Instruct" \
  --format "window" \
  --output_parquet "/path/to/dataset.parquet"
```

### Important Arguments

- `--moatless_path`: Path to the moatless-tree-search repository
- `--output_dir`: Directory to store output JSON files
- `--instances`: List of SWE-bench instances to run
- `--model`: LLM model to use
- `--format`: Format for moatless (window, etc.)
- `--output_parquet`: Output path for the parquet dataset

## Step 2: Train VERL with Pre-computed Trajectories

Configure VERL to use the `MoatlessRollout` class:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/path/to/dataset.parquet" \
    data.val_files="/path/to/dataset.parquet" \
    # ... other VERL parameters ... \
    actor_rollout_ref.rollout.name=moatless \
    actor_rollout_ref.rollout.dataset_path="/path/to/dataset.parquet" \
    # ... other VERL parameters ...
```

### Key Configuration

- `actor_rollout_ref.rollout.name=moatless`: Use the MoatlessRollout class
- `actor_rollout_ref.rollout.dataset_path`: Path to the dataset generated in Step 1

## Using the Pipeline Script

For convenience, a script is provided to run the entire pipeline:

```bash
bash scripts/run_moatless_pipeline.sh
```

You may need to modify the script to match your environment and requirements.

## Data Format

The generated dataset contains the following fields:

- `prompt`: The input prompt
- `response`: The generated response
- `response_tokens`: Tokenized response
- `logprobs`: Log probabilities for each token

## Debugging Tips

1. **Start with a small set of instances**: Start with 1-2 instances to verify the pipeline works
2. **Check the generated dataset**: Inspect the parquet file to ensure it contains the expected data
3. **Monitor GPU memory**: The process can be memory-intensive

## Advanced Configuration

### Customizing the MoatlessRollout

You can customize the `MoatlessRollout` class behavior with these options:

- `randomize`: Whether to randomize the order of examples (default: True)
- Custom example selection strategies by overriding `_get_next_batch`

### Extending to Other Datasets

To use with other datasets beyond SWE-bench:

1. Modify the `MoatlessDataGenerator` to support your dataset
2. Update the build_prompts.py script to process your dataset format

## Troubleshooting

### Common Issues

- **VLLM Connection Errors**: Ensure the VLLM server is running and accessible
- **Missing Dataset**: Verify the dataset path exists and contains valid data
- **Memory Errors**: Reduce batch size or number of workers

### Logging

Set the logging level for more detailed output:

```bash
export VERL_LOG_LEVEL=DEBUG
``` 