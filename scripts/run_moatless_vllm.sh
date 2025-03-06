#!/bin/bash
# Script to run moatless-tree-search with veRL's vLLM engine

set -e  # Exit on error

# Configuration
MOATLESS_PATH="/share/edc/home/antonis/_swe-planner/moatless-tree-search"
VERL_PATH="/share/edc/home/antonis/swe-gym-setup/verl"
OUTPUT_DIR="${VERL_PATH}/data/moatless_trajectories"
DATASET_PATH="${VERL_PATH}/data/moatless_dataset.parquet"

# Model configuration
MODEL="/share/edc/home/antonis/weights/huggingface/models--Qwen--Qwen2.5-0.5B"
FORMAT="window"

# Training configuration
N_GPUS=4
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=16
EXPERIMENT_NAME="verl_moatless_vllm_swe_bench"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate data and train in one go
echo "Running moatless-tree-search with veRL's vLLM engine"

# Set environment variables
export VERL_LOG_LEVEL=INFO
export PYTHONPATH="${MOATLESS_PATH}:${PYTHONPATH}"

# Run VERL with moatless_vllm rollout
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${VERL_PATH}/data/gsm8k/train.parquet" \
    data.val_files="${VERL_PATH}/data/gsm8k/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_BATCH_SIZE / 4)) \
    +actor_rollout_ref.actor.micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=moatless_vllm \
    +actor_rollout_ref.rollout.moatless_path="$MOATLESS_PATH" \
    +actor_rollout_ref.rollout.output_dir="$OUTPUT_DIR" \
    +actor_rollout_ref.rollout.model="$MODEL" \
    +actor_rollout_ref.rollout.format="$FORMAT" \
    "+actor_rollout_ref.rollout.instances=[\"django__django-11179\",\"astropy__astropy-14365\",\"django__django-13033\",\"django__django-14155\",\"django__django-14999\"]" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$EXPERIMENT_NAME \
    trainer.experiment_name='moatless_vllm_swe_bench' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100

echo "Pipeline completed successfully!" 