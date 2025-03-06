#!/bin/bash
# Script to run the moatless-VERL pipeline

set -e  # Exit on error

# Configuration
MOATLESS_PATH="/share/edc/home/antonis/swe-gym-setup/moatless-tree-search"
VERL_PATH="/share/edc/home/antonis/swe-gym-setup/verl"
OUTPUT_DIR="${VERL_PATH}/data/moatless_trajectories"
DATASET_PATH="${VERL_PATH}/data/moatless_dataset.parquet"

# Model configuration
MODEL="openai/Qwen/Qwen2.5-Coder-32B-Instruct"
FORMAT="window"

# Training configuration
N_GPUS=4
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=16
EXPERIMENT_NAME="verl_moatless_swe_bench"

# SWE-bench instances to run
INSTANCES=(
  "django__django-11179"
  "astropy__astropy-14365"
  "django__django-13033"
  "django__django-14155"
  "django__django-14999"
)

# Step 1: Generate data using moatless-tree-search
echo "Step 1: Generating data using moatless-tree-search"
mkdir -p "$OUTPUT_DIR"

# Use join to build a space-separated list of instances
INSTANCES_STR=$(IFS=" " ; echo "${INSTANCES[*]}")

python -m verl.workers.rollout.moatless_generator \
  --moatless_path "$MOATLESS_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --instances $INSTANCES_STR \
  --model "$MODEL" \
  --format "$FORMAT" \
  --output_parquet "$DATASET_PATH"

# Step 2: Run VERL training with the generated dataset
echo "Step 2: Running VERL training with the generated dataset"

# Set environment variables
export VERL_LOG_LEVEL=INFO

# Run VERL training
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATASET_PATH" \
    data.val_files="$DATASET_PATH" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_BATCH_SIZE / 4)) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=moatless \
    actor_rollout_ref.rollout.dataset_path="$DATASET_PATH" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$EXPERIMENT_NAME \
    trainer.experiment_name='moatless_swe_bench' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100

echo "Pipeline completed successfully!" 