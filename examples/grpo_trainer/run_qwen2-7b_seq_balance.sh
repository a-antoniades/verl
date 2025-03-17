set -x

# Add this line to set logging level
export VERL_LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

export VLLM_ATTENTION_BACKEND=XFORMERS
# Automatically detect number of available GPUs using nvidia-smi
export N_GPUS=2
export BASE_MODEL="/share/edc/home/antonis/weights/huggingface/models--Qwen--Qwen2.5-0.5B"
export DATA_DIR="/share/edc/home/antonis/swe-gym-setup/verl/data/gsm8k"
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME='verl_grpo_length_gsm8k_small'
export TRAIN_BATCH_SIZE=640  # Reduce this
export PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE / 4))
export VAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE))  # 1.28x train batch size
export MAX_TOKEN_LEN_PER_GPU=$((TRAIN_BATCH_SIZE * 24))  # 24x train batch size

# Memory management
export RAY_memory_threshold=0.90
export RAY_memory_monitor_refresh_ms=100
export RAY_object_store_memory=50000000000  # 50GB limit for Ray's object store
export RAY_memory_usage_threshold=0.95

# Reduce number of Ray workers
export RAY_actor_concurrency=2  # Limit concurrent actor tasks
export RAY_max_actor_restarts=3

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$EXPERIMENT_NAME \
    trainer.experiment_name='length_exp_no_pen' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1000 \
    +actor_rollout_ref.actor.use_length_penalty=false \
    +actor_rollout_ref.actor.length_window_size=250 \
    +actor_rollout_ref.actor.length_penalty_coef=0.1 \
    +actor_rollout_ref.actor.max_concurrent_workers=2 \
    +actor_rollout_ref.actor.worker_cpu_limit=4 \
    $@