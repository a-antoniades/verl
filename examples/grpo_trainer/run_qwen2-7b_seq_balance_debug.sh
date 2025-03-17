set -x

# Remove GPU-specific environment variables
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export N_GPUS=4
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# # set debug logging
export VERL_LOG_LEVEL=DEBUG
## set moatless testbed logging
export MOATLESS_LOG_LEVEL=DEBUG

# Set to use CPU
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_WORKER_LOG_TO_DRIVER=1
export N_GPUS=2
export ROLLOUT_TP_SIZE=1
export BATCH_SIZE=2
LEN_PROMPTS=20000

export BASE_MODEL="/share/edc/home/antonis/weights/huggingface/models--Qwen--Qwen2.5-0.5B"
export DATA_DIR="/share/edc/home/antonis/swe-gym-setup/verl/data/swe-bench/swe-verifier-50/"
# /share/edc/home/antonis/swe-gym-setup/verl/data/swe-bench/debug
export EXPERIMENT_NAME='verl_grpo_example_swe-bench'

# Or disable OOM killing (not recommended)
# export RAY_memory_monitor_refresh_ms=0

python3 -m verl.trainer.main_ppo_swe \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/train.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$LEN_PROMPTS \
    data.max_response_length=$LEN_PROMPTS \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$N_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(($LEN_PROMPTS + 5000)) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype='bf16' \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype='bf16' \
    actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype='fp32' \
    actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype='fp32' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name="vllm" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +model.enable_gradient_checkpointing=true \
    +trainer.offload_activations=true \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_penalty="reverse_kl" \
    +trainer.mixed_precision=true \
    +trainer.max_grad_norm=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$EXPERIMENT_NAME \
    trainer.experiment_name='debug_cpu_run' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=500 \
    +trainer.max_steps=2 \
    +bypass_testbed=true \
    +actor_rollout_ref.model.enable_lora=false \
    +actor_rollout_ref.model.max_lora_rank=8 \
    +actor_rollout_ref.model.max_loras=4 \
    +actor_rollout_ref.model.lora_dtype='auto' \
    +actor_rollout_ref.model.quantization=null $@