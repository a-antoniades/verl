# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, swe_bench
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import logging

# Setup logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Get log level from environment variable, default to INFO if not set
log_level = os.getenv('VERL_LOG_LEVEL', 'DEBUG')
logging.basicConfig(level=getattr(logging, log_level))

logger.setLevel(logging.DEBUG)

def _default_compute_score(data_source, solution_str, ground_truth, **kwargs):
    """Default score computation function with GRPO support"""
    if data_source == 'grpo':  # Add this case for GRPO
        # For GRPO, we don't need solution_str or ground_truth
        # Return a constant reward or compute based on policy behavior
        return 1.0  # Simple constant reward for testing
    elif data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    elif data_source == 'lighteval/MATH':
        return math.compute_score(solution_str, ground_truth)
    elif 'swe-bench' in data_source:
        return swe_bench.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError

class RewardManager():
    """The reward manager."""   

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
 
    def __call__(self, data: DataProto):
        self.logger.debug(f"Input data batch size: {len(data)}")
        
        if 'rm_scores' in data.batch.keys():
            self.logger.debug("Using pre-computed rm_scores")
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        self.logger.debug(f"Created reward_tensor with shape: {reward_tensor.shape}")

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            self.logger.debug(f"\nProcessing item {i}:")
            self.logger.debug(f"Data item: {data_item}")
            
            prompt_ids = data_item.batch['prompts']
            self.logger.debug(f"Prompt ids shape: {prompt_ids.shape}")

            prompt_length = prompt_ids.shape[-1]
            self.logger.debug(f"Prompt length: {prompt_length}")

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            self.logger.debug(f"Valid prompt length: {valid_prompt_length}, Valid prompt ids shape: {valid_prompt_ids.shape}")

            response_ids = data_item.batch['responses']
            self.logger.debug(f"Response ids shape: {response_ids.shape}")
            
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            self.logger.debug(f"Valid response length: {valid_response_length}, Valid response ids shape: {valid_response_ids.shape}")

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            self.logger.debug(f"Combined sequences shape: {sequences.shape}")
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']
            self.logger.debug(f"Data source: {data_source}")

            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            self.logger.debug(f"Computed score: {score}")
            
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                self.logger.debug("\nFull sequence:")
                self.logger.debug(sequences_str)

        self.logger.debug(f"Final reward_tensor shape: {reward_tensor.shape}")
        return reward_tensor

import ray
import hydra

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    logger.debug("Starting main function")
    run_ppo(config)

def run_ppo(config, compute_score=None):
    logger.debug("Initializing PPO training")
    if not ray.is_initialized():
        logger.debug("Initializing Ray")
        ray.init(
            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
            logging_level=logging.DEBUG,
            log_to_driver=True
        )

    logger.debug("Starting main PPO task")
    ray.get(main_task.remote(config, compute_score))



@ray.remote
def main_task(config, compute_score=None):
    logger = logging.getLogger(__name__)
    logger.debug("Inside main_task")
    
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    logger.debug("Initialized RayPPOTrainer")
    
    logger.debug("Initializing workers")
    trainer.init_workers()
    
    logger.debug("Starting training")
    trainer.fit()


if __name__ == '__main__':
    main()
