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

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, swe_bench
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import logging
from moatless.runtime.testbed import TestbedEnvironment
import hashlib
import os
import sys

from moatless.benchmark.evaluation import load_instances
from moatless.benchmark.swebench import create_repository, create_index
from moatless.benchmark.evaluation import load_instances
import shutil

# Get log level from environment variable, default to INFO if not set
log_level = os.getenv('VERL_LOG_LEVEL', 'DEBUG')
logging.basicConfig(level=getattr(logging, log_level))

# Configure testbed server URL
os.environ['TESTBED_SERVER_URL'] = 'http://testbeds.moatless.ai'  # This is from your working test logs

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create console handler if it doesn't exist
if not root_logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

# Get our logger
logger = logging.getLogger(__name__)

# Ensure testbed loggers are not filtered
for logger_name in ['testbeds.sdk.sdk', 'testbeds.sdk.client', 'moatless.runtime.testbed']:
    logging.getLogger(logger_name).setLevel(logging.INFO)
    logging.getLogger(logger_name).propagate = True


def _default_compute_score(data_source, solution_str, ground_truth, testbed_manager=None, **kwargs):
    logger.debug(f"Computing score for {data_source}")
    """Default score computation function with SWE-bench support"""
    if 'swe-bench' in data_source:
        logger.debug(f"Computing SWE-bench score for {data_source}")
        if testbed_manager is None:
            logger.warning("No testbed manager provided, skipping SWE-bench score computation")
            return 0.0
        
        # Extract instance_id from the extra_info that was passed through
        instance_id = kwargs.get('extra_info', {}).get('instance_id')
        if not instance_id:
            logger.warning("No instance_id found in extra_info")
            return 0.0
            
        # Parse instance_id to get repository and instance
        try:
            repository, instance = instance_id.split('__', 1)
        except ValueError:
            logger.error(f"Invalid instance_id format: {instance_id}")
            return 0.0
        
        logger.debug(f"Computing SWE-bench score for {data_source} with testbed_manager")
        return testbed_manager.compute_score(
            solution_str=solution_str,
            repository=repository,
            instance=instance,
            log_dir=kwargs.get('log_dir', '/tmp/swe-bench-logs')
        )
    elif data_source == 'grpo':
        logger.debug(f"Computing GRPO score for {data_source}")
        return 1.0

class RewardManager():
    """The reward manager."""   

    def __init__(self, tokenizer, num_examine, compute_score=None, testbed_manager=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.testbed_manager = testbed_manager
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.testbeds = {}
 
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
            self.logger.debug(f"Full non_tensor_batch: {data_item.non_tensor_batch}")
            self.logger.debug(f"Data source: {data_item.non_tensor_batch.get('data_source')}")
            self.logger.debug(f"Extra info: {data_item.non_tensor_batch.get('extra_info')}")
            
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

            # Only pass testbed_manager if it's a swe-bench task
            kwargs = {
                'data_source': data_source,
                'solution_str': sequences_str,
                'ground_truth': ground_truth,
                'extra_info': data_item.non_tensor_batch.get('extra_info', {})
            }
            if 'swe-bench' in data_source and self.testbed_manager is not None:
                kwargs['testbed_manager'] = self.testbed_manager

            score = self.compute_score(**kwargs)
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

    # Check if train_files contains swe-bench
    testbed_manager = None
    if hasattr(config.data, 'train_files') and 'swe-bench' in config.data.train_files:
        logger.debug("Creating SWEBenchRewardManager for SWE-bench tasks")
        # Get bypass_testbed from config, defaulting to False if not present
        bypass_testbed = config.get('bypass_testbed', False)
        logger.debug(f"bypass_testbed setting: {bypass_testbed}")
        testbed_manager = SWEBenchRewardManager(debug=True, bypass_testbed=bypass_testbed)
    
    logger.debug("Starting main PPO task")
    ray.get(main_task.remote(config, compute_score, testbed_manager))



@ray.remote
def main_task(config, compute_score=None, testbed_manager=None):
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

    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0, 
        compute_score=compute_score,
        testbed_manager=testbed_manager
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=1, 
        compute_score=compute_score,
        testbed_manager=testbed_manager
    )

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

def create_sha256_hash(text):
    """Create SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

class SWEBenchRewardManager:
    """Manages reward computation for SWE-bench tasks."""
    
    def __init__(self, debug=False, bypass_testbed=False):
        """Initialize the reward manager."""
        self.debug = debug
        self.bypass_testbed = bypass_testbed  # New flag to bypass testbed evaluation
        self.runtimes = {}  # Dictionary to store TestbedEnvironment instances
        self.patch_cache = {}  # Cache for patch evaluation results
        # Load verified instances
        all_instances = load_instances('verified')
        self.instances = {instance["instance_id"]: instance for instance in all_instances}
        logger.info(f"Loaded {len(self.instances)} instances for SWE-bench evaluation")
        if self.debug:
            logger.info("Running in debug mode - will use golden patches for testing")
        if self.bypass_testbed:
            logger.info("Running in bypass mode - will return success without running testbed")
        
    def get_runtime(self, repository, instance):
        """Get or create a TestbedEnvironment instance for the given repository and instance."""
        key = f"{repository}__{instance}"
        
        if key not in self.runtimes:
            instance_metadata = self.instances.get(key)
            if instance_metadata is None:
                logger.error(f"No metadata found for instance {key}")
                raise ValueError(f"Instance {key} not found in SWE-bench dataset")
            
            # Setup paths
            repo_base_dir = f"/tmp/test_repos_{key}"
            log_dir = f"/tmp/test_logs_{key}"
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(repo_base_dir, exist_ok=True)
            
            try:
                # Create repository
                logger.info(f"Creating repository for instance {key}...")
                repo = create_repository(instance_metadata, repo_base_dir=repo_base_dir)
                
                # Create code index
                logger.info("Creating code index...")
                code_index = create_index(instance_metadata, repository=repo)
                
                # Initialize testbed
                logger.info("Initializing testbed environment...")
                self.runtimes[key] = TestbedEnvironment(
                    repository=repo,
                    instance=instance_metadata,
                    log_dir=log_dir,
                    dataset_name="princeton-nlp/SWE-bench_Verified",
                    timeout=2000
                )
                logger.info(f"Created new runtime for {key}")
            except Exception as e:
                logger.error(f"Error creating runtime for {key}: {str(e)}")
                if os.path.exists(repo_base_dir):
                    shutil.rmtree(repo_base_dir, ignore_errors=True)
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir, ignore_errors=True)
                raise
                
        return self.runtimes[key]
        
    def compute_score(self, solution_str, repository, instance, log_dir='/tmp/swe-bench-logs', format_score=0., score=1.):
        """Compute reward score for SWE-bench responses using testbed evaluation."""
        # Add bypass check at the start
        if self.bypass_testbed:
            logger.debug("Bypass mode: Returning success without running testbed")
            return score
            
        key = f"{repository}__{instance}"
        
        # Use golden patch if in debug mode
        if self.debug:
            instance_metadata = self.instances.get(key)
            if instance_metadata is None:
                logger.error(f"No metadata found for instance {key}")
                return 0.0
                
            solution_str = instance_metadata.get("golden_patch", "")
            if not solution_str:
                logger.error("No golden patch found in instance")
                return 0.0
                
            logger.info(f"Debug mode: Using golden patch from instance {key}:\n{solution_str}")
        else:
            logger.debug(f"Computing score with inputs: solution_str={solution_str[:100]}..., repository={repository}, instance={instance}")
        
        if not solution_str or not solution_str.strip():
            logger.warning("Empty solution string provided")
            return 0.0
            
        # Basic validation that this looks like a patch
        if not any(marker in solution_str.lower() for marker in ['diff', '+++', '---', '@@ ', 'index ']):
            logger.warning("Solution string doesn't appear to be a patch format")
            logger.debug(f"Full solution string: {solution_str}")
            return 0.0
        
        # Create hash of the patch for caching
        patch_hash = create_sha256_hash(solution_str)
        
        # Check cache first
        if patch_hash in self.patch_cache:
            logger.debug("Found result in cache")
            return self.patch_cache[patch_hash]
        
        try:
            # Get or create the appropriate runtime
            runtime = self.get_runtime(repository, instance)
            
            # Evaluate the patch with detailed error capture
            logger.debug("Attempting to evaluate patch")
            try:
                result = runtime.evaluate(patch=solution_str)
                logger.debug(f"Raw evaluation result: {result}")
                
                if result:
                    logger.debug(f"Resolved: {result.resolved}")
                    if result.tests_status:
                        logger.debug(f"Test status: {result.tests_status.status}")
                        if result.tests_status.fail_to_pass:
                            logger.debug(f"Failed tests: {result.tests_status.fail_to_pass.failure}")
                            logger.debug(f"Passed tests: {result.tests_status.fail_to_pass.success}")
                        if hasattr(result.tests_status, 'error_message') and result.tests_status.error_message:
                            logger.debug(f"Error message: {result.tests_status.error_message}")
                else:
                    logger.error("Evaluation returned None - this typically means the testbed failed to process the patch")
                    logger.debug(f"Patch content being evaluated: {solution_str[:500]}...")
                
                # Convert result to reward score
                reward = score if (result and getattr(result, 'resolved', False)) else format_score
                
            except Exception as eval_error:
                logger.error(f"Evaluation error details: {str(eval_error)}")
                import traceback
                logger.error(f"Evaluation error traceback: {traceback.format_exc()}")
                reward = format_score
            
            # Cache the result
            self.patch_cache[patch_hash] = reward
            return reward
            
        except Exception as e:
            logger.error(f"Error in compute_score: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0

    def __del__(self):
        """Cleanup when the manager is destroyed."""
        logger.info("Cleaning up SWEBenchRewardManager resources...")
        for key, runtime in self.runtimes.items():
            try:
                repo_base_dir = f"/tmp/test_repos_{key}"
                log_dir = f"/tmp/test_logs_{key}"
                if os.path.exists(repo_base_dir):
                    shutil.rmtree(repo_base_dir, ignore_errors=True)
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Error cleaning up runtime {key}: {str(e)}")

if __name__ == '__main__':
    main()
