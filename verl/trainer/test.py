import json
import logging
import torch
import numpy as np
from tensordict import TensorDict
from transformers import AutoTokenizer
from verl import DataProto
from verl.trainer.main_ppo import RewardManager, run_ppo
import hydra
from omegaconf import OmegaConf
import ray
import pandas as pd
import os
from verl.single_controller.base import Worker
from unittest.mock import Mock, patch
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from contextlib import contextmanager
from verl.trainer.ppo.ray_trainer import Role  # Import the Role enum
from verl.workers.rollout.base import BaseRollout

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen2.5-0.5B"

def load_conversation_data(file_path):
    """Loads conversation data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_mock_data_from_conversation(conversation_data, tokenizer):
    """Converts conversation data into DataProto format"""
    # Get the assistant's response and logprobs
    assistant_msg = conversation_data["node_1_step_0"][-1]
    content = assistant_msg["content"]
    logprobs = assistant_msg["logprobs"]["content"]
    
    # Create tensors from the conversation
    batch_size = 1
    
    # Tokenize the content and ensure it's the right shape
    tokens = tokenizer(content, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(0)  # Remove batch dimension
    
    # Extract logprobs into a tensor and match the shape
    logprob_values = torch.tensor([token["logprob"] for token in logprobs])
    
    # Create attention mask based on logprobs availability
    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    # Set 1 for positions where we have logprobs (unmasked)
    attention_mask[:len(logprob_values)] = 1
    
    # Pad logprobs if needed to match input_ids length
    if len(logprob_values) < len(input_ids):
        logprob_values = torch.nn.functional.pad(
            logprob_values, 
            (0, len(input_ids) - len(logprob_values)), 
            value=0.0
        )
    
    logger.debug(f"input_ids shape: {input_ids.shape}")
    logger.debug(f"logprobs shape: {logprob_values.shape}")
    logger.debug(f"attention_mask shape: {attention_mask.shape}")
    logger.debug(f"Number of unmasked tokens: {attention_mask.sum()}")
    logger.debug(f"Number of logprobs: {len(logprobs)}")
    
    # Create TensorDict with consistent shapes
    batch = TensorDict({
        'prompts': input_ids.unsqueeze(0),  # Add batch dimension back: [1, seq_len]
        'responses': input_ids.unsqueeze(0),  # Add batch dimension back: [1, seq_len]
        'attention_mask': attention_mask.unsqueeze(0),  # Add batch dimension back: [1, seq_len]
        'logprobs': logprob_values.unsqueeze(0)  # Add batch dimension back: [1, seq_len]
    }, batch_size=[batch_size])

    # Create non-tensor batch
    non_tensor_batch = {
        'reward_model': np.array([
            {'ground_truth': '42'}
        ], dtype=object),
        'data_source': np.array(['openai/gsm8k'], dtype=object)
    }

    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch
    )

def create_mock_parquet(mock_data, tokenizer):
    """Creates a parquet file from our mock data"""
    # Extract data from mock_data
    batch = mock_data.batch
    non_tensor_batch = mock_data.non_tensor_batch
    
    # Create DataFrame with the required structure
    data = {
        'prompt': tokenizer.decode(batch['prompts'][0].tolist()),
        'response': tokenizer.decode(batch['responses'][0].tolist()),
        'ground_truth': non_tensor_batch['reward_model'][0]['ground_truth'],
        'data_source': non_tensor_batch['data_source'][0],
        'reward_model': non_tensor_batch['reward_model'].tolist()
    }
    df = pd.DataFrame([data])  # Single row DataFrame
    
    # Save to parquet
    parquet_path = '/tmp/mock_data.parquet'
    df.to_parquet(parquet_path)
    logger.debug(f"Created parquet file at {parquet_path}")
    return parquet_path

class MockRolloutWorker(Worker):
    """A minimal mock rollout worker that returns predefined data"""
    
    def __init__(self, config, role='actor_rollout'):
        super().__init__()
        self.config = config
        self.role = role
        
    def init_model(self):
        """No real model initialization needed"""
        pass
        
    def generate_sequences(self, batch):
        """Return our mock data instead of generating sequences"""
        # Return the same batch since our mock data already contains responses
        return batch
        
    def compute_log_prob(self, batch):
        """Return mock log probabilities"""
        # Create mock log probs of the same shape as responses
        responses = batch.batch['responses']
        mock_log_probs = torch.zeros_like(responses, dtype=torch.float)
        return DataProto(batch={'old_log_probs': mock_log_probs})
        
    def update_actor(self, batch):
        """Mock actor update"""
        return DataProto(meta_info={'metrics': {'actor_loss': 0.0}})

class MockVLLMRollout:
    """Mock that matches vLLMRollout's interface"""
    def __init__(self, actor_module, config, tokenizer, model_hf_config, **kwargs):
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id if tokenizer else 0
        print("MockVLLMRollout initialized")
    
    @contextmanager
    def update_sampling_params(self, **kwargs):
        print(f"update_sampling_params called with {kwargs}")
        yield
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        print("generate_sequences called")
        
        # Get batch size from input
        idx = prompts.batch['input_ids']
        batch_size = idx.size(0)
        device = idx.device
        
        # Create mock response data
        response_length = self.config.response_length
        response = torch.randint(0, 1000, (batch_size, response_length), device=device)
        log_probs = torch.randn(batch_size, response_length, device=device)
        
        # Concatenate sequences
        seq = torch.cat([idx, response], dim=-1)
        
        # Create position IDs and attention mask
        position_ids = prompts.batch['position_ids']
        attention_mask = prompts.batch['attention_mask']
        
        # Create response position IDs and attention mask
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # Mock response attention mask (all 1s for simplicity)
        response_attention_mask = torch.ones(batch_size, response_length, device=device)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Return DataProto with batch
        batch = {
            'prompts': idx,
            'responses': response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        
        return DataProto(batch=batch)

def create_minimal_config():
    """Creates minimal configuration for testing"""
    config = {
        'algorithm': {
            'adv_estimator': 'grpo',
            'kl_ctrl': {'type': 'fixed', 'kl_coef': 0.001},
            'gamma': 1.0,
            'lam': 1.0
        },
        'actor_rollout_ref': {
            'hybrid_engine': True,
            'model': {
                'path': 'dummy/model',
                'external_lib': None,
                'override_config': {},
                'enable_gradient_checkpointing': False,
                'use_remove_padding': False
            },
            'actor': {
                'strategy': 'fsdp',
                'ppo_mini_batch_size': 256,
                'ppo_micro_batch_size_per_gpu': 8,
                'use_dynamic_bsz': False,
                'ppo_max_token_len_per_gpu': 16384,
                'grad_clip': 1.0,
                'clip_ratio': 0.2,
                'entropy_coeff': 0.001,
                'use_kl_loss': False,
                'ppo_epochs': 1,
                'shuffle': False,
                'optim': {
                    'total_training_steps': 10,  # Will be overwritten by trainer
                    'lr': 1e-5,
                    'weight_decay': 0.0,
                    'beta1': 0.9,
                    'beta2': 0.95,
                    'eps': 1e-8,
                    'warmup_ratio': 0.0
                }
            },
            'rollout': {
                'name': 'vllm',
                'temperature': 1.0,
                'top_k': -1,
                'top_p': 1.0,
                'n': 1,
                'prompt_length': 512,
                'response_length': 128,
                'dtype': 'bfloat16',
                'gpu_memory_utilization': 0.5,
                'enforce_eager': True,
                'free_cache_engine': True,
                'load_format': 'dummy_dtensor',
                'tensor_model_parallel_size': 1,
                'log_prob_micro_batch_size_per_gpu': 16,
                'log_prob_use_dynamic_bsz': False,
                'disable_log_stats': True,
                'enable_chunked_prefill': False
            }
        },
        'data': {
            'train_files': ['/tmp/mock_data.parquet'],
            'val_files': ['/tmp/mock_data.parquet'],
            'train_batch_size': 1,
            'val_batch_size': 1,
            'max_prompt_length': 512,
            'max_response_length': 128,
            'prompt_key': 'prompt'
        },
        'trainer': {
            'n_gpus_per_node': 1,
            'nnodes': 1,
            'total_epochs': 1,
            'total_training_steps': 10,
            'project_name': 'test_project',
            'experiment_name': 'test_experiment',
            'logger': 'wandb',
            'default_local_dir': '/tmp/test_checkpoints',
            'default_hdfs_dir': None,
            'critic_warmup': 0,
            'test_freq': 0,
            'save_freq': 0,
            'val_before_train': False
        }
    }
    return OmegaConf.create(config)

def compute_grpo_score(data_source, solution_str, ground_truth):
    """Compute GRPO score with the same signature as _default_compute_score"""
    # For GRPO, we ignore the input parameters and return a constant reward
    return 1.0

def create_mock_data():
    """Creates mock data matching GSM8k structure plus rollout data"""
    df = pd.DataFrame({
        'data_source': ['openai/gsm8k'],
        'prompt': [[{
            'content': 'Test math problem',
            'role': 'user'
        }]],
        'ability': ['math'],
        'reward_model': [{
            'ground_truth': '42',
            'style': 'rule'
        }],
        'extra_info': [{
            'answer': 'The answer is 42\n#### 42',
            'index': 0,
            'split': 'train'
        }],
        # Add rollout data
        'response': ['Let me solve this:\n1. Step one\n2. Step two\n#### 42'],
        'input_ids': [torch.randint(0, 1000, (15,)).tolist()],
        'attention_mask': [torch.ones(15).tolist()],
        'position_ids': [torch.arange(15).tolist()],
        'logprobs': [torch.randn(15).tolist()]
    })
    
    parquet_path = '/tmp/mock_data.parquet'
    df.to_parquet(parquet_path)
    return parquet_path

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def decode(self, token_ids):
        return "Mocked decoded text"

@patch('verl.workers.sharding_manager.fsdp_ulysses.FSDPUlyssesShardingManager')
def test_single_training_step(mock_sharding_manager):
    """Test function that runs a single training step with mocked components"""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    
    config = create_minimal_config()
    mock_data = create_mock_data()
    
    # Create mock tokenizer
    tokenizer = MockTokenizer()
    
    # Create mock role worker mapping with correct roles
    role_worker_mapping = {
        Role.ActorRollout: ['localhost:10001'],  # Required by hybrid_engine assertion
    }
    
    # Create mock resource pool manager
    resource_pool_manager = Mock()
    resource_pool_manager.get_worker_address.return_value = 'localhost:10001'
    resource_pool_manager.get_resource_pool.return_value = Mock()
    
    # Create mock worker that will use our MockRollout
    mock_worker = Mock()
    mock_worker.generate_sequences = MockRollout().generate_sequences
    
    # Create mock worker group
    mock_worker_group = Mock()
    mock_worker_group.world_size = 1
    mock_worker_group.generate_sequences.side_effect = mock_worker.generate_sequences
    
    print("Created config and mock data")
    with patch('verl.trainer.ppo.ray_trainer.RayWorkerGroup') as mock_ray_worker_group:
        mock_ray_worker_group.return_value = mock_worker_group
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager
        )
        print("Created trainer")
        trainer.train()

class MockRollout(BaseRollout):
    """Simple mock rollout that returns predefined responses"""
    def __init__(self):
        super().__init__()
        print("MockRollout initialized")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate mock sequences from prompts"""
        print("MockRollout.generate_sequences called")
        
        # Get input tensors from prompts
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        
        batch_size = idx.size(0)
        device = idx.device
        
        # Create mock response with fixed length (e.g., 15 tokens)
        response_length = 15
        response = torch.randint(0, 1000, (batch_size, response_length), device=device)
        
        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)
        
        # Create position IDs and attention mask for the response
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # Create attention mask for response (all 1s for simplicity)
        response_attention_mask = torch.ones(batch_size, response_length, device=device)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Create return batch
        batch = {
            'prompts': idx,
            'responses': response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        
        return DataProto(batch=batch)

if __name__ == "__main__":
    test_single_training_step()