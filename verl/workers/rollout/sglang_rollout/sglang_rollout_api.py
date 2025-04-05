# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
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

from __future__ import annotations
import os
import socket
import requests
import random
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, List
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from sglang.srt.entrypoints.verl_engine import VerlEngine
from sglang.srt.server_args import ServerArgs
from torch.distributed.device_mesh import init_device_mesh
from sglang.srt.sampling.sampling_params import SamplingParams
from verl.third_party.sglang import parallel_state as sglang_ps
import torch
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
import logging

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Helper function to remove padding from input token ids
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def find_available_port(base_port: int):
    """Find an available port starting from base_port."""
    port = base_port + random.randint(100, 1000)
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1
            if port > 65535:
                port = base_port

class SGLangHttpServer:
    def __init__(self, host, port, worker_rank, config, device_mesh_cpu, actor_module):
        self.host = host
        self.port = self._initialize_port(port, worker_rank)
        self.url = f"http://{self.host}:{self.port}/generate"
        self.api_base_url = f"http://{self.host}:{self.port}/v1"
        self.engine = self._initialize_engine(config, device_mesh_cpu, actor_module, worker_rank)
        
    def _initialize_port(self, port, worker_rank):
        if port is None:
            return find_available_port(32000 + worker_rank * 100)
        return port + worker_rank
        
    def _initialize_engine(self, config, device_mesh_cpu, actor_module, worker_rank):
        # worker_rank is now the tp_rank within the group
        if torch.cuda.is_available():
            torch.cuda.set_device(worker_rank)
        
        # VerlEngine will only start the server on tp_rank 0 
        return VerlEngine(
            model_path=actor_module,
            dtype=config.dtype,
            mem_fraction_static=config.gpu_memory_utilization,
            device_mesh_cpu=device_mesh_cpu["tp"],
            base_gpu_id=0,  # base_gpu_id should be 0 as we set the device above
            gpu_id_step=1,
            log_level="INFO",
            log_requests=True,
            log_requests_level=2,
            max_running_requests=1,
            launch_server=True,  # This flag tells VerlEngine to conditionally launch server only on tp_rank 0
            server_args=ServerArgs(
                model_path=actor_module,
                tp_size=config.get("tensor_model_parallel_size", 1),
                port=self.port,
                mem_fraction_static=config.gpu_memory_utilization,
            ),
        )
    
    def check_health(self, max_retries=5):
        """Check if server is healthy and responding"""
        for retry in range(max_retries):
            try:
                health_url = f"http://{self.host}:{self.port}/health_generate"
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Successfully connected to SGLang server")
                    return True
            except Exception as e:
                logger.warning(f"Server health check failed (attempt {retry+1}/{max_retries}): {e}")
                time.sleep(1)
        return False


class SGLangRolloutAPI(BaseRollout):
    """SGLang rollout that uses Moatless API to generate completions."""

    def __init__(
        self,
        actor_module: nn.Module | str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
        """Initialize a SGLangRolloutAPI that uses the Moatless API for generation.

        Args:
            actor_module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initialize the generating model
            **kwargs: additional arguments
        """
        super().__init__()
        self.config = config
        
        # Setup for tensor parallelism
        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (tensor_parallel_size <= torch.distributed.get_world_size()
               ), "tensor parallel size should be less than or equal to the world size"

        if kwargs.get("train_tp", None) is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        assert (model_hf_config.max_position_embeddings >= config.prompt_length +
                config.response_length), "model context length should be greater than total sequence length"

        # Initialize device mesh
        tp_size = tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        device_mesh_kwargs = dict(
            mesh_shape=(world_size // tp_size, tp_size, 1),
            mesh_dim_names=["dp", "tp", "pp"],
        )
        device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
        self.device_mesh_cpu = device_mesh_cpu

        # Setup for the worker
        worker_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        tp_group_rank = worker_rank % tp_size
        
        # Set CUDA device based on TP rank
        if torch.cuda.is_available():
            torch.cuda.set_device(tp_group_rank)
        
        # Initialize HTTP server for SGLang
        server_port = config.get("server_port", None)
        if server_port is not None:
            # Make server port unique per DP group
            dp_group_id = worker_rank // tp_size
            server_port = server_port + dp_group_id
        
        # Initialize the SGLang server
        self.http_server = SGLangHttpServer(
            host=config.get("server_host", "localhost"),
            port=server_port,
            worker_rank=tp_group_rank,
            config=config,
            device_mesh_cpu=device_mesh_cpu,
            actor_module=actor_module
        )
        
        if not self.http_server.check_health():
            raise Exception("Failed to initialize HTTP server")
        
        # self.api = EvaluationWrapper()
            
        # Set up environment variable for Moatless API
        os.environ["CUSTOM_LLM_API_BASE"] = self.http_server.api_base_url
        os.environ["CUSTOM_LLM_API_KEY"] = "not-needed"
        logger.info(f"Set CUSTOM_LLM_API_BASE to {self.http_server.api_base_url}")
        
        # Initialize sampling parameters
        self.sampling_params = dict(
            n=1,
            max_new_tokens=config.response_length,
            temperature=config.get("temperature", 1.0),
            top_p=config.get("top_p", 1.0),
            top_k=config.get("top_k", -1)
        )
                
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.model_name = config.get("model_name", "default_model")
        
        # Try to import Moatless
        try:
            from moatless.benchmark.api.evaluation_wrapper import EvaluationWrapper
            self.has_moatless = True
        except ImportError:
            logger.warning("Moatless not found. Will use direct API requests.")
            self.has_moatless = False

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if key in self.sampling_params:
                    old_value = self.sampling_params[key]
                    old_sampling_params_args[key] = old_value
                    self.sampling_params[key] = value
        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            self.sampling_params[key] = value

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences using the Moatless API.

        Args:
            prompts: DataProto object containing input information (used primarily for batch size and device).
                     The actual prompt content comes from the API call based on instance_id.
            **kwargs: Additional parameters for generation.

        Returns:
            DataProto: DataProto object containing the generated sequences corresponding to the
                     processed instance, repeated n_samples times and matched to the original batch size.
        """
        # Determine batch size and device from the input prompts object
        if "input_ids" in prompts.batch:
            original_batch_size = prompts.batch["input_ids"].size(0)
            device = prompts.batch["input_ids"].device
        elif "prompts" in prompts.batch: # Fallback if input_ids is not directly available
            original_batch_size = prompts.batch["prompts"].size(0)
            device = prompts.batch["prompts"].device
        else:
            try:
                some_tensor = next(iter(prompts.batch.values()))
                original_batch_size = some_tensor.size(0)
                device = some_tensor.device
            except (StopIteration, AttributeError):
                 raise ValueError("Could not determine batch size or device from input DataProto.")

        eos_token_id = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id

        # Update sampling parameters for this generation
        with self.update_sampling_params(**kwargs):
            n_samples = self.sampling_params.get("n", 1)

            single_instance_prompts = []
            single_instance_responses = []

            # Removed loop over original_batch_size.
            # Now processes only the single instance defined in the EvaluationWrapper/API call.
            instance_id_for_api = "django__django-16569" # Example, matching user's edit
            logger.info(f"Processing single instance via API: {instance_id_for_api}")

            try:
                from moatless.benchmark.api.evaluation_wrapper import EvaluationWrapper

                # Initialize wrapper for the single instance
                wrapper = EvaluationWrapper(
                    model="gpt-4o-mini-2024-07-18", # Example, matching user's edit
                    eval_dir="./evaluations/custom_run",
                    dataset_name="princeton-nlp/SWE-bench_Lite",
                    split="lite",
                    max_finished_nodes=1,
                    overwrite=True,
                )

                # Single API call for the specified instance
                result = wrapper.run_and_process_instance(
                    instance_id=instance_id_for_api,
                    message_format="messages",
                    split_messages=False,
                    verl_format=True
                )

                # Extract prompt messages and response text from the single result
                if isinstance(result, dict) and "message_groups" in result and result["message_groups"]:
                    first_group = result["message_groups"][0]
                    input_messages = first_group.get("input", [])
                    response_text = first_group.get("response", "")

                    if not input_messages or not response_text:
                         logger.warning(f"Instance {instance_id_for_api}: Missing input or response in result group: {first_group}")
                         input_messages = [{"role": "user", "content": ""}] # Placeholder
                         response_text = ""
                else:
                    logger.warning(f"Instance {instance_id_for_api}: Unexpected result format: {result}")
                    input_messages = [{"role": "user", "content": ""}] # Placeholder
                    response_text = ""

                # Tokenize the single result
                prompt_tokens = self.tokenizer.apply_chat_template(
                    input_messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )[0]

                response_tokens = self.tokenizer.encode(
                    response_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )[0]

                # Repeat the single result n_samples times
                single_instance_prompts = [prompt_tokens] * n_samples
                single_instance_responses = [response_tokens] * n_samples

            except ImportError:
                 logger.error("Moatless is required but not installed.")
                 single_instance_prompts = [torch.tensor([], dtype=torch.long)] * n_samples
                 single_instance_responses = [torch.tensor([], dtype=torch.long)] * n_samples
                 self.has_moatless = False
            except Exception as e:
                logger.error(f"Error processing instance {instance_id_for_api}: {e}")
                single_instance_prompts = [torch.tensor([], dtype=torch.long)] * n_samples
                single_instance_responses = [torch.tensor([], dtype=torch.long)] * n_samples

            # We now have lists containing n_samples copies of the tokenized result from the single API call.
            batched_prompt_token_ids = single_instance_prompts
            batched_response_token_ids = single_instance_responses

            # Pad sequences
            if not batched_prompt_token_ids or not batched_response_token_ids:
                # Handle cases where API call failed entirely
                 logger.warning("No valid tokens generated from API call. Returning empty tensors.")
                 # Create empty tensors with the correct target batch size and sequence lengths
                 target_batch_size = original_batch_size * n_samples
                 prompts_tensor = torch.full((target_batch_size, 0), pad_token_id, dtype=torch.long, device=device)
                 responses_tensor = torch.full((target_batch_size, self.config.response_length), pad_token_id, dtype=torch.long, device=device)
                 log_probs = torch.zeros_like(responses_tensor, dtype=torch.float).to(device)
            else:
                # Pad prompts based on the (repeated) single prompt result
                prompts_tensor = pad_sequence(
                    batched_prompt_token_ids, batch_first=True, padding_value=pad_token_id
                )

                # Pad responses based on the (repeated) single response result
                responses_tensor = pad_sequence(
                    batched_response_token_ids, batch_first=True, padding_value=pad_token_id
                )

                # Truncate or pad responses tensor to exact response_length
                current_response_len = responses_tensor.shape[1]
                if current_response_len > self.config.response_length:
                    responses_tensor = responses_tensor[:, :self.config.response_length]
                elif current_response_len < self.config.response_length:
                    padding_size = self.config.response_length - current_response_len
                    padding_tensor = torch.full(
                        (responses_tensor.shape[0], padding_size),
                        pad_token_id,
                        dtype=responses_tensor.dtype,
                        device=responses_tensor.device # Use device of responses_tensor
                    )
                    responses_tensor = torch.cat([responses_tensor, padding_tensor], dim=1)

                # Move tensors to the target device
                prompts_tensor = prompts_tensor.to(device)
                responses_tensor = responses_tensor.to(device)

                # Create dummy log probabilities
                log_probs = torch.zeros_like(responses_tensor, dtype=torch.float).to(device)

        # We have n_samples results from ONE instance.
        # The target batch size is original_batch_size * n_samples.
        # We need to repeat the n_samples results `original_batch_size` times.
        target_batch_size = original_batch_size * n_samples
        current_batch_size = prompts_tensor.size(0) # Should be n_samples

        if current_batch_size != target_batch_size:
            if current_batch_size == 0: # Handle case where API failed and tensors are empty shells
                if target_batch_size > 0:
                    logger.warning(f"API failed, creating empty tensors for target batch size {target_batch_size}")
                    # Recreate empty tensors with correct target size if needed (already handled above, but double-check)
                    prompts_tensor = torch.full((target_batch_size, prompts_tensor.shape[1]), pad_token_id, dtype=torch.long, device=device)
                    responses_tensor = torch.full((target_batch_size, self.config.response_length), pad_token_id, dtype=torch.long, device=device)
                    log_probs = torch.zeros_like(responses_tensor, dtype=torch.float).to(device)
                # else: target_batch_size is 0, nothing to do
            elif current_batch_size > 0 and target_batch_size % current_batch_size == 0:
                repeat_factor = target_batch_size // current_batch_size # Should be original_batch_size
                logger.info(f"Repeating single instance result {repeat_factor} times to match target batch size {target_batch_size}.")
                prompts_tensor = prompts_tensor.repeat_interleave(repeat_factor, dim=0)
                responses_tensor = responses_tensor.repeat_interleave(repeat_factor, dim=0)
                log_probs = log_probs.repeat_interleave(repeat_factor, dim=0)
            else:
                # This case should ideally not happen if current_batch_size is n_samples
                # and target_batch_size is original_batch_size * n_samples.
                logger.warning(f"Unexpected batch size mismatch. current={current_batch_size}, target={target_batch_size}. Attempting to reshape, but results may be incorrect.")
                # Fallback: just take the first target_batch_size elements if too many, or pad if too few (less likely)
                if current_batch_size > target_batch_size:
                    prompts_tensor = prompts_tensor[:target_batch_size]
                    responses_tensor = responses_tensor[:target_batch_size]
                    log_probs = log_probs[:target_batch_size]
                # If current_batch_size < target_batch_size, padding is complex, log error
                elif current_batch_size < target_batch_size:
                     logger.error(f"Cannot properly expand batch size from {current_batch_size} to {target_batch_size}. Results will be incomplete.")
                     # Optionally pad with default values, but this indicates a logic error

        # Final assertions to ensure correct batch size
        assert prompts_tensor.size(0) == target_batch_size, f"Final prompt tensor batch size mismatch: {prompts_tensor.size(0)} vs {target_batch_size}"
        assert responses_tensor.size(0) == target_batch_size, f"Final response tensor batch size mismatch: {responses_tensor.size(0)} vs {target_batch_size}"
        assert log_probs.size(0) == target_batch_size, f"Final log probs tensor batch size mismatch: {log_probs.size(0)} vs {target_batch_size}"

        # Concatenate prompt and response
        seq = torch.cat([prompts_tensor, responses_tensor], dim=-1)

        # Calculate position IDs
        seq_length = seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(target_batch_size, 1)

        # Calculate attention mask (mask out padding tokens)
        attention_mask = (seq != pad_token_id).long()

        # Create output batch
        batch = TensorDict(
            {
                "prompts": prompts_tensor,
                "responses": responses_tensor,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=target_batch_size,
        )

        return DataProto(batch=batch) 