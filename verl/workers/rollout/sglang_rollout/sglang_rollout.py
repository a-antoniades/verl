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
import requests
import socket
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length, pad_2d_list_to_length
from sglang.srt.entrypoints.verl_engine import VerlEngine
from sglang.srt.server_args import ServerArgs
from torch.distributed.device_mesh import init_device_mesh
from sglang.srt.sampling.sampling_params import SamplingParams
from verl.third_party.sglang import parallel_state as sglang_ps
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
import json
import random
import time

if TYPE_CHECKING:
    from torch import nn


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# NOTE(linjunrong): adhoc
def _post_process_outputs(tokenizer, output):

    def _map_each_response(l):
        # output_token_ids = torch.tensor(l['token_ids'])
        log_probs = []
        output_token_ids = []
        for log_prob, token_ids, _ in l["meta_info"]["output_token_logprobs"]:
            log_probs.append(log_prob)
            output_token_ids.append(token_ids)
        log_probs = torch.tensor(log_probs)
        output_token_ids = torch.tensor(output_token_ids)
        return output_token_ids, log_probs

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs


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
                    print(f"Successfully connected to SGLang server")
                    return True
            except Exception as e:
                print(f"Server health check failed (attempt {retry+1}/{max_retries}): {e}")
                time.sleep(1)
        return False

    def generate(self, prompt_tokens, sampling_params, tokenizer):
        """Handle generation request to server with fallback"""
        try:
            decoded_text = tokenizer.decode(prompt_tokens)
            request_data = {
                "text": decoded_text,
                "max_tokens": min(sampling_params.get("max_new_tokens", 1024), sampling_params.get("response_length", 1024)),
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
                "top_k": sampling_params.get("top_k", -1),
                "n": sampling_params.get("n", 1)
            }
            
            response = requests.post(self.url, json=request_data, timeout=120)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error in HTTP request: {e}")
            return None


class SGLangRollout(BaseRollout):

    def __init__(
        self,
        actor_module: nn.Module | str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
        """A SGLang rollout. It requires the module is supported by the SGLang.

        Args:
            actor_module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in SGLang
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        assert not (not config.enforce_eager and
                    config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

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

        tp_size = tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        device_mesh_kwargs = dict(
            mesh_shape=(world_size // tp_size, tp_size, 1),
            mesh_dim_names=["dp", "tp", "pp"],
        )
        device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
        # Save device_mesh_cpu for later use
        self.device_mesh_cpu = device_mesh_cpu
        # device_mesh_device = init_device_mesh("cuda", **device_mesh_kwargs)

        # get tp_rank of this process in this tp group
        visible_devices = [None] * device_mesh_cpu.size(1)
        torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"],
                                          device_mesh_cpu.get_group("tp"))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

        # Initialize HTTP server if enabled
        self.http_server = None
        self.inference_engine = None  # Initialize to None first
        
        # Get the worker rank for proper device assignment
        worker_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Calculate proper TP rank for this worker based on the device mesh
        tp_size = config.get("tensor_model_parallel_size", 1)
        tp_group_rank = worker_rank % tp_size
        
        # Set CUDA device based on TP rank
        if torch.cuda.is_available():
            torch.cuda.set_device(tp_group_rank)
        
        if config.get("use_http_server", False):
            try:
                server_port = config.get("server_port", None)
                if server_port is not None:
                    # Make server port unique per DP group
                    dp_group_id = worker_rank // tp_size
                    server_port = server_port + dp_group_id
                
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
                    
                # Store a reference to the engine created by the HTTP server
                self.inference_engine = self.http_server.engine
                    
            except Exception as e:
                print(f"Failed to start SGLang HTTP server: {e}")
                print("Falling back to in-process engine")
                self.http_server = None
        
        # Initialize in-process engine if HTTP server is not used or failed
        if self.http_server is None:
            self.inference_engine = self._initialize_in_process_engine(
                actor_module, config, device_mesh_cpu, worker_rank=tp_group_rank
            )
            if not config.get("use_http_server", False):
                self.inference_engine.release_memory_occupation()

        kwargs = dict(n=1,
                      max_new_tokens=config.response_length,
                      presence_penalty=0.0,
                      frequency_penalty=0.0,
                      repetition_penalty=1.0)
        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        self.sampling_params = kwargs

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def _initialize_in_process_engine(self, actor_module, config, device_mesh_cpu, worker_rank=None):
        # Set CUDA device if worker_rank is provided
        if worker_rank is not None and torch.cuda.is_available():
            torch.cuda.set_device(worker_rank)
            
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
        )

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
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            self.sampling_params[key] = value

    def _post_process_server_outputs(self, tokenizer, outputs, batch_size, n=1):
        """Process outputs from the SGLang HTTP server into the expected format."""
        try:
            print(f"Server output format: {outputs}")
            print(f"Expected batch size: {batch_size}, n={n}")
            
            # Convert server response to the expected format
            batched_output_token_ids = []
            batched_logprobs = []
            
            # For each prompt in the batch
            for i in range(len(outputs)):
                response_data = outputs[i]
                
                # The response should have 'n' completions
                if isinstance(response_data, list):
                    # Server returned a list of completions
                    responses = response_data
                elif isinstance(response_data, dict) and 'text' in response_data:
                    # Server returned a single completion
                    responses = [response_data]
                else:
                    print(f"Unexpected server response format: {response_data}")
                    # Create empty tensors for this sample
                    empty_ids = torch.tensor([tokenizer.eos_token_id])
                    empty_probs = torch.tensor([0.0])
                    for _ in range(n):
                        batched_output_token_ids.append(empty_ids)
                        batched_logprobs.append(empty_probs)
                    continue
                
                # Process each of the 'n' completions for this prompt
                for j in range(len(responses)):
                    response = responses[j]
                    if 'text' in response:
                        # Extract just the assistant's response (not including prompt)
                        output_text = response.get('text', '')
                        try:
                            # Use Hugging Face tokenizer to convert text to tokens
                            output_token_ids = tokenizer.encode(output_text, add_special_tokens=False)
                            output_token_ids = torch.tensor(output_token_ids)
                            
                            # Create dummy logprobs since the server might not return them
                            log_probs = torch.zeros(len(output_token_ids))
                            
                            batched_output_token_ids.append(output_token_ids)
                            batched_logprobs.append(log_probs)
                        except Exception as e:
                            print(f"Error tokenizing response: {e}")
                            batched_output_token_ids.append(torch.tensor([tokenizer.eos_token_id]))
                            batched_logprobs.append(torch.tensor([0.0]))
                    elif 'token_ids' in response:
                        # If the server returns token IDs directly
                        output_token_ids = torch.tensor(response['token_ids'])
                        # Create dummy logprobs if not provided
                        log_probs = torch.zeros(len(output_token_ids))
                        if 'logprobs' in response:
                            log_probs = torch.tensor(response['logprobs'])
                        
                        batched_output_token_ids.append(output_token_ids)
                        batched_logprobs.append(log_probs)
                    else:
                        print(f"Unexpected completion format: {response}")
                        batched_output_token_ids.append(torch.tensor([tokenizer.eos_token_id]))
                        batched_logprobs.append(torch.tensor([0.0]))
                
                # Make sure we have exactly n completions for this prompt
                while len(batched_output_token_ids) < (i+1) * n:
                    batched_output_token_ids.append(torch.tensor([tokenizer.eos_token_id]))
                    batched_logprobs.append(torch.tensor([0.0]))
            
            # Ensure we have the right total number of samples (batch_size * n)
            expected_samples = batch_size * n
            while len(batched_output_token_ids) < expected_samples:
                batched_output_token_ids.append(torch.tensor([tokenizer.eos_token_id]))
                batched_logprobs.append(torch.tensor([0.0]))
            
            # Print the number of samples we have
            print(f"Number of processed samples: {len(batched_output_token_ids)}")
            
            # Pad sequences to the same length
            pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
            if len(batched_logprobs) > 0:
                batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=0.0)
            
            print(f"Final tensor shape: {batched_output_token_ids.shape}")
            return batched_output_token_ids, batched_logprobs
        except Exception as e:
            print(f"Error processing server outputs: {e}")
            # Return empty tensors with the right batch dimension
            return torch.full((batch_size * n, 1), tokenizer.eos_token_id), torch.zeros((batch_size * n, 1))

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # Get original batch parameters
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # Record the original batch size - we need to maintain this exact size
        original_batch_size = idx.size(0)
        print(f"Original batch size: {original_batch_size}")

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=4096,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        
        # Memory management: Check current CUDA memory usage
        if torch.cuda.is_available():
            try:
                free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Free memory in GB
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Total memory in GB
                print(f"CUDA Memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total")
                
                # Adjust batch size or n based on available memory if needed
                n_samples = self.sampling_params.get("n", 1)
                if free_memory < 10 and n_samples > 2:  # Less than 10GB free and using multiple samples
                    old_n = n_samples
                    n_samples = max(1, min(n_samples, int(free_memory / 2)))  # Reduce n based on available memory
                    print(f"Memory low, reducing samples from {old_n} to {n_samples}")
                    kwargs["n"] = n_samples  # Update n in kwargs to pass to sampling params
            except Exception as e:
                print(f"Error checking CUDA memory: {e}")
            
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            print(f"{self.sampling_params=}")
            n_samples = self.sampling_params.get("n", 1)
            print(f"Using n={n_samples} samples")
            
            # Limit max_new_tokens to prevent OOM
            max_new_tokens = min(self.sampling_params.get("max_new_tokens", 4096), self.config.response_length)
            if "max_new_tokens" in self.sampling_params and self.sampling_params["max_new_tokens"] > max_new_tokens:
                print(f"Limiting max_new_tokens from {self.sampling_params['max_new_tokens']} to {max_new_tokens}")
                self.sampling_params["max_new_tokens"] = max_new_tokens
            
            if self.http_server:
                all_outputs = []
                max_batch_size = max(1, 8 // n_samples)
                
                for batch_start in range(0, len(idx_list), max_batch_size):
                    batch_end = min(batch_start + max_batch_size, len(idx_list))
                    batch_idx_list = idx_list[batch_start:batch_end]
                    
                    for prompt_tokens in batch_idx_list:
                        output = self.http_server.generate(
                            prompt_tokens, 
                            self.sampling_params,
                            self.tokenizer
                        )
                        if output is None:
                            # Fallback to in-process generation using a temporary engine
                            # Don't use self.inference_engine directly as it's the server's engine
                            worker_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                            tp_size = self.config.get("tensor_model_parallel_size", 1)
                            tp_group_rank = worker_rank % tp_size
                            
                            # Set CUDA device for this fallback operation
                            if torch.cuda.is_available():
                                torch.cuda.set_device(tp_group_rank)
                                
                            temp_engine = self._initialize_in_process_engine(
                                actor_module=self.config.model.path,
                                config=self.config,
                                device_mesh_cpu=self.device_mesh_cpu,
                                worker_rank=tp_group_rank
                            )
                            try:
                                output = temp_engine.generate(
                                    prompt=None,
                                    sampling_params=self.sampling_params,
                                    return_logprob=True,
                                    input_ids=[prompt_tokens],
                                )[0]
                            finally:
                                # Clean up the temporary engine
                                del temp_engine
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        all_outputs.append(output)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                out = self._post_process_server_outputs(
                    self.tokenizer, all_outputs, batch_size=batch_size, n=n_samples
                )
            else:
                output = self.inference_engine.generate(
                    prompt=None,
                    sampling_params=self.sampling_params,
                    return_logprob=True,
                    input_ids=idx_list,
                )
                out = _post_process_outputs(self.tokenizer, output)

        response = out[0].to(idx.device)
        log_probs = out[1].to(idx.device)

        print(f"Response tensor shape: {response.shape}")
        print(f"IDX tensor shape: {idx.shape}")
        print(f"log_probs tensor shape: {log_probs.shape}")

        # Ensure we don't exceed the configured response length
        if response.shape[1] > self.config.response_length:
            print(f"Truncating response from {response.shape[1]} to {self.config.response_length}")
            response = response[:, :self.config.response_length]
            log_probs = log_probs[:, :self.config.response_length] if log_probs.shape[1] > self.config.response_length else log_probs
        
        # Pad if needed
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        # CRITICAL: The final batch size must match the original batch size
        # When using n>1, we need to ensure we return exactly original_batch_size * n samples
        target_batch_size = original_batch_size
        if self.config.n > 1 and do_sample:
            target_batch_size = original_batch_size * self.config.n
        
        print(f"Target batch size: {target_batch_size}, Current response shape: {response.shape}")
        
        # Adjust the response batch size to match the target
        if response.size(0) > target_batch_size:
            # Truncate if we have too many samples
            print(f"Truncating response from {response.size(0)} to {target_batch_size} samples")
            response = response[:target_batch_size]
            log_probs = log_probs[:target_batch_size]
        elif response.size(0) < target_batch_size:
            # Repeat if we have too few samples
            print(f"Expanding response from {response.size(0)} to {target_batch_size} samples")
            repeat_factor = (target_batch_size + response.size(0) - 1) // response.size(0)  # Ceiling division
            response = response.repeat_interleave(repeat_factor, dim=0)[:target_batch_size]
            log_probs = log_probs.repeat_interleave(repeat_factor, dim=0)[:target_batch_size]
        
        # Now adjust idx, attention_mask and position_ids to match
        if idx.size(0) != target_batch_size:
            if idx.size(0) < target_batch_size:
                # Repeat idx to match target batch size
                repeat_factor = (target_batch_size + idx.size(0) - 1) // idx.size(0)  # Ceiling division
                idx = idx.repeat_interleave(repeat_factor, dim=0)[:target_batch_size]
                attention_mask = attention_mask.repeat_interleave(repeat_factor, dim=0)[:target_batch_size]
                position_ids = position_ids.repeat_interleave(repeat_factor, dim=0)[:target_batch_size]
            else:
                # Truncate idx to match target batch size
                idx = idx[:target_batch_size]
                attention_mask = attention_mask[:target_batch_size]
                position_ids = position_ids[:target_batch_size]
        
        batch_size = target_batch_size  # Update batch size after adjustments
        
        print(f"Final sizes - idx: {idx.shape}, response: {response.shape}, batch_size: {batch_size}")
        
        # Sanity check the batch sizes
        assert idx.size(0) == target_batch_size, f"idx size {idx.size(0)} doesn't match target {target_batch_size}"
        assert response.size(0) == target_batch_size, f"response size {response.size(0)} doesn't match target {target_batch_size}"
        
        # Clear any unused memory before concatenation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free cache engine
        if self.config.free_cache_engine and not self.http_server and self.inference_engine._engine is not None:
            self.inference_engine._engine.tokenizer_manager.flush_cache()

        return DataProto(batch=batch)
