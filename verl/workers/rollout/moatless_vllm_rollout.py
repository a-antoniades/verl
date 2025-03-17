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
"""
Moatless Tree Search rollout that directly uses veRL's vLLM engine.
"""
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import os
import subprocess
import tempfile
import json
import logging
import threading
import queue
import time
from pathlib import Path
import datetime
import shutil
import filelock

from omegaconf import DictConfig
import torch
from tensordict import TensorDict
from torch import nn

from verl.workers.rollout.base import BaseRollout, DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout

# Import vLLM components
try:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.utils import random_uuid
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose logging

class VLLMServerAdapter:
    """
    Adapter that makes a vLLM engine accessible via a local API-compatible interface.
    This allows moatless-tools to use the vLLM engine as if it were a remote server.
    """
    def __init__(self, llm_engine):
        """
        Initialize the adapter with a vLLM engine.
        
        Args:
            llm_engine: A vLLM engine instance
        """
        self.llm_engine = llm_engine
        self.request_queue = queue.Queue()
        self.response_queues = {}
        self.running = True
        
        # Start the worker thread
        self.worker_thread = threading.Thread(target=self._process_requests)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("VLLM Server Adapter initialized")
    
    def _process_requests(self):
        """Worker thread that processes requests from the queue."""
        while self.running:
            try:
                request_id, params, response_queue = self.request_queue.get(timeout=0.1)
                
                try:
                    # Process the request using the vLLM engine
                    if params["type"] == "completions":
                        result = self._handle_completions(params)
                    elif params["type"] == "chat_completions":
                        result = self._handle_chat_completions(params)
                    else:
                        result = {"error": f"Unsupported request type: {params['type']}"}
                    
                    # Put the result in the response queue
                    response_queue.put(result)
                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {e}")
                    response_queue.put({"error": str(e)})
                
                self.request_queue.task_done()
            except queue.Empty:
                # No requests in the queue, continue
                pass
            except Exception as e:
                logger.error(f"Error in request processing loop: {e}")
    
    def _handle_completions(self, params):
        """
        Handle a completions request.
        
        Args:
            params: Request parameters
            
        Returns:
            dict: Response data
        """
        prompt = params.get("prompt", "")
        # Create sampling parameters with logprobs if requested
        sampling_params = SamplingParams(
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 1.0),
            max_tokens=params.get("max_tokens", 16),
            logprobs=params.get("logprobs") is not None
        )
        
        # Generate completions
        try:
            logger.info(f"Generating completion for prompt of length {len(prompt)}")
            
            # Generate using vLLM
            outputs = self.llm_engine.generate(prompt, sampling_params)
            
            # Format the response to match OpenAI API
            completions = []
            for output in outputs:
                completion = {
                    "id": output.request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": params.get("model", "unknown"),
                    "choices": [{
                        "text": output.outputs[0].text,
                        "index": 0,
                        "finish_reason": output.outputs[0].finish_reason,
                        "logprobs": output.outputs[0].logprobs if hasattr(output.outputs[0], "logprobs") else None
                    }]
                }
                completions.append(completion)
            
            return {"completions": completions}
        
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return {"error": f"Completion generation failed: {str(e)}"}
    
    def _handle_chat_completions(self, params):
        """
        Handle a chat completions request.
        
        Args:
            params: Request parameters
            
        Returns:
            dict: Response data
        """
        messages = params.get("messages", [])
        
        # Convert messages to a prompt format that vLLM can understand
        try:
            # Try to use the tokenizer to format chat messages
            if hasattr(self.llm_engine, "get_tokenizer"):
                tokenizer = self.llm_engine.get_tokenizer()
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    # Fallback to simple format
                    prompt = self._format_chat_messages(messages)
            else:
                # Fallback to simple format
                prompt = self._format_chat_messages(messages)
        except Exception as e:
            logger.warning(f"Error formatting chat messages: {e}, falling back to simple format")
            prompt = self._format_chat_messages(messages)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 1.0),
            max_tokens=params.get("max_tokens", 16)
        )
        
        # Enable logprobs if requested
        logprobs_requested = params.get("logprobs") is not None
        if logprobs_requested and hasattr(sampling_params, 'logprobs'):
            sampling_params.logprobs = True
        
        # Generate completions
        try:
            outputs = self.llm_engine.generate(prompt, sampling_params)
            
            # Format the response to match OpenAI API
            completions = []
            for output in outputs:
                completion = {
                    "id": output.request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": params.get("model", "unknown"),
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": output.outputs[0].text,
                        },
                        "index": 0,
                        "finish_reason": output.outputs[0].finish_reason,
                    }]
                }
                
                # Add logprobs if available
                if logprobs_requested:
                    # Try different ways to access logprobs based on vLLM version
                    logprobs_value = None
                    if hasattr(output.outputs[0], "logprobs"):
                        logprobs_value = output.outputs[0].logprobs
                    elif hasattr(output, "logprobs"):
                        logprobs_value = output.logprobs
                    
                    if logprobs_value is not None:
                        completion["choices"][0]["logprobs"] = logprobs_value
                    else:
                        # Provide empty logprobs if requested but not available
                        completion["choices"][0]["logprobs"] = []
                
                completions.append(completion)
            
            return {"chat_completions": completions}
            
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            return {"error": f"Chat completion generation failed: {str(e)}"}
    
    def _format_chat_messages(self, messages):
        """
        Format chat messages into a simple text prompt.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Formatted prompt
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def make_request(self, params):
        """
        Make a request to the vLLM engine.
        
        Args:
            params: Request parameters
            
        Returns:
            dict: Response data
        """
        request_id = random_uuid()
        response_queue = queue.Queue()
        
        logger.info(f"Queuing request {request_id} of type {params.get('type', 'unknown')}")
        
        # Put the request in the queue
        self.request_queue.put((request_id, params, response_queue))
        
        # Wait for the response
        try:
            result = response_queue.get(timeout=300)  # 5-minute timeout
            logger.info(f"Got response for request {request_id}")
            return result
        except queue.Empty:
            logger.error(f"Timeout waiting for response to request {request_id}")
            return {"error": "Request timed out"}
    
    def stop(self):
        """Stop the adapter's worker thread."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)


class MoatlessVLLMRollout(BaseRollout):
    """
    Moatless Tree Search rollout that directly uses veRL's vLLM engine.
    """
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """
        Initialize the MoatlessVLLM rollout.
        
        Args:
            actor_module: The actor model module
            config: Configuration dictionary
            tokenizer: The tokenizer
            model_hf_config: HuggingFace model configuration
            **kwargs: Additional keyword arguments
        """
        if not HAS_VLLM:
            raise ImportError("vLLM is required for MoatlessVLLMRollout but not installed")
        
        super().__init__()
        self.actor_module = actor_module
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        
        # Initialize vLLM engine
        logger.info("Initializing vLLM engine for MoatlessVLLMRollout")
        self.vllm_rollout = vLLMRollout(actor_module, config, tokenizer, model_hf_config, **kwargs)
        self.llm_engine = self.vllm_rollout.inference_engine
        
        # Create the vLLM server adapter
        self.server_adapter = VLLMServerAdapter(self.llm_engine)
        
        # Moatless specific configuration
        self.moatless_path = kwargs.get('moatless_path', '/share/edc/home/antonis/_swe-planner/moatless-tree-search')
        self.output_dir = kwargs.get('output_dir', os.path.join(os.getcwd(), 'moatless_results'))
        self.model = kwargs.get('model', 'openai/Qwen/Qwen2.5-Coder-32B-Instruct')
        
        # Create model_path from model for use in moatless commands
        self.model_path = self.model  # This fixes the attribute error
        
        self.format = kwargs.get('format', 'window')
        self.max_iterations = kwargs.get('max_iterations', 500)
        self.max_expansions = kwargs.get('max_expansions', 5)
        self.feedback_type = kwargs.get('feedback_type', 'diff_agent')
        self.temp = kwargs.get('temp', 0.7)
        
        # Parse the instances if they come as a string (which seems to be the case from the error)
        instance_input = kwargs.get('instances', ['django__django-11179'])
        if isinstance(instance_input, str):
            try:
                # Try to parse as JSON if it's a string representation of a list
                import json
                self.instance_set = json.loads(instance_input)
            except json.JSONDecodeError:
                # If not valid JSON, treat as a single instance ID
                self.instance_set = [instance_input]
        else:
            self.instance_set = instance_input
            
        self.dataset_path = kwargs.get('dataset_path', None)
        
        # Create temporary directory for results if not using a pre-computed dataset
        if not self.dataset_path:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")
        else:
            logger.info(f"Using pre-computed dataset: {self.dataset_path}")
        
        # Set up environment for moatless-tools
        self._setup_env()
        
        # If using pre-computed dataset, load it
        if self.dataset_path:
            import pandas as pd
            self.dataset = pd.read_parquet(self.dataset_path)
            self.current_position = 0
            logger.info(f"Loaded {len(self.dataset)} examples from {self.dataset_path}")


    def _setup_env(self):
        """Set up environment variables for moatless-tools."""
        # Set dummy values for API endpoints - these are just placeholders
        os.environ["CUSTOM_LLM_API_BASE"] = "http://localhost:8000/v1"
        os.environ["CUSTOM_LLM_API_KEY"] = "not-needed"
        
        # Set Hugging Face environment variables
        os.environ["HUGGINGFACE_API_BASE"] = "http://localhost:8000/v1"
        os.environ["HUGGINGFACE_API_KEY"] = "not-needed"
        
        # Update PYTHONPATH
        os.environ["PYTHONPATH"] = f"{self.moatless_path}:{os.environ.get('PYTHONPATH', '')}"
        
        # Set a flag to indicate we're using the adapter
        os.environ["VERL_VLLM_ADAPTER"] = "1"
        
        # Patch moatless-tools to use our adapter
        self._patch_moatless()
    
    def _patch_moatless(self):
        """
        Patch moatless-tools to use our vLLM server adapter instead of making HTTP requests.
        """
        try:
            import litellm
            import types
            import functools
            
            # Store the original completion function
            original_completion = litellm.completion
            
            # Create a reference to self that can be used in the patched function
            adapter = self.server_adapter
            logger_ref = logger
            
            # Define a replacement function that uses our adapter
            def patched_completion(*args, **kwargs):
                model = kwargs.get("model", "")
                logger_ref.info(f"LiteLLM completion called with model: {model}")
                
                # Always intercept when VERL_VLLM_ADAPTER is enabled
                if os.environ.get("VERL_VLLM_ADAPTER") == "1":
                    logger_ref.info(f"Intercepting LiteLLM completion call for model {model}")
                    
                    # Extract parameters from kwargs
                    prompt = kwargs.get("prompt", "")
                    messages = kwargs.get("messages", None)
                    temperature = kwargs.get("temperature", 1.0)
                    top_p = kwargs.get("top_p", 1.0)
                    max_tokens = kwargs.get("max_tokens", 1024)
                    logprobs = kwargs.get("logprobs", None)
                    
                    # Determine request type based on input
                    if messages is not None:
                        # This is a chat completion request
                        adapter_params = {
                            "type": "chat_completions",
                            "model": model,  # Use the original model string
                            "messages": messages,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": max_tokens,
                            "logprobs": logprobs
                        }
                    else:
                        # This is a text completion request
                        adapter_params = {
                            "type": "completions",
                            "model": model,  # Use the original model string
                            "prompt": prompt,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": max_tokens,
                            "logprobs": logprobs
                        }
                    
                    # Make the request to our adapter
                    result = adapter.make_request(adapter_params)
                    
                    # Check for errors
                    if "error" in result:
                        logger_ref.error(f"Error in adapter request: {result['error']}")
                        # Return empty response to avoid breaking the pipeline
                        if messages is not None:
                            return {
                                "choices": [{
                                    "message": {"content": f"Error: {result['error']}"},
                                    "logprobs": None
                                }]
                            }
                        else:
                            return {
                                "choices": [{
                                    "text": f"Error: {result['error']}",
                                    "logprobs": None
                                }]
                            }
                    
                    # Return the appropriate result
                    if messages is not None:
                        chat_response = result.get("chat_completions", [{}])[0]
                        logger_ref.debug(f"Adapter request successful for model: {model}")
                        return chat_response
                    else:
                        completion_response = result.get("completions", [{}])[0]
                        logger_ref.debug(f"Adapter request successful for model: {model}")
                        return completion_response
                else:
                    # For requests we're not handling, use the original function
                    return original_completion(*args, **kwargs)
            
            # Apply the patch
            litellm.completion = patched_completion
            logger.info("Successfully patched litellm.completion")
            
        except ImportError:
            logger.warning("Could not patch litellm - it may not be installed")
    
    @contextmanager
    def update_sampling_params(self, **kwargs):
        """
        Context manager to temporarily update sampling parameters.
        
        Args:
            **kwargs: Sampling parameters to update
        """
        with self.vllm_rollout.update_sampling_params(**kwargs):
            yield
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using moatless-tree-search.
        
        Args:
            prompts: DataProto containing input prompts
            **kwargs: Additional keyword arguments
            
        Returns:
            DataProto: DataProto containing generated responses
        """
        # If using a pre-computed dataset, load from it
        if self.dataset_path:
            return self._generate_from_dataset(prompts, **kwargs)
        
        # Otherwise, run moatless-tree-search
        return self._generate_with_moatless(prompts, **kwargs)
    
    def _generate_from_dataset(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences by loading pre-computed trajectories.
        
        Args:
            prompts: DataProto containing input prompts
            **kwargs: Additional keyword arguments
            
        Returns:
            DataProto: DataProto containing loaded responses and logprobs
        """
        batch_size = len(prompts["prompt_tokens"])
        logger.info(f"Loading {batch_size} pre-computed trajectories from dataset")
        
        # Get the next batch of examples
        
        examples = self._get_next_batch(batch_size)
        
        # Convert to DataProto format
        response_tokens = []
        logprobs = []
        
        for example in examples:
            # Convert response tokens
            if 'response_tokens' in example:
                # If tokens are already stored as a list
                tokens = example['response_tokens']
                if isinstance(tokens, str):
                    # If tokens are stored as a string representation of a list
                    tokens = json.loads(tokens)
            else:
                # Convert text to tokens if needed
                response_text = example.get('response', '')
                tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
            
            response_tokens.append(torch.tensor(tokens))
            
            # Process logprobs
            if 'logprobs' in example:
                # Load logprobs depending on format
                if isinstance(example['logprobs'], str):
                    lp = json.loads(example['logprobs'])
                else:
                    lp = example['logprobs']
                
                # Convert to appropriate format
                if isinstance(lp, list):
                    # Pad logprobs to match token length if needed
                    if len(lp) < len(tokens):
                        lp.extend([0.0] * (len(tokens) - len(lp)))
                    elif len(lp) > len(tokens):
                        lp = lp[:len(tokens)]
                    
                    logprobs.append(torch.tensor(lp))
                else:
                    # Default logprob if not available
                    logprobs.append(torch.zeros(len(tokens)))
            else:
                # Default logprob if not available
                logprobs.append(torch.zeros(len(tokens)))
        
        # Create output DataProto
        output = DataProto()
        output["response_tokens"] = response_tokens
        output["logprobs"] = logprobs
        
        # Copy prompt information
        output["prompt_tokens"] = prompts.get("prompt_tokens", [])
        
        return output
    
    def _get_next_batch(self, batch_size: int) -> List[Dict]:
        """
        Get the next batch of examples from the dataset.
        
        Args:
            batch_size: Batch size
            
        Returns:
            List[Dict]: List of examples
        """
        # Get sequential indices
        start = self.current_position
        end = min(start + batch_size, len(self.dataset))
        
        # Reset if we reach the end
        if start >= len(self.dataset):
            start = 0
            end = min(batch_size, len(self.dataset))
            self.current_position = end
        else:
            self.current_position = end
            # Handle wraparound if batch extends beyond the end
            if end - start < batch_size:
                remaining = batch_size - (end - start)
                return [self.dataset.iloc[i].to_dict() for i in range(start, end)] + \
                       [self.dataset.iloc[i].to_dict() for i in range(min(remaining, len(self.dataset)))]
        
        return [self.dataset.iloc[i].to_dict() for i in range(start, end)]
    
    def _generate_with_moatless(self, prompts, instance_set=None):
        """
        Generate responses using moatless-tree-search on the given instances.
        
        Args:
            prompts: DataProto containing input_ids, attention_mask, etc.
            instance_set: List of instance IDs to run evaluation on
        
        Returns:
            TensorDict containing the generated responses
        """
        batch_size = prompts.batch.batch_size[0]
        
        # Select instances to use
        instances = self._select_instances(batch_size)
        
        # Run moatless for each instance
        trajectory_paths = []
        for instance_id in instances:
            trajectory_path = self._run_moatless_for_instance(instance_id)
            if trajectory_path:
                trajectory_paths.append(trajectory_path)
        
        # Build prompts from trajectories
        prompts_path = self._build_prompts_from_trajectories()
        
        if prompts_path and os.path.exists(prompts_path):
            # Load the generated prompts
            import pandas as pd
            df = pd.read_parquet(prompts_path)
            
            # Convert to appropriate format for return
            responses = []
            for i in range(batch_size):
                # In case we generated fewer trajectories than the batch size
                index = i % len(df)
                responses.append(df.iloc[index]["response"])
            
            # Create a TensorDict with the responses
            import torch
            from tensordict import TensorDict
            
            response_tensor = torch.tensor(
                [self.tokenizer.encode(r) for r in responses], 
                dtype=torch.long
            )
            
            result = TensorDict({
                "output_ids": response_tensor,
            }, batch_size=[batch_size])
            
            return result
        
        # Fallback in case of errors
        return TensorDict({"output_ids": torch.zeros((batch_size, 1), dtype=torch.long)}, batch_size=[batch_size])
        
    def _select_instances(self, batch_size):
        """
        Select instance_ids to use for this batch.
        
        Args:
            batch_size: Number of instances needed
        
        Returns:
            List of instance IDs to use
        """
        # If no instances available, return empty list
        if not hasattr(self, 'instance_set') or not self.instance_set:
            return []
        
        # Repeat instances if needed to match batch_size
        instances = []
        for i in range(batch_size):
            instances.append(self.instance_set[i % len(self.instance_set)])
        
        return instances
    
    def _run_moatless_for_instance(self, instance_id, **kwargs):
        """Run moatless-tree-search for a specific SWE-bench instance."""
        # Create output directory structure
        eval_output_dir = os.path.join(self.moatless_path, "evaluations", "verl_integration", "verl_integration")
        os.makedirs(eval_output_dir, exist_ok=True)
        
        # Create a unique worker ID by combining process ID and timestamp
        worker_id = f"{os.getpid()}_{time.time()}"
        worker_repo_dir = os.path.join(self.moatless_path, "repos", f"worker_{worker_id}")
        os.makedirs(worker_repo_dir, exist_ok=True)
        
        # Determine if model_path is a local path or a model identifier
        model_arg = self.model_path
        if os.path.exists(self.model_path):
            # It's a local path, use it directly
            model_arg = self.model_path
        else:
            # It might be an identifier like 'openai/Qwen/Qwen2.5-0.5B'
            # For local model path, we might need to extract just the model name
            if '/' in self.model_path:
                model_parts = self.model_path.split('/')
                if len(model_parts) > 1 and model_parts[0] in ['openai', 'anthropic']:
                    # This looks like an API model identifier, leave it as is
                    model_arg = self.model_path
                else:
                    # This might be a local path with slashes, take the last part
                    model_arg = model_parts[-1]
        
        # Add environment variables to influence Moatless repository handling
        env = os.environ.copy()
        env["MOATLESS_USE_READ_ONLY_REPOS"] = "1"
        env["MOATLESS_SHARED_REPO_DIR"] = "/path/to/shared/repos"
        
        # Set up the command with the correct arguments
        command = [
            "python", f"{self.moatless_path}/moatless/benchmark/run_evaluation.py",
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--split", "verified",
            "--model", model_arg,
            "--repo_base_dir", worker_repo_dir,
            "--eval_dir", f"{self.moatless_path}/evaluations/verl_integration",
            "--eval_name", "verl_integration",
            "--temp", str(self.temp),
            "--num_workers", "1",
            "--format", self.format,
            "--max_iterations", str(self.max_iterations),
            "--max_expansions", str(self.max_expansions),
            "--reward_threshold", "101",
            "--max_finished_nodes", "5",
            "--use_edit_actions",
            "--feedback_type", self.feedback_type,
            "--feedback",
            "--selector_type", "depth_first",
            "--max_trajectory_depth", "20",
            "--use_testbed",
            "--overwrite",
            "--instance_ids", instance_id  
        ]
        
        # Determine lock file path
        repo_name = f"swe-bench_{instance_id.split('__')[0]}__{instance_id.split('__')[1].split('-')[0]}"
        lock_path = os.path.join(self.moatless_path, "repos", f"{repo_name}.lock")
        
        # Acquire lock before accessing repository
        with filelock.FileLock(lock_path, timeout=300):  # 5-minute timeout
            # Run the evaluation with exclusive access to repository
            try:
                # Run with environment variables
                subprocess.run(command, check=True, env=env)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running moatless for {instance_id}: {e}")
                return None
        
        # The expected output location by moatless's default behavior
        instance_dir = os.path.join(eval_output_dir, instance_id)
        trajectory_path = os.path.join(instance_dir, "trajectory.json")
        
        # Verify the output exists
        if os.path.exists(trajectory_path):
            logger.info(f"Successfully generated trajectory at {trajectory_path}")
            return trajectory_path
        else:
            logger.error(f"Trajectory file not found at {trajectory_path}")
            return None

    def _is_valid_repo(self, path):
        """Check if a repository exists and has a valid structure"""
        if not os.path.exists(path):
            return False
        
        try:
            # Check for .git directory and branches
            result = subprocess.run(
                ["git", "-C", path, "branch"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            return result.returncode == 0
        except:
            return False

    def _get_litellm_model_path(self, model_path):
        """
        Convert a model path to the format expected by LiteLLM.
        
        Args:
            model_path: Original model path
            
        Returns:
            str: Model path formatted for LiteLLM
        """
        # If it's already prefixed, return as is
        if model_path.startswith('openai/') or model_path.startswith('anthropic/') or model_path.startswith('huggingface/'):
            return model_path
        
        # If it's a local path, add huggingface/ prefix
        if model_path.startswith('/'):
            return f"huggingface/{model_path}"
        
        # Default to openai/ prefix for other cases
        return f"openai/{model_path}"
