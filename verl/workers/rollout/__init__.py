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

from .base import BaseRollout
from .naive import NaiveRollout
from .hf_rollout import HFRollout

__all__ = ["BaseRollout", "NaiveRollout", "HFRollout"]

# Register the rollout types
def get_rollout_cls(name):
    rollout_map = _get_rollout_map()
    rollout_info = rollout_map.get(name, None)
    if rollout_info is None:
        raise ValueError(f"Rollout {name} not found")
    
    from importlib import import_module
    module_path, class_name = rollout_info["cls"].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def _get_rollout_map():
    return {
        "naive": {"cls": "verl.workers.rollout.naive.NaiveRollout"},
        "hf": {"cls": "verl.workers.rollout.hf_rollout.hf_rollout.HFRollout"},
        "moatless_vllm": {"cls": "verl.workers.rollout.moatless_vllm_rollout.MoatlessVLLMRollout"},
        # Add vLLM if it's available
        "vllm": {"cls": "verl.workers.rollout.vllm_rollout.vllm_rollout.vLLMRollout"},
    }
