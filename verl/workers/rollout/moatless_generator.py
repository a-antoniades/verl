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
Moatless Tree Search data generator for VERL
"""
import os
import subprocess
import tempfile
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MoatlessDataGenerator:
    """
    Generates data using moatless-tree-search for VERL training.
    """
    def __init__(
        self,
        moatless_path: str,
        output_dir: str,
        model: str = "openai/Qwen/Qwen2.5-Coder-32B-Instruct",
        format: str = "window",
        max_iterations: int = 500,
        max_expansions: int = 5,
        feedback_type: str = "diff_agent",
        temp: float = 0.7,
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "verified",
    ):
        """
        Initialize the Moatless data generator.
        
        Args:
            moatless_path: Path to moatless-tree-search
            output_dir: Directory to store the output data
            model: LLM model to use
            format: Format for moatless (window, etc.)
            max_iterations: Maximum iterations for tree search
            max_expansions: Maximum expansions for tree search
            feedback_type: Feedback type for moatless
            temp: Temperature for generation
            dataset_name: Dataset name for SWE-bench
            split: Dataset split
        """
        self.moatless_path = moatless_path
        self.output_dir = output_dir
        self.model = model
        self.format = format
        self.max_iterations = max_iterations
        self.max_expansions = max_expansions
        self.feedback_type = feedback_type
        self.temp = temp
        self.dataset_name = dataset_name
        self.split = split
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up moatless environment variables
        self.setup_env()
        
    def setup_env(self):
        """Set up environment variables for moatless."""
        os.environ["CUSTOM_LLM_API_BASE"] = "http://localhost:8000/v1"
        os.environ["CUSTOM_LLM_API_KEY"] = "not-needed"
        os.environ["PYTHONPATH"] = f"{self.moatless_path}:{os.environ.get('PYTHONPATH', '')}"
    
    def generate_data(self, instances: List[str], enable_logprobs: bool = True):
        """
        Generate data for the specified instances.
        
        Args:
            instances: List of repository instances (e.g., ["django__django-11179"])
            enable_logprobs: Whether to enable logging probabilities
        """
        for instance in instances:
            output_path = os.path.join(self.output_dir, f"{instance}.json")
            logger.info(f"Generating data for {instance}, output to {output_path}")
            
            # Run moatless-tree-search for this instance
            self._run_moatless_for_instance(instance, output_path, enable_logprobs)
    
    def _run_moatless_for_instance(self, instance: str, output_path: str, enable_logprobs: bool):
        """
        Run moatless-tree-search for a specific instance.
        
        Args:
            instance: Repository instance ID
            output_path: Path to save the output
            enable_logprobs: Whether to enable logging probabilities
        """
        # Build command
        cmd = [
            "python", f"{self.moatless_path}/moatless/benchmark/run_evaluation.py",
            "--dataset_name", self.dataset_name,
            "--split", self.split,
            "--model", self.model,
            "--repo_base_dir", f"{self.moatless_path}/repos",
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
            "--instance_id", instance,
            "--output_json", output_path
        ]
        
        # Add logprobs if enabled
        if enable_logprobs:
            cmd.append("--enable_logprobs")
        
        # Run the command
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.moatless_path)
            logger.info(f"Successfully generated data for {instance}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running moatless for {instance}: {e}")
    
    def build_dataset(self, output_parquet: str):
        """
        Build a dataset from the generated trajectories using build_prompts.py.
        
        Args:
            output_parquet: Path to save the output parquet file
        """
        # Build command to run build_prompts.py
        cmd = [
            "python", f"{self.moatless_path}/build_prompts.py",
            "--data_dir", self.output_dir,
            "--output", output_parquet
        ]
        
        # Run the command
        try:
            logger.info(f"Building dataset with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.moatless_path)
            logger.info(f"Successfully built dataset at {output_parquet}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building dataset: {e}")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate data using moatless-tree-search for VERL training")
    parser.add_argument("--moatless_path", type=str, required=True, help="Path to moatless-tree-search")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output data")
    parser.add_argument("--instances", type=str, nargs="+", required=True, help="Repository instances to run")
    parser.add_argument("--model", type=str, default="openai/Qwen/Qwen2.5-Coder-32B-Instruct", help="LLM model to use")
    parser.add_argument("--format", type=str, default="window", help="Format for moatless")
    parser.add_argument("--output_parquet", type=str, help="Output path for the parquet dataset")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create generator
    generator = MoatlessDataGenerator(
        moatless_path=args.moatless_path,
        output_dir=args.output_dir,
        model=args.model,
        format=args.format
    )
    
    # Generate data
    generator.generate_data(args.instances)
    
    # Build dataset if output_parquet is specified
    if args.output_parquet:
        generator.build_dataset(args.output_parquet)

if __name__ == "__main__":
    main() 