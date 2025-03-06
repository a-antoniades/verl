import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def print_completion(completion: Dict[str, Any]) -> None:
    """Print completion information in a structured format."""
    if not completion:
        return
        
    print("\nModel Information:")
    print(json.dumps({
        "model": completion.get("model"),
        "usage": completion.get("usage")
    }, indent=2))
    
    if completion.get("input"):
        print("\nInput prompts:")
        for input_idx, input_msg in enumerate(completion["input"]):
            try:
                if "content" in input_msg:
                    content = input_msg["content"]
                    if isinstance(content, list) and input_msg["role"] == "user":
                        content = "\n\n".join(c.get("content") or c.get("text") for c in content)
                    elif not isinstance(content, str):
                        content = json.dumps(content, indent=2)

                    print(f"\nMessage {input_idx + 1} by {input_msg['role']}:")
                    print(content)
                else:
                    print(f"\nMessage {input_idx + 1} by {input_msg['role']}:")
                    print(json.dumps(input_msg, indent=2))
            except Exception as e:
                logger.exception(f"Failed to parse message: {json.dumps(input_msg, indent=2)}")
                print(f"Error parsing message: {str(e)}")

    if completion.get("response"):
        print("\nCompletion response:")
        print(json.dumps(completion["response"], indent=2))

def load_trajectory(file_path: str) -> Optional[Dict[str, Any]]:
    """Load trajectory data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Failed to load trajectory from {file_path}")
        print(f"Error loading trajectory: {str(e)}")
        return None

def print_build_info(node_data: Dict[str, Any]) -> None:
    """Print build information for a specific node."""
    if not node_data.get("completions", {}).get("build_action"):
        print("No build action completion data available for this node.")
        return
        
    build_completion = node_data["completions"]["build_action"]
    print_completion(build_completion)

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <trajectory_file_path>")
        return
        
    trajectory_path = sys.argv[1]
    trajectory_data = load_trajectory(trajectory_path)
    
    if not trajectory_data:
        return
        
    # Print build info for all nodes
    nodes = trajectory_data.get("nodes", [])
    if not nodes:
        print("No nodes found in trajectory")
        return
        
    for i, node in enumerate(nodes):
        print(f"\n{'='*80}")
        print(f"Node {node.get`('node_id', i)}")
        print('='*80)
        print_build_info(node)

if __name__ == "__main__":
    main()