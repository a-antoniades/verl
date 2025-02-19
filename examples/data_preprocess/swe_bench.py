import json
import os
from pathlib import Path
import pandas as pd

def format_node_data(instance_dir, output_dir=None, log_dir="/tmp/swe-bench-logs"):
    """
    Reformats node data into a structured format for training and saves as train.parquet.
    
    Args:
        instance_dir (str): Directory containing trajectory_inputs.json and trajectory.json
        output_dir (str, optional): Directory to save the formatted data
        log_dir (str, optional): Directory for testbed logs
    """
    # Load both JSON files
    with open(os.path.join(instance_dir, 'trajectory_inputs.json'), 'r') as f:
        inputs_data = json.load(f)
    
    with open(os.path.join(instance_dir, 'trajectory.json'), 'r') as f:
        trajectory_data = json.load(f)
    
    # Extract metadata
    instance_id = trajectory_data['metadata']['instance_id']
    evaluation_name = trajectory_data['metadata']['evaluation_name']
    
    formatted_data = []
    
    for node_key, messages in inputs_data.items():
        prompt_messages = []
        for msg in messages:
            cleaned_msg = {
                'role': msg['role'],
                'content': msg['content']
            }
            prompt_messages.append(cleaned_msg)
        
        entry = {
            'data_source': 'swe-bench',
            'prompt': prompt_messages,
            'ability': 'code',
            'reward_model': {
                'ground_truth': 'placeholder_answer',
                'style': 'rule'
            },
            'extra_info': {
                'answer': 'placeholder_answer',
                'index': 0,
                'question': prompt_messages[-1]['content'],
                'split': 'train',
                'instance_id': instance_id,  # Add instance_id here
                'evaluation_name': evaluation_name  # Optionally include evaluation name
            }
        }
        
        formatted_data.append(entry)
    
    df = pd.DataFrame(formatted_data)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'train.parquet')
        df.to_parquet(output_path, index=False)
        print(f"Saved formatted data to: {output_path}")
        print(f"Processed {len(formatted_data)} entries for instance {instance_id}")
    
    return df

def inspect_parquet_data(parquet_path, row_index=0):
    """
    Load parquet file and print contents of a single row.
    
    Args:
        parquet_path (str): Path to the parquet file
        row_index (int): Index of row to inspect (default: 0)
    """
    # Load the parquet file
    df = pd.read_parquet(parquet_path)
    
    print(f"Total rows in dataset: {len(df)}\n")
    print(f"Inspecting row {row_index}:\n")
    
    # Get the specified row
    row = df.iloc[row_index]
    
    # Print each column's value
    for i, column in enumerate(df.columns):
        print(f"Column {i}: {column}")
        print(f"{row[column]}\n")

# Example usage:
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    
    instance_dir = "/share/edc/home/antonis/_swe-planner/moatless-tree-search/evaluations/swe-1/debug/swe-1/debug/swe-gym/django__django-16255/"
    output_dir = os.path.join(ROOT_DIR, 'data', 'swe-bench')
    log_dir = os.path.join(ROOT_DIR, 'logs', 'swe-bench')
    
    df = format_node_data(instance_dir, output_dir=output_dir, log_dir=log_dir)
    print(f"df: {df}")

    # Print first instance columns (for verification)
    if len(df) > 0:
        print("\nFirst instance columns:")
        for column in df.columns:
            print(f"\n{column}:")
            print(df.iloc[0][column])
            
    # Add verification of saved parquet
    print("\nVerifying saved parquet file:")
    saved_df = pd.read_parquet(os.path.join(output_dir, 'train.parquet'))
    print("\nFirst row extra_info from saved parquet:")
    print(saved_df.iloc[0]['extra_info'])