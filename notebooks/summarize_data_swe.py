import pandas as pd
from litellm import completion
import os
import json
import numpy as np
import tiktoken

# Load the raw parquet data
parquet_path = "/share/edc/home/antonis/swe-gym-setup/verl/data/swe-bench/swe-verifier-50/train.parquet"
raw_data = pd.read_parquet(parquet_path)
df = raw_data
print(f"Data loaded. Number of rows: {len(df)}")

# Get the first row
first_row = df.iloc[0]
print(f"first row keys: {first_row.keys()}")

# Define the API base URL and model name
api_base = "http://avior.mlfoundry.com/live-inference/v1"
api_key = os.getenv("CUSTOM_LLM_API_KEY", "default_api_key")
model_name = "openai/Qwen/Qwen2.5-Coder-32B-Instruct"

# Initialize tokenizer
encoding = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding

# Check the type of the prompt field
print(f"Type of prompt field: {type(first_row['prompt'])}")
print(f"First few characters of prompt: {str(first_row['prompt'])[:100]}")

# Convert numpy array to list if needed
prompt_data = first_row['prompt']
if isinstance(prompt_data, np.ndarray):
    prompt_data = prompt_data.tolist()
    print("Converted numpy array to list")

# Now check if prompt_data is a list
if isinstance(prompt_data, list):
    print(f"Number of messages in prompt: {len(prompt_data)}")
    
    # Print the roles of each message
    for i, msg in enumerate(prompt_data):
        if isinstance(msg, dict) and 'role' in msg:
            content_preview = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
            print(f"Message {i+1}: {msg['role']} - {content_preview}")
    
    # Calculate token count for full content
    full_content = ""
    for msg in prompt_data:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            full_content += f"{msg['role'].upper()}: {msg['content']}\n\n"
    
    full_token_count = len(encoding.encode(full_content))
    print(f"\nFull content token count: {full_token_count}")
    
    # Prepare the prompt for summarization with truncated content
    summary_prompt = "Please summarize the following conversation thread from a coding assistant:\n\n"
    truncated_content = ""
    for msg in prompt_data:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            truncated_msg = f"{msg['role'].upper()}: {msg['content'][:200]}...\n\n"
            truncated_content += truncated_msg
            summary_prompt += truncated_msg
    
    truncated_token_count = len(encoding.encode(truncated_content))
    print(f"Truncated content token count: {truncated_token_count}")
    print(f"Token reduction: {full_token_count - truncated_token_count} tokens ({(1 - truncated_token_count/full_token_count)*100:.2f}%)")
    
    # Get summary from the model
    print("\nGenerating summary...")
    messages = [{"role": "user", "content": summary_prompt}]
    try:
        response = completion(
            model=model_name,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            max_tokens=500
        )
        summary = response['choices'][0]['message']['content']
        summary_token_count = len(encoding.encode(summary))
        print(f"\nSummary token count: {summary_token_count}")
        print(f"\nSummary of the first row's prompt:\n{summary}")
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
else:
    print(f"Prompt data is not a list. Type: {type(prompt_data)}")
    # If it's a dictionary, try to extract messages
    if isinstance(prompt_data, dict) and 'messages' in prompt_data:
        messages = prompt_data['messages']
        print(f"Found messages in prompt dictionary. Count: {len(messages)}")
    else:
        print("Could not find a list of messages in the prompt data")