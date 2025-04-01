import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd
import os
def analyze_token_distribution(parquet_files, tokenizer_name="gpt2", 
                               prompt_key="prompt",
                               filename="token-distribution.png"):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Read and combine dataframes
    dataframes = []
    for parquet_file in parquet_files:
        dataframe = pd.read_parquet(parquet_file)
        dataframes.append(dataframe)
    df = pd.concat(dataframes)
    
    # Calculate token lengths for each prompt
    token_lengths = []
    for _, row in df.iterrows():
        chat = row[prompt_key]
        prompt_with_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        token_length = len(prompt_with_template)
        token_lengths.append(token_length)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, edgecolor='black')
    plt.title('Distribution of Token Lengths in Dataset')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    # Add some statistics as text
    stats_text = f'Mean: {np.mean(token_lengths):.1f}\n'
    stats_text += f'Median: {np.median(token_lengths):.1f}\n'
    stats_text += f'Max: {np.max(token_lengths)}\n'
    stats_text += f'Min: {np.min(token_lengths)}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(os.path.dirname(parquet_files[0]), filename))
    plt.show()
    print(f"Token distribution saved to {os.path.join(os.path.dirname(parquet_files[0]), filename)}")

# Example usage:
# parquet_files = ["path/to/your/data.parquet"]
# analyze_token_distribution(parquet_files, tokenizer_name="your_tokenizer_name")

if __name__ == "__main__":
    parquet_files = ["/share/edc/home/antonis/swe-gym-setup/verl/data/swe-bench/swe-verifier-50/no-feedback/train_debug.parquet"]
    analyze_token_distribution(parquet_files, tokenizer_name="Qwen/Qwen2.5-0.5B")
