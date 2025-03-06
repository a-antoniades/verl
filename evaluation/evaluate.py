#!/usr/bin/env python3

import os
import argparse
import re
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_final_answer(text):
    """
    Extract the final numerical answer from the text.
    This function looks for the last number in the text.
    """
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        return matches[-1].strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face model on GSM8K.")
    parser.add_argument("--model", type=str, default="gpt2", 
                        help="Hugging Face model name or local path.")
    parser.add_argument("--max_samples", type=int, default=50, 
                        help="Max number of test samples to evaluate for quick testing.")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate for each answer.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: 'cuda' or 'cpu'.")
    args = parser.parse_args()

    # Load GSM8K dataset (test split)
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]
    log_file = os.path.join(os.path.dirname(__file__), "gsm8k", f"{args.model}_results.json")

    # Optionally limit the number of samples (for experimentation)
    test_data = test_data.select(range(min(len(test_data), args.max_samples)))

    # Load model & tokenizer
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)

    # Evaluate
    correct = 0

    for idx, sample in enumerate(test_data):
        question, target_answer = sample["question"], sample["answer"]

        # Create a simple prompt. You can improve prompt formatting for better performance.
        prompt = question + "\nThe answer is"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the predicted answer
        predicted_answer = extract_final_answer(generated_text)
        target_final_answer = extract_final_answer(target_answer)

        # Determine if the prediction is correct
        if predicted_answer == target_final_answer:
            correct += 1

    accuracy = correct / len(test_data)
    print(f"Accuracy on {len(test_data)} samples: {accuracy:.3f}")

    # Write the final accuracy score to a JSON file
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    with open(log_file, 'w') as f:
        json.dump({"accuracy": accuracy}, f, indent=4)
        print(f"Accuracy saved to {log_file}")

if __name__ == "__main__":
    main()