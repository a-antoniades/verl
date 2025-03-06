from litellm import completion
import math
import os


# Define the API base URL and model name
api_base = "http://avior.mlfoundry.com/live-inference/v1"
api_key = os.getenv("CUSTOM_LLM_API_KEY")
model_name = "openai/Qwen/Qwen2.5-Coder-32B-Instruct"
# model_name = "openai/nvidia/OpenMath2-Llama3.1-8B"

# Prepare the message for the model
messages = [{"role": "user", "content": "What is the capital of France?"}]

# Make the completion call
response = completion(
    model=model_name,
    messages=messages,
    api_base=api_base,
    api_key=api_key,
    logprobs=True,
    top_logprobs=10
)

# Print the entire response for debugging
print("Full response:", response)

# Print the response content
print(response['choices'][0]['message']['content'])
choice_logprobs = response['choices'][0]['logprobs']

# Check if 'content' key exists in choice_logprobs
if 'content' in choice_logprobs:
    token_logprobs = choice_logprobs['content']

    for i, token_info in enumerate(token_logprobs):
        token = token_info['token']
        logprob = token_info['logprob']
        top_candidates = token_info['top_logprobs']
        
        print(f"Token {i + 1}: '{token}' (Log Probability: {logprob})")
        print("Top potential tokens and their probabilities:")

        for candidate_info in top_candidates:
            candidate = candidate_info.token  # Accessing attribute directly
            candidate_logprob = candidate_info.logprob  # Accessing attribute directly
            probability = math.exp(candidate_logprob)  # Convert log probability to probability
            print(f"  Token: '{candidate}', Probability: {probability:.4f}")

        print("\n" + "-"*50 + "\n")
else:
    print("The 'content' key is not present in the logprobs.")

#exit()