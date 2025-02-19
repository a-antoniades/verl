import torch
from transformers import AutoModelForCausalLM, AutoConfig
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_qwen_model():
    model_path = "/share/edc/home/antonis/weights/huggingface/models--Qwen--Qwen2.5-0.5B"
    
    # Load config first
    logger.info("Loading model config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"\nConfig:\n{config}")
    
    # Load the model
    logger.info("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Inspect the first transformer layer's attention
    logger.info("\nInspecting attention implementation...")
    first_layer = model.model.layers[0]
    
    # Print the layer's structure
    logger.info(f"\nLayer structure:")
    logger.info(f"Available attributes: {dir(first_layer)}")
    
    # Try to find attention module
    for name, module in first_layer.named_children():
        logger.info(f"\nFound module: {name}")
        logger.info(f"Type: {type(module)}")
        logger.info(f"Structure: {module}")
    
    # Print attention-specific parameters
    logger.info("\nAttention-related parameters in first layer:")
    for name, param in first_layer.named_parameters():
        if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention']):
            logger.info(f"{name}: {param.shape}")
    
    # Print key attention configuration
    logger.info("\nAttention configuration:")
    logger.info(f"Hidden size: {config.hidden_size}")
    logger.info(f"Num attention heads: {config.num_attention_heads}")
    logger.info(f"Num KV heads: {config.num_key_value_heads}")
    logger.info(f"Head dimension: {config.hidden_size // config.num_attention_heads}")
    logger.info(f"Max position embeddings: {config.max_position_embeddings}")
    if hasattr(config, 'use_sliding_window'):
        logger.info(f"Using sliding window: {config.use_sliding_window}")
        logger.info(f"Window size (if applicable): {config.sliding_window}")
    
    return model, config

if __name__ == "__main__":
    model, config = inspect_qwen_model()