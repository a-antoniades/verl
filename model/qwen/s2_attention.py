import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
import math
from transformers import AutoModelForCausalLM
from flash_attn.bert_padding import pad_input, unpad_input
import logging

logger = logging.getLogger(__name__)

class S2Attention(Qwen2SdpaAttention):
    """Memory-efficient Shift Short Attention implementation for Qwen2"""
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.shift_size = config.hidden_size // 4
        self.window_size = getattr(config, 'window_size', 256)
        self.use_flash_attn = True  # Enable by default, will fallback if not available
        self.layer_idx = layer_idx  # Store layer_idx
        logger.info(f"Initialized S2-Attention with window_size={self.window_size}, shift_size={self.shift_size}")
        
    def _shift_blocks(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Memory-efficient block shifting with optional unpadding."""
        batch_size, seq_len, hidden_size = x.shape
        
        # Handle unpadded sequence if mask is provided
        if attention_mask is not None:
            x_unpad, indices, _ = unpad_input(x, attention_mask)
            
            # Reshape and shift unpadded sequence
            block_size = self.window_size
            pad_len = (block_size - x_unpad.size(0) % block_size) % block_size
            if pad_len > 0:
                x_unpad = F.pad(x_unpad, (0, 0, 0, pad_len))
            
            # Perform shift on unpadded data
            blocks = x_unpad.size(0) // block_size
            x_unpad = x_unpad.view(blocks, block_size, hidden_size)
            x_unpad = torch.roll(x_unpad, shifts=-1, dims=0)
            x_unpad = x_unpad.view(-1, hidden_size)
            
            # Remove padding and pad back to original shape
            if pad_len > 0:
                x_unpad = x_unpad[:-pad_len]
            x = pad_input(x_unpad.unsqueeze(-1), indices, batch_size, seq_len)
            return x.squeeze(-1)
        
        # Standard block shifting for padded sequences
        block_size = self.window_size
        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        blocks = (seq_len + pad_len) // block_size
        x = x.view(batch_size, blocks, block_size, hidden_size)
        x = torch.roll(x, shifts=-1, dims=1)
        x = x.view(batch_size, -1, hidden_size)
        
        if pad_len > 0:
            x = x[:, :seq_len, :]
        
        return x
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        
        # Add memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / 1024**2

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            batch_size, seq_length, _ = hidden_states.shape
            
            # Apply shifting with optional unpadding
            shifted_hidden_states = self._shift_blocks(hidden_states, attention_mask)
            
            def process_branch(branch_hidden_states):
                # Process in chunks to save memory
                chunk_size = min(seq_length, 2048)
                queries, keys, values = [], [], []
                
                for chunk_start in range(0, seq_length, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, seq_length)
                    chunk = branch_hidden_states[:, chunk_start:chunk_end]
                    
                    q = self.q_proj(chunk)
                    k = self.k_proj(chunk)
                    v = self.v_proj(chunk)
                    
                    q = q.view(batch_size, -1, self.num_heads, self.head_dim)
                    k = k.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
                    v = v.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
                    
                    if position_ids is not None:
                        chunk_positions = position_ids[:, chunk_start:chunk_end]
                        q, k = self.rotary_emb(q, k, position_ids=chunk_positions)
                    
                    queries.append(q)
                    keys.append(k)
                    values.append(v)
                
                return torch.cat(queries, dim=1), torch.cat(keys, dim=1), torch.cat(values, dim=1)
            
            # Process both branches
            query1, key1, value1 = process_branch(hidden_states)
            query2, key2, value2 = process_branch(shifted_hidden_states)
            
            # Combine branches efficiently
            query = torch.cat([query1, query2], dim=2)  # Along heads dimension
            key = torch.cat([key1, key2], dim=2)
            value = torch.cat([value1, value2], dim=2)
            
            # Use flash attention when possible
            if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention') and attention_mask is None:
                attn_output = F.scaled_dot_product_attention(
                    query, key, value,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True
                )
            else:
                # Fallback to regular attention with memory-efficient computation
                scale = 1 / math.sqrt(self.head_dim)
                
                # Compute attention scores in chunks
                attn_output = torch.zeros_like(query)
                chunk_size = min(seq_length, 1024)
                
                for chunk_start in range(0, seq_length, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, seq_length)
                    q_chunk = query[:, chunk_start:chunk_end]
                    
                    # Compute attention scores for this chunk
                    attn_weights = torch.matmul(q_chunk, key.transpose(-2, -1)) * scale
                    
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask[:, :, chunk_start:chunk_end]
                    
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                    attn_weights = attn_weights.to(query.dtype)
                    
                    if self.training and self.attention_dropout > 0:
                        attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
                    
                    attn_output[:, chunk_start:chunk_end] = torch.matmul(attn_weights, value)
            
            # Final output projection
            attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                curr_mem = torch.cuda.memory_allocated() / 1024**2
                logger.info(f"S2-Attention memory usage: Start={start_mem:.1f}MB, Peak={peak_mem:.1f}MB, Current={curr_mem:.1f}MB")

            return attn_output, None, None

def replace_attention_with_s2(model):
    """Replace standard attention with S2-Attention"""
    count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        # Create new S2-Attention instance with layer_idx
        s2_attn = S2Attention(model.config, layer_idx=layer_idx)
        
        # Get device and dtype from the parameters instead of the module
        device = next(layer.self_attn.parameters()).device
        dtype = next(layer.self_attn.parameters()).dtype
        
        # Move to same device and dtype as original attention
        s2_attn = s2_attn.to(device=device, dtype=dtype)
        
        # Copy weights from original attention
        s2_attn.load_state_dict(layer.self_attn.state_dict())
        
        # Replace attention module
        layer.self_attn = s2_attn
        count += 1
    
    logger.info(f"Replaced {count} attention layers with S2-Attention")
    return model

def test_s2_attention():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = replace_attention_with_s2(model)
    print("Model structure after S2-Attention replacement:")
    print(model)
    print(f"Successfully replaced attention with memory-efficient S2-Attention")

if __name__ == "__main__":
    test_s2_attention()