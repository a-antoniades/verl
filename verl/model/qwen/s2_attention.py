import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
import math
from flash_attn.bert_padding import pad_input, unpad_input

class S2Attention(Qwen2SdpaAttention):
    """Memory-efficient S2-Attention implementation for Qwen2"""
    
    def __init__(self, config):
        super().__init__(config)
        self.shift_size = config.hidden_size // 4
        self.window_size = getattr(config, 'window_size', 256)
        
    def _shift_blocks(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """Memory-efficient block shifting"""
        batch_size, seq_len, hidden_size = x.shape
        
        # Unpad if attention mask is provided
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
        
        # If no attention mask, process normally but in chunks
        block_size = self.window_size
        chunks = []
        for i in range(0, seq_len, block_size):
            chunk = x[:, i:min(i+block_size, seq_len), :]
            chunks.append(chunk)
        
        # Shift chunks
        shifted_chunks = chunks[1:] + [chunks[0]]
        return torch.cat(shifted_chunks, dim=1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        batch_size, seq_length, _ = hidden_states.shape

        # Process in half precision
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Apply shifting to create two branches
            shifted_hidden_states = self._shift_blocks(hidden_states, attention_mask)
            
            def process_branch(branch_hidden_states):
                # Unpad if attention mask is provided
                if attention_mask is not None and self.use_remove_padding:
                    branch_unpad, indices, _ = unpad_input(branch_hidden_states, attention_mask)
                    
                    # Process unpadded sequence
                    query = self.q_proj(branch_unpad)
                    key = self.k_proj(branch_unpad)
                    value = self.v_proj(branch_unpad)
                    
                    # Reshape for attention
                    query = query.view(-1, self.num_heads, self.head_dim)
                    key = key.view(-1, self.num_key_value_heads, self.head_dim)
                    value = value.view(-1, self.num_key_value_heads, self.head_dim)
                else:
                    # Process normally
                    query = self.q_proj(branch_hidden_states)
                    key = self.k_proj(branch_hidden_states)
                    value = self.v_proj(branch_hidden_states)
                    
                    query = query.view(batch_size, -1, self.num_heads, self.head_dim)
                    key = key.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
                    value = value.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
                
                # Apply rotary embeddings
                query, key = self.rotary_emb(query, key, position_ids=position_ids)
                
                return query, key, value, indices if attention_mask is not None else None
            
            # Process both branches
            q1, k1, v1, indices = process_branch(hidden_states)
            q2, k2, v2, _ = process_branch(shifted_hidden_states)
            
            # Combine branches efficiently
            query = torch.cat([q1, q2], dim=-2)  # Concatenate along sequence length
            key = torch.cat([k1, k2], dim=-2)
            value = torch.cat([v1, v2], dim=-2)
            
            # Compute attention with flash attention when possible
            if hasattr(F, 'scaled_dot_product_attention') and attention_mask is None:
                attn_output = F.scaled_dot_product_attention(
                    query, key, value,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True
                )
            else:
                # Fall back to regular attention with mask
                attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
                attn_output = torch.matmul(attn_weights, value)
            
            # Reshape and project output
            attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            return attn_output, None, None 