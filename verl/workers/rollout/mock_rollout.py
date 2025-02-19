import torch
import pandas as pd
import os
import ast
from verl.protocol import DataProto
from verl.workers.rollout import BaseRollout
from tensordict import TensorDict

class MockRollout(BaseRollout):
    """
    A mock rollout class that pulls saved trajectories from a local DataFrame.
    This can be used for debugging/integration tests without requiring calling agent.
    """

    def __init__(self, module: torch.nn.Module, config):
        """
        module: The model module (we'll ignore it for mock purposes)
        config: The rollout config section
        """
        super().__init__()
        self.config = config
        self.module = module
        
        # Load the data
        train_file = os.getenv('DATA_DIR')
        if train_file:
            train_file = os.path.join(train_file, 'train.parquet')
        else:
            raise ValueError("DATA_DIR environment variable not set")
            
        df = pd.read_parquet(train_file)
        self.saved_row = df.iloc[0]
        
        # The prompt is already a list of dicts
        self.prompt_list = self.saved_row['prompt']
        
        # Store messages by role while preserving order
        self.messages = []
        for msg in self.prompt_list:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Invalid message format: {msg}")
            self.messages.append({
                'role': msg['role'],
                'content': msg['content']
            })

    @staticmethod
    def _get_loss_mask(input_ids: torch.Tensor, message_token_lengths: list, dtype: torch.dtype) -> torch.Tensor:
        """Create a mask more memory efficiently"""
        batch_size, seq_len = input_ids.shape
        # Create mask in CPU first, then transfer to GPU
        mask = torch.zeros((batch_size, seq_len), dtype=dtype, device='cpu')
        
        current_pos = 0
        for role, length in message_token_lengths:
            if role in ['user', 'assistant']:
                # Set 1s for user and assistant message portions
                mask[:, current_pos:current_pos + length] = 1.0
            current_pos += length
        
        # Transfer to GPU only after filling
        return mask.to(input_ids.device)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Match HFRollout's generate_sequences interface exactly
        """
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # Get input tensors - move to CPU first, then GPU only when needed
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        
        batch_size = idx.size(0)
        
        # Create mock response on CPU first
        mock_response = torch.tensor(
            [ord(c) for c in self.messages[-1]['content']], 
            dtype=torch.int32,
            device='cpu'
        )
        mock_response = mock_response.unsqueeze(0).expand(batch_size, -1)
        
        # Do concatenation on CPU before moving to GPU
        seq = torch.cat([idx.cpu(), mock_response], dim=1)
        
        # Create attention masks on CPU
        response_attention_mask = torch.ones_like(mock_response, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat((attention_mask.cpu(), response_attention_mask), dim=-1)
        
        # Create loss mask on CPU
        message_token_lengths = [(msg['role'], len(msg['content'])) for msg in self.messages]
        loss_mask = self._get_loss_mask(
            input_ids=seq,
            message_token_lengths=message_token_lengths,
            dtype=attention_mask.dtype
        )
        
        # Position IDs computation on CPU
        response_length = mock_response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, 
            device='cpu'
        )
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:].cpu() + delta_position_id
        position_ids = torch.cat([position_ids.cpu(), response_position_ids], dim=-1)
        
        # Move everything to GPU only at the end
        batch = TensorDict(
            {
                'prompts': idx.cuda(),
                'responses': mock_response.cuda(),
                'input_ids': seq.cuda(),
                'attention_mask': full_attention_mask.cuda(),
                'loss_mask': loss_mask.cuda(),
                'position_ids': position_ids.cuda()
            },
            batch_size=batch_size
        )
        
        return DataProto(batch=batch)

    # Remove train_on_data as it's not part of the BaseRollout interface