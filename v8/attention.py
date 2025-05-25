# attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SharedSelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, attention_heads=1, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads

        if self.attention_dim % self.attention_heads != 0:
            raise ValueError(
                f"Attention dim ({self.attention_dim}) must be divisible by "
                f"the number of heads ({self.attention_heads})."
            )
        
        self.head_dim = self.attention_dim // self.attention_heads

        self.query_proj = nn.Linear(input_dim, self.attention_dim)
        self.key_proj = nn.Linear(input_dim, self.attention_dim)
        self.value_proj = nn.Linear(input_dim, self.attention_dim)
        
        self.output_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sequence_features, mask=None):
        # sequence_features: (batch_size, seq_len, input_dim)
        if sequence_features.ndim == 2: # (seq_len, input_dim)
            sequence_features = sequence_features.unsqueeze(0) 

        batch_size, seq_len, _ = sequence_features.shape

        Q = self.query_proj(sequence_features)
        K = self.key_proj(sequence_features)
        V = self.value_proj(sequence_features)

        Q = Q.view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.attention_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        if mask is not None: # mask should be (batch_size, 1, 1, seq_len) for broadcasting
            energy = energy.masked_fill(mask == 0, -1e10 if energy.dtype == torch.float32 else -1e4)


        attention_weights = F.softmax(energy, dim=-1)
        attention_weights = self.dropout(attention_weights)

        weighted_values = torch.matmul(attention_weights, V)
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()
        weighted_values = weighted_values.view(batch_size, seq_len, self.attention_dim)

        output = self.output_proj(weighted_values)
        
        # Use the output corresponding to the *last element* of the sequence as the context
        # if we want attention to be causal and focus on summarizing up to the current point.
        # Or, average over the sequence length to get one context vector.
        # Let's average for a general summary.
        context_vector = output.mean(dim=1) 
        
        if batch_size == 1:
            return context_vector.squeeze(0)
        return context_vector

# Global instance of the shared attention layer
shared_attention_layer = None
attention_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_shared_attention(input_dim, attention_dim, attention_heads=1, dropout_rate=0.1):
    global shared_attention_layer
    shared_attention_layer = SharedSelfAttention(input_dim, attention_dim, attention_heads, dropout_rate)
    shared_attention_layer.to(attention_device)
    # The parameters (weights of query_proj, key_proj, value_proj, output_proj)
    # are initialized by PyTorch. They will remain FIXED during NEAT evolution in this version.
    # NEAT learns to use the output of this fixed attention mechanism.
    shared_attention_layer.eval() # Set to evaluation mode as we are not training it
    print(f"Shared attention layer initialized on {attention_device} with input_dim={input_dim}, attention_dim={attention_dim}, heads={attention_heads}. Weights are FIXED.")

def get_attention_output(sequence_features_np, current_seq_len, target_seq_len, feature_dim):
    global shared_attention_layer
    if shared_attention_layer is None:
        raise RuntimeError("Shared attention layer not initialized.")

    # Handle padding for sequences shorter than target_seq_len (e.g., at the start of data)
    padded_sequence_np = np.zeros((target_seq_len, feature_dim), dtype=np.float32)
    if current_seq_len > 0:
        if current_seq_len <= target_seq_len:
            padded_sequence_np[-current_seq_len:] = sequence_features_np[-current_seq_len:]
        else: # current_seq_len > target_seq_len, take the last target_seq_len elements
            padded_sequence_np = sequence_features_np[-target_seq_len:]
    
    # Create a mask for padded elements if needed by attention mechanism
    # (1 for real data, 0 for padding). Mask shape (1, 1, target_seq_len)
    # For `output.mean(dim=1)` as context, explicit masking of softmax input might be less critical
    # if padding is zeros, but good practice.
    # For simplicity in this version, we'll rely on zero-padding influencing softmax less.
    # A more robust mask would be:
    # attention_mask = torch.zeros(1, 1, target_seq_len, device=attention_device)
    # attention_mask[:, :, -current_seq_len:] = 1 # Mark valid parts of the sequence

    sequence_tensor = torch.tensor(padded_sequence_np, dtype=torch.float32).to(attention_device)
    if sequence_tensor.ndim == 2:
        sequence_tensor = sequence_tensor.unsqueeze(0)
    
    with torch.no_grad():
        attention_context_vector = shared_attention_layer(sequence_tensor) #, mask=attention_mask if current_seq_len < target_seq_len else None)
    
    return attention_context_vector.cpu().numpy()

# Placeholder for evolving attention parameters (Q,K,V) - NOT USED IN THIS VERSION
def evolve_attention_parameters_heuristic(best_fitness_current_gen, historical_best_fitness):
    print("DEBUG: evolve_attention_parameters_heuristic called, but attention parameters are currently fixed.")
    pass

def train_attention_on_high_impact_data(high_impact_sequences, target_attention_patterns):
    print("DEBUG: train_attention_on_high_impact_data called, but attention parameters are currently fixed.")
    pass
