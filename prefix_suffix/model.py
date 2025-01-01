"""
Heavily inspired by nanoGPT:
https://github.com/karpathy/nanoGPT
"""

import math
import inspect

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, num_dimensions, bias):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_dimensions))
        self.beta = nn.Parameter(torch.zeros(num_dimensions)) if bias else None
        self.eps = 1e-6

    def forward(self, x):
        return F.layer_norm(x, self.gamma.shape, self.gamma, self.beta, self.eps)
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert (config.n_hidden % config.n_head) == 0        
        self.n_head = config.n_head
        self.n_hidden = config.n_hidden
        self.head_shape = config.n_hidden // config.n_head
        
        self.qkv = nn.Linear(config.n_hidden, config.n_hidden * 3, bias = config.bias)
        self.w_o = nn.Linear(config.n_hidden, config.n_hidden, bias = config.bias)
        
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))    

    def forward(self, x):  
        B, T, C = x.size()
        
        # Extract query, key, and value vectors
        q, k, v = self.qkv(x).split(self.n_hidden, dim = 2)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_shape)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # ...
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # ...
        
        if self.flash:
            hidden_state = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            A = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_shape)
            A = A.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  
            A = F.softmax(A, dim = -1)
            A = self.attn_dropout(A)
            hidden_state = A @ v
            
        hidden_state = hidden_state.transpose(1, 2).contiguous().view(B, T, C)
        
        output = self.w_o(hidden_state)
        output = self.residual_dropout(output)
        
        return output
        

class FeedForward(nn.Module):   
    def __init__(self, config):
        super().__init__()

        self.linear_1 = nn.Linear(config.n_hidden, config.n_hidden * 4, bias = config.bias)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(config.n_hidden * 4, config.n_hidden, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        hidden_state = self.linear_1(x)
        hidden_state = self.gelu(hidden_state)
        hidden_state = self.linear_2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        
        return hidden_state

@dataclass
class CharConfig:
    context_length: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_head: int = 12
    n_hidden: int = 768
    dropout: float = 0.0
    bias: bool = True


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.ln_1 = LayerNorm(config.n_hidden, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_hidden, config.bias)
        self.ff = FeedForward(config)
        
    def forward(self, hidden_state):
        hidden_state = hidden_state + self.attn(self.ln_1(hidden_state))
        hidden_state = hidden_state + self.ff(self.ln_2(hidden_state))
        
        return hidden_state
        

class CharModel(nn.Module):
    def __init__(self, config, space_id = 1):
        super().__init__()
        assert config.vocab_size is not None
        assert config.n_layers is not None
        self.config = config
        self.space_id = space_id
        
        self.transformer = nn.ModuleDict(dict(
            embedding_lookup = nn.Embedding(config.vocab_size, config.n_hidden),
            positional_embedding = nn.Embedding(config.context_length, config.n_hidden),
            dropout = nn.Dropout(config.dropout),
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            layer_norm = LayerNorm(config.n_hidden, config.bias)
        ))
        
        # Initialize prefix and suffix embeddings
        self.prefix_embeddings = nn.Parameter(torch.randn(1, 1, config.n_hidden) * 0.02)
        self.suffix_embeddings = nn.Parameter(torch.randn(1, 1, config.n_hidden) * 0.02)
        
        # Project residual stream to vocab space
        self.vocab_projection = nn.Linear(config.n_hidden, config.vocab_size, bias=False)
        
        # Copy embedding weights from vocab_projection to embedding_lookup
        self.transformer.embedding_lookup.weight = self.vocab_projection.weight
        
        # Initialize weights recursively
        self.apply(self.init_weights)
        
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith('c_proj.weight'):
                divisor = math.sqrt(2 * config.n_layers)
                torch.nn.init.normal_(parameter, mean = 0.0, std = 0.02 / divisor)

        print(f"Initialized GPT Model! Number of Parameters: {self.get_num_params() / 1e6}")
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())   

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def forward(self, input, targets = None):
        B, T = input.size()
        assert T <= self.config.context_length, f"Input length {T} is longer than context length {self.config.context_length}" 
        
        device = input.device
        positional = torch.arange(0, T, dtype = torch.long, device = device)
        
        char_embeddings = self.transformer['embedding_lookup'](input)
        positional_embeddings = self.transformer['positional_embedding'](positional) 
        
        # Add prefix and suffix embeddings to word start and end            
        word_starts, word_ends = self.get_word_boundaries(input)  
        char_embeddings = char_embeddings + (word_starts.unsqueeze(-1) * self.prefix_embeddings)
        char_embeddings = char_embeddings + (word_ends.unsqueeze(-1) * self.suffix_embeddings)

        hidden_state = self.transformer['dropout'](char_embeddings + positional_embeddings)
        
        for layer in self.transformer['layers']:
            hidden_state = layer(hidden_state)
        hidden_state = self.transformer['layer_norm'](hidden_state)
        
        # Project hidden to vocab space
        if targets is not None:       
            logits = self.vocab_projection(hidden_state) 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.vocab_projection(hidden_state[:, [-1], :]) 
            loss = None    

        return logits, loss        
    
    def get_word_boundaries(self, char_ids):
        B, T = char_ids.size()
        device = char_ids.device

        word_starts = torch.zeros_like(char_ids, dtype=torch.bool, device=device)
        word_ends = torch.zeros_like(char_ids, dtype=torch.bool, device=device)
        
        is_space = char_ids == self.space_id
        is_space_next = torch.roll(is_space, shifts=-1, dims=1)
        is_space_prev = torch.roll(is_space, shifts=1, dims=1)
        
        word_starts[:, 1:] = is_space[:, :-1] & ~is_space_next[:, :-1]
        word_ends[:, :-1] = is_space[:, 1:] & ~is_space_prev[:, 1:]
        
        word_starts[:, 0] = ~is_space[:, 0]
        word_ends[:, -1] = ~is_space[:, -1]
        
        return word_starts, word_ends
    
    # Do not update params using gradients
    @torch.no_grad() 
    def generate(self, tokens, n_tokens, temperature = 1.0):
        for _ in range(n_tokens):
            logits, _ = self(tokens)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim = -1)
            next_token = torch.multinomial(probs, num_samples = 1)
            
            tokens = torch.cat([tokens, next_token], dim = -1)
        
        return tokens
            
            