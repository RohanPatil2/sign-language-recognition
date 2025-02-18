import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from einops import rearrange, repeat
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    input_size: int = 768
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    dropout: float = 0.1
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 512
    use_rotary_emb: bool = True
    pooling_type: str = 'attention'
    stochastic_depth: float = 0.2
    activation: str = 'gelu_new'
    pre_norm: bool = True
    use_flash_attention: bool = True

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation"""
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

class FlashAttention(nn.Module):
    """Memory-efficient scaled dot-product attention"""
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.attn_dropout)
        self.head_dim = config.hidden_size // config.num_attention_heads
        
    def forward(self, q, k, v, attention_mask=None):
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

class TransformerBlock(nn.Module):
    """Improved transformer block with RoPE and stochastic depth"""
    def __init__(self, config, layer_id):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = FlashAttention(config)
        self.drop_path = StochasticDepth(config.stochastic_depth * layer_id / config.num_hidden_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        if config.use_rotary_emb:
            self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        else:
            self.rotary_emb = None

    def forward(self, x, attention_mask=None):
        # Multi-head attention
        B, T, C = x.shape
        qkv = self.norm1(x) if self.config.pre_norm else x
        
        q = k = v = qkv  # For simplicity, can be expanded to separate projections
        
        if self.rotary_emb is not None:
            pos_emb = self.rotary_emb(T, x.device)
            q = apply_rotary_pos_emb(pos_emb, q)
            k = apply_rotary_pos_emb(pos_emb, k)
            
        attn_output = self.attn(q, k, v, attention_mask)
        x = x + self.drop_path(attn_output)
        
        # Feed-forward network
        mlp_input = self.norm2(x) if self.config.pre_norm else x
        x = x + self.drop_path(self.mlp(mlp_input))
        return x

class MultiPooling(nn.Module):
    """Combined pooling strategy with learnable weights"""
    def __init__(self, hidden_size):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3))
        self.attention_pool = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Attention pooling
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)
        attn_pool = torch.sum(attn_weights * x, dim=1)
        
        # Standard pooling
        max_pool = torch.max(x, dim=1).values
        mean_pool = torch.mean(x, dim=1)
        
        # Combine with learned weights
        weights = F.softmax(self.weights, dim=0)
        return weights[0] * attn_pool + weights[1] * max_pool + weights[2] * mean_pool

class AdvancedTransformer(nn.Module):
    def __init__(self, config, n_classes=50):
        super().__init__()
        self.config = config
        
        # Input processing
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks with stochastic depth
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_id=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Pooling and projection
        self.pooling = MultiPooling(config.hidden_size)
        self.projection = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, n_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x, attention_mask=None):
        x = self.input_proj(x)
        x = self.embed_dropout(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        pooled = self.pooling(x)
        projected = self.projection(pooled)
        return self.classifier(projected)