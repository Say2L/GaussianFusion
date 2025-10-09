import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

@ATTENTION.register_module()
class CrossAttention(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        is_global: bool = True,
        rope=None,
        **kwargs
    ) -> None:
        super().__init__()
        assert embed_dims % num_heads == 0, "dim should be divisible by num_heads"
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.is_global = is_global

        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
    def forward(self,
                query,
                key,
                query_pos=None,
                key_pos=None,
                **kwargs):
        
        identity = query
        B, N, C = query.shape
        q = self.q(query + query_pos).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key + key_pos).reshape(B, key.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        v = self.v(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) + identity
        
        return x