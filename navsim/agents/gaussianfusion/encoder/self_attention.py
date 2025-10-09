from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION

@ATTENTION.register_module()
class SelfAttention(nn.Module):
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

        self.qk = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
        self.v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, *args, query_pos=None, **kwargs) -> Tensor:
        # if not self.is_global:
        #     B, N, C = x.shape
        #     x = x.view(B, 2, N // 2, C).reshape(B * 2, N // 2, C)
        #     pos = pos.view(B, 2, N // 2, -1).reshape(B * 2, N // 2, -1) if pos is not None else None

        identity = x
        B, N, C = x.shape
        qk = self.qk(x + query_pos).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k= qk.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    
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
    