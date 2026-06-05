import torch
import torch.nn as nn
from cs336_basics.RoPE import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.linear import Linear

class causal_multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, rope=False, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads

        self.W_q = Linear(d_model, d_model,  device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model,  device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model,  device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model,  device=device, dtype=dtype)

        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device) if rope else None

    def forward(self, x, token_positions=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # view在不改变内存中数据的情况下，重新调整张量的形状（Reshape): d_model -> (num_heads, d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # view 的特性：它绝对不会复制内存。它只是创建了一个新的张量头部（Tensor Header），改变了对底层同一块连续内存的“解读方式”。如果内存不连续（比如刚做完 transpose），view 会直接报错。O(1)开销。
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2) # 而reshape呢：如果内存连续，reshape 和 view 的行为是一样的；如果内存不连续，reshape 会在后台默默地调用 .clone().view(...)，帮你复制一份内存，让它变连续。是情况而定的，可能是 O(1) 也可能是 O(n) 的开销。

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) # (seq_len,) -> (batch_size, seq_len)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        ).view(1, 1, seq_len, seq_len)

        attn = scaled_dot_product_attention(Q, K, V, causal_mask)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model) # 等价于 attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.W_o(attn)