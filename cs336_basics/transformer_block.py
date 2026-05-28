
from torch import nn, Tensor
import torch

from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.causal_multihead_self_attention import causal_multihead_self_attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0, max_seq_len: int = 512):
        """
        Construct a Transformer block. This function should accept the following parameters:
            d_model: dimension of the input and output features
            num_heads: number of attention heads to use in the multi-headed attention module
            d_ff: dimension of the hidden layer in the feedforward network
            theta: base period of the RoPE positional encoding (default: 10,000)
            max_seq_len: maximum sequence length that the block should be able to process (default: 512)
        """
        super().__init__()

        self.attn = causal_multihead_self_attention(d_model, num_heads, rope = True, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)
       
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        y = x + self.attn(self.norm1(x))
        z = y + self.ffn(self.norm2(y))
        return z