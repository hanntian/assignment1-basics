from torch import nn, Tensor
import torch

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        """
        Args:
            vocab_size: 词表大小，token embedding 矩阵的行数
            context_length: 最大上下文长度，用于 RoPE sin/cos 的预先计算
            d_model: 模型隐藏维度
            num_layers: Transformer block 的层数
            num_heads: 注意力头数量
            d_ff: FFN (SwiGLU) 内部隐藏维度
            rope_theta: RoPE 的 theta 基础周期
        """
        super().__init__()
        # token embedding
        self.token_embedding = Embedding(vocab_size, d_model)
        # num_layers 个 transformer block，使用 RoPE
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    theta=rope_theta,
                    max_seq_len=context_length,
                )
                for _ in range(num_layers)
            ]
        )
        # 最终的 RMSNorm（模型其他位置都是 RMSNorm，这里同样要用 RMSNorm 而不是 nn.LayerNorm）
        self.norm = RMSNorm(d_model)
        # 输出投影：把 d_model 投到 vocab_size 的 logits
        self.output_projection = Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) 的 LongTensor
        Returns:
            logits: (batch_size, seq_len, vocab_size) 未归一化的下一个 token 分布
        """
        # token embedding -> (batch_size, seq_len, d_model)
        x = self.token_embedding(input_ids)
        # 依次过每个 transformer block
        for layer in self.layers:
            x = layer(x)
        # final RMSNorm
        x = self.norm(x)
        # 投影到词表 logits -> (batch_size, seq_len, vocab_size)
        logits = self.output_projection(x)
        return logits
