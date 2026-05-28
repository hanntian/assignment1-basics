
from torch import nn
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.transformer_block import TransformerBlock

class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.output_projection = Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, seq_len = input_ids.shape
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        for layer in self.layers:
            x = layer(x)
        logits = self.output_projection(x)
        return logits