from __future__ import annotations

import math
import torch
from torch import Tensor, nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        w1 = torch.empty((d_ff, d_model), device=device, dtype=dtype)
        w2 = torch.empty((d_model, d_ff), device=device, dtype=dtype)
        w3 = torch.empty((d_ff, d_model), device=device, dtype=dtype)

        std = math.sqrt(2.0 / (d_model + d_ff))
        torch.nn.init.trunc_normal_(w1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(w2, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(w3, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.W1 = nn.Parameter(w1)
        self.W2 = nn.Parameter(w2)
        self.W3 = nn.Parameter(w3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
        x (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.
        """
        z1 = x @ self.W1.T
        gate = z1 * torch.sigmoid(z1)
        value = x @ self.W3.T
        return (gate * value) @ self.W2.T
