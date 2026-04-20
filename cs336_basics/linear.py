from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
            in_features: final dimension of the input
            out_features: final dimension of the output
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.W = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return x @ self.W.transpose(0, 1)

