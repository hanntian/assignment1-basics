
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Implements the RoPE (Rotary Position Embedding) module.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float  Θ value for the RoPE
        d_k: int  dimension of query and key vectors
        max_seq_len: int  Maximum sequence length that will be input
        device: torch.device | None = None  Device to store the buffer on

        R: Tensor[max_seq_len, d_k // 2, 2, 2]  “位置 0、位置 1、位置 2 ...”各自的旋转矩阵.
        """
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")
        
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32) # 1 dimesion:(max_seq_len,)
        pair_indices = torch.arange(d_k // 2, device=device, dtype=torch.float32) # 1 dimension:(d_k // 2,)
        
        angles = positions[:, None] / (theta ** ((2 * pair_indices[None, :]) / d_k)) #(max_seq_len, d_k // 2)
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        
        R = torch.zeros(max_seq_len, d_k // 2, 2, 2, device=device)
        R[..., 0, 0] = cos_values
        R[..., 0, 1] = -sin_values
        R[..., 1, 0] = sin_values
        R[..., 1, 1] = cos_values

        self.register_buffer("R", R, persistent=False) #每个batch都是需要这么计算的


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note 
        that you should tolerate x with an arbitrary number of batch dimensions. 
        You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        return: (..., seq_len, d_k)
        """
        *batch_dims, seq_len, d_k = x.shape
        # R: (..., seq_len, d_k // 2, 2, 2)
        R = self.R[token_positions.to(device=self.R.device)].to(dtype=x.dtype) #(batch, seq_len, d_k // 2, 2, 2)
        
        x_pairs = x.reshape(*batch_dims, seq_len, d_k // 2, 2)
        rotated = torch.einsum("...spij,...spj->...spi", R, x_pairs) #(*batch_dims, seq_len, d_k // 2, 2)
        
        return rotated.reshape(*batch_dims, seq_len, d_k) 

