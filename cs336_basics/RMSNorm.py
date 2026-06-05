
from torch import Tensor, nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int  Hidden dimension of the model
        eps: float = 1e-5  Epsilon value for numerical stability
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameter
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))  # gain: (d_model,)


    def forward(self, x: Tensor) -> Tensor:
        # should upcast the input to torch.float32 to prevent overflow when we square the input.
        in_type = x.dtype 
        x = x.to(torch.float32)  # x: (..., d_model)
        # Step 1: compute the root mean square (RMS) of the input
        # x ** 2: (..., d_model) -> sum(dim=-1, keepdim=True): (..., 1) -> sqrt: (..., 1)
        rms = torch.sqrt( (1 / self.d_model) * torch.sum(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Step 2: normalize the input by dividing by the RMS
        # x / rms: (..., d_model) / (..., 1) 广播后仍是 (..., d_model)
        # * self.gain: 乘上 (d_model,)，广播到最后一维，结果 (..., d_model)
        normalized_x = x / rms * self.gain

        return normalized_x.to(in_type)  # (..., d_model)，dtype 还原回输入类型
