
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
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
   

    def forward(self, x: Tensor) -> Tensor:
        # should upcast the input to torch.float32 to prevent overflow when we square the input.
        in_type = x.dtype
        x = x.to(torch.float32)
        # Step 1: compute the root mean square (RMS) of the input
        rms = torch.sqrt( (1 / self.d_model) * torch.sum(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Step 2: normalize the input by dividing by the RMS
        normalized_x = x / rms * self.gain
       
        return normalized_x.to(in_type)
