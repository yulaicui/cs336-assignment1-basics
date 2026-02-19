import torch
from jaxtyping import Float

class Linear(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, weights: Float[torch.Tensor, " d_out d_in"], device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.d_in: int = d_in # input dimension
        self.d_out: int = d_out # output dimension
        self.weights: Float[torch.Tensor, " d_out d_in"] = weights
        self._device: torch.device | None = device # device to store parameters on
        self._dtype: torch.dtype | None = dtype # data type of parameters
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies linear transformation to an input tensor

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return torch.einsum("...i,ji->...j", x, self.weights)
    