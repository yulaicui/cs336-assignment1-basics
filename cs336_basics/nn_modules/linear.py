import torch
from einops import einsum
from jaxtyping import Float

class Linear(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, weights: Float[torch.Tensor, " d_out d_in"], device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_in: int = d_in # input dimension
        self.d_out: int = d_out # output dimension
        self.weights: Float[torch.Tensor, " d_out d_in"] = torch.nn.Parameter(weights)
        self._device: torch.device = device

        if self._device is not None:
            self.weights.to(device=self._device)
    
    def forward(self, x: Float[torch.Tensor, "b ... d_in"]) -> Float[torch.Tensor, "b ... d_out"]:
        """
        Applies linear transformation to an input tensor

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return einsum(
            x,
            self.weights,
            "b ... d_in, d_out d_in -> b ... d_out",
        )
    