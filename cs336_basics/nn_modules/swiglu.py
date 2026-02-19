import torch
from jaxtyping import Float

class SwiGLU(torch.nn.Module):
    def __init__(
            self, 
            d_model: int, 
            d_ff: int, 
            w1_weight: Float[torch.Tensor, "d_ff d_model"], 
            w2_weight: Float[torch.Tensor, " d_model d_ff"], 
            w3_weight: Float[torch.Tensor, " d_ff d_model"],
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.w1_weight: Float[torch.Tensor, "d_ff d_model"] = w1_weight
        self.w2_weight: Float[torch.Tensor, " d_model d_ff"] = w2_weight
        self.w3_weight: Float[torch.Tensor, " d_ff d_model"] = w3_weight
        self._device: torch.device | None = device # device to store parameters on
        self._dtype: torch.dtype | None = dtype # data type of parameters
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        Applies SwiGLU activation.

        Args:
            in_features (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        y1: Float[torch.Tensor, " ... d_ff"] = torch.einsum("...i,ji->...j", in_features, self.w1_weight)
        silu: Float[torch.Tensor, " ... d_ff"] = y1 * torch.sigmoid(y1)
        y3: Float[torch.Tensor, " ... d_ff"] = torch.einsum("...i,ji->...j", in_features, self.w3_weight)
        swiglu: Float[torch.Tensor, " ... d_model"] = torch.einsum("...i,ji->...j", silu * y3, self.w2_weight)

        return swiglu
    