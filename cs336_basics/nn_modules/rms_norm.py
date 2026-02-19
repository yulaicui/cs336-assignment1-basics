import torch
from jaxtyping import Float

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, weights: Float[torch.Tensor, " d_model"], eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.d_model: int = d_model # hidden dimension of the model
        self.weights: Float[torch.Tensor, " d_model"] = weights # RMS normalization layer weights
        self._eps: float = eps # Epsilon value for numerical stability
        self._device: torch.device | None = device # device to store parameters on
        self._dtype: torch.dtype | None = dtype # data type of parameters
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        Compute Root Mean Square Layer Normalization.

        Args:
            in_features (torch.Tensor): batched feature vector input.

        Returns:
            torch.Tensor: RMSNorm output.
        """
        original_dtype: torch.dtype = in_features.dtype
        in_features = in_features.to(torch.float32) # upcast to float32 to prevent overflow from sqrt
        rms: torch.Tensor = torch.sqrt((torch.sum(torch.square(in_features), dim=-1) / self.d_model) + self._eps)
        
        return (in_features / torch.unsqueeze(rms, -1) * self.weights).to(original_dtype)
