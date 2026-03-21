import torch
from jaxtyping import Float, Int

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self._theta: float = theta
        self._max_seq_len: int = max_seq_len # Maximum sequence length that will be inputted

        self._inv_freq: Float[torch.Tensor, "half_k"] = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', self._inv_freq)

        # Seed sin and cos values to avoid re-computation
        self._compute_sin_cos_cache()
    
    def _compute_sin_cos_cache(self) -> None:
        """
        Cache sin, cos values of rotation angles for all possible token positions.
        """
        indices: Int[torch.Tensor, "max_seq_len"] = torch.arange(self._max_seq_len)
        rotation_angles: Float[torch.Tensor, "max_seq_len half_k"] = torch.outer(indices, self._inv_freq)
        self._cos: Float[torch.Tensor, "max_seq_len half_k"]= torch.cos(rotation_angles)
        self._sin: Float[torch.Tensor, "max_seq_len half_k"] = torch.sin(rotation_angles)
    
    def forward(
        self, 
        x: Float[torch.Tensor, " ... sequence_length d_k"], 
        token_positions: Float[torch.Tensor, " ... sequence_length"] | None = None
    ) -> Float[torch.Tensor, " ... sequence_length d_k"]:
        """
        Run RoPE on an input tensor.

        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
            token_positions (Float[Tensor, "... sequence_length"]): Token potions of sequence element.
        Returns:
            Input tensor with positional info after RoPE
        """
        cos_slice = self._cos[token_positions] if token_positions is not None else self._cos[: x.size(-2)]
        sin_slice = self._sin[token_positions] if token_positions is not None else self._sin[: x.size(-2)]
        # cos_slice, sin_slice = self._cos[token_positions], self._sin[token_positions]

        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        rotated_x1 = x1 * cos_slice - x2 * sin_slice
        rotated_x2 = x1 * sin_slice + x2 * cos_slice

        return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
