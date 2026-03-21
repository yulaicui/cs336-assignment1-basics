import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int

from cs336_basics.nn_modules.linear import Linear
from cs336_basics.nn_modules.rope import RotaryPositionalEmbedding
from cs336_basics.nn_modules.utils import scaled_dot_product_attention


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[torch.Tensor, " d_k d_in"],
        k_proj_weight: Float[torch.Tensor, " d_k d_in"],
        v_proj_weight: Float[torch.Tensor, " d_v d_in"],
        o_weight: Float[torch.Tensor, " d_model d_v"],
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"model dimension must be divisible by num_heads={num_heads}"

        self._d_model: int = d_model
        self._num_heads: int = num_heads
        self._q_proj_weight: Float[torch.Tensor, " d_k d_in"] = torch.nn.Parameter(q_proj_weight)
        self._k_proj_weight: Float[torch.Tensor, " d_k d_in"] = torch.nn.Parameter(k_proj_weight)
        self._v_proj_weight: Float[torch.Tensor, " d_v d_in"] = torch.nn.Parameter(v_proj_weight)
        self._output_layer: Linear = Linear(d_model, d_model, o_weight)

        apply_rope = max_seq_len is not None and theta is not None
        self.rope: RotaryPositionalEmbedding | None = RotaryPositionalEmbedding(theta, d_k=d_model // num_heads, max_seq_len=max_seq_len) if apply_rope else None
    
    def forward(
            self, 
            in_features: Float[torch.Tensor, " ... sequence_length d_in"],
            token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None
        ) -> Float[torch.Tensor, " ... sequence_length d_out"]:
        # Apply QKV projection from input_features (assuming d_k == d_v)
        q, k, v = einsum(
            in_features, 
            torch.cat([self._q_proj_weight, self._k_proj_weight, self._v_proj_weight], dim=0),
            "b sequence_length d_in, d_qkv d_in -> b sequence_length d_qkv"
            ).chunk(3, dim=-1)
        # q, k, v = torch.einsum("...ji,ki->...jk", in_features, torch.cat([self._q_proj_weight, self._k_proj_weight, self._v_proj_weight], dim=0)).chunk(3, dim=-1)
        q = rearrange(q, "b sequence_length (num_heads d_k) -> (num_heads b) sequence_length d_k", num_heads=self._num_heads)
        k = rearrange(k, "b sequence_length (num_heads d_k) -> (num_heads b) sequence_length d_k", num_heads=self._num_heads)
        v = rearrange(v, "b sequence_length (num_heads d_k) -> (num_heads b) sequence_length d_k", num_heads=self._num_heads)

        # Apply positional encoding
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Compute mask
        seq_len = in_features.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()

        # Compute scaled dot-product attention
        attentions = rearrange(scaled_dot_product_attention(q, k, v, mask), "(num_heads b) sequence_length d_v -> b sequence_length (num_heads d_v)", num_heads=self._num_heads)
        return self._output_layer(attentions)
