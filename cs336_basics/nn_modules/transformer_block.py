import torch
from jaxtyping import Float

from cs336_basics.nn_modules.attention import MultiHeadSelfAttention
from cs336_basics.nn_modules.rms_norm import RMSNorm
from cs336_basics.nn_modules.swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, torch.Tensor],
    ):
        super().__init__()
        self._d_model: int = d_model
        self._num_heads: int = num_heads

        self._ln1: RMSNorm = RMSNorm(d_model, weights['ln1.weight'])
        self._ln2: RMSNorm = RMSNorm(d_model, weights['ln2.weight'])
        self._mha: MultiHeadSelfAttention = MultiHeadSelfAttention(
            d_model, 
            num_heads, 
            weights['attn.q_proj.weight'],
            weights['attn.k_proj.weight'],
            weights['attn.v_proj.weight'],
            weights['attn.output_proj.weight'],
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self._ffn: SwiGLU = SwiGLU(
            d_model,
            d_ff,
            weights['ffn.w1.weight'],
            weights['ffn.w2.weight'],
            weights['ffn.w3.weight'],
        )
    
    def forward(
        self, 
        in_features: Float[torch.Tensor, " batch sequence_length d_model"]
    ) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        ln1_output = self._ln1(in_features)
        mha_output = self._mha(ln1_output)
        ln2_input = in_features + mha_output
        ln2_output = self._ln2(ln2_input)
        ffn_output = self._ffn(ln2_output)

        return ffn_output + ln2_input
