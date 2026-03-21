import torch
from jaxtyping import Float, Int

from cs336_basics.nn_modules.embedding import Embedding
from cs336_basics.nn_modules.linear import Linear
from cs336_basics.nn_modules.rms_norm import RMSNorm
from cs336_basics.nn_modules.transformer_block import TransformerBlock


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, torch.Tensor],
    ):
        super().__init__()

        self._embedding: Embedding = Embedding(vocab_size, d_model, weights['token_embeddings.weight'])
        self._transformer_blocks: list[TransformerBlock] = [
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, self._get_weights_for_transformer_block(weights, i))
            for i in range(num_layers)
        ]
        self._ln_final: RMSNorm = RMSNorm(d_model, weights['ln_final.weight'])
        self._output: Linear = Linear(d_model, vocab_size, weights['lm_head.weight'])
    
    def _get_weights_for_transformer_block(self, weights: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
        return {
            "attn.q_proj.weight": weights[f"layers.{index}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{index}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{index}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{index}.attn.output_proj.weight"],
            "ln1.weight": weights[f"layers.{index}.ln1.weight"],
            "ffn.w1.weight": weights[f"layers.{index}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{index}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{index}.ffn.w3.weight"],
            "ln2.weight": weights[f"layers.{index}.ln2.weight"],
        }
    
    def forward(self, in_indices: Int[torch.Tensor, "batch_size sequence_length"]) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        # Get token embeddings from input indices
        embeddings: Float[torch.Tensor, "batch_size sequence_length d_model"] = self._embedding(in_indices)
        
        # Run transformer blocks
        transformer_block_output = embeddings
        for transformer_block in self._transformer_blocks:
            transformer_block_output = transformer_block(transformer_block_output)
        
        # Apply layer norm
        layer_norm: Float[torch.Tensor, "batch_size sequence_length d_model"] = self._ln_final(transformer_block_output)

        # Run final linear layer
        output: Float[torch.Tensor,"... d_model"] = self._output(layer_norm)

        # Convert output into probabilities
        return output
