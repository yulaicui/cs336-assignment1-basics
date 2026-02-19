import torch
from einops import rearrange
from jaxtyping import Float

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, weights: Float[torch.Tensor, " vocab_size d_model"], device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.vocab_size: int = vocab_size # size of vocabulary
        self.embedding_dim: int = embedding_dim # dimension of embedding vectors
        self.weights: Float[torch.Tensor, " vocab_size d_model"] = weights # embedding matrix
        self._device: torch.device | None = device # device to store parameters on
        self._dtype: torch.dtype | None = dtype # data type of parameters
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Converts batches of token ids to batches of embedding vectors.

        Args:
            token_ids (torch.Tensor): batched token ids.

        Returns:
            torch.Tensor: batched embedding vectors.
        """
        embeddings: Float[torch.Tensor, "... d_model"] = torch.index_select(self.weights, 0, token_ids.flatten()).reshape(token_ids.shape + (self.embedding_dim,))
        return embeddings
