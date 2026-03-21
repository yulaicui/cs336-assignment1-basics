import torch
from einops import einsum
from jaxtyping import Bool, Float


def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    """
    Utility function to compute softmax over a tensor's dimension.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # Subtract max value over the target dimension for numerical stability
    X_max = torch.max(in_features, dim=dim, keepdim=True).values
    X_stable = in_features - X_max
    
    # Exponentiate each term
    X_exp = torch.exp(X_stable)

    # Compute normalized value by dividing exponents by sum of exponents over the target dimension
    return X_exp / torch.sum(X_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    """
    Compute scaled dot product attention from key (K), query (Q), and value (V) tensors.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1] # embedding dimensions

    # Compute unmasked scores
    unmasked_scores: Float[torch.Tensor, "... queries keys"] = einsum(Q, K, "... q d, ... k d-> ... q k") / torch.sqrt(torch.tensor(d_k))

    # Mask scores
    masked_scores: Float[torch.Tensor, "... queries keys"] = (unmasked_scores * mask).masked_fill(mask == 0, float('-inf')) if mask is not None else unmasked_scores

    # Apply softmax
    normalized_masked_scores: Float[torch.Tensor, "... queries keys"] = softmax(masked_scores, -1)

    # Compute attention output
    attn_output: Float[torch.Tensor, " ... queries d_v"] = einsum(normalized_masked_scores, V, "... q k, ... k v-> ... q v" )
    return attn_output
