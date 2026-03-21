import torch
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[torch.Tensor, " batch_size vocab_size"], 
    targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    logits_stable = logits - torch.max(logits, dim=-1, keepdim=True).values
    logits_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
    target_logits = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = -target_logits + logits_sum_exp
    return loss.mean()


