from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    """The number of unique items in the output vocabulary to be predicted."""

    context_length: int
    """The maximum number of tokens to process at once."""

    d_model: int
    """The dimensionality of the Transformer block input."""

    num_layers: int
    """The number of Transformer layers to use."""

    num_heads: int
    """Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`."""

    d_ff: int
    """Dimensionality of the feed-forward inner layer."""

    rope_theta: float
    """The RoPE (positional encoding) parameter."""


@dataclass(frozen=True)
class TrainingConfig:
    epocs: int
    """Number of training epocs."""

    batch_size: int
    """Training batch size."""

    context_length: int
    """Training context length."""

    random_seed: int = 42
    """Random seed for anything stochastic."""


@dataclass(frozen=True)
class ValidationConfig:
    val_every: int
    """Validate every (this number) of epocs"""

    batch_size: int
    """Training batch size."""

    context_length: int
    """Training context length."""

    random_seed: int = 42
    """Random seed for anything stochastic."""
