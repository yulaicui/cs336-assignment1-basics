import importlib.metadata
from cs336_basics.nn_modules.attention import MultiHeadSelfAttention
from cs336_basics.nn_modules.embedding import Embedding
from cs336_basics.nn_modules.linear import Linear
from cs336_basics.nn_modules.rms_norm import RMSNorm
from cs336_basics.nn_modules.rope import RotaryPositionalEmbedding
from cs336_basics.nn_modules.swiglu import SwiGLU
from cs336_basics.nn_modules.transformer_block import TransformerBlock
from cs336_basics.nn_modules.transformer_lm import TransformerLM
from cs336_basics.nn_modules.utils import scaled_dot_product_attention, softmax

__version__ = importlib.metadata.version("cs336_basics")

__all__ = [
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "scaled_dot_product_attention",
    "softmax"
]
