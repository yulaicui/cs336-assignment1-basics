import importlib.metadata
from cs336_basics.nn_modules.embedding import Embedding
from cs336_basics.nn_modules.linear import Linear
from cs336_basics.nn_modules.rms_norm import RMSNorm
from cs336_basics.nn_modules.swiglu import SwiGLU

__version__ = importlib.metadata.version("cs336_basics")

__all__ = [
    "Embedding",
    "Linear",
    "RMSNorm",
    "SwiGLU"
]
