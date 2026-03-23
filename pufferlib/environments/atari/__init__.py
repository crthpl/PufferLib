from .environment import env_creator

try:
    import torch
except ImportError:
    pass
else:
    from .torch import Encoder
