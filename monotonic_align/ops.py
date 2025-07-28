import torch
from torch import Tensor

__all__ = ["maximum_path_cpp"]


def maximum_path_cpp(a: Tensor, b: Tensor) -> Tensor:
    """Monotonic alignment C++ implementation."""
    return torch.ops.monotonic_align.maximum_path_cpp.default(a, b)


@torch.library.register_fake("monotonic_align::maximum_path_cpp")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)
