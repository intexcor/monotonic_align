import time
import torch
import monotonic_align_cpp
v, m = torch.randn(14, 636, 281, requires_grad=False), torch.randn(14, 636, 281, requires_grad=False)
t_start = time.time()
print(monotonic_align_cpp.maximum_path_cpp(v, m))
t_end = time.time()
print(t_end - t_start)