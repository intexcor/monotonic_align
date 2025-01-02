import torch
import monotonic_align

v, m = torch.randn(14, 636, 281, requires_grad=True).cuda(), torch.randn(14, 636, 281, requires_grad=True).cuda()
print(monotonic_align.maximum_path_cpp(v, m))
