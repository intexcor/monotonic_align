import torch
import monotonic_align

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

v, m = (torch.randn(14, 636, 281, requires_grad=False, device=device), 
        torch.randn(14, 636, 281, requires_grad=False, device=device).cuda())
print(monotonic_align.maximum_path_cpp(v, m))
