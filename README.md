# Monotonic align
Implementation of search monotonic alignment by cpp extension of PyTorch.

## Installation
```sh
pip install git+https://github.com/intexcor/monotonic_align.git
```
## Example
```python
import torch
import monotonic_align

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

v, m = (torch.randn(14, 636, 281, requires_grad=False, device=device), 
        torch.randn(14, 636, 281, requires_grad=False, device=device))
print(monotonic_align.maximum_path_cpp(v, m))
```
