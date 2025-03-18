# ... existing code ...
import torch
import torch.distributed as dist

# Updated import for DeviceMesh and DTensor
try:
    from torch.distributed.tensor import DeviceMesh, DTensor
    print("torch.distributed.tensor is available in torch.distributed")
except ImportError:
    # For newer PyTorch versions
    from torch.distributed._tensor import DeviceMesh, DTensor
    print("torch.distributed._tensor is available in torch.distributed")
    # Alternative path if the above doesn't work:
    # from torch.distributed.device_mesh import DeviceMesh
    # from torch.distributed.tensor.tensor import DTensor
# ... existing code ...