"""Share unused device memory with the separately allocated native backend."""
import torch


def prepare_native_memory(device):
    """Release inactive Torch cache only when native allocation headroom is low."""
    free, total = torch.cuda.mem_get_info(device)
    inactive = max(0, torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device))
    released = free < max(4*1024**3, total//8) and inactive >= 256*1024**2
    if released:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return dict(cache_released=released, free_bytes_before=free,
                inactive_torch_bytes_before=inactive)
