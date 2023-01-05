import torch
import torch.distributed as dist
import numpy as np
from einops import rearrange
from typing import List, Optional, Union
def gather(
    tensor: torch.Tensor,
    world_size: int,
    rank: Optional[int] = None,
    preserve_gradients: bool = False,
) -> List[torch.Tensor]:
    containers = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.barrier()
    dist.all_gather(containers, tensor)
    if preserve_gradients:
        # If rank is provided, use it; if not, infer it from distributed
        if rank is None:
            assert dist.is_initialized(), "Distributed is not initialized."
            rank = dist.get_rank()
        containers[rank] = tensor.clone()
    return containers


def gather_reduce(
    tensor: torch.Tensor,
    world_size: int,
    reduction: str = "mean",
    rank: Optional[int] = None,
    preserve_gradients: bool = False,
):
    containers = gather(
        tensor, world_size, rank=rank, preserve_gradients=preserve_gradients
    )
    pre_reduced = torch.stack(containers)
    if reduction == "mean":
        return pre_reduced.mean(0)
    elif reduction == "sum":
        return pre_reduced.sum(0)
    elif reduction == "stack":
        return pre_reduced
    elif reduction == "cat":
        if pre_reduced.dim() > 1:
            return rearrange(pre_reduced, "worker batch ... -> (worker batch) ...")
        else:
            return pre_reduced
    else:
        raise NotImplementedError(f"Reduction mode {reduction} not implemented.")


def sync_values(
    value,
    reduction: str,
    device: Union[str, torch.device],
    world_size: int,
    rank: Optional[int] = None,
    preserve_gradients: Optional[bool] = False,
):
    if not torch.is_tensor(value):
        if isinstance(value, np.ndarray):
            value_type = "ndarray"
        elif isinstance(value, (float, int)):
            value_type = "py_scalar"
        else:
            raise TypeError
        value = torch.tensor(value).to(device)
        original_device = "cpu"
    else:
        value_type = "tensor"
        original_device = value.device
        value = value.to(device)
    gathered_value = gather_reduce(
        value,
        world_size,
        reduction=reduction,
        rank=rank,
        preserve_gradients=preserve_gradients,
    )
    # Ship the tensor back to where it was
    gathered_value = gathered_value.to(original_device)
    if value_type != "tensor":
        if value_type == "ndarray":
            gathered_value = gathered_value.numpy()
        elif value_type == "py_scalar":
            gathered_value = gathered_value.item()
    return gathered_value

