import os
import torch
import torch.distributed as dist

"""
Distributed utilities for PyTorch training.

This module provides helper functions to initialize and manage
PyTorch distributed training, including setup, synchronization,
tensor reduction, and gathering across processes.
"""


def is_dist_avail_and_initialized():
    """
    Check if torch.distributed is available and initialized.

    Returns:
        bool: True if distributed is available and initialized, else False.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Get the number of processes in the current distributed group.

    Returns:
        int: World size if distributed initialized, else 1.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process in the distributed group.

    Returns:
        int: Rank if initialized, else 0.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Determine if the current process is the main (rank 0).

    Returns:
        bool: True if rank is 0, else False.
    """
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    Redefine print to only output on the master process.

    Args:
        is_master (bool): True if this process is master, else False.
    """
    import builtins as __builtins__

    builtin_print = __builtins__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtins__.print = print


def init_distributed_mode(backend="nccl", init_method="env://"):
    """
    - Initialize torch.distributed process group based on environment variables.
    - This function sets up the distributed training environment
    and initializes the process group.
    - It **MUST** be called before any distributed operations.

    Args:
        backend (str): Backend to use (e.g., 'nccl', 'gloo').
        init_method (str): URL specifying how to initialize the process group.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        print(
            f"[init] rank = {rank}, world_size = {world_size}, local_rank = {local_rank}, "
            f"device = cuda:{local_rank}",
            flush=True,
        )
        setup_for_distributed(rank == 0)
        dist.barrier()

    else:
        raise RuntimeError(
            "Distributed training requires RANK and WORLD_SIZE environment variables to be set."
            "\nAre you forgetting to use torchrun or torch.distributed.launch?"
        )


def cleanup():
    """
    - Cleanup the distributed process group.
    - **MUST** be called after training is done.
    """
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
        setup_for_distributed(False)
        print("Distributed process group destroyed")
    else:
        print("No distributed process group to destroy")


def synchronize():
    """
    Synchronize all processes (barrier) if distributed is initialized.
    """
    if not is_dist_avail_and_initialized():
        return
    dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM):
    """
    Reduce a tensor from all processes.

    Args:
        tensor (torch.Tensor): Tensor to reduce.
        op (torch.distributed.ReduceOp): Reduction operation.

    Returns:
        torch.Tensor: Reduced tensor.
    """
    if not is_dist_avail_and_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    return tensor


def gather_tensors(tensor):
    """
    Gather tensors from all processes.

    Args:
        tensor (torch.Tensor): Local tensor to gather.

    Returns:
        list[torch.Tensor]: List of tensors gathered from each process.
    """
    if not is_dist_avail_and_initialized():
        return [tensor]
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered
