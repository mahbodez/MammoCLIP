import os, re, shutil
import torch
import torch.nn as nn
from logging import Logger
from custom.model import MammoCLIP
from .dist_utils import (
    is_main_process,
)


def cleanup_checkpoints(
    project_dir: str, prefix: str, max_checkpoints: int, logger: Logger
) -> None:
    """
    Remove older checkpoints matching prefix_<num>
    (and if you passed "xxx_resumed", also cleans "xxx_<num>")
    keeping only the newest max_checkpoints per prefix.
    """
    assert is_main_process(), "Only main process can clean checkpoints"
    max_checkpoints = max(1, max_checkpoints)  # at least keep one checkpoint
    # build list of prefixes to clean
    prefixes = [prefix]
    if prefix.endswith("_resumed"):
        base = prefix[: -len("_resumed")]
        prefixes.append(base)

    for pref in prefixes:
        # escape in case pref contains regex chars
        pattern = re.compile(rf"^{re.escape(pref)}_(\d+)$")
        # collect (dirname, epoch) tuples
        all_ckpts = []
        for d in os.listdir(project_dir):
            m = pattern.match(d)
            if m and os.path.isdir(os.path.join(project_dir, d)):
                epoch = int(m.group(1))
                all_ckpts.append((d, epoch))

        # sort by epoch, keep only the last `max_checkpoints`
        all_ckpts.sort(key=lambda x: x[1])
        keep = {d for d, _ in all_ckpts[-max_checkpoints:]}
        for d, _ in all_ckpts:
            if d not in keep:
                logger.info(f"Removing {d} ...")
                shutil.rmtree(os.path.join(project_dir, d))


def save_checkpoint(
    model: MammoCLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    project_dir: str,
    prefix: str,
    epoch: int,
    logger: Logger,
) -> None:
    """
    Save model, optimizer, and scheduler under project_dir/prefix_{epoch:03d}.
    """
    if not is_main_process():
        raise RuntimeError("Only main process can save checkpoints")
    if (
        not isinstance(model, nn.Module)
        or not isinstance(optimizer, torch.optim.Optimizer)
        or not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
    ):
        raise ValueError("Invalid model, optimizer, or scheduler type")
    if not isinstance(logger, Logger):
        raise ValueError("Invalid logger type")

    logger.info("Saving checkpoint ...")
    path = os.path.join(project_dir, f"{prefix}_{epoch:03d}")
    os.makedirs(path, exist_ok=True)

    # Save model
    try:
        model.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model checkpoint {path}: {e}")

    # Save optimizer state
    opt_path = os.path.join(path, "optimizer.pt")
    try:
        torch.save(optimizer.state_dict(), opt_path)
        logger.info(f"Optimizer state saved to {opt_path}")
    except Exception as e:
        logger.error(f"Error saving optimizer state {opt_path}: {e}")

    # Save scheduler state
    sch_path = os.path.join(path, "scheduler.pt")
    try:
        torch.save(scheduler.state_dict(), sch_path)
        logger.info(f"Scheduler state saved to {sch_path}")
    except Exception as e:
        logger.error(f"Error saving scheduler state {sch_path}: {e}")
