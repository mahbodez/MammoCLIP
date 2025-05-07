import os, re, shutil
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from logging import Logger
from torch.utils.tensorboard import SummaryWriter
from custom.model import MammoCLIP
from .dist_utils import (
    is_dist_avail_and_initialized,
    reduce_tensor,
    is_main_process,
    synchronize,
    get_rank,
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
    assert is_main_process(), "Only main process can save checkpoints"
    logger.info("Saving model ...")
    path = os.path.join(project_dir, f"{prefix}_{epoch:03d}")
    try:
        model.save_pretrained(path)
        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
    except Exception as e:
        logger.error(f"Error saving checkpoint {path}: {e}")
        logger.info("Skipping ...")
    logger.info(f"Model saved to {path}")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    val_dl: torch.utils.data.DataLoader,
    logger: Logger | None,
    tb_writer: SummaryWriter,
    step: int,
):
    """
    Evaluate model on validation loader, aggregate loss across DDP, and log on main process.
    """
    gpu_id = get_rank()
    pbar = tqdm(
        total=len(val_dl),
        disable=not is_main_process(),
        desc="Validation",
        leave=False,
        dynamic_ncols=True,
        colour="blue",
    )
    losses = []
    # Dont forget to set_epoch for the sampler
    if hasattr(val_dl.sampler, "set_epoch"):
        val_dl.sampler.set_epoch(step)
    model.eval()
    # -------------------- DDP-aware evaluation loop -------------------
    synchronize()  # synchronize before starting
    for batch in val_dl:
        # move batch to local rank GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(gpu_id)
        # forward
        loss = model(**batch, return_loss=True).loss.item()
        losses.append(loss)
        pbar.update(1)
        pbar.set_postfix({"val_loss": loss})
    pbar.close()
    # ------------------------------------------------------------------
    # DDP-aware evaluation statistics
    local_sum = sum(losses)
    local_count = len(losses)
    if is_dist_avail_and_initialized():
        device = next(model.parameters()).device
        stats = torch.tensor([local_sum, local_count], device=device)
        stats = reduce_tensor(stats)
        total_sum, total_count = stats.tolist()
        avg = total_sum / total_count if total_count > 0 else 0.0
    else:
        avg = local_sum / local_count if local_count > 0 else 0.0
    # log only from the main process
    if is_main_process():
        logger.info(f"Val loss @ step {step}: {avg:.4f}")
        tb_writer.add_scalar("val/loss", avg, global_step=step)
    model.train()
    return avg
