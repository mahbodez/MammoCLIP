import os, re, shutil
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from logging import Logger
from custom import MammoCLIP


def cleanup_checkpoints(
    project_dir: str, prefix: str, max_checkpoints: int, logger: Logger
) -> None:
    """
    Remove older checkpoints matching prefix_<num>
    keeping only the newest max_checkpoints.
    """
    pattern = rf"^{prefix}_(\d+)$"
    names = [
        d
        for d in os.listdir(project_dir)
        if os.path.isdir(os.path.join(project_dir, d)) and re.match(pattern, d)
    ]
    # sort by epoch number
    names.sort(key=lambda x: int(re.match(pattern, x).group(1)))
    keep = set(names[-max_checkpoints:])
    for d in names:
        if d not in keep:
            logger.info(f"Removing {d} ...")
            shutil.rmtree(os.path.join(project_dir, d))


def save_checkpoint(
    model: MammoCLIP, project_dir: str, prefix: str, epoch: int, logger: Logger
) -> None:
    """
    Save model under project_dir/prefix_{epoch:03d}.
    """
    logger.info("Saving model ...")
    path = os.path.join(project_dir, f"{prefix}_{epoch:03d}")
    try:
        model.save_pretrained(path)
    except Exception as e:
        logger.error(f"Error saving checkpoint {path}: {e}")
        logger.info("Skipping ...")


@torch.inference_mode()
def evaluate(
    model,
    val_dl: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    logger: Logger,
    step: int,
):
    model.eval()
    losses = []
    pbar = tqdm(
        total=len(val_dl),
        disable=not accelerator.is_local_main_process,
        desc="Validation",
    )
    for batch in val_dl:
        loss = model(**batch, return_loss=True).loss.item()
        losses.append(loss)
        pbar.update(1)
        pbar.set_postfix({"val_loss": loss})
    pbar.close()
    avg = sum(losses) / len(losses) if losses else 0.0
    if accelerator.is_main_process:
        logger.info(f"Val loss @ step {step}: {avg:.4f}")
    accelerator.log({"val_loss": avg}, step=step)
    model.train()
    return avg
