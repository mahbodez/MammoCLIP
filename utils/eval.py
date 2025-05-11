import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from custom.config import Config
from .dist_utils import (
    is_main_process,
    get_rank,
    synchronize,
    is_dist_avail_and_initialized,
    reduce_tensor,
)
from .checkpoint import cleanup_checkpoints, save_checkpoint
from tensorboard import SummaryWriter
from logging import Logger


def eval_and_checkpoint(
    epoch: int,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    val_dl: torch.utils.data.DataLoader,
    logger: Logger | None,
    tb_writer: SummaryWriter | None,
    resuming: bool,
    optimization_steps: float,
    best_metric: float = float("inf"),
):
    # evaluation
    metric = None
    if (epoch + 1) % config.eval_interval == 0:
        metric = evaluate_loss(
            model, val_dl, logger, tb_writer, int(optimization_steps)
        )
    # synchronize processes before saving
    synchronize()
    if is_main_process():
        last = epoch == config.training_params["num_epochs"] - 1
        # if best metric is lower than current metric, save the model
        # regardless of the epoch
        if metric is not None:
            if metric < best_metric:
                best_metric = metric
                logger.info(f"New best metric: {best_metric}")
                tb_writer.add_scalar(
                    "val/best_metric",
                    best_metric,
                    global_step=optimization_steps,
                )
                # save the model
                prefix = "model_best"
                cleanup_checkpoints(config.project_dir, prefix, 1, logger)
                # if wrapped in DDP, unwrap
                base_model = model.module if hasattr(model, "module") else model
                save_checkpoint(
                    base_model,
                    optimizer,
                    scheduler,
                    config.project_dir,
                    prefix,
                    epoch + 1,
                    logger,
                )
        # save the model every save_interval epochs
        if ((epoch + 1) % config.save_interval == 0) or last:
            prefix = "model_resumed" if resuming else "model"
            cleanup_checkpoints(
                config.project_dir, prefix, config.max_checkpoints, logger
            )
            # if wrapped in DDP, unwrap
            base_model = model.module if hasattr(model, "module") else model
            save_checkpoint(
                base_model,
                optimizer,
                scheduler,
                config.project_dir,
                prefix,
                epoch + 1,
                logger,
            )
            if not resuming:
                config.to_yaml(os.path.join(config.project_dir, "config.yaml"))
    return best_metric


@torch.inference_mode()
def evaluate_loss(
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
        output = model(**batch, return_loss=True)
        loss = output["loss"].item()
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
