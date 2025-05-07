import os
from tqdm.auto import tqdm
import torch
from torch.amp import autocast, GradScaler
from .dist_utils import is_main_process, synchronize, get_rank, get_world_size
from typing import Tuple, Dict
from custom.config import Config
from .stats import (
    get_gpu_memory_usage,
    get_gpu_power_usage,
    get_gpu_temperature,
)
from .logger import setup_logger
from .train_utils import (
    save_checkpoint,
    cleanup_checkpoints,
    evaluate,
)
from logging import Logger
from torch.utils.tensorboard import SummaryWriter


def init_logger(config: Config) -> Logger:
    """
    Initialize logger on main process.
    """
    # Setup logger only on main process
    logger = None
    if is_main_process():
        os.makedirs(config.project_dir, exist_ok=True)
        logger = setup_logger(
            "training",
            log_to_file=os.path.join(config.project_dir, "logs", "train.log"),
        )
    return logger


def init_tensorboard(config: Config) -> SummaryWriter | None:
    """
    Initialize TensorBoard writer on main process.
    """
    tb_writer = None
    if is_main_process():
        path = os.path.join(config.project_dir, "logs", "tensorboard")
        os.makedirs(path, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=path)
    return tb_writer


def poll_gpu_stats() -> Dict[str, float]:
    return {
        "vram": get_gpu_memory_usage().get("percentage", 0.0),
        "pwr": get_gpu_power_usage().get("power", 0.0),
        "temp": get_gpu_temperature().get("temp", 0.0),
    }


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_dl: torch.utils.data.DataLoader,
    config: Config,
    epoch: int,
    logger: Logger | None,
    tb_writer: SummaryWriter | None,
    optimization_steps: float = 0.0,
) -> float:
    gpu_id = get_rank()
    grad_acc = config.training_params.get("gradient_accumulation_steps", 1)
    pbar = tqdm(
        total=len(train_dl) / grad_acc,
        disable=not is_main_process(),
        desc=f"Epoch {epoch+1}/{config.training_params['num_epochs']}",
        leave=False,
        dynamic_ncols=True,
        colour="green",
    )
    gpu_stats = {}
    opt_steps = optimization_steps
    # Mixed precision setting
    mixed_precision = config.training_params.get("mixed_precision")
    dtype = None
    if mixed_precision is not None:
        dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else None
        )
    use_amp = dtype is not None
    scaler = GradScaler() if use_amp else None
    # Don't forget to set_epoch for distributed sampler
    if hasattr(train_dl.sampler, "set_epoch"):
        train_dl.sampler.set_epoch(epoch)
    model.train()
    synchronize()
    for i, batch in enumerate(train_dl):
        # Move batch to local rank GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(gpu_id)
        # log the batch stats
        if logger is not None and i == 0:
            logger.info(
                f"Local rank {gpu_id} - "
                f"Batch size {batch['pixel_values'].shape[0]} - "
                f"Pixel values shape {batch['pixel_values'].shape} - "
                f"Input IDs shape {batch['input_ids'].shape}"
            )
        # END of logging
        # forward with optional autocast
        with autocast(enabled=use_amp, device_type="cuda", dtype=dtype):
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss / grad_acc
        # backward, with scaling if fp16/bf16
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # optimization step if gradient accumulation is done
        if (i + 1) % grad_acc == 0 or i == len(train_dl) - 1:
            # gradient clipping and optimizer step
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training_params.get("max_grad_norm", 1.0)
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1
            # update GPU stats periodically
            if is_main_process() and opt_steps % 10 == 0:
                gpu_stats = poll_gpu_stats()

            # log into tensorboard
            if is_main_process():
                tb_writer.add_scalar(
                    "train/loss",
                    loss.item() * grad_acc,
                    global_step=opt_steps,
                )
                tb_writer.add_scalar(
                    "train/lr",
                    scheduler.get_last_lr()[0],
                    global_step=opt_steps,
                )
                for k, v in gpu_stats.items():
                    tb_writer.add_scalar(f"gpu/{k}", v, global_step=opt_steps)

        # log progress
        pbar.update(1 / grad_acc)
        pbar.set_postfix(
            {
                "loss": loss.item() * grad_acc,
                "lr": scheduler.get_last_lr()[0],
                **gpu_stats,
            }
        )

    pbar.close()
    return opt_steps


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
        metric = evaluate(model, val_dl, logger, tb_writer, int(optimization_steps))
    # synchronize processes before saving
    synchronize()
    # Consolidate optimizer state for ZeroRedundancyOptimizer across all ranks
    if hasattr(optimizer, "consolidate_state_dict"):
        optimizer.consolidate_state_dict(to=0)
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
