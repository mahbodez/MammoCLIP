import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from custom.config import Config
from custom.inference import evaluate_batch
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
import pandas as pd
from typing import Optional, Tuple


def _maybe_eval_loss(
    epoch: int,
    config: Config,
    model: torch.nn.Module,
    val_dl: torch.utils.data.DataLoader,
    logger: Optional[Logger],
    tb_writer: Optional[SummaryWriter],
    optimization_steps: float,
) -> Optional[float]:
    eval_loss = None
    if (epoch + 1) % config.eval_interval == 0:
        eval_loss = evaluate_loss(
            model, val_dl, logger, tb_writer, int(optimization_steps)
        )
    return eval_loss


def _maybe_eval_metrics(
    epoch: int,
    config: Config,
    model: torch.nn.Module,
    logger: Optional[Logger],
    tb_writer: Optional[SummaryWriter],
    optimization_steps: float,
    metric_query: str,
    metric: str,
) -> Optional[float]:
    metric_value = None
    if is_main_process() and (epoch + 1) % config.infer_interval == 0:
        metrics_dict = evaluate_metrics(
            model, config, logger, tb_writer, int(optimization_steps)
        )
        metric_value = metrics_dict[metric_query][metric]
    return metric_value


def _save_best_loss(
    eval_loss: Optional[float],
    lowest_loss: float,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: Logger,
    tb_writer: SummaryWriter,
    optimization_steps: float,
) -> float:
    if eval_loss is not None:
        if eval_loss < lowest_loss:
            lowest_loss = eval_loss
            logger.info(f"New lowest loss: {lowest_loss}")
            tb_writer.add_scalar(
                "val/lowest_loss", lowest_loss, global_step=int(optimization_steps)
            )
    return lowest_loss


def _save_best_metric(
    metric_value: Optional[float],
    best_metric: float,
    metric_criterion: str,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    logger: Logger,
) -> float:
    if metric_value is not None:
        if (metric_criterion == "highest" and metric_value > best_metric) or (
            metric_criterion == "lowest" and metric_value < best_metric
        ):
            best_metric = metric_value
            cleanup_checkpoints(config.project_dir, "model_best", 1, logger)
            base_model = model.module if hasattr(model, "module") else model
            save_checkpoint(
                base_model,
                optimizer,
                scheduler,
                config.project_dir,
                "model_best",
                epoch + 1,
                logger,
            )
    return best_metric


def _save_periodic_checkpoint(
    epoch: int,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: Logger,
    resuming: bool,
) -> None:
    last = epoch == config.training_params["num_epochs"] - 1
    if ((epoch + 1) % config.save_interval == 0) or last:
        prefix = "model_resumed" if resuming else "model"
        cleanup_checkpoints(config.project_dir, prefix, config.max_checkpoints, logger)
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


def eval_and_checkpoint(
    epoch: int,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    val_dl: torch.utils.data.DataLoader,
    lowest_loss: float,
    best_metric: float,
    logger: Optional[Logger],
    tb_writer: Optional[SummaryWriter],
    resuming: bool,
    optimization_steps: float,
    metric_query: str = "birads",
    metric: str = "accuracy",
    metric_criterion: str = "highest",
) -> Tuple[float, float]:
    # run conditional loss and metrics evaluation
    eval_loss = _maybe_eval_loss(
        epoch, config, model, val_dl, logger, tb_writer, optimization_steps
    )
    metric_value = _maybe_eval_metrics(
        epoch,
        config,
        model,
        logger,
        tb_writer,
        optimization_steps,
        metric_query,
        metric,
    )
    # synchronize processes before saving
    synchronize()
    if is_main_process():
        # update lowest loss and best metric, then save checkpoints
        lowest_loss = _save_best_loss(
            eval_loss,
            lowest_loss,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            tb_writer,
            optimization_steps,
        )
        best_metric = _save_best_metric(
            metric_value,
            best_metric,
            metric_criterion,
            config,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
        )
        _save_periodic_checkpoint(
            epoch, config, model, optimizer, scheduler, logger, resuming
        )
    synchronize()
    return lowest_loss, best_metric


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


@torch.inference_mode()
def evaluate_metrics(
    model: torch.nn.Module,
    config: Config,
    logger: Logger,
    tb_writer: SummaryWriter,
    global_step: int,
) -> dict[str, dict]:
    assert (
        is_main_process()
    ), "This evaluation must be run on the main process (for now)."
    # extract base model if not unwrapped
    base_model = model.module if hasattr(model, "module") else model

    df = pd.read_csv(config.infer_settings["csv_path"])

    weight_col = config.infer_settings.get("weight_col", None)
    sample_size = config.infer_settings["sample_size"]
    replace = config.infer_settings.get("replace", True)
    image_cols = config.infer_settings["view_cols"]
    batch_size = config.infer_settings.get("batch_size", 8)

    weights = None
    if weight_col:
        weights = df[weight_col]

    sample = df.sample(
        n=sample_size,
        replace=replace,
        weights=weights,
    )
    # Evaluate each query
    metrics_dict = {}
    for query_name, specs in config.infer_settings["query_dict"].items():
        try:
            metrics_dict[query_name] = evaluate_batch(
                dataframe=sample,
                view_cols=image_cols,
                label_col=specs["label_col"],
                query2label=specs["query2label"],
                model=base_model,
                config=config,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.error(f"Exception @ inference: {e}")
            metrics[query_name] = None
    # Log the metrics
    for qname, metrics in metrics_dict.items():
        if metrics is None:
            continue
        for metric in config.infer_settings["logger_metrics"]:
            logger.info(f"{qname.upper()} {metric}:\n{metrics[metric]}")
        for metric in config.infer_settings["tensorboard_metrics"]:
            tb_writer.add_scalar(
                f"metric/{qname}/{metric}", metrics[metric], global_step=global_step
            )
    return metrics_dict
