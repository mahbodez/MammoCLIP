from tqdm.auto import tqdm
import torch
from torch.amp import autocast, GradScaler
from .dist_utils import is_main_process, synchronize, get_rank
from custom.config import Config
from .stats import poll_gpu_stats
from logging import Logger
from torch.utils.tensorboard import SummaryWriter


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
                for group in range(len(optimizer.param_groups)):
                    tb_writer.add_scalar(
                        f"train/lr{group}",
                        scheduler.get_last_lr()[group],
                        global_step=opt_steps,
                    )
                for k, v in gpu_stats.items():
                    tb_writer.add_scalar(f"gpu/{k}", v, global_step=opt_steps)

        # log progress
        pbar.update(1 / grad_acc)
        pbar.set_postfix(
            {
                "loss": loss.item() * grad_acc,
                **{
                    f"lr{group}": scheduler.get_last_lr()[group]
                    for group in range(len(optimizer.param_groups))
                },
                **gpu_stats,
            }
        )

    pbar.close()
    return opt_steps
