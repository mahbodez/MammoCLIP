import os
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from typing import Tuple, Dict
from custom import Config
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


def init_accelerator_and_logger(
    config: Config, ddp_kwargs: DistributedDataParallelKwargs = None
) -> Tuple[Accelerator, Logger]:
    accel = Accelerator(
        mixed_precision=config.training_params["mixed_precision"],
        gradient_accumulation_steps=config.training_params[
            "gradient_accumulation_steps"
        ],
        log_with=["tensorboard"],
        project_dir=os.path.join(config.project_dir, "logs"),
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else [],
    )
    logger = None
    if accel.is_main_process:
        os.makedirs(config.project_dir, exist_ok=True)
        logger = setup_logger(
            "training",
            log_to_file=os.path.join(config.project_dir, "logs", "train.log"),
        )
        accel.init_trackers(config.project_dir)
    return accel, logger


def poll_gpu_stats() -> Dict[str, float]:
    return {
        "vram": get_gpu_memory_usage().get("percentage", 0.0),
        "pwr": get_gpu_power_usage().get("power", 0.0),
        "temp": get_gpu_temperature().get("temp", 0.0),
    }


def train_one_epoch(
    epoch: int,
    config: Config,
    model,
    optimizer,
    scheduler,
    train_dl,
    accelerator: Accelerator,
    total_steps: int,
    starting_epoch: int,
    optimization_steps: float = 0.0,
) -> float:
    grad_acc = config.training_params["gradient_accumulation_steps"]
    pbar = tqdm(
        total=len(train_dl) / grad_acc,
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch+1}/{starting_epoch+config.training_params['num_epochs']}",
    )
    gpu_stats = poll_gpu_stats()
    opt_steps = optimization_steps
    model.train()
    for i, batch in enumerate(train_dl):
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, return_loss=True).loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), config.training_params["max_grad_norm"]
                )
            optimizer.step()
            scheduler.step()

        opt_steps += 1 / grad_acc
        if accelerator.is_main_process and int(opt_steps) % 10 == 0:
            gpu_stats = poll_gpu_stats()

        logs = {
            "step": f"{int(opt_steps)}/{total_steps}",
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            **gpu_stats,
        }
        pbar.update(1 / grad_acc)
        pbar.set_postfix(**logs)
        accelerator.log(logs, step=int(opt_steps))

    pbar.close()
    return opt_steps


def eval_and_checkpoint(
    epoch: int,
    config: Config,
    model,
    val_dl,
    accelerator: Accelerator,
    logger: Logger,
    resuming: bool,
    optimization_steps: float,
):
    # eval
    if (epoch + 1) % config.eval_interval == 0:
        evaluate(model, val_dl, accelerator, logger, int(optimization_steps))

    # save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        last = epoch == config.training_params["num_epochs"] - 1
        if ((epoch + 1) % config.save_interval == 0) or last:
            prefix = "model_resumed" if resuming else "model"
            cleanup_checkpoints(
                config.project_dir, prefix, config.max_checkpoints, logger
            )
            save_checkpoint(
                accelerator.unwrap_model(model),
                config.project_dir,
                prefix,
                epoch + 1,
                logger,
            )
            if not resuming:
                config.to_yaml(os.path.join(config.project_dir, "config.yaml"))
