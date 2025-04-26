import re
from typing import Tuple
import torch
from custom import (
    MammoCLIP,
    Config,
)
from torch.utils.data import DataLoader
import os
import torch.distributed as dist
from accelerate import DistributedDataParallelKwargs, Accelerator
from logging import Logger
import argparse as ap
from utils import (
    build_model_and_optim,
    prepare_dataloaders,
    set_seed,
    init_accelerator_and_logger,
    train_one_epoch,
    eval_and_checkpoint,
)


def training_loop(
    config: Config,
    accelerator: Accelerator,
    logger: Logger,
    model: MammoCLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_dl: DataLoader,
    val_dl: DataLoader,
    stats: dict = None,
    resuming: bool = False,
    starting_epoch: int = 0,  # for resuming
):
    assert (not resuming and starting_epoch == 0) or (
        resuming and starting_epoch > 0
    ), "Starting epoch must be 0 if not resuming, or > 0 if resuming."

    # Prepare everything
    (
        model,
        optimizer,
        scheduler,
        train_dl,
        val_dl,
    ) = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        train_dl,
        val_dl,
    )

    opt_steps = 0.0
    total_steps = stats["total_optimization_steps"]

    # Now you train the model
    for epoch in range(
        starting_epoch, config.training_params["num_epochs"] + starting_epoch
    ):
        opt_steps = train_one_epoch(
            epoch=epoch,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            accelerator=accelerator,
            total_steps=total_steps,
            starting_epoch=starting_epoch,
            optimization_steps=opt_steps,
        )

        eval_and_checkpoint(
            epoch=epoch,
            config=config,
            model=model,
            val_dl=val_dl,
            accelerator=accelerator,
            logger=logger,
            resuming=resuming,
            optimization_steps=opt_steps,
        )
    accelerator.clear()


def _find_latest_checkpoint(project_dir: str) -> Tuple[int, str]:
    """
    Returns (latest_epoch, checkpoint_dir) or (0, "") if none found.
    """
    model_dirs = [
        f
        for f in os.listdir(project_dir)
        if re.match(r"(model_resumed|model)_(\d+)$", f)
        and os.path.isdir(os.path.join(project_dir, f))
    ]
    if not model_dirs:
        print("No previous epochs found.")
        return 0, ""
    most_recent = max(model_dirs, key=lambda d: int(re.search(r"\d+", d).group()))
    epoch = int(re.search(r"\d+", most_recent).group())
    return epoch, os.path.join(project_dir, most_recent)


def _patch_config_for_resume(config: Config, latest_epoch: int) -> Config:
    """
    Adjusts lr schedule and remaining epochs in-place.
    """
    total_epochs = config.training_params["num_epochs"]
    # no work to do if already completed
    if latest_epoch >= total_epochs:
        return config
    # linear decay of max lr
    lr_max = config.training_params["lr_max"] * (1 - latest_epoch / total_epochs)
    config.training_params["lr_max"] = lr_max
    config.training_params["num_epochs"] = total_epochs - latest_epoch
    return config


def build_training_run(config: Config, resume_dir: str = None):
    set_seed(config.seed)
    train_dl, val_dl, n_train, _ = prepare_dataloaders(
        config=config,
        resuming=resume_dir is not None,
    )
    model, opt, sched, stats, warmup, steady, total = build_model_and_optim(
        config=config,
        dataset_size=n_train,
        resume_from=resume_dir,
    )
    return (train_dl, val_dl, model, opt, sched, stats, warmup, steady, total)


def train_from_scratch(cfg_path: str):
    # torch.autograd.set_detect_anomaly(True)
    if cfg_path.lower().endswith(".yaml") or cfg_path.lower().endswith(".yml"):
        config = Config.from_yaml(cfg_path)
    elif cfg_path.lower().endswith(".json"):
        config = Config.from_json(cfg_path)
    else:
        raise ValueError("Unsupported config file format. Use .yaml or .json.")

    train_dl, val_dl, model, opt, sched, stats, warmup, steady, total = (
        build_training_run(config, None)
    )

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True, broadcast_buffers=False
    )
    # Initialize accelerator and tensorboard logging
    accelerator, logger = init_accelerator_and_logger(
        config=config,
        # ddp_kwargs=ddp_kwargs,
    )

    # number of parameters
    if accelerator.is_main_process:
        model_size = sum(p.numel() for p in model.parameters())
        trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Starting training from scratch...")
        logger.info(config)
        logger.info(f"Model size: {model_size:,}")
        logger.info(
            f"Trainable parameters: {trainable_size:,} ({100 * trainable_size / model_size:.2f}%)"
        )
        logger.info(f"Effective batch size: {stats['effective_batch_size']}")
        logger.info(
            f"Optimization steps per epoch: {stats['optimization_steps_per_epoch']}"
        )
        logger.info(f"Total Optimization steps: {stats['total_optimization_steps']}")
        logger.info(f"Warmup steps: {warmup}")
        logger.info(f"Steady steps: {steady}")
        logger.info(f"Total steps: {total}")

    training_loop(
        config=config,
        accelerator=accelerator,
        logger=logger,
        model=model,
        optimizer=opt,
        scheduler=sched,
        train_dl=train_dl,
        val_dl=val_dl,
        stats=stats,
    )


def resume(project_dir: str):
    latest_epoch, resume_dir = _find_latest_checkpoint(project_dir)
    if latest_epoch == 0 or resume_dir == "":
        print("No previous epochs found.")
        return

    config = Config.from_yaml(os.path.join(project_dir, "config.yaml"))
    config = _patch_config_for_resume(config, latest_epoch)
    if config.training_params["num_epochs"] == 0:
        print("No remaining epochs to resume.")
        return

    # delegate seeding, dataloaders, model/optim/sched setup
    train_dl, val_dl, model, opt, sched, stats, warmup, steady, total = (
        build_training_run(config, resume_dir)
    )

    # Initialize accelerator and tensorboard logging
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True, broadcast_buffers=False
    )
    # Initialize accelerator and tensorboard logging
    accelerator, logger = init_accelerator_and_logger(
        config=config,
        # ddp_kwargs=ddp_kwargs,
    )

    # number of parameters
    if accelerator.is_main_process:
        model_size = sum(p.numel() for p in model.parameters())
        trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Resuming training from epoch {latest_epoch} in {resume_dir}...")
        logger.info(config)
        logger.info(f"Model size: {model_size:,}")
        logger.info(
            f"Trainable parameters: {trainable_size:,} ({100 * trainable_size / model_size:.2f}%)"
        )
        logger.info(f"Effective batch size: {stats['effective_batch_size']}")
        logger.info(
            f"Optimization steps per epoch: {stats['optimization_steps_per_epoch']}"
        )
        logger.info(f"Total Optimization steps: {stats['total_optimization_steps']}")
        logger.info(f"Warmup steps: {warmup}")
        logger.info(f"Steady steps: {steady}")
        logger.info(f"Total steps: {total}")

    # hand off to your existing loop
    training_loop(
        config=config,
        accelerator=accelerator,
        logger=logger,
        model=model,
        optimizer=opt,
        scheduler=sched,
        train_dl=train_dl,
        val_dl=val_dl,
        stats=stats,
        resuming=True,
        starting_epoch=latest_epoch,
    )


def parse():
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="",
        help="Path to the project directory [Used for resuming training]",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to the config file (json or yaml) [Used for training from scratch]",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint [Used to continue training from a saved state]",
    )
    args = parser.parse_args()

    assert args.resume or args.config, "Either --resume or --config must be provided."
    if args.dir and args.config and args.resume:
        raise ValueError(
            "Cannot provide both --dir and --config with --resume. Use one or the other."
        )
    return args


if __name__ == "__main__":
    args = parse()
    try:
        if args.resume and os.path.exists(args.dir):
            # Resume training from checkpoint
            resume(args.dir)
        elif not args.resume and args.config and not args.dir:
            train_from_scratch(args.config)
        else:
            raise ValueError(
                "Invalid arguments: Either provide a valid directory to resume or a config file to train from scratch."
            )
    except KeyboardInterrupt:
        print(f"Process {os.getpid()} received KeyboardInterrupt.")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
