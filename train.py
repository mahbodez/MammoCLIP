import re
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import os
from logging import Logger
from torch.utils.tensorboard import SummaryWriter
import argparse as ap

from custom.model import MammoCLIP
from custom.config import Config
from utils.dist_utils import is_main_process, init_distributed_mode, cleanup
from utils.model_utils import build_model_and_optim
from utils.data_utils import prepare_dataloaders
from utils.seed_utils import set_seed
from utils.train_loop_utils import train_one_epoch, eval_and_checkpoint
from utils.resume import find_latest_checkpoint
from utils.logger import init_logger, init_tensorboard


def training_loop(
    config: Config,
    logger: Logger,
    tb_writer: SummaryWriter,
    model: MammoCLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_dl: DataLoader,
    val_dl: DataLoader,
    resuming: bool = False,
    starting_epoch: int = 0,  # for resuming
):
    assert (not resuming and starting_epoch == 0) or (
        resuming and starting_epoch > 0
    ), "Starting epoch must be 0 if not resuming, or > 0 if resuming."

    opt_steps = 0.0

    best_metric = float("inf")

    # Now you train the model
    for epoch in range(starting_epoch, config.training_params["num_epochs"]):
        opt_steps = train_one_epoch(
            epoch=epoch,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            logger=logger,
            tb_writer=tb_writer,
            optimization_steps=opt_steps,
        )

        best_metric = eval_and_checkpoint(
            epoch=epoch,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_dl=val_dl,
            logger=logger,
            tb_writer=tb_writer,
            resuming=resuming,
            optimization_steps=opt_steps,
            best_metric=best_metric,
        )
    torch.cuda.empty_cache()


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

    # Initialize logger
    logger = init_logger(config)
    tb_writer = init_tensorboard(config)

    # number of parameters
    if is_main_process():
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
        logger=logger,
        tb_writer=tb_writer,
        model=model,
        optimizer=opt,
        scheduler=sched,
        train_dl=train_dl,
        val_dl=val_dl,
    )


def resume(project_dir: str):
    latest_epoch, resume_dir = find_latest_checkpoint(project_dir)
    if latest_epoch == 0 or resume_dir == "":
        print("No previous epochs found.")
        return

    config = Config.from_yaml(os.path.join(project_dir, "config.yaml"))

    if latest_epoch == config.training_params["num_epochs"]:
        print("Training already completed.")
        return

    # delegate seeding, dataloaders, model/optim/sched setup
    train_dl, val_dl, model, opt, sched, stats, warmup, steady, total = (
        build_training_run(config, resume_dir)
    )

    # Initialize logger
    logger = init_logger(config)
    tb_writer = init_tensorboard(config)

    # number of parameters
    if is_main_process():
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
        logger=logger,
        tb_writer=tb_writer,
        model=model,
        optimizer=opt,
        scheduler=sched,
        train_dl=train_dl,
        val_dl=val_dl,
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


def main():
    args = parse()
    try:
        init_distributed_mode()
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
        cleanup()


if __name__ == "__main__":
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
