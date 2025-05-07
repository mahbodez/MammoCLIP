import torch
import os
from transformers import get_wsd_schedule
from custom.model import MammoCLIP
from custom.config import Config
from .stats import stats_from_epochs
from .freezer import freeze_submodules
from torch.nn.parallel import DistributedDataParallel as DDP
from .dist_utils import get_rank, is_dist_avail_and_initialized


def build_model_and_optim(
    config: Config,
    dataset_size: int,
    resume_from: str = None,
) -> tuple:
    """
    Build the model, optimizer, and scheduler for training.
    Args:
        config (Config): Configuration object.
        dataset_size (int): Size of the dataset.
        resume_from (str, optional): Path to resume from. Defaults to None.
    Returns:
        tuple: Model, optimizer, scheduler, stats, warmup, steady, total.
    """
    resuming = resume_from is not None

    # load model
    if resuming:
        model = MammoCLIP.from_pretrained(resume_from, **config.pretrained_model_cfg)
    else:
        model = MammoCLIP.from_vision_text_pretrained(**config.pretrained_model_cfg)

    # freeze if requested
    if config.freeze_text_model:
        freeze_submodules(model, ["text_model"], True)
    if config.freeze_vision_model:
        freeze_submodules(model, ["vision_model"], True)

    # pick optimizer class
    optimizer_cls = {
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adafactor": torch.optim.Adafactor,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
    }.get(config.training_params["optimizer"].lower(), torch.optim.AdamW)

    # --- COMMON: compute stats and instantiate optimizer -----------------
    stats = stats_from_epochs(
        num_epochs=config.training_params["num_epochs"],
        accumulation_steps=config.training_params["gradient_accumulation_steps"],
        per_gpu_batch_size=config.training_params["batch_size"],
        dataset_size=dataset_size,
    )
    total = stats["total_optimization_steps"]
    warmup = int(total * config.training_params["warmup_fraction"])
    steady = int(total * config.training_params["steady_fraction"])

    optimizer = optimizer_cls(
        params=model.parameters(),
        lr=config.training_params["lr_max"],
        **config.training_params["optimizer_kwargs"],
    )

    # --- RESUME ONLY: load optimizer state ------------------------------
    if resuming:
        opt_path = os.path.join(resume_from, "optimizer.pt")
        sch_path = os.path.join(resume_from, "scheduler.pt")
        if not os.path.isfile(opt_path):
            raise FileNotFoundError(f"Optimizer checkpoint not found at {opt_path}")
        if not os.path.isfile(sch_path):
            raise FileNotFoundError(f"Scheduler checkpoint not found at {sch_path}")
        optimizer.load_state_dict(torch.load(opt_path))

    # --- COMMON: build scheduler ----------------------------------------
    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup,
        num_stable_steps=steady,
        num_decay_steps=total - warmup - steady,
        min_lr_ratio=config.training_params["lr_min"]
        / config.training_params["lr_max"],
    )

    # --- RESUME ONLY: load scheduler state ------------------------------
    if resuming:
        scheduler.load_state_dict(torch.load(sch_path))

    # move to device / wrap DDP
    rank = get_rank()
    model = model.to(rank)
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[rank], **config.ddp_kwargs)
    optimizer.to(rank)
    scheduler.to(rank)

    return model, optimizer, scheduler, stats, warmup, steady, total
