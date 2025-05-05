import torch
from transformers import get_wsd_schedule
from custom.model import MammoCLIP
from custom.config import Config
from .stats import stats_from_epochs
from .freezer import freeze_submodules
from torch.nn.parallel import DistributedDataParallel as DDP
from .dist_utils import get_rank, is_dist_avail_and_initialized
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel.distributed import _MixedPrecision
from .dist_utils import get_world_size


def build_model_and_optim(
    config: Config, dataset_size: int, resume_from: str = None
) -> tuple:
    if resume_from is not None:
        # resume training
        model = MammoCLIP.from_pretrained(resume_from, **config.pretrained_model_cfg)
    else:
        # from scratch
        model = MammoCLIP.from_vision_text_pretrained(**config.pretrained_model_cfg)

    # freeze submodules if needed
    if config.freeze_text_model:
        freeze_submodules(model, ["text_model"], True)
    if config.freeze_vision_model:
        freeze_submodules(model, ["vision_model"], True)

    optimizer_cls = {
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adafactor": torch.optim.Adafactor,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
    }.get(config.training_params["optimizer"].lower(), torch.optim.AdamW)

    if is_dist_avail_and_initialized():
        optimizer = ZeroRedundancyOptimizer(
            params=model.parameters(),
            optimizer_class=optimizer_cls,
            lr=config.training_params["lr_max"],
            weight_decay=config.training_params["weight_decay"],
        )
    else:
        optimizer = optimizer_cls(
            params=model.parameters(),
            lr=config.training_params["lr_max"],
            weight_decay=config.training_params["weight_decay"],
        )

    stats = stats_from_epochs(
        num_epochs=config.training_params["num_epochs"],
        num_gpus=torch.cuda.device_count(),
        accumulation_steps=config.training_params["gradient_accumulation_steps"],
        per_gpu_batch_size=config.training_params["batch_size"],
        dataset_size=dataset_size,
    )

    warmup = (
        int(
            stats["total_optimization_steps"]
            * config.training_params["warmup_fraction"]
        )
        // get_world_size()
    )
    warmup = 0 if resume_from else warmup
    steady = (
        int(
            stats["total_optimization_steps"]
            * config.training_params["steady_fraction"]
        )
        // get_world_size()
    )
    steady = 0 if resume_from else steady
    total = stats["total_optimization_steps"] // get_world_size()

    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup,
        num_stable_steps=steady,
        num_decay_steps=total - warmup - steady,
        min_lr_ratio=config.training_params["lr_min"]
        / config.training_params["lr_max"],
    )

    rank = get_rank()
    model = model.to(rank)
    if is_dist_avail_and_initialized():
        model = DDP(
            model,
            device_ids=[rank],
            **config.ddp_kwargs,
        )

    return model, optimizer, scheduler, stats, warmup, steady, total
