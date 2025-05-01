import torch
from transformers import get_wsd_schedule
from custom.model import MammoCLIP
from custom.config import Config
from .stats import stats_from_epochs
from .freezer import freeze_submodules
from torch.nn.parallel import DistributedDataParallel as DDP
from .dist_utils import get_rank, is_dist_avail_and_initialized
from torch.distributed.optim import ZeroRedundancyOptimizer


def build_model_and_optim(
    config: Config, dataset_size: int, resume_from: str = None
) -> tuple:
    if resume_from is not None:
        # resume training
        model = MammoCLIP.from_pretrained(resume_from)
    else:
        # from scratch
        model = MammoCLIP.from_vision_text_pretrained(**config.pretrained_model_cfg)

    # freeze submodules if needed
    if config.freeze_text_model:
        freeze_submodules(model, ["text_model"], True)
    if config.freeze_vision_model:
        freeze_submodules(model, ["vision_model"], True)

    optimizer = ZeroRedundancyOptimizer(
        params=[p for p in model.parameters() if p.requires_grad],
        optimizer_class=torch.optim.Adafactor,
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

    warmup = int(
        stats["total_optimization_steps"] * config.training_params["warmup_fraction"]
    )
    warmup = 0 if resume_from else warmup
    steady = int(
        stats["total_optimization_steps"] * config.training_params["steady_fraction"]
    )
    steady = 0 if resume_from else steady
    total = stats["total_optimization_steps"]

    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup,
        num_stable_steps=steady,
        num_decay_steps=total - warmup - steady,
        min_lr_ratio=config.training_params["lr_min"]
        / config.training_params["lr_max"],
    )

    if is_dist_avail_and_initialized():
        rank = get_rank()
        model = model.to(rank)
        model = DDP(
            model,
            device_ids=[rank],
            **config.ddp_kwargs,
            mixed_precision=config.training_params["mixed_precision"]
        )

    return model, optimizer, scheduler, stats, warmup, steady, total
