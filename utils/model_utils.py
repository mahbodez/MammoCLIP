import torch
import os
from transformers import get_wsd_schedule
from custom.model import MammoCLIP
from custom.config import Config
from .stats import stats_from_epochs
from .freezer import freeze_submodules
from torch.nn.parallel import DistributedDataParallel as DDP
from .dist_utils import get_rank, is_dist_avail_and_initialized


def build_param_groups(
    model,
    base_lr,
    lr_mul,
    wd,
    vision_prefix="vision_model",
    text_prefix="text_model",
    fusion_prefix="fusion_",
):
    """Return a list of param groups with no duplicates."""
    all_seen = set()  # ids we have already added
    groups = []

    def add_block(named_iter, lr_factor):
        decay, no_decay = [], []
        for n, p in named_iter:
            if not p.requires_grad or id(p) in all_seen:
                continue  # already in another block
            all_seen.add(id(p))

            if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)

        if decay:
            groups.append(
                {"params": decay, "lr": base_lr * lr_factor, "weight_decay": wd}
            )
        if no_decay:
            groups.append(
                {"params": no_decay, "lr": base_lr * lr_factor, "weight_decay": 0.0}
            )

    # ---- vision ------------------------------------------------------------
    add_block(
        ((n, p) for n, p in model.named_parameters() if n.startswith(vision_prefix)),
        lr_mul["vision"],
    )

    # ---- text --------------------------------------------------------------
    add_block(
        ((n, p) for n, p in model.named_parameters() if n.startswith(text_prefix)),
        lr_mul["text"],
    )

    # ---- fusion (= everything else) ----------------------------------------
    add_block(
        ((n, p) for n, p in model.named_parameters() if n.startswith(fusion_prefix)),
        lr_mul["fusion"],
    )

    return groups


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
    rank = get_rank()

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

    # collect parameter groups ------------------------------------------------
    base_lr = config.training_params["lr_max"]
    min_lr_ratio = config.training_params["lr_min"] / config.training_params["lr_max"]
    wd = config.training_params.get("weight_decay", 0.05)

    lr_mul = {
        "vision": config.training_params.get("vision_lr_mul", 1.0),
        "text": config.training_params.get("text_lr_mul", 1.0),
        "fusion": config.training_params.get("fusion_lr_mul", 5.0),
    }

    param_groups = build_param_groups(
        model,
        base_lr=base_lr,
        lr_mul=lr_mul,
        wd=wd,
        vision_prefix="vision_model",
        text_prefix="text_model",
        fusion_prefix="fusion_",
    )

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

    betas = config.training_params.get("betas", (0.9, 0.98))
    eps = config.training_params.get("eps", 1e-8)

    assert len({id(p) for g in param_groups for p in g["params"]}) == sum(
        len(g["params"]) for g in param_groups
    ), "Duplicate parameter detected!"

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    # --- RESUME ONLY: load optimizer state ------------------------------
    if resuming:
        opt_path = os.path.join(resume_from, "optimizer.pt")
        sch_path = os.path.join(resume_from, "scheduler.pt")
        if not os.path.isfile(opt_path):
            raise FileNotFoundError(f"Optimizer checkpoint not found at {opt_path}")
        if not os.path.isfile(sch_path):
            raise FileNotFoundError(f"Scheduler checkpoint not found at {sch_path}")
        optimizer.load_state_dict(torch.load(opt_path))
        # move optimizer state tensors to the correct device after loading
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(rank)

    # --- COMMON: build scheduler ----------------------------------------
    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup,
        num_stable_steps=steady,
        num_decay_steps=total - warmup - steady,
        min_lr_ratio=min_lr_ratio,
    )

    # --- RESUME ONLY: load scheduler state ------------------------------
    if resuming:
        scheduler.load_state_dict(torch.load(sch_path))

    # move to device / wrap DDP
    model = model.to(rank)
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[rank], **config.ddp_kwargs)

    return model, optimizer, scheduler, stats, warmup, steady, total
