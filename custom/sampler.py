import math, torch
from torch.utils.data import Sampler
import torch.distributed as dist
from logging import getLogger

log = getLogger(__name__)


class DistributedWeightedRandomSampler(Sampler[int]):
    """
    Draws `num_samples_total` weighted samples from the whole dataset, then
    hands each distributed rank an equally‑sized slice.  Works under
    torch.distributed (DDP), HuggingFace `accelerate`, or single‑process mode.

    - With replacement: behaves like torch.utils.data.WeightedRandomSampler.
    - Without replacement: raises if you ask for more samples than items.
    - Call   sampler.set_epoch(epoch)   at the start of every epoch.
    """

    def __init__(
        self,
        weights,  # sequence‑like of length = len(dataset)
        num_samples_total: int = None,  # if None → len(weights)
        replacement: bool = True,
        generator: torch.Generator | None = None,
        seed: int = 42,
        drop_last: bool = False,  # like DistributedSampler
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1‑D")

        self.replacement = replacement
        self.generator = generator or torch.Generator(device="cpu")
        self.base_seed = seed
        self.epoch = 0

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_samples_total = num_samples_total or len(self.weights)

        if not replacement and self.num_samples_total > len(self.weights):
            raise ValueError(
                f"With replacement=False you asked for "
                f"{self.num_samples_total} samples but dataset has only "
                f"{len(self.weights)} items."
            )

        # -----   make per‑rank bookkeeping identical to DistributedSampler -----
        if drop_last:
            self.num_samples_per_rank = self.num_samples_total // self.world_size
        else:
            self.num_samples_per_rank = math.ceil(
                self.num_samples_total / self.world_size
            )
        self.total_size = self.num_samples_per_rank * self.world_size
        # ----------------------------------------------------------------------
        log.info("Make sure to use `set_epoch(epoch)` at the start of every epoch.")
        log.info(
            "Initialized DistributedWeightedRandomSampler with total samples: %d",
            self.num_samples_total,
        )
        log.info("Rank %d will sample %d items", self.rank, self.num_samples_per_rank)

    # accelerate will call this automatically; DDP won't, so do it yourself.
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.base_seed + self.epoch)

        # 1. Draw the **global** list of indices
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=g,
        )

        # 2. Slice out this rank's view
        indices_rank = global_indices[self.rank : self.total_size : self.world_size]

        # 3. Return python ints (what DataLoader wants)
        return iter(indices_rank.tolist())

    def __len__(self):
        return self.num_samples_per_rank
