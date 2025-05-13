import pytest
import torch.distributed as dist
from custom.sampler import DistributedWeightedRandomSampler


@pytest.fixture(autouse=True)
def reset_dist(monkeypatch):
    # Default to single-process (no distributed)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    yield


def test_single_rank_reproducible_same_seed():
    weights = [0.1, 0.2, 0.7]
    sampler1 = DistributedWeightedRandomSampler(
        weights, num_samples_total=10, replacement=True, seed=123
    )
    sampler2 = DistributedWeightedRandomSampler(
        weights, num_samples_total=10, replacement=True, seed=123
    )
    sampler1.set_epoch(0)
    sampler2.set_epoch(0)
    out1 = list(iter(sampler1))
    out2 = list(iter(sampler2))
    assert out1 == out2


def test_single_rank_without_replacement_unique():
    weights = [1, 1, 1, 1, 1]
    sampler = DistributedWeightedRandomSampler(
        weights, num_samples_total=5, replacement=False, seed=0
    )
    sampler.set_epoch(2)
    out = list(iter(sampler))
    assert sorted(out) == list(range(len(weights)))
    assert len(set(out)) == len(weights)


def test_multi_rank_combined_counts(monkeypatch):
    # Simulate two distributed ranks
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    results = []
    num_samples_total = 6
    weights = [1, 1, 1, 1]
    for rank in [0, 1]:
        monkeypatch.setattr(dist, "get_rank", lambda r=rank: r)
        sampler = DistributedWeightedRandomSampler(
            weights,
            num_samples_total=num_samples_total,
            replacement=True,
            seed=42,
            drop_last=True,
        )
        sampler.set_epoch(1)
        out = list(iter(sampler))
        assert len(out) == len(sampler)
        results.extend(out)
    # Check total collected equals world_size * num_samples_per_rank
    assert len(results) == sampler.world_size * sampler.num_samples_per_rank


def test_len_reports_correct_samples():
    weights = [1] * 10
    sampler = DistributedWeightedRandomSampler(
        weights, num_samples_total=7, replacement=True, seed=0
    )
    assert len(sampler) == 7
