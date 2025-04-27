import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from custom import MammogramDataset, Config


def split_df(df: pd.DataFrame, config: Config):
    id_col = config.train_ds["attrs_"]["pid_col"]
    unique_patients = df[id_col].unique()
    np.random.shuffle(unique_patients)
    train_patients = unique_patients[
        : int(config.training_params["train_fraction"] * len(unique_patients))
    ]
    val_patients = unique_patients[
        int(config.training_params["train_fraction"] * len(unique_patients)) :
    ]
    df_train = df[df[id_col].isin(train_patients)].copy(True).reset_index(drop=True)
    df_val = df[df[id_col].isin(val_patients)].copy(True).reset_index(drop=True)
    return df_train, df_val


def _make_sampler(weights, dataset_len):
    return (
        WeightedRandomSampler(
            weights=weights, num_samples=dataset_len, replacement=True
        )
        if weights is not None
        else None
    )


def _make_dataloader(ds, sampler, batch_size, num_workers, shuffle_if_no_sampler: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle_if_no_sampler),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


def prepare_dataloaders(config: Config, resuming: bool = False):
    # 1) load + split if starting fresh
    if not resuming:
        clean_cols = config.train_ds["attrs_"]["image_cols"] + [
            config.train_ds["attrs_"]["text_col"],
            config.train_ds["attrs_"]["pid_col"],
            config.train_ds["attrs_"]["weights_col"],
        ]
        df = pd.read_csv(config.csv_path)
        df = df.dropna(subset=clean_cols)
        train_df, val_df = split_df(df, config)
        os.makedirs(config.project_dir, exist_ok=True)
        train_df.to_csv(config.train_ds["attrs_"]["path_to_df"], index=False)
        val_df.to_csv(config.val_ds["attrs_"]["path_to_df"], index=False)

    # 2) build Datasets
    train_ds = MammogramDataset.from_dict(config.train_ds)
    val_ds = MammogramDataset.from_dict(config.val_ds)

    # 3) samplers
    train_sampler = _make_sampler(train_ds.get_weights(), len(train_ds))
    val_sampler = _make_sampler(val_ds.get_weights(), len(val_ds))

    # 4) dataloaders
    bs = config.training_params["batch_size"]
    workers = config.dl_workers
    train_dl = _make_dataloader(
        train_ds, train_sampler, bs, workers["train"], shuffle_if_no_sampler=True
    )
    val_dl = _make_dataloader(
        val_ds, val_sampler, bs, workers["val"], shuffle_if_no_sampler=False
    )

    return train_dl, val_dl, len(train_ds), len(val_ds)
