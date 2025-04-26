from typing import List, Tuple
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import pandas as pd
import os
from transformers import PreTrainedTokenizerBase, BatchEncoding, AutoTokenizer
import yaml
from .dictable import Dictable
from .preprocessing import MammogramPreprocessor, MammogramTransform
import hashlib
from copy import deepcopy


class MammogramDataset(Dataset, Dictable):
    """
    A PyTorch Dataset for loading and preprocessing mammogram images and associated text data.

    This dataset supports image caching, preprocessing, and optional augmentation.
    Images are loaded from file paths specified in a pandas DataFrame, preprocessed, and optionally augmented before being returned along with their corresponding text.

    Args:
        df (pandas.DataFrame): DataFrame containing metadata, image paths, and text.
        path_to_df (str): Path to the CSV file containing metadata, image paths, and text.
        pid_col (str): Column name for patient or sample IDs.
        image_cols (List[str]): List of column names containing image file paths.
        text_col (str): Column name for the primary text annotation.
        alt_text_col (str): Column name for alternative text annotation.
        image_preprocessor (MammogramPreprocessor): Callable for preprocessing images.
        transform_function (MammogramTransform, optional): Callable for augmenting images. Defaults to None.
        tokenizer (PreTrainedTokenizerBase, optional): Tokenizer for text processing. Defaults to None.
        tokenizer_kwargs (dict, optional): Additional arguments for the tokenizer when calling it. Defaults to {}.
        alt_text_prob (float, optional): Probability of using alternative text instead of primary text. Defaults to 0.0.
        cache_dir (str, optional): Directory for caching preprocessed images. Defaults to ".cache".

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Loads and returns the (possibly cached and augmented) images and text for the given index.
        clear_cache(): Clears the cache directory for this dataset instance.
        clear_cache_dir(cache_dir): Static method to clear a specified cache directory.

    Returns:
        BatchEncoding: A dictionary containing the preprocessed and augmented images and tokenized text.
    """

    def __init__(
        self,
        pid_col: str,
        image_cols: List[str],
        text_col: str,
        image_preprocessor: MammogramPreprocessor,
        alt_text_cols: List[str] = None,
        weights_col: str = None,
        df: pd.DataFrame = None,
        path_to_df: str = None,
        transform_function: MammogramTransform = None,
        tokenizer: PreTrainedTokenizerBase = None,
        tokenizer_kwargs: dict = {},
        alt_text_prob: float = 0.0,
        cache_dir: str = None,
    ):
        if path_to_df:
            self.path_to_df = path_to_df
            self.df = pd.read_csv(self.path_to_df).dropna(
                how="any", subset=[pid_col] + image_cols + [text_col]
            )
        elif df is not None:
            self.path_to_df = None
            self.df = df.dropna(how="any", subset=[pid_col] + image_cols + [text_col])
        else:
            raise ValueError("Either `df` or `path_to_df` have to be specified!")
        if weights_col and isinstance(weights_col, str):
            assert (
                weights_col in self.df.columns
            ), f"{weights_col} not in dataframe columns!"
            self.weights_col = weights_col
        else:
            self.weights_col = None
        self.pid_col = pid_col
        self.image_cols = image_cols
        self.text_col = text_col
        self.alt_text_cols = alt_text_cols or []
        # Normalize & validate alt_text_cols
        if self.alt_text_cols:
            if isinstance(self.alt_text_cols, str):
                self.alt_text_cols = [self.alt_text_cols]
            missing = [c for c in self.alt_text_cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"alt_text_cols not in dataframe columns: {missing}")
        self.image_preprocessor = image_preprocessor
        self.transform_function = transform_function
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.alt_text_prob = alt_text_prob
        self.cache_dir = cache_dir or ".cache"

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __len__(self):
        return len(self.df)

    def get_weights(self):
        """
        Get the weights for each sample in the dataset.
        Returns:
            torch.Tensor: A tensor containing the weights for each sample.
        """
        if self.weights_col is not None:
            return self.df[self.weights_col].values
        else:
            return None

    @property
    def name(self):
        return "MammogramDataset"

    def to_dict(self):
        """
        Convert the dataset configuration to a dictionary.
        Returns:
            dict: Configuration dictionary.
        """
        d = super(MammogramDataset, self).to_dict()
        d["attrs_"]["tokenizer"] = self.tokenizer.name_or_path or "unspecified"
        return d

    def save_config(self, config_path: str):
        """
        Save the dataset configuration to a YAML file.
        Args:
            config_path (str): Path to save the configuration file.
        """
        config = self.to_dict()
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    @classmethod
    def from_dict(cls, configuration: dict) -> "MammogramDataset":
        """
        Load the dataset configuration from a dictionary.
        Args:
            configuration (dict): Configuration dictionary.
        Returns:
            MammogramDataset: An instance of the dataset with the loaded configuration.
        """
        config = deepcopy(configuration)
        if config["class_"] != "MammogramDataset":
            raise ValueError(
                f"Expected class_ to be 'MammogramDataset', but got {config['class_']}"
            )
        image_preprocessor_cfg = config["attrs_"].pop("image_preprocessor")
        image_preprocessor_class = image_preprocessor_cfg.pop("class_")
        image_preprocessor_params = image_preprocessor_cfg.pop("attrs_")

        image_preprocessor_class = globals().get(
            image_preprocessor_class, MammogramPreprocessor
        )
        image_preprocessor = (
            image_preprocessor_class.from_dict(image_preprocessor_params)
            if image_preprocessor_class
            else None
        )

        transform_function_cfg = config["attrs_"].pop("transform_function")
        transform_function_class = transform_function_cfg.pop("class_")
        transform_function_params = transform_function_cfg.pop("attrs_")

        transform_function_class = globals().get(transform_function_class, None)
        transform_function_params = (
            transform_function_params if transform_function_class else {}
        )
        transform_function = (
            transform_function_class.from_dict(transform_function_params)
            if transform_function_class
            else None
        )

        tokenizer = (
            AutoTokenizer.from_pretrained(config["attrs_"].pop("tokenizer")) or None
        )

        if tokenizer is None:
            raise ValueError("Tokenizer class not found or could not be initialized.")

        try:
            return_obj = cls(
                image_preprocessor=image_preprocessor,
                transform_function=transform_function,
                tokenizer=tokenizer,
                **config["attrs_"],
            )
        except Exception as e:
            raise ValueError(
                f"Error initializing MammogramDataset from dict: {e}"
            ) from e
        return return_obj

    @classmethod
    def from_config(cls, config_path: str) -> "MammogramDataset":
        """
        Load the dataset configuration from a YAML file.
        Args:
            config_path (str): Path to the configuration file.
        Returns:
            MammogramDataset: An instance of the dataset with the loaded configuration.
        """
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(config)

    def _save_to_cache(self, pid: str, img: np.ndarray):
        os.makedirs(self.cache_dir, exist_ok=True)
        hashed_pid = hashlib.md5(pid.encode()).hexdigest()[:12]
        cache_path = os.path.join(self.cache_dir, f"{hashed_pid}.npy")
        # save the image to the cache
        np.save(cache_path, img)

    def _load_from_cache(self, pid: str):
        hashed_pid = hashlib.md5(pid.encode()).hexdigest()[:12]
        cache_path = os.path.join(self.cache_dir, f"{hashed_pid}.npy")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data
        else:
            return None

    @staticmethod
    def clear_cache_dir(cache_dir: str):
        """
        Clear the cache directory.
        """
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path) and file.endswith(".npz"):
                    os.remove(file_path)
            os.rmdir(cache_dir)

    def clear_cache(self):
        """
        Clear the cache directory.
        """
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                if os.path.isfile(file_path) and file.endswith(".npz"):
                    os.remove(file_path)
            os.rmdir(self.cache_dir)

    def __getitem__(self, idx):
        pid = str(self.df[self.pid_col].iloc[idx])
        text = str(self.df[self.text_col].iloc[idx])
        # Robust alternative‐text sampling
        if self.alt_text_cols:
            raw = self.df.iloc[idx][self.alt_text_cols]
            valid_alt = [
                at for at in raw if pd.notna(at) and isinstance(at, str) and at.strip()
            ]
            if valid_alt and np.random.rand() < self.alt_text_prob:
                text = np.random.choice(valid_alt).strip()

        # Check if the image is already cached
        cached_img = self._load_from_cache(pid)
        if cached_img is not None:
            imgs = cached_img
        else:
            img_paths = self.df[self.image_cols].iloc[idx]

            # Load images (assuming grayscale)
            imgs = [
                cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                for img_path in img_paths
            ]

            # Preprocessing
            imgs = [self.image_preprocessor(img) for img in imgs]
            imgs = np.concatenate(imgs, axis=0)  # shape: (N, H, W)
            # Save to cache
            self._save_to_cache(pid, imgs)

        # Augmentation
        if self.transform_function:
            imgs = [self.transform_function(img) for img in imgs]
            imgs = torch.stack(imgs, dim=0).squeeze()
        else:
            imgs = torch.from_numpy(imgs).float()

        output = BatchEncoding()
        # Tokenization
        if self.tokenizer:
            output = self.tokenizer(text, **self.tokenizer_kwargs)
            output["input_ids"] = output["input_ids"].squeeze(0)
            output["attention_mask"] = output["attention_mask"].squeeze(0)
            output["token_type_ids"] = output["token_type_ids"].squeeze(0)
        else:
            token_ids = torch.tensor([0], dtype=torch.long)
            output["input_ids"] = token_ids

        output["pixel_values"] = imgs

        return output
