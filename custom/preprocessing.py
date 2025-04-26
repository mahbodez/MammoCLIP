from .dictable import Dictable
from typing import Tuple
import torch
import numpy as np
import cv2
from skimage.measure import label, regionprops
from torchvision import transforms as T
import torch.nn as nn


class MammogramPreprocessor(Dictable):
    """
    Robust preprocessing pipeline for mammogram images.
    Includes:
    - Largest connected component extraction
    - CLAHE contrast enhancement
    - Intensity normalization
    - Resolution standardization
    """

    def __init__(
        self,
        output_size=(384, 384),
        extract_largest_cc=True,
        use_clahe=True,
        *args,
        **kwargs
    ):
        """
        Args:
            output_size (tuple): Desired output size for images.
            extract_largest_cc (bool): Whether to apply largest connected component extraction.
            use_clahe (bool): Whether to apply CLAHE.
        """
        if not isinstance(output_size, (tuple, list)) or len(output_size) != 2:
            raise ValueError("output_size must be a tuple/list of (height, width).")
        self.output_size = output_size
        self.use_clahe = use_clahe
        self.extract_largest_cc = extract_largest_cc
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @property
    def name(self):
        return "MammogramPreprocessor"

    def largest_connected_component(self, img: np.ndarray) -> np.ndarray:
        """
        Extract the largest connected component from the image.
        Args:
            img (np.ndarray): Input image.
        Returns:
            np.ndarray: Image with only the largest connected component.
        """
        if img.ndim != 2:
            raise ValueError("Input image must be a 2D grayscale image.")

        binary = img > np.median(img)
        labeled_img = label(binary)
        regions = regionprops(labeled_img)

        if len(regions) == 0:
            return img  # Fallback if no regions found

        largest_region = max(regions, key=lambda x: x.area)
        min_row, min_col, max_row, max_col = largest_region.bbox
        cropped_img = img[min_row:max_row, min_col:max_col]
        return cropped_img

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image.
        Args:
            img (np.ndarray): Input image.
        Returns:
            np.ndarray: Image after applying CLAHE.
        """
        if img.ndim != 2:
            raise ValueError("Input image must be a 2D grayscale image.")

        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        img_clahe = self.clahe.apply(img_uint8)
        return img_clahe.astype(np.float32) / 255.0

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified output size.
        Args:
            img (np.ndarray): Input image.
        Returns:
            np.ndarray: Resized image.
        """
        if img.ndim != 2:
            raise ValueError("Input image must be a 2D grayscale image.")

        resized_img = cv2.resize(
            img, self.output_size, interpolation=cv2.INTER_LINEAR_EXACT
        )
        return resized_img

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        Args:
            img (np.ndarray): Input image.
        Returns:
            np.ndarray: Preprocessed image.
        """
        img = self.largest_connected_component(img) if self.extract_largest_cc else img
        img = self.apply_clahe(img) if self.use_clahe else img
        img = self.resize_image(img)
        img = np.clip(img, 0, 1)  # Ensure pixel values are in [0, 1]
        return img.reshape(-1, *self.output_size)  # Reshape to (1, H, W)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Call method to preprocess the image.
        Args:
            img (np.ndarray): Input image.
        Returns:
            np.ndarray: Preprocessed image.
        """
        return self.preprocess(img)


class GaussianNoise(Dictable):
    """
    Add Gaussian noise to an image.
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: Tuple[float, float] = (0.01, 0.03),
        *args,
        **kwargs
    ):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std (tuple): Standard deviation of the Gaussian noise.
        """
        if not isinstance(std, (tuple, list)) or len(std) != 2:
            raise ValueError("std must be a tuple/list of (min_std, max_std).")
        if std[0] < 0 or std[1] < 0:
            raise ValueError("std values must be non-negative.")
        if std[0] > std[1]:
            raise ValueError("min_std must be less than max_std.")

        self.mean = mean
        self.std = std

    @property
    def name(self):
        return "GaussianNoise"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise to the image.
        Args:
            img (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Image with added Gaussian noise.
        """
        sigma = torch.empty(1).uniform_(*self.std)
        noise = torch.randn_like(img) * sigma + self.mean
        return img + noise

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Call method to apply Gaussian noise.
        Args:
            img (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Image with added Gaussian noise.
        """
        return self.forward(img)


class MammogramTransform(Dictable):
    """
    Modular and robust augmentation pipeline using Torchvision.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (384, 384),
        degrees: int = 15,
        translate: Tuple[float, float] = (0.1, 0.1),
        scale: Tuple[float, float] = (0.9, 1.1),
        shear: Tuple = (0, 0, 0, 0),
        mean: Tuple[float] = (0.5,),
        std: Tuple[float] = (0.5,),
        noise_std: Tuple[float, float] = (0.01, 0.03),
        is_validation: bool = False,
        *args,
        **kwargs
    ):
        self.size = size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.mean = mean
        self.std = std
        self.noise_std = noise_std
        self.is_validation = is_validation

        self.aug = T.Compose(
            [
                T.ToTensor(),
                (
                    T.RandomAffine(
                        degrees=degrees,
                        translate=translate,
                        scale=scale,
                        shear=shear,
                        interpolation=T.InterpolationMode.BILINEAR,
                    )
                    if not is_validation
                    else nn.Identity()
                ),
                (
                    T.RandomCrop(size=size)
                    if not is_validation
                    else T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)
                ),
                (
                    GaussianNoise(mean=0.0, std=noise_std)
                    if not is_validation
                    else nn.Identity()
                ),
                T.Normalize(mean=mean, std=std),
            ]
        )

    @property
    def name(self):
        return "MammogramTransform"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to the image.
        Args:
            img (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Augmented image tensor.
        """
        return self.aug(img)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Call method to apply augmentations.
        Args:
            img (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Augmented image tensor.
        """
        return self.forward(img)
