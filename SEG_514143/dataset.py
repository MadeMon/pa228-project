# STUDENT's UCO: 514143

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from albumentations import Compose
from tqdm import tqdm
from label_dict import label_dict
from pandas import DataFrame
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SegDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, df: DataFrame, transforms: Compose | None = None, preload_samples: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.color_to_class = {
            color: class_idx for class_idx, color in enumerate(label_dict.values())
        }
        self.samples = self._preload_samples() if preload_samples else []

    def __len__(self) -> int:
        return len(self.df)

    def _rgb_mask_to_class(self, mask_rgb: np.ndarray, mask_path: Path) -> np.ndarray:
        class_mask = np.full(mask_rgb.shape[:2], -1, dtype=np.int64)

        for color, class_idx in self.color_to_class.items():
            color_match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
            class_mask[color_match] = class_idx

        unknown_pixels = class_mask == -1
        if np.any(unknown_pixels):
            unknown_colors = np.unique(mask_rgb[unknown_pixels], axis=0)
            raise ValueError(
                f"Unknown mask colors in {mask_path}: {unknown_colors.tolist()}"
            )

        return class_mask

    def _load_single_sample(self, img_path: Path, mask_path: Path) -> dict[str, Any]:
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        mask = self._rgb_mask_to_class(mask_rgb, mask_path)
        return {
            "image": image,
            "mask": mask,
            "img_path": img_path,
            "mask_path": mask_path,
        }

    def _preload_samples(self) -> list[dict[str, Any]]:
        indexed_paths = [
            (idx, Path(row.img_path), Path(row.mask_path))
            for idx, row in enumerate(self.df.itertuples(index=False))
        ]
        if not indexed_paths:
            return []

        loaded: list[tuple[int, dict[str, Any]]] = []

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._load_single_sample, img_path, mask_path): idx
                for idx, img_path, mask_path in indexed_paths
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Preloading samples",
            ):
                idx = futures[future]
                loaded.append((idx, future.result()))

        loaded.sort(key=lambda item: item[0])
        return [sample for _, sample in loaded]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.samples:
            sample = self.samples[idx]
        else:
            row = self.df.iloc[idx]
            sample = self._load_single_sample(Path(row.img_path), Path(row.mask_path))

        image = sample["image"].copy()
        mask = sample["mask"].copy()

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).long()
        return image_tensor, mask_tensor
