# STUDENT's UČO: 514143

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple
import random

import albumentations as A
import numpy as np
from PIL import Image

from label_dict import label_dict


def _pad_bottom_right(
    array: np.ndarray, pad_h: int, pad_w: int, fill_value: int
) -> np.ndarray:
    if pad_h <= 0 and pad_w <= 0:
        return array

    pad_width = [(0, pad_h), (0, pad_w)]
    if array.ndim > 2:
        pad_width.extend([(0, 0)] * (array.ndim - 2))

    return np.pad(array, pad_width, mode="constant", constant_values=fill_value)


def _crop_with_padding(
    array: np.ndarray,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
    pad_h: int,
    pad_w: int,
    fill_value: int,
) -> np.ndarray:
    padded = _pad_bottom_right(array, pad_h, pad_w, fill_value)
    return padded[top : top + crop_h, left : left + crop_w, ...]


def _sample_crop_top_left(
    height: int,
    width: int,
    crop_h: int,
    crop_w: int,
    center_y: Optional[int] = None,
    center_x: Optional[int] = None,
    jitter_ratio: float = 0.25,
) -> tuple[int, int]:
    max_top = max(0, height - crop_h)
    max_left = max(0, width - crop_w)

    if center_y is None or center_x is None:
        return random.randint(0, max_top), random.randint(0, max_left)

    jitter_y = max(0, int(round(crop_h * jitter_ratio)))
    jitter_x = max(0, int(round(crop_w * jitter_ratio)))
    top = (
        center_y
        - crop_h // 2
        + (random.randint(-jitter_y, jitter_y) if jitter_y > 0 else 0)
    )
    left = (
        center_x
        - crop_w // 2
        + (random.randint(-jitter_x, jitter_x) if jitter_x > 0 else 0)
    )

    return max(0, min(top, max_top)), max(0, min(left, max_left))


class MixedCropTransform(A.DualTransform):
    def __init__(
        self,
        width,
        height,
        rare_classes: Sequence[int],
        rare_prob: float = 0.5,
        p: float = 1.0,
        debug_dir: Optional[str] = None,
    ) -> None:
        super().__init__(p=p)
        self.width = width
        self.height = height
        self.rare_classes = rare_classes
        self.rare_prob = rare_prob
        self.debug_dir = debug_dir

    @property
    def targets_as_params(self) -> list[str]:
        return ["mask"]

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        mask = np.asarray(data["mask"])
        if mask.ndim > 2:
            mask = np.squeeze(mask)

        height, width = mask.shape[:2]
        pad_h = max(0, self.height - height)
        pad_w = max(0, self.width - width)

        use_rare_crop = random.random() < self.rare_prob
        top = left = 0

        if use_rare_crop and len(self.rare_classes) > 0:
            rare_mask = np.isin(mask, self.rare_classes)
            ys, xs = np.where(rare_mask)
            if len(ys) > 0:
                index = random.randrange(len(ys))
                top, left = _sample_crop_top_left(
                    height + pad_h,
                    width + pad_w,
                    self.height,
                    self.width,
                    int(ys[index]),
                    int(xs[index]),
                )
            else:
                top, left = _sample_crop_top_left(
                    height + pad_h, width + pad_w, self.height, self.width
                )
        else:
            top, left = _sample_crop_top_left(
                height + pad_h, width + pad_w, self.height, self.width
            )

        return {
            "top": top,
            "left": left,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "fill_value": 0,
            "crop_h": self.height,
            "crop_w": self.width,
        }

    def apply(
        self,
        image: np.ndarray,
        top: int = 0,
        left: int = 0,
        pad_h: int = 0,
        pad_w: int = 0,
        crop_h: int = 0,
        crop_w: int = 0,
        **params: Any,
    ) -> np.ndarray:
        crop = _crop_with_padding(image, top, left, crop_h, crop_w, pad_h, pad_w, 0)
        # if self.debug_dir is not None:
        #     self._save_debug_crop(crop, kind="image")

        return crop

    def apply_to_mask(
        self,
        mask: np.ndarray,
        top: int = 0,
        left: int = 0,
        pad_h: int = 0,
        pad_w: int = 0,
        crop_h: int = 0,
        crop_w: int = 0,
        fill_value: int = 0,
        **params: Any,
    ) -> np.ndarray:
        crop = _crop_with_padding(
            mask, top, left, crop_h, crop_w, pad_h, pad_w, fill_value
        )
        if self.debug_dir is not None:
            self._save_debug_crop(crop, kind="mask")
        return crop

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("rare_prob",)

    def _mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        palette = np.array(list(label_dict.values()), dtype=np.uint8)
        mask_array = np.asarray(mask)
        if mask_array.ndim > 2:
            mask_array = np.squeeze(mask_array)

        mask_array = np.clip(mask_array.astype(np.int64), 0, len(palette) - 1)
        return palette[mask_array]

    def _save_debug_crop(self, crop: np.ndarray, kind: str) -> None:
        if self.debug_dir is None:
            return
        debug_path = Path(self.debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        save_array = self._mask_to_rgb(crop) if kind == "mask" else crop
        if save_array.dtype != np.uint8:
            save_array = np.clip(save_array, 0, 255).astype(np.uint8)

        Image.fromarray(save_array).save(
            debug_path / f"{kind}_{random.randint(0, 999999)}.png"
        )
