# STUDENT's UCO: 514143

# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <dataset_path> <model_path>

from argparse import ArgumentParser
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CONFIG
from dataset import SegDataset
from label_dict import label_dict
from network import MobilnetASPP

from utils import device

# Pre-built color lookup table: shape (num_classes, 3)
_COLOR_LUT = np.array(list(label_dict.values()), dtype=np.uint8)


# declaration for this function should not be changed
@torch.no_grad()  # do not calculate the gradients
def inference(dataset_path: Path, model_path: Path) -> None:
    """Performs inference on the given dataset using the specified model.

    Args:
        dataset_path: Path to the dataset. The function processes all PNG images in
            this directory (optionally recursively in its subdirectories).
        model_path: Path to the model file.

    Saves:
        predictions to 'output_predictions' folder. The files can be saved in a flat
            structure with the same name as the input file.
    """
    # Check for available GPU
    print("Computing with {}!".format(device))

    # loading the model
    model = MobilnetASPP(len(label_dict))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    output_dir = Path("output_predictions")
    output_dir.mkdir(exist_ok=True)

    # find all images in the dataset
    img_dir = dataset_path / "img"
    if not img_dir.exists():
        print(f"Error: Image directory not found at {img_dir}")
        return
    image_files = list(img_dir.rglob("*_leftImg8bit.png"))
    if not image_files:
        print(f"No images found in {img_dir}")
        return
    print(f"Found {len(image_files)} images to process")

    # dataset and dataloader
    infer_transforms = A.Compose(
        [
            A.Normalize(
                mean=CONFIG["transforms_normalize_mean"],
                std=CONFIG["transforms_normalize_std"],
            ),
        ]
    )
    df = pd.DataFrame({"img_path": [str(p) for p in image_files]})
    dataset = SegDataset(
        df,
        transforms=infer_transforms,
        inference_mode=True,
        preload_samples=CONFIG["preload_samples"],
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=CONFIG["runtime_num_workers"],
        pin_memory=CONFIG["runtime_pin_memory"],
    )

    # inference loop
    for images, _, img_paths in tqdm(loader, desc="Processing images"):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=CONFIG["runtime_amp"]):
            output = model(images)

        logits = output["out"] if isinstance(output, dict) else output
        predictions_np = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for pred, img_path in zip(predictions_np, img_paths):
            pred_rgb = _COLOR_LUT[pred]
            Image.fromarray(pred_rgb).save(output_dir / Path(img_path).name)


# #### code below should not be changed ############################################################################
def main() -> None:
    parser = ArgumentParser(description="Inference script for a neural network.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    parser.add_argument("model_path", type=Path, help="Path to the model weights")
    args = parser.parse_args()
    inference(args.dataset_path, args.model_path)


if __name__ == "__main__":
    main()
