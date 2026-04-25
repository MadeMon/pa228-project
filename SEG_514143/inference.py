# STUDENT's UCO: 514143

# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <dataset_path> <model_path>

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from network import ModelLRASPP
from PIL import Image

from utils import device


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
    model = ModelLRASPP()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    for i in range(0, 3):
        # generate a random image and save it to output_predictions
        random_image = np.astype(np.random.rand(50, 50, 3) * 255, np.uint8)
        Image.fromarray(random_image).save(f"output_predictions/random_image_{i}.png")


# #### code below should not be changed ############################################################################
def main() -> None:
    parser = ArgumentParser(description="Inference script for a neural network.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    parser.add_argument("model_path", type=Path, help="Path to the model weights")
    args = parser.parse_args()
    inference(args.dataset_path, args.model_path)


if __name__ == "__main__":
    main()
