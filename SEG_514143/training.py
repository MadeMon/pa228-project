# STUDENT's UCO: 514143

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dataset import SampleDataset
from network import ModelExample
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchview import draw_graph


# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(net: nn.Module, input_sample: Tensor) -> None:
    # saves visualization of model architecture to the model_architecture.png
    draw_graph(
        net,
        input_sample,
        graph_dir="TB",
        save_graph=True,
        filename="model_architecture",
        expand_nested=True,
    )


# sample function for losses visualization
def plot_learning_curves(
    train_losses: list[float], validation_losses: list[float]
) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


# sample function for training
def fit(
    net: nn.Module,
    batch_size: int,
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    train_losses: list[float] = []
    val_losses: list[float] = []

    running_loss = 0.0
    for epoch in range(epochs):
        for batch_idx in range(0, 10):
            # add current loss
            running_loss += 0.1

            # graph variables
            train_losses.append(running_loss)
            val_losses.append(running_loss - 0.1)

        # print training info
        print(
            "Epoch {}, train loss: {:.5f}, val loss: {:.5f}".format(
                epoch, running_loss / 42, running_loss / 42
            )
        )

    print("Training finished!")
    return train_losses, val_losses


# declaration for this function should not be changed
def training(dataset_path: Path) -> None:
    """Performs training on the given dataset.

    Args:
        dataset_path: Path to the dataset.

    Saves:
        - model.pt (trained model)
        - learning_curves.png (learning curves generated during training)
        - model_architecture.png (a scheme of model's architecture)
    """
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))

    batch_size = 64
    train_dataset, val_dataset = SampleDataset(), SampleDataset()
    train_dataloader, val_dataloader = None, None

    net = ModelExample()
    input_sample = torch.zeros((1, 512, 1024))
    draw_network_architecture(net, input_sample)

    # define optimizer and learning rate
    optimizer = None

    # define loss function
    loss = None

    # train the network
    train_losses, val_losses = fit(
        net, batch_size, 3, train_dataloader, val_dataloader, loss, optimizer, device
    )

    # save the trained model and plot the losses, feel free to create your own functions
    torch.save(net.state_dict(), "model.pt")
    plot_learning_curves(train_losses, val_losses)


# #### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
