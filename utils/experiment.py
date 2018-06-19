import argparse
import os
import numpy as np

import torch

from utils.maze import MazeDataset
from utils.mechanism import DifferentialDrive, NorthEastWestSouth, Moore


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Environment parameters
    parser.add_argument(
        "--datafile", type=str, default="mazes.npz", help="Path to data file.")
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor. (keeps value within reasonable range).")
    parser.add_argument(
        "--mechanism", type=str, default="news",
        help="Maze transition mechanism. (news|diffdrive|moore)")

    # Log parameters
    parser.add_argument(
        "--save-directory", type=str, default="log/",
        help="Directory to save the graphs and models.")
    parser.add_argument(
        "--save-intermediate", default=False,
        help="Whether to save every epoch.")
    parser.add_argument(
        "--use-percent-successful", default=False,
        help="Use % successful instead of % optimal to decide best models.")

    # Optimization parameters
    parser.add_argument(
        "--optimizer", type=str, default="RMSprop",
        help="Which optimizer to use.")
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--lr-decay", type=float, default=1.0,
        help="Learning rate decay when CE goes up.")
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="Epsilon for denominator.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--clip-grad", type=float, default=40,
        help="Whether to clip the gradient norms. (0 for none)")

    # Model parameters
    parser.add_argument(
        "--model", type=str, default="models.VIN",
        help="Which model architecture to train.")
    parser.add_argument(
        "--load-file", type=str, default="",
        help="Model weights to load. (leave empty for none)")
    parser.add_argument(
        "--load-best", default=False,
        help="Whether to load the best weights from the load-file.")
    parser.add_argument(
        "--k", type=int, default=10, help="Number of Value Iterations.")
    parser.add_argument(
        "--l-i", type=int, default=5,
        help="Number of channels in input layer.")
    parser.add_argument(
        "--l-h", type=int, default=150,
        help="Number of channels in first hidden layer.")
    parser.add_argument(
        "--l-q", type=int, default=600,
        help="Number of channels in q layer (~actions) in VI-module.")
    parser.add_argument(
        "--f", type=int, default=3, help="Kernel size")

    args = parser.parse_args()

    # Automatic switch of GPU mode if available
    args.use_gpu = torch.cuda.is_available()

    return args


def get_mechanism(mechanism):
    if mechanism == "news":
        print("Using NEWS Drive")
        return NorthEastWestSouth()
    elif mechanism == "diffdrive":
        print("Using Differential Drive")
        return DifferentialDrive()
    elif mechanism == "moore":
        print("Using Moore Drive")
        return Moore()
    else:
        raise ValueError("Unsupported mechanism: %s" % mechanism)


def create_dataloader(datafile, dataset_type, batch_size, mechanism, shuffle=False):
    """
    Creates a maze DataLoader.
    Args:
      datafile (string): Path to the dataset
      dataset_type (string): One of "train", "valid", or "test"
      batch_size (int): The batch size
      shuffle (bool): Whether to shuffle the data
    """
    dataset = MazeDataset(datafile, dataset_type)
    assert dataset.num_actions == mechanism.num_actions
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def create_save_dir(save_directory):
    """
    Creates and returns path to the save directory.
    """
    try:
        os.makedirs(save_directory)
    except OSError:
        if not os.path.isdir(save_directory):
            raise
    return save_directory + "/planner"


def print_row(width, items):
    """
    Prints the given items.
    Args:
      width (int): Character length for each column.
      items (list): List of items to print.
    """
    def fmt_item(x):
        if isinstance(x, np.ndarray):
            assert x.ndim == 0
            x = x.item()
        if isinstance(x, float):
            rep = "%.3f" % x
        else:
            rep = str(x)
        return rep.ljust(width)

    print(" | ".join(fmt_item(item) for item in items))


def print_stats(info):
    """Prints performance statistics output from Runner."""
    print_row(10, ["Loss", "Err", "% Optimal", "% Success"])
    print_row(10, [
        info["avg_loss"], info["avg_error"],
        info["avg_optimal"], info["avg_success"]])
    return info
