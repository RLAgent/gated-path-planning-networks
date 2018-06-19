"""
Trains a planner model.

Example usage:
  python train.py --datafile mazes.npz --mechanism news --model models.GPPN \
    --k 15 --f 5 --save-directory log/gppn-k15-f5
"""
from __future__ import print_function

import argparse
import time
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import torch
    
from utils.experiment import (parse_args, create_save_dir, get_mechanism,
    create_dataloader, print_row, print_stats)
from utils.runner import Runner


def main():
    args = parse_args()

    save_path = create_save_dir(args.save_directory)
    mechanism = get_mechanism(args.mechanism)

    # Create DataLoaders
    trainloader = create_dataloader(
        args.datafile, "train", args.batch_size, mechanism, shuffle=True)
    validloader = create_dataloader(
        args.datafile, "valid", args.batch_size, mechanism, shuffle=False)
    testloader = create_dataloader(
        args.datafile, "test", args.batch_size, mechanism, shuffle=False)

    runner = Runner(args, mechanism)

    # Print header
    col_width = 5
    print("\n      |            Train              |            Valid              |")  # pylint: disable=line-too-long
    print_row(col_width, [
        "Epoch", "CE", "Err", "%Opt", "%Suc", "CE", "Err", "%Opt", "%Suc", "W",
        "dW", "Time", "Best"
    ])

    tr_total_loss, tr_total_error, tr_total_optimal, tr_total_success = [], [], [], []
    v_total_loss,   v_total_error,  v_total_optimal,  v_total_success = [], [], [], []
    for epoch in range(args.epochs):
        start_time = time.time()

        # Train the model
        tr_info = runner.train(trainloader, args.batch_size)

        # Compute validation stats and save the best model
        v_info = runner.validate(validloader)
        time_duration = time.time() - start_time

        # Print epoch logs
        print_row(col_width, [
            epoch + 1, tr_info["avg_loss"], tr_info["avg_error"],
            tr_info["avg_optimal"], tr_info["avg_success"], v_info["avg_loss"],
            v_info["avg_error"], v_info["avg_optimal"], v_info["avg_success"],
            tr_info["weight_norm"], tr_info["grad_norm"],
            time_duration,
            "!" if v_info["is_best"] else " "
        ])

        # Keep track of metrics:
        tr_total_loss.append(tr_info["avg_loss"])
        tr_total_error.append(tr_info["avg_error"])
        tr_total_optimal.append(tr_info["avg_optimal"])
        tr_total_success.append(tr_info["avg_success"])
        v_total_loss.append(v_info["avg_loss"])
        v_total_error.append(v_info["avg_error"])
        v_total_optimal.append(v_info["avg_optimal"])
        v_total_success.append(v_info["avg_success"])

        # Plot learning curves.
        def _plot(train, valid, name):
            plt.clf()
            x = np.array(range(len(train)))
            y = np.array(valid)
            plt.plot(x, np.array(train), label="train")
            plt.plot(x, np.array(valid), label="valid")
            plt.legend()
            plt.savefig(name)
        _plot(tr_total_loss, v_total_loss, save_path + "_total_loss.pdf")
        _plot(tr_total_error, v_total_error, save_path + "_total_error.pdf")
        _plot(tr_total_optimal, v_total_optimal,
              save_path + "_total_optimal.pdf")
        _plot(tr_total_success, v_total_success,
              save_path + "_total_success.pdf")

        # Save intermediate model.
        if args.save_intermediate:
            torch.save({
                "model": runner.model.state_dict(),
                "best_model": runner.best_model.state_dict(),
                "tr_total_loss": tr_total_loss,
                "tr_total_error": tr_total_error,
                "tr_total_optimal": tr_total_optimal,
                "tr_total_success": tr_total_success,
                "v_total_loss": v_total_loss,
                "v_total_error": v_total_error,
                "v_total_optimal": v_total_optimal,
                "v_total_success": v_total_success,
            }, save_path + ".e" + str(epoch) + ".pth")

    # Test accuracy
    print("\nFinal test performance:")
    t_final_info = runner.test(testloader)
    print_stats(t_final_info)

    print("\nBest test performance:")
    t_best_info = runner.test(testloader, use_best=True)
    print_stats(t_best_info)

    # Save the final trained model
    torch.save({
        "model": runner.model.state_dict(),
        "best_model": runner.best_model.state_dict(),
        "tr_total_loss": tr_total_loss,
        "tr_total_error": tr_total_error,
        "tr_total_optimal": tr_total_optimal,
        "tr_total_success": tr_total_success,
        "v_total_loss": v_total_loss,
        "v_total_error": v_total_error,
        "v_total_optimal": v_total_optimal,
        "v_total_success": v_total_success,
        "t_final_loss": t_final_info["avg_loss"],
        "t_final_error": t_final_info["avg_error"],
        "t_final_optimal": t_final_info["avg_optimal"],
        "t_final_success": t_final_info["avg_success"],
        "t_best_loss": t_best_info["avg_loss"],
        "t_best_error": t_best_info["avg_error"],
        "t_best_optimal": t_best_info["avg_optimal"],
        "t_best_success": t_best_info["avg_success"],
    }, save_path + ".final.pth")


if __name__ == "__main__":
    main()
