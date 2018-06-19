"""
Evaluates a trained model on a dataset.

Example usage:
  python eval.py --datafile mazes.npz --mechanism news --model models.GPPN \
    --k 15 --f 5 --load-file log/gppn-k15-f5/planner.final.pth
"""
from __future__ import print_function

from utils.experiment import (parse_args, create_save_dir, get_mechanism,
    create_dataloader, print_stats)
from utils.runner import Runner


def main():
    args = parse_args()

    create_save_dir(args.save_directory)
    mechanism = get_mechanism(args.mechanism)

    # Create DataLoaders.
    trainloader = create_dataloader(
        args.datafile, "train", args.batch_size, mechanism, shuffle=True)
    validloader = create_dataloader(
        args.datafile, "valid", args.batch_size, mechanism, shuffle=False)
    testloader = create_dataloader(
        args.datafile, "test", args.batch_size, mechanism, shuffle=False)

    runner = Runner(args, mechanism)

    print("\n------------- Evaluating final model -------------")
    print("\nTrain performance:")
    print_stats(runner.test(trainloader))

    print("\nValidation performance:")
    print_stats(runner.test(testloader))

    print("\nTest performance:")
    print_stats(runner.test(validloader))

    print("\n------------- Evaluating best model -------------")
    print("\nTrain performance:")
    print_stats(runner.test(trainloader, use_best=True))

    print("\nValidation performance:")
    print_stats(runner.test(testloader, use_best=True))

    print("\nTest performance:")
    print_stats(runner.test(validloader, use_best=True))


if __name__ == "__main__":
    main()
