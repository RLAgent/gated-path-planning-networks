"""
Generates a 2D maze dataset.

Example usage:
  python generate_dataset.py --output-path mazes.npz --mechanism news \
    --maze-size 9 --train-size 5000 --valid-size 1000 --test-size 1000
"""
from __future__ import print_function
import sys
import argparse
import numpy as np

from utils.dijkstra import dijkstra_dist
from utils.experiment import get_mechanism
from utils.maze import RandomMaze, extract_policy


def generate_data(filename,
                  train_size,
                  valid_size,
                  test_size,
                  mechanism,
                  maze_size,
                  min_decimation,
                  max_decimation,
                  start_pos=(1, 1)):
    maze_class = RandomMaze(
        mechanism,
        maze_size,
        maze_size,
        min_decimation,
        max_decimation,
        start_pos=start_pos)

    def hash_maze_to_string(maze):
        maze = np.array(maze, dtype=np.uint8).reshape((-1))
        mazekey = ""
        for i in range(maze.shape[0]):
            mazekey += str(maze[i])
        return mazekey

    def hashed_check_maze_exists(mazekey, mazehash):
        if mazehash is None:
            return False
        if mazekey in mazehash:
            return True
        return False

    def check_maze_exists(maze, compare_mazes):
        if compare_mazes is None:
            return False
        diff = np.sum(
            np.abs(compare_mazes - maze).reshape((len(compare_mazes), -1)),
            axis=1)
        if np.sum(diff == 0):
            return True
        return False

    def extract_goal(goal_map):
        for o in range(mechanism.num_orient):
            for y in range(maze_size):
                for x in range(maze_size):
                    if goal_map[o][y][x] == 1.:
                        return (o, y, x)

    def create_dataset(data_size, compare_mazes=None):
        mazes = np.zeros((data_size, maze_size, maze_size))
        goal_maps = np.zeros((data_size, mechanism.num_orient, maze_size,
                              maze_size))
        opt_policies = np.zeros((data_size, mechanism.num_actions,
                                 mechanism.num_orient, maze_size, maze_size))

        mazehash = {}
        if compare_mazes is not None:
            for i in range(compare_mazes.shape[0]):
                maze = compare_mazes[i]
                mazekey = hash_maze_to_string(maze)
                mazehash[mazekey] = 1
        for i in range(data_size):
            maze, goal_map = None, None
            while True:
                maze, _, goal_map = maze_class.reset()
                mazekey = hash_maze_to_string(maze)

                # Make sure we sampled a unique maze from the compare set
                if hashed_check_maze_exists(mazekey, mazehash):
                    continue
                mazehash[mazekey] = 1
                break

            # Use Dijkstra's to construct the optimal policy
            opt_value = dijkstra_dist(maze, mechanism, extract_goal(goal_map))
            opt_policy = extract_policy(maze, mechanism, opt_value)

            mazes[i, :, :] = maze
            goal_maps[i, :, :, :] = goal_map
            opt_policies[i, :, :, :, :] = opt_policy

            sys.stdout.write("\r%0.4f" % (float(i) / data_size * 100) + "%")
            sys.stdout.flush()
        sys.stdout.write("\r100%\n")

        return mazes, goal_maps, opt_policies

    # Generate test set first
    print("Creating valid+test dataset...")
    validtest_mazes, validtest_goal_maps, validtest_opt_policies = create_dataset(
        test_size + valid_size)

    # Split valid and test
    valid_mazes = validtest_mazes[0:valid_size]
    test_mazes = validtest_mazes[valid_size:]
    valid_goal_maps = validtest_goal_maps[0:valid_size]
    test_goal_maps = validtest_goal_maps[valid_size:]
    valid_opt_policies = validtest_opt_policies[0:valid_size]
    test_opt_policies = validtest_opt_policies[valid_size:]

    # Generate train set while avoiding test geometries
    print("Creating training dataset...")
    train_mazes, train_goal_maps, train_opt_policies = create_dataset(
        train_size, compare_mazes=validtest_mazes)

    # Re-shuffle
    mazes = np.concatenate((train_mazes, valid_mazes, test_mazes), 0)
    goal_maps = np.concatenate(
        (train_goal_maps, valid_goal_maps, test_goal_maps), 0)
    opt_policies = np.concatenate(
        (train_opt_policies, valid_opt_policies, test_opt_policies), 0)

    shuffidx = np.random.permutation(mazes.shape[0])
    mazes = mazes[shuffidx]
    goal_maps = goal_maps[shuffidx]
    opt_policies = opt_policies[shuffidx]

    train_mazes = mazes[:train_size]
    train_goal_maps = goal_maps[:train_size]
    train_opt_policies = opt_policies[:train_size]

    valid_mazes = mazes[train_size:train_size + valid_size]
    valid_goal_maps = goal_maps[train_size:train_size + valid_size]
    valid_opt_policies = opt_policies[train_size:train_size + valid_size]

    test_mazes = mazes[train_size + valid_size:]
    test_goal_maps = goal_maps[train_size + valid_size:]
    test_opt_policies = opt_policies[train_size + valid_size:]

    # Save to numpy
    np.savez_compressed(filename, train_mazes, train_goal_maps,
                        train_opt_policies, valid_mazes, valid_goal_maps,
                        valid_opt_policies, test_mazes, test_goal_maps,
                        test_opt_policies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path", type=str, default="mazes.npz",
        help="Filename to save the dataset to.")
    parser.add_argument(
        "--train-size", type=int, default=10000,
        help="Number of training mazes.")
    parser.add_argument(
        "--valid-size", type=int, default=1000,
        help="Number of validation mazes.")
    parser.add_argument(
        "--test-size", type=int, default=1000,
        help="Number of test mazes.")
    parser.add_argument(
        "--maze-size", type=int, default=9,
        help="Size of mazes.")
    parser.add_argument(
        "--min-decimation", type=float, default=0.0,
        help="How likely a wall is to be destroyed (minimum).")
    parser.add_argument("--max-decimation", type=float, default=1.0,
        help="How likely a wall is to be destroyed (maximum).")
    parser.add_argument(
        "--start-pos-x", type=int, default=1,
        help="Maze start X-axis position.")
    parser.add_argument(
        "--start-pos-y", type=int, default=1,
        help="Maze start Y-axis position.")
    parser.add_argument(
        "--mechanism", type=str, default="news",
        help="Maze transition mechanism. (news|diffdrive|moore)")
    args = parser.parse_args()

    mechanism = get_mechanism(args.mechanism)
    generate_data(
        args.output_path,
        args.train_size,
        args.valid_size,
        args.test_size,
        mechanism,
        args.maze_size,
        args.min_decimation,
        args.max_decimation,
        start_pos=(args.start_pos_y, args.start_pos_x))


if __name__ == "__main__":
    main()