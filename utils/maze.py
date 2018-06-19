# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data


class MazeDataset(data.Dataset):

    def __init__(self, filename, dataset_type):
        """
        Args:
          filename (str): Dataset filename (must be .npz format).
          dataset_type (str): One of "train", "valid", or "test".
        """
        assert filename.endswith("npz")  # Must be .npz format
        self.filename = filename
        self.dataset_type = dataset_type  # train, valid, test

        self.mazes, self.goal_maps, self.opt_policies = self._process(filename)

        self.num_actions = self.opt_policies.shape[1]
        self.num_orient = self.opt_policies.shape[2]

    def _process(self, filename):
        """
        Data format: list, [train data, test data]
        """
        with np.load(filename) as f:
            dataset2idx = {"train": 0, "valid": 3, "test": 6}
            idx = dataset2idx[self.dataset_type]
            mazes = f["arr_" + str(idx)]
            goal_maps = f["arr_" + str(idx + 1)]
            opt_policies = f["arr_" + str(idx + 2)]

        # Set proper datatypes
        mazes = mazes.astype(np.float32)
        goal_maps = goal_maps.astype(np.float32)
        opt_policies = opt_policies.astype(np.float32)

        # Print number of samples
        if self.dataset_type == "train":
            print("Number of Train Samples: {0}".format(mazes.shape[0]))
        elif self.dataset_type == "valid":
            print("Number of Validation Samples: {0}".format(mazes.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(mazes.shape[0]))
        print("\tSize: {}x{}".format(mazes.shape[1], mazes.shape[2]))
        return mazes, goal_maps, opt_policies

    def __getitem__(self, index):
        maze = self.mazes[index]
        goal_map = self.goal_maps[index]
        opt_policy = self.opt_policies[index]

        maze = torch.from_numpy(maze)
        goal_map = torch.from_numpy(goal_map)
        opt_policy = torch.from_numpy(opt_policy)

        return maze, goal_map, opt_policy

    def __len__(self):
        return self.mazes.shape[0]


def generate_maze(maze_size, decimation, start_pos=(1, 1)):
    maze = np.zeros((maze_size, maze_size))

    stack = [((start_pos[0], start_pos[1]), (0, 0))]

    def add_stack(next_pos, next_dir):
        if (next_pos[0] < 0) or (next_pos[0] >= maze_size):
            return
        if (next_pos[1] < 0) or (next_pos[1] >= maze_size):
            return
        if maze[next_pos[0]][next_pos[1]] == 0.:
            stack.append((next_pos, next_dir))

    while stack:
        pos, prev_dir = stack.pop()
        # Has this not been filled since being added?
        if maze[pos[0]][pos[1]] == 1.:
            continue

        # Fill in this point + break down wall from previous position
        maze[pos[0]][pos[1]] = 1.
        maze[pos[0] - prev_dir[0]][pos[1] - prev_dir[1]] = 1.

        choices = []
        choices.append(((pos[0] - 2, pos[1]), (-1, 0)))
        choices.append(((pos[0], pos[1] + 2), (0, 1)))
        choices.append(((pos[0], pos[1] - 2), (0, -1)))
        choices.append(((pos[0] + 2, pos[1]), (1, 0)))

        perm = np.random.permutation(np.array(range(4)))
        for i in range(4):
            choice = choices[perm[i]]
            add_stack(choice[0], choice[1])

    for y in range(1, maze_size - 1):
        for x in range(1, maze_size - 1):
            if np.random.uniform() < decimation:
                maze[y][x] = 1.

    return maze


class RandomMaze:

    def __init__(self,
                 mechanism,
                 min_maze_size,
                 max_maze_size,
                 min_decimation,
                 max_decimation,
                 start_pos=(1, 1)):
        self.mechanism = mechanism
        self.min_maze_size = min_maze_size
        self.max_maze_size = max_maze_size
        self.min_decimation = min_decimation
        self.max_decimation = max_decimation
        self.start_pos = start_pos

    def _isGoalPos(self, pos):
        """Returns true if pos is equal to the goal position."""
        return pos[0] == self.goal_pos[0] and pos[1] == self.goal_pos[1]

    def _getState(self):
        """Returns the current state."""
        goal_map = np.zeros((self.mechanism.num_orient, self.maze_size,
                             self.maze_size))
        goal_map[self.goal_orient, self.goal_pos[0], self.goal_pos[1]] = 1.

        player_map = np.zeros((self.mechanism.num_orient, self.maze_size,
                               self.maze_size))
        player_map[self.player_orient, self.player_pos[0],
                   self.player_pos[1]] = 1.

        # Check if agent has reached the goal state
        reward = 0
        terminal = False
        if (self.player_orient == self.goal_orient) and self._isGoalPos(
                self.player_pos):
            reward = 1
            terminal = True

        return np.copy(self.maze), player_map, goal_map, reward, terminal

    def reset(self):
        """Resets the maze."""
        if self.min_maze_size == self.max_maze_size:
            self.maze_size = self.min_maze_size
        else:
            self.maze_size = self.min_maze_size + 2 * npr.randint(
                math.floor((self.max_maze_size - self.min_maze_size) / 2))
        if self.min_decimation == self.max_decimation:
            self.decimation = self.min_decimation
        else:
            self.decimation = npr.uniform(self.min_decimation,
                                          self.max_decimation)
        self.maze = generate_maze(
            self.maze_size, self.decimation, start_pos=self.start_pos)

        # Randomly sample a goal location
        self.goal_pos = (npr.randint(1, self.maze_size - 1),
                         npr.randint(1, self.maze_size - 1))
        while self._isGoalPos(self.start_pos):
            self.goal_pos = (npr.randint(1, self.maze_size - 1),
                             npr.randint(1, self.maze_size - 1))
        self.goal_orient = npr.randint(self.mechanism.num_orient)

        # Free the maze at the goal location
        self.maze[self.goal_pos[0]][self.goal_pos[1]] = 1.

        # Player start position
        self.player_pos = (self.start_pos[0], self.start_pos[1])

        # Sample player orientation
        self.player_orient = npr.randint(self.mechanism.num_orient)

        screen, player_map, goal_map, _, _ = self._getState()
        return screen, player_map, goal_map

    def step(self, action):
        # Compute neighbors for the current state.
        neighbors = self.neighbors_func(self.maze, self.player_orient,
                                        self.player_pos[0], self.player_pos[1])
        assert (action > 0) and (action < len(neighbors))
        self.player_orient, self.player_pos[0], self.player_pos[1] = neighbors[
            action]
        return self._getState()


def extract_policy(maze, mechanism, value):
    """Extracts the policy from the given values."""
    policy = np.zeros((mechanism.num_actions, value.shape[0], value.shape[1],
                       value.shape[2]))
    for p_orient in range(value.shape[0]):
        for p_y in range(value.shape[1]):
            for p_x in range(value.shape[2]):
                # Find the neighbor w/ max value (assuming deterministic
                # transitions)
                max_val = -sys.maxsize
                max_acts = [0]
                neighbors = mechanism.neighbors_func(maze, p_orient, p_y, p_x)
                for i in range(len(neighbors)):
                    n = neighbors[i]
                    nval = value[n[0]][n[1]][n[2]]
                    if nval > max_val:
                        max_val = nval
                        max_acts = [i]
                    elif nval == max_val:
                        max_acts.append(i)

                # Choose max actions if several w/ same value
                max_act = max_acts[np.random.randint(len(max_acts))]
                policy[max_act][p_orient][p_y][p_x] = 1.
    return policy


def extract_goal(goal_map):
    """Returns the goal location."""
    for o in range(goal_map.shape[0]):
        for y in range(goal_map.shape[1]):
            for x in range(goal_map.shape[2]):
                if goal_map[o][y][x] == 1.:
                    return (o, y, x)
    assert False
