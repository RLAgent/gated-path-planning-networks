# -*- coding: utf-8 -*-
from __future__ import print_function
import abc


class Mechanism(abc.ABC):
    """Base class for maze transition mechanisms."""

    def __init__(self, num_actions, num_orient):
        self.num_actions = num_actions
        self.num_orient = num_orient

    @abc.abstractmethod
    def neighbors_func(self, maze, p_orient, p_y, p_x):
        """Computes next states for each action."""

    @abc.abstractmethod
    def invneighbors_func(self, maze, p_orient, p_y, p_x):
        """Computes previous states for each action."""

    @abc.abstractmethod
    def print_policy(self, maze, goal, policy):
        """Prints the given policy."""


class DifferentialDrive(Mechanism):
    """
    In Differential Drive, the agent can move forward along its current
    orientation, or turn left/right by 90 degrees.
    """

    def __init__(self):
        super(DifferentialDrive, self).__init__(num_actions=3, num_orient=4)
        self.clockwise = [1, 3, 0, 2]  # E S N W
        self.cclockwise = [2, 0, 3, 1]  # W N S E

    def _is_out_of_bounds(self, maze, p_y, p_x):
        return (p_x < 0 or p_x >= maze.shape[1] or p_y < 0 or
                p_y >= maze.shape[0])

    def _forward(self, maze, p_orient, p_y, p_x):
        assert p_orient < self.num_orient, p_orient

        next_p_y, next_p_x = p_y, p_x
        if p_orient == 0:  # North
            next_p_y -= 1
        elif p_orient == 1:  # East
            next_p_x += 1
        elif p_orient == 2:  # West
            next_p_x -= 1
        else:  # South
            next_p_y += 1

        # If position is out of bounds, simply return the current state.
        if (self._is_out_of_bounds(maze, next_p_y, next_p_x) or
                maze[p_y][p_x] == 0.):
            return p_orient, p_y, p_x

        return p_orient, next_p_y, next_p_x

    def _turnright(self, p_orient, p_y, p_x):
        assert p_orient < self.num_orient, p_orient
        return self.clockwise[p_orient], p_y, p_x

    def _turnleft(self, p_orient, p_y, p_x):
        assert p_orient < self.num_orient, p_orient
        return self.cclockwise[p_orient], p_y, p_x

    def _backward(self, maze, p_orient, p_y, p_x):
        assert p_orient < self.num_orient, p_orient

        next_p_y, next_p_x = p_y, p_x
        if p_orient == 0:  # North
            next_p_y += 1
        elif p_orient == 1:  # East
            next_p_x -= 1
        elif p_orient == 2:  # West
            next_p_x += 1
        else:  # South
            next_p_y -= 1

        # If position is out of bounds, simply return the current state.
        if (self._is_out_of_bounds(maze, next_p_y, next_p_x) or
                maze[p_y][p_x] == 0.):
            return p_orient, p_y, p_x

        return p_orient, next_p_y, next_p_x

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._forward(maze, p_orient, p_y, p_x),
            self._turnright(p_orient, p_y, p_x),
            self._turnleft(p_orient, p_y, p_x),
        ]

    def invneighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._backward(maze, p_orient, p_y, p_x),
            self._turnleft(p_orient, p_y, p_x),
            self._turnright(p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        orient2str = ["↑", "→", "←", "↓"]
        action2str = ["F", "R", "L"]
        for o in range(self.num_orient):
            print(orient2str[o])
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.:
                        print(u"\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(action2str[a], end="")
                print("")


class NorthEastWestSouth(Mechanism):
    """
    In NEWS, the agent can move North, East, West, or South.
    """

    def __init__(self):
        super(NorthEastWestSouth, self).__init__(num_actions=4, num_orient=1)

    def _north(self, maze, p_orient, p_y, p_x):
        if (p_y > 0) and (maze[p_y - 1][p_x] != 0.):
            return p_orient, p_y - 1, p_x
        return p_orient, p_y, p_x

    def _east(self, maze, p_orient, p_y, p_x):
        if (p_x < (maze.shape[1] - 1)) and (maze[p_y][p_x + 1] != 0.):
            return p_orient, p_y, p_x + 1
        return p_orient, p_y, p_x

    def _west(self, maze, p_orient, p_y, p_x):
        if (p_x > 0) and (maze[p_y][p_x - 1] != 0.):
            return p_orient, p_y, p_x - 1
        return p_orient, p_y, p_x

    def _south(self, maze, p_orient, p_y, p_x):
        if (p_y < (maze.shape[0] - 1)) and (maze[p_y + 1][p_x] != 0.):
            return p_orient, p_y + 1, p_x
        return p_orient, p_y, p_x

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._north(maze, p_orient, p_y, p_x),
            self._east(maze, p_orient, p_y, p_x),
            self._west(maze, p_orient, p_y, p_x),
            self._south(maze, p_orient, p_y, p_x),
        ]

    def invneighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._south(maze, p_orient, p_y, p_x),
            self._west(maze, p_orient, p_y, p_x),
            self._east(maze, p_orient, p_y, p_x),
            self._north(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        action2str = ["↑", "→", "←", "↓"]
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.:
                        print(u"\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(action2str[a], end="")
                print("")
            print("")


class Moore(Mechanism):
    """
    In Moore, the agent can move to any of the eight cells in its Moore
    neighborhood.
    """

    def __init__(self):
        super(Moore, self).__init__(num_actions=8, num_orient=1)

    def _north(self, maze, p_orient, p_y, p_x):
        if (p_y > 0) and (maze[p_y - 1][p_x] != 0.):
            return p_orient, p_y - 1, p_x
        return p_orient, p_y, p_x

    def _east(self, maze, p_orient, p_y, p_x):
        if (p_x < (maze.shape[1] - 1)) and (maze[p_y][p_x + 1] != 0.):
            return p_orient, p_y, p_x + 1
        return p_orient, p_y, p_x

    def _west(self, maze, p_orient, p_y, p_x):
        if (p_x > 0) and (maze[p_y][p_x - 1] != 0.):
            return p_orient, p_y, p_x - 1
        return p_orient, p_y, p_x

    def _south(self, maze, p_orient, p_y, p_x):
        if (p_y < (maze.shape[0] - 1)) and (maze[p_y + 1][p_x] != 0.):
            return p_orient, p_y + 1, p_x
        return p_orient, p_y, p_x

    def _northeast(self, maze, p_orient, p_y, p_x):
        if (p_y > 0) and (p_x < (maze.shape[1] - 1)) and (maze[p_y - 1][p_x + 1]
                                                          != 0.):
            return p_orient, p_y - 1, p_x + 1
        return p_orient, p_y, p_x

    def _northwest(self, maze, p_orient, p_y, p_x):
        if (p_y > 0) and (p_x > 0) and (maze[p_y - 1][p_x - 1] != 0.):
            return p_orient, p_y - 1, p_x - 1
        return p_orient, p_y, p_x

    def _southeast(self, maze, p_orient, p_y, p_x):
        if (p_y < (maze.shape[0] - 1)) and (p_x < (maze.shape[1] - 1)) and (
                maze[p_y + 1][p_x + 1] != 0.):
            return p_orient, p_y + 1, p_x + 1
        return p_orient, p_y, p_x

    def _southwest(self, maze, p_orient, p_y, p_x):
        if (p_y < (maze.shape[0] - 1)) and (p_x > 0) and (maze[p_y + 1][p_x - 1]
                                                          != 0.):
            return p_orient, p_y + 1, p_x - 1
        return p_orient, p_y, p_x

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._north(maze, p_orient, p_y, p_x),
            self._east(maze, p_orient, p_y, p_x),
            self._west(maze, p_orient, p_y, p_x),
            self._south(maze, p_orient, p_y, p_x),
            self._northeast(maze, p_orient, p_y, p_x),
            self._northwest(maze, p_orient, p_y, p_x),
            self._southeast(maze, p_orient, p_y, p_x),
            self._southwest(maze, p_orient, p_y, p_x),
        ]

    def invneighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._south(maze, p_orient, p_y, p_x),
            self._west(maze, p_orient, p_y, p_x),
            self._east(maze, p_orient, p_y, p_x),
            self._north(maze, p_orient, p_y, p_x),
            self._southwest(maze, p_orient, p_y, p_x),
            self._southeast(maze, p_orient, p_y, p_x),
            self._northwest(maze, p_orient, p_y, p_x),
            self._northeast(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        action2str = ["↑", "→", "←", "↓", "↗", "↖", "↘", "↙"]
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.:
                        print(u"\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(action2str[a], end="")
                print("")
            print("")
