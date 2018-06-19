import math
import numpy as np


class MinHeap:

    def __init__(self):
        self.heap = []  # binary min-heap
        self.heapdict = {}  # [key]->index dict
        self.invheapdict = {}  # [index]->key dict
        self.heap_length = 0  # number of elements

    def empty(self):
        return self.heap_length == 0

    def insert(self, key, val):
        # Insert the value at the bottom of heap
        if self.heap_length == len(self.heap):
            self.heap.append(val)
        else:
            self.heap[self.heap_length] = val
        add_idx = self.heap_length
        self.heap_length += 1

        # Update the dictionaries
        self.heapdict[key] = add_idx
        self.invheapdict[add_idx] = key

        # percolate upwards
        self._percolate_up(add_idx)

    def decrease(self, key, new_val):
        # Find the index and value of this key
        curr_idx = self.heapdict[key]
        curr_val = self.heap[curr_idx]
        assert new_val <= curr_val

        # Update with new lower value
        self.heap[curr_idx] = new_val

        # Percolate upwards
        self._percolate_up(curr_idx)

    def extract(self):
        assert self.heap_length > 0

        retval = self.heap[0]
        retkey = self.invheapdict[0]

        # Swap the root with a leaf
        self._swap_index(0, self.heap_length - 1)

        # Delete the last element from the dictionaries
        del self.heapdict[retkey]
        del self.invheapdict[self.heap_length - 1]
        self.heap_length -= 1

        # Percolate downwards
        self._percolate_down(0)
        return retkey, retval

    def _swap_index(self, idx1, idx2):
        # get keys
        key1 = self.invheapdict[idx1]
        key2 = self.invheapdict[idx2]

        # Swap values in the heap
        tmp1_ = self.heap[idx1]
        self.heap[idx1] = self.heap[idx2]
        self.heap[idx2] = tmp1_

        # Swap indices in the [key]->index dict
        tmp2_ = self.heapdict[key1]
        self.heapdict[key1] = self.heapdict[key2]
        self.heapdict[key2] = tmp2_

        # Swap keys in the [index]->key dict
        tmp3_ = self.invheapdict[idx1]
        self.invheapdict[idx1] = self.invheapdict[idx2]
        self.invheapdict[idx2] = tmp3_

    def _percolate_up(self, curr_idx):
        while curr_idx != 0:
            parent_idx = int(math.floor((curr_idx - 1) / 2))
            if self.heap[parent_idx] > self.heap[curr_idx]:
                self._swap_index(curr_idx, parent_idx)
                curr_idx = parent_idx
            else:
                break

    def _percolate_down(self, curr_idx):
        while curr_idx < self.heap_length:
            child1 = 2 * curr_idx + 1
            child2 = child1 + 1
            if child1 >= self.heap_length:
                break
            minchild = child1
            maxchild = child2
            if child2 >= self.heap_length:
                maxchild = None
            if (maxchild is not None) and (self.heap[child1] >
                                           self.heap[child2]):
                minchild = child2
                maxchild = child1

            if self.heap[minchild] < self.heap[curr_idx]:
                self._swap_index(curr_idx, minchild)
                curr_idx = minchild
                continue

            if (maxchild is not None) and (self.heap[maxchild] <
                                           self.heap[curr_idx]):
                self._swap_index(curr_idx, maxchild)
                curr_idx = maxchild
                continue
            break


def dijkstra_dist(maze, mechanism, goal):
    # Initialize distance to largest possible distance
    dist = (np.zeros((mechanism.num_orient, maze.shape[0], maze.shape[1])) +
            mechanism.num_orient * maze.shape[0] * maze.shape[1])

    pq = MinHeap()
    pq.insert(goal, 0)
    for orient in range(mechanism.num_orient):
        for y in range(maze.shape[0]):
            for x in range(maze.shape[1]):
                if (orient == goal[0]) and (y == goal[1]) and (x == goal[2]):
                    continue
                pq.insert((orient, y, x),
                          mechanism.num_orient * maze.shape[0] * maze.shape[1])

    while not pq.empty():
        # extract minimum distance position
        ((p_orient, p_y, p_x), val) = pq.extract()
        dist[p_orient][p_y][p_x] = val

        # Update neighbors
        for n in mechanism.invneighbors_func(maze, p_orient, p_y, p_x):
            if (n[1] < 0) or (n[1] >= maze.shape[0]):
                continue
            if (n[2] < 0) or (n[2] >= maze.shape[1]):
                continue

            if maze[n[1]][n[2]] == 0.:
                continue

            curr_to_n = val + 1
            if curr_to_n < dist[n[0]][n[1]][n[2]]:
                dist[n[0]][n[1]][n[2]] = curr_to_n
                pq.decrease(n, curr_to_n)
    return -dist  # negative distance ~= value


def dijkstra_policy(maze, mechanism, goal, policy):
    # Initialize distance to largest possible distance
    dist = np.zeros(
        (mechanism.num_orient, maze.shape[0],
         maze.shape[1])) + mechanism.num_orient * maze.shape[0] * maze.shape[1]

    pq = MinHeap()
    pq.insert(goal, 0)
    for orient in range(mechanism.num_orient):
        for y in range(maze.shape[0]):
            for x in range(maze.shape[1]):
                if (orient == goal[0]) and (y == goal[1]) and (x == goal[2]):
                    continue
                pq.insert((orient, y, x),
                          mechanism.num_orient * maze.shape[0] * maze.shape[1])

    while not pq.empty():
        # extract minimum distance position
        ((p_orient, p_y, p_x), val) = pq.extract()
        dist[p_orient][p_y][p_x] = val

        # Update neighboring predecessors
        predecessors = mechanism.invneighbors_func(maze, p_orient, p_y, p_x)
        for i in range(len(predecessors)):
            n = predecessors[i]
            if (n[1] < 0) or (n[1] >= maze.shape[0]):
                continue
            if (n[2] < 0) or (n[2] >= maze.shape[1]):
                continue

            if maze[n[1]][n[2]] == 0.:
                continue

            # What are the successor from this predecessor state?
            succ_pred = mechanism.neighbors_func(maze, n[0], n[1], n[2])

            # Does following the policy on the predecessor state transition to
            # the current state?
            succ_pred_pol = succ_pred[policy[n[0]][n[1]][n[2]]]
            if (succ_pred_pol[0] == p_orient) and (
                    succ_pred_pol[1] == p_y) and (succ_pred_pol[2] == p_x):
                # Update value
                curr_to_n = val + 1
                if curr_to_n < dist[n[0]][n[1]][n[2]]:
                    dist[n[0]][n[1]][n[2]] = curr_to_n
                    pq.decrease(n, curr_to_n)
    return -dist  # negative distance ~= value
