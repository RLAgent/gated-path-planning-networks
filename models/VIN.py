import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# VIN planner module
class Planner(nn.Module):
    """
    Implementation of the Value Iteration Network.
    """
    def __init__(self, num_orient, num_actions, args):
        super(Planner, self).__init__()

        self.num_orient = num_orient
        self.num_actions = num_actions

        self.l_q = args.l_q
        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f

        self.h = nn.Conv2d(
            in_channels=(num_orient + 1),  # maze map + goal location
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)

        self.r = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=num_orient,  # reward per orientation
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)

        self.q = nn.Conv2d(
            in_channels=num_orient,
            out_channels=self.l_q * num_orient,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            bias=False)

        self.policy = nn.Conv2d(
            in_channels=self.l_q * num_orient,
            out_channels=num_actions * num_orient,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)

        self.w = Parameter(
            torch.zeros(self.l_q * num_orient, num_orient, self.f,
                        self.f),
            requires_grad=True)

        self.sm = nn.Softmax2d()

    def forward(self, map_design, goal_map):
        maze_size = map_design.size()[-1]
        X = torch.cat([map_design, goal_map], 1)

        h = self.h(X)
        r = self.r(h)
        q = self.q(r)
        q = q.view(-1, self.num_orient, self.l_q, maze_size, maze_size)
        v, _ = torch.max(q, dim=2, keepdim=True)
        v = v.view(-1, self.num_orient, maze_size, maze_size)
        for _ in range(0, self.k - 1):
            q = F.conv2d(
                torch.cat([r, v], 1),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=int((self.f - 1.0) / 2))
            q = q.view(-1, self.num_orient, self.l_q, maze_size, maze_size)
            v, _ = torch.max(q, dim=2)
            v = v.view(-1, self.num_orient, maze_size, maze_size)

        q = F.conv2d(
            torch.cat([r, v], 1),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
            padding=int((self.f - 1.0) / 2))

        logits = self.policy(q)

        # Normalize over actions
        logits = logits.view(-1, self.num_actions, maze_size, maze_size)
        probs = self.sm(logits)

        # Reshape to output dimensions
        logits = logits.view(-1, self.num_orient, self.num_actions, maze_size,
                             maze_size)
        probs = probs.view(-1, self.num_orient, self.num_actions, maze_size,
                           maze_size)
        logits = torch.transpose(logits, 1, 2).contiguous()
        probs = torch.transpose(probs, 1, 2).contiguous()

        return logits, probs, v, r
