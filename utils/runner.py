import importlib
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from utils.dijkstra import dijkstra_dist, dijkstra_policy
from utils.maze import extract_goal


def get_optimizer(args, parameters):
    if args.optimizer == "RMSprop":
        return optim.RMSprop(parameters, lr=args.lr, eps=args.eps)
    elif args.optimizer == "Adam":
        return optim.Adam(parameters, lr=args.lr, eps=args.eps)
    elif args.optimizer == "SGD":
        return optim.SGD(parameters, lr=args.lr, momentum=0.95)
    else:
        raise ValueError("Unsupported optimizer: %s" % args.optimizer)


class Runner():
    """
    The Runner class runs a planner model on a given dataset and records
    statistics such as loss, prediction error, % Optimal, and % Success.
    """

    def __init__(self, args, mechanism):
        """
        Args:
          model (torch.nn.Module): The Planner model
          mechanism (utils.mechanism.Mechanism): Environment transition kernel
          args (Namespace): Arguments
        """
        self.use_gpu = args.use_gpu
        self.clip_grad = args.clip_grad
        self.lr_decay = args.lr_decay
        self.use_percent_successful = args.use_percent_successful

        self.mechanism = mechanism
        self.criterion = nn.CrossEntropyLoss()

        # Instantiate the model
        model_module = importlib.import_module(args.model)
        self.model = model_module.Planner(
            mechanism.num_orient, mechanism.num_actions, args)
        self.best_model = model_module.Planner(
            mechanism.num_orient, mechanism.num_actions, args)

        # Load model from file if provided
        if args.load_file != "":
            saved_model = torch.load(args.load_file)
            if args.load_best:
                self.model.load_state_dict(saved_model["best_model"])
            else:
                self.model.load_state_dict(saved_model["model"])
            self.best_model.load_state_dict(saved_model["best_model"])
        else:
            self.best_model.load_state_dict(self.model.state_dict())

        # Track the best performing model so far
        self.best_metric = 0.

        # Use GPU if available
        if self.use_gpu:
            self.model = self.model.cuda()
            self.best_model = self.best_model.cuda()

        self.optimizer = get_optimizer(args, self.model.parameters())

    def _compute_stats(self, batch_size, map_design, goal_map,
              outputs, predictions, labels,
              loss, opt_policy, sample=False):
        # Select argmax policy
        _, pred_pol = torch.max(outputs, dim=1, keepdim=True)
    
        # Convert to numpy arrays
        map_design = map_design.cpu().data.numpy()
        goal_map = goal_map.cpu().data.numpy()
        outputs = outputs.cpu().data.numpy()
        predictions = predictions.cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        opt_policy = opt_policy.cpu().data.numpy()
        pred_pol = pred_pol.cpu().data.numpy()
    
        max_pred = (predictions == predictions.max(axis=1)[:, None]).astype(
            np.float32)
        match_action = np.sum((max_pred != opt_policy).astype(np.float32), axis=1)
        match_action = (match_action == 0).astype(np.float32)
        match_action = np.reshape(match_action, (batch_size, -1))
        batch_error = 1 - np.mean(match_action)
    
        def calc_optimal_and_success(i):
            # Get current sample
            md = map_design[i][0]
            gm = goal_map[i]
            op = opt_policy[i]
            pp = pred_pol[i][0]
            ll = labels[i][0]
    
            # Extract the goal in 2D coordinates
            goal = extract_goal(gm)
    
            # Check how different the predicted policy is from the optimal one
            # in terms of path lengths
            pred_dist = dijkstra_policy(md, self.mechanism, goal, pp)
            opt_dist = dijkstra_dist(md, self.mechanism, goal)
            diff_dist = pred_dist - opt_dist

            wall_dist = np.min(pred_dist)  # impossible distance

            for o in range(self.mechanism.num_orient):
                # Refill the walls in the difference with the impossible distance
                diff_dist[o] += (1 - md) * wall_dist

                # Mask out the walls in the prediction distances
                pred_dist[o] = pred_dist[o] - np.multiply(1 - md, pred_dist[o])

            num_open = md.sum() * self.mechanism.num_orient  # number of reachable locations
            return (diff_dist == 0).sum() / num_open, 1. - (
                pred_dist == wall_dist).sum() / num_open

        if sample:
            percent_optimal, percent_successful = calc_optimal_and_success(
                np.random.randint(batch_size))
        else:
            percent_optimal, percent_successful = 0, 0
            for i in range(batch_size):
                po, ps = calc_optimal_and_success(i)
                percent_optimal += po
                percent_successful += ps
            percent_optimal = percent_optimal / float(batch_size)
            percent_successful = percent_successful / float(batch_size)

        return loss.data.item(), batch_error, percent_optimal, percent_successful

    def _run(self, model, dataloader, train=False, batch_size=-1,
             store_best=False):
        """
        Runs the model on the given data.
        Args:
          model (torch.nn.Module): The Planner model
          dataloader (torch.utils.data.Dataset): Dataset loader
          train (bool): Whether to train the model
          batch_size (int): Only used if train=True
          store_best (bool): Whether to store the best model
        Returns:
          info (dict): Performance statistics, including
          info["avg_loss"] (float): Average loss
          info["avg_error"] (float): Average error
          info["avg_optimal"] (float): Average % Optimal
          info["avg_success"] (float): Average % Success
          info["weight_norm"] (float): Model weight norm, stored if train=True
          info["grad_norm"]: Gradient norm, stored if train=True
          info["is_best"] (bool): Whether the model is best, stored if store_best=True
        """
        info = {}
        for key in ["avg_loss", "avg_error", "avg_optimal", "avg_success"]:
            info[key] = 0.0
        num_batches = 0

        for i, data in enumerate(dataloader):
            # Get input batch.
            map_design, goal_map, opt_policy = data

            if train:
                if map_design.size()[0] != batch_size:
                    continue # Drop those data, if not enough for a batch
                self.optimizer.zero_grad()  # Zero the parameter gradients
            else:
                batch_size = map_design.size()[0]

            # Send tensor to GPU if available
            if self.use_gpu:
                map_design = map_design.cuda()
                goal_map = goal_map.cuda()
                opt_policy = opt_policy.cuda()
            map_design = Variable(map_design)
            goal_map = Variable(goal_map)
            opt_policy = Variable(opt_policy)
    
            # Reshape batch-wise if necessary
            if map_design.dim() == 3:
                map_design = map_design.unsqueeze(1)
    
            # Forward pass
            outputs, predictions, _, _ = model(map_design, goal_map)
    
            # Loss
            flat_outputs = outputs.transpose(1, 4).contiguous()
            flat_outputs = flat_outputs.view(
                -1, flat_outputs.size()[-1]).contiguous()
            _, labels = opt_policy.max(1, keepdim=True)
            flat_labels = labels.transpose(1, 4).contiguous()
            flat_labels = flat_labels.view(-1).contiguous()
            loss = self.criterion(flat_outputs, flat_labels)
    
            # Select actions with max scores (logits)
            _, predicted = torch.max(outputs, dim=1, keepdim=True)
    
            # Update parameters
            if train:
                # Backward pass
                loss.backward()
    
                # Clip the gradient norm
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.clip_grad)

                # Update parameters
                self.optimizer.step()

            # Compute loss and error
            loss_batch, batch_error, p_opt, p_suc = self._compute_stats(
                batch_size, map_design, goal_map,
                outputs, predictions, labels,
                loss, opt_policy, sample=train)
            info["avg_loss"] += loss_batch
            info["avg_error"] += batch_error
            info["avg_optimal"] += p_opt
            info["avg_success"] += p_suc
            num_batches += 1

        info["avg_loss"] = info["avg_loss"] / num_batches
        info["avg_error"] =  info["avg_error"] / num_batches
        info["avg_optimal"] =  info["avg_optimal"] / num_batches
        info["avg_success"] =  info["avg_success"] / num_batches
        
        if train:
            # Calculate weight norm
            weight_norm = 0
            grad_norm = 0
            for p in model.parameters():
                weight_norm += torch.norm(p)**2
                if p.grad is not None:
                    grad_norm += torch.norm(p.grad)**2
            info["weight_norm"] = float(np.sqrt(weight_norm.cpu().data.numpy().item()))
            info["grad_norm"] = float(np.sqrt(grad_norm.cpu().data.numpy().item()))

        if store_best:
            # Was the validation accuracy greater than the best one?
            metric = (info["avg_success"] if self.use_percent_successful else
                      info["avg_optimal"])
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_model.load_state_dict(model.state_dict())
                info["is_best"] = True
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * self.lr_decay
                info["is_best"] = False
        return info
    
    def train(self, dataloader, batch_size):
        """
        Trains the model on the given training dataset.
        """
        return self._run(self.model, dataloader, train=True,
                         batch_size=batch_size)
    
    def validate(self, dataloader):
        """
        Evaluates the model on the given validation dataset. Stores the
        current model if it achieves the best validation performance.
        """
        return self._run(self.model, dataloader, store_best=True)
    
    def test(self, dataloader, use_best=False):
        """
        Tests the model on the given dataset.
        """
        if use_best:
            model = self.best_model
        else:
            model = self.model
        model.eval()
        return self._run(model, dataloader, store_best=True)
