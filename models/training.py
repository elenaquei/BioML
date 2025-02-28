#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
"""
import json
import torch.nn as nn
import numpy as np
import scipy
from numpy import mean
import torch
# from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from torch.utils import data as data
from torch.utils.data import DataLoader, TensorDataset

losses = {'mse': nn.MSELoss(),
          'cross_entropy': nn.CrossEntropyLoss(),
          'ell1': nn.SmoothL1Loss()
          }


class easyTrainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the
    weights+biases.
    -- eps: Set a strength for the extra loss term that penalizes the gradients of the original loss
    -- The float eps_comp records the gradient of the standard loss even when robust training is not active (for comparison). Only to be used with eps = 0
    ***
    """

    def __init__(self, model, optimizer, device,
                 print_freq=10, record_freq=10, verbose=1, save_dir=None,
                 l2_factor=0, db_type='l1'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose

        self.histories = {'loss_history': [],
                          'epoch_loss_history': []}
        self.buffer = {'loss': []}
        self.is_resnet = (type(self.model).__name__ == 'resnet')
        self.is_nODE = (type(self.model).__name__ == 'nODE')
        self.l2_factor = l2_factor
        self.db_type = db_type

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader):
        epoch_loss = 0.

        for i, (x_batch, y_batch) in enumerate(data_loader):
            # if i == 0:
            #     print('first data batch', x_batch[0], y_batch[0])
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if self.is_resnet:
                y_pred, _, _ = self.model(x_batch)
            elif not self.is_nODE:
                y_pred, _ = self.model(x_batch)
            else:
                y_pred = self.model(x_batch)
            # # Classical empirical risk minimization
            loss = self.loss_func(y_pred, y_batch)

            if self.l2_factor > 0:
                for param in self.model.parameters():
                    l2_regularization = param.norm()
                    loss += self.l2_factor * l2_regularization

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if i % self.print_freq == 0:
                if self.verbose > 1:
                    print("\nIteration {}/{}".format(i, len(data_loader)))
                    print("Loss: {:.3f}".format(loss))

            self.buffer['loss'].append(loss.item())

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))

                # Clear buffer
                self.buffer['loss'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))

        return epoch_loss / len(data_loader)


class maskedTrainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the
    weights+biases
    ***
    """

    def __init__(self, model, optimizer, device, mask, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None,
                 bound=0.):
        self.model = model
        self.optimizer = optimizer
        self.mask = mask
        self.device = device
        self.cross_entropy = cross_entropy
        self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        # In case we consider L1-reg. we threshold the norm.
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound

        self.histories = {'loss_history': [],
                          'epoch_loss_history': []}
        self.buffer = {'loss': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')

        # logging_dir='runs/our_experiment'
        # writer = SummaryWriter(logging_dir)

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_acc = 0.

        for i, (x_batch, y_batch) in enumerate(data_loader):
            # if i == 0:
            #     print('first data batch', x_batch[0], y_batch[0])
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred, traj = self.model(x_batch)
            time_steps = self.model.time_steps
            T = self.model.T
            dt = T / time_steps

            ## Classical empirical risk minimization

            loss = self.loss_func(y_pred, y_batch)
            '''loss = 0
            for j in range(2):
                x_i = x_batch[j]
                y_i = y_batch[j]
                y_pred = self.model(x_i)[0]
                loss += torch.sum((y_pred - y_i) ** 2)
                # loss_trainer += self.loss_func(y_pred, y_i)
            '''

            masked_loss = 0
            for k in range(self.model.time_steps):
                if self.model.flow.dynamics.architecture == 1:
                    weights = self.model.flow.dynamics.fc1_time[k].weight.matmul(
                        self.model.flow.dynamics.fc3_time[k].weight)
                elif self.model.flow.dynamics.architecture == 0:
                    weights = self.model.flow.dynamics.fc2_time[k].weight
                elif self.model.flow.dynamics.architecture == 3:
                    weights = self.model.flow.dynamics.fc2_time[k].weight
                else:
                    weights = self.model.flow.dynamics.fc2_time[k].weight
                unexpected_connections = weights * (1 - self.mask)
                masked_loss += unexpected_connections.norm(1)

            loss += masked_loss

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nIteration {}/{}".format(i, len(data_loader)))
                    print("Loss: {:.3f}".format(loss))
                    print("Masked loss: {:.3f}".format(masked_loss))

            self.buffer['loss'].append(loss.item())

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                if self.cross_entropy:
                    self.histories['acc_history'].append(mean(self.buffer['accuracy']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['accuracy'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))

        return epoch_loss / len(data_loader)

def weights_to_dataset(integration_time, Gamma, Win=None, bin=None, Wout=None, bout=None, batch_size=3000):
    dim = np.max(Gamma.shape)
    if Win is None:
        Win = np.eye(dim)
    if bin is None:
        bin = np.zeros([1, dim])
    if Wout is None:
        Wout = np.eye(dim)
    if bout is None:
        bout = np.zeros([1, dim]).T
    Gamma_mat = np.diag(Gamma)

    def x_vert(x):
        if len(x.shape) == 2:
            if x.shape[0] == 1:
                return x.T
        else:
            x = x[None, :]
            x = x_vert(x)
        return x

    def x_hor(x):
        return x.squeeze()

    rhs = lambda x, t: x_hor(np.matmul(Gamma_mat, x_vert(x)) + np.matmul(Wout, np.tanh(np.matmul(Win, x_vert(x)) + bin)) + bout)
    map = lambda x: scipy.integrate.odeint(lambda val, t: rhs(val, t).squeeze(), x, [0, integration_time])[-1, :]
    X, Y = from_map_to_XYdataset(map, dim, batch_size=batch_size)
    train, test = from_numpyXY_to_dataloader(X, Y)
    return train, test


def from_map_to_XYdataset(map, dim, batch_size=3000, bounds=[0, 5], random_state=None, noise=0.):
    if random_state:
        g = torch.Generator()
        g.manual_seed(random_state)

    size = [batch_size, dim]  # dimension of the pytorch tensor to be generated
    low, high = bounds  # range of uniform distribution

    X = np.array(torch.distributions.uniform.Uniform(low, high).sample(size))
    Y = np.array([map(np.squeeze(X[i, :])) for i in range(batch_size)])
    Y += noise * np.random.randn(Y.shape[0], Y.shape[1])

    return X, Y


def from_numpyXY_to_dataloader(X, Y, random_state=None, label='vector'):
    g = torch.Generator()
    if random_state:
        g.manual_seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.5, random_state=random_state, shuffle=True)

    X_train = torch.Tensor(X_train)  # transform to torch tensor for dataloader
    y_train = torch.Tensor(y_train)  # transform to torch tensor for dataloader

    X_test = torch.Tensor(X_test)  # transform to torch tensor for dataloader
    y_test = torch.Tensor(y_test)  # transform to torch tensor for dataloader

    if label == 'scalar':
        X_train = X_train.type(torch.float32)  # type of orginial pickle.load data
        y_train = y_train.type(torch.int64)  # dtype of original picle.load data

        X_test = X_test.type(torch.float32)  # type of orginial pickle.load data
        y_test = y_test.type(torch.int64)  # dtype of original picle.load data

    else:
        X_train = X_train.type(torch.float32)  # type of orginial pickle.load data
        y_train = y_train.type(torch.float32)  # dtype of original picle.load data

        X_test = X_test.type(torch.float32)  # type of orginial pickle.load data
        y_test = y_test.type(torch.float32)  # dtype of original picle.load data

    train_data = TensorDataset(X_train, y_train)  # create your datset
    test_data = TensorDataset(X_test, y_test)

    train = DataLoader(train_data, batch_size=64, shuffle=True, generator=g)
    test = DataLoader(test_data, batch_size=256, shuffle=True, generator=g)  # 128 before

    return train, test