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


class doublebackTrainer():
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

    def __init__(self, model, optimizer, device, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None,
                 turnpike=True, bound=0., fixed_projector=False, eps=0.01, l2_factor=0, eps_comp=0., db_type='l1'):
        self.model = model
        self.optimizer = optimizer
        self.cross_entropy = cross_entropy
        self.device = device
        if cross_entropy:
            self.loss_func = losses['cross_entropy']
        else:
            # self.loss_func = losses['mse']
            self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        self.turnpike = turnpike
        # In case we consider L1-reg. we threshold the norm.
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound
        self.fixed_projector = fixed_projector

        self.histories = {'loss_history': [], 'loss_rob_history': [], 'acc_history': [],
                          'epoch_loss_history': [], 'epoch_loss_rob_history': [], 'epoch_acc_history': []}
        self.buffer = {'loss': [], 'loss_rob': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')
        self.l2_factor = l2_factor
        self.db_type = db_type

        # logging_dir='runs/our_experiment'
        # writer = SummaryWriter(logging_dir)

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_loss_rob = 0.
        epoch_acc = 0.

        loss_max = torch.tensor(0.)

        x_batch_grad = torch.tensor(0.).to(self.device)

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
            
            #loss = self.loss_func(y_pred, y_batch)
            loss = 0
            for i in range(2):
                x_i = x_batch[i]
                y_i = y_batch[i]
                y_pred = self.model(x_i)[0]
                loss += torch.sum((y_pred - y_i)**2)
                # loss_trainer += self.loss_func(y_pred, y_i)
            
            if self.l2_factor > 0:
                for param in self.model.parameters():
                    l2_regularization = param.norm()
                    loss += self.l2_factor * l2_regularization

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nIteration {}/{}".format(i, len(data_loader)))
                    if self.cross_entropy:
                        print("Loss: {:.3f}".format(loss))
                        print("Robust Term Loss: {:.3f}".format(loss_rob))

                        print("Accuracy: {:.3f}".format((softpred == y_batch).sum().item() / (y_batch.size(0))))

                    else:
                        print("Loss: {:.3f}".format(loss))

            self.buffer['loss'].append(loss.item())

            if not self.fixed_projector and self.cross_entropy:
                self.buffer['accuracy'].append((softpred == y_batch).sum().item() / (y_batch.size(0)))

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                if not self.fixed_projector and self.cross_entropy:
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

        if not self.fixed_projector:
            self.histories['epoch_acc_history'].append(epoch_acc / len(data_loader))

        return epoch_loss / len(data_loader)


def create_dataloader(data_type, batch_size=3000, noise=0.15, factor=0.15, random_state=1, shuffle=True,
                      plotlim=[-2, 2], label='scalar'):
    label_types = ['scalar', 'vector']
    if label not in label_types:
        raise ValueError("Invalid label type. Expected one of: %s" % label_types)

    if data_type == 'circles':
        X, y = make_circles(batch_size, noise=noise, factor=factor, random_state=random_state, shuffle=shuffle)



    elif data_type == 'blobs':
        centers = [[-1, -1], [1, 1]]
        X, y = make_blobs(
            n_samples=batch_size, centers=centers, cluster_std=noise, random_state=random_state)


    elif data_type == 'moons':
        X, y = make_moons(batch_size, noise=noise, shuffle=shuffle, random_state=random_state)


    elif data_type == 'TS':
        size = [batch_size, 2]  # dimension of the pytorch tensor to be generated
        low, high = 0, 1  # range of uniform distribution

        X = torch.distributions.uniform.Uniform(low, high).sample(size)

        def toggleswitch(x, t):
            # p = (0.25,5,1,1)
            # (S,n,k21,k12) = p
            S = 0.25
            n = 5
            k21 = 1
            k12 = 1
            A = np.array([[0, k21], [k12, 0]])
            Ax = np.matmul(A, x)
            act_x = np.tanh(Ax)  # S**n/(S**n + Ax**n)
            y = act_x - 0.5 * x
            return y

        deltat = 0.5
        y = np.array([scipy.integrate.odeint(toggleswitch, X[i, :], [0, deltat])[-1, :] for i in range(batch_size)])

        # np.array((X[:, 0] > X[:, 1]).float())
        # y = y.to(torch.int64)
        X = torch.abs(X + noise * torch.randn(X.shape))
    
    elif data_type == 'restrictedTS':
        if batch_size > 2:
            stopHere
        
        size = [batch_size, 2]  # dimension of the pytorch tensor to be generated
        low, high = 0, 1  # range of uniform distribution

        X = torch.Tensor([[1,2],[4.,3.]])

        def toggleswitch(x, t):
            W1 = np.array([[0, -1], [-1, 0]])
            b1 = np.array([2.,2.])
            interior = np.matmul(W1, x) + b1
            act_x = np.tanh(interior)  # S**n/(S**n + Ax**n)
            W2 = np.array([[2.,0],[0.,2.]])
            b2 = np.array([2.,2.])
            exterior = np.matmul(W2, act_x) + b2
            y = exterior - x
            return y

        deltat = 1
        y = np.array([scipy.integrate.odeint(toggleswitch, X[i, :], [0, deltat])[-1, :] for i in range(batch_size)])

    elif data_type == 'repr':  # REPRESSILATOR

        size = [batch_size, 3]  # dimension of the pytorch tensor to be generated
        low, high = plotlim  # range of uniform distribution

        X = np.array(torch.distributions.uniform.Uniform(low, high).sample(size))

        def repressilator(xyz, t):
            x, y, z = xyz[0], xyz[1], xyz[2]
            n = 5
            gamma, lx, ly, lz, deltax, deltay, deltaz, thetax, thetay, thetaz = 0.05, 0.01, 0.02, 0.03, 3.1, 3.2, 2.7, 1, 1.1, 1.2
            x_dot = - gamma * x + lx + deltax * thetax ** n / (thetax ** n + z ** n)
            y_dot = - gamma * y + ly + deltay * thetay ** n / (thetay ** n + x ** n)
            z_dot = - gamma * z + lz + deltaz * thetaz ** n / (thetaz ** n + y ** n)
            return np.array([x_dot, y_dot, z_dot])

        deltat = 0.5
        # forward the random points in time a lot
        small_sample_size = int(np.floor(batch_size*0.75))
        X[:,:small_sample_size] = np.array([scipy.integrate.odeint(repressilator, 
                                                                  X[i, :small_sample_size], [0, 100*deltat])[-1, :] for i in range(batch_size)])
        y = np.array([scipy.integrate.odeint(repressilator, X[i, :], [0, deltat])[-1, :] for i in range(batch_size)])

        # np.array((X[:, 0] > X[:, 1]).float())
        # y = y.to(torch.int64)
        X = torch.abs(torch.from_numpy(X) + noise * torch.randn(X.shape))

    elif data_type == 'xor':
        X = torch.randint(low=0, high=2, size=(batch_size, 2), dtype=torch.float32)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).float()
        # y = y.to(torch.int64)
        X += noise * torch.randn(X.shape)


    else:
        print('datatype not supported')
        return None, None

    if label == 'vector':
        if data_type == 'TS' or data_type == 'repr' or data_type == 'restrictedTS':
            print('No change  applied to TS or repr data')
            # y = np.array([(1., 0.) if label == 1 else (0., 1.) for label in y])
        else:
            y = np.array([(2., 0.) if label == 1 else (-2., 0.) for label in y])

    g = torch.Generator()
    g.manual_seed(random_state)

    # X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_state, shuffle=shuffle)
    if data_type == 'restrictedTS':
        X_test, X_train = X, X
        y_train, y_test = y, y

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

    train = DataLoader(train_data, batch_size=64, shuffle=shuffle, generator=g)
    test = DataLoader(test_data, batch_size=256, shuffle=shuffle, generator=g)  # 128 before
    if label == 'scalar':
        data_0 = X_train[y_train == 0]
        data_1 = X_train[y_train == 1]
    else:
        data_0 = X_train[y_train[:, 0] > 0]
        data_1 = X_train[y_train[:, 0] < 0]
    fig = plt.figure(figsize=(5, 5), dpi=100)
    
    if data_type == 'repr':
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], edgecolor="#333", alpha=0.5)
        ax.scatter(y_train[:, 0], y_train[:, 1], y_train[:, 2], edgecolor="#333", alpha=0.5)
        #plt.xlim(plotlim)
        #plt.ylim(plotlim)
        #plt.zlim(plotlim)
    else:
        plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", alpha=0.5)
        plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", alpha=0.5)
        plt.xlim(plotlim)
        plt.ylim(plotlim)
        ax = plt.gca()
        ax.set_aspect('equal')
    plt.savefig('trainingset.png', bbox_inches='tight', dpi=300, format='png', facecolor='white')
    plt.show()

    return train, test


def create_dataloader_other(data_type, batch_size=3000, noise=0.15, factor=0.15, random_state=1, shuffle=True,
                            plotlim=[-2, 2], label='scalar', deltat=0.5):
    label_types = ['scalar', 'vector']
    if label not in label_types:
        raise ValueError("Invalid label type. Expected one of: %s" % label_types)

    if data_type == 'TS':
        size = [batch_size, 2]  # dimension of the pytorch tensor to be generated
        low, high = 0, 5  # range of uniform distribution

        X = torch.distributions.uniform.Uniform(low, high).sample(size)

        def toggleswitch(x, t):
            gamma_x, ell_x, delta_x, theta_x = 1, 0.4, 4, 2
            gamma_y, ell_y, delta_y, theta_y = 1, 0.4, 4, 2

            hill_func = lambda x: ell_x + delta_x / 2 * (np.tanh(theta_x - x) + 1)

            Gamma = np.array([gamma_x, gamma_y])
            Ell = np.array([ell_x, ell_y])
            Delta = np.array([delta_x, delta_y])
            Theta = np.array([theta_x, theta_y])
            Connection_matrix = np.array([[0, 1], [1, 0]])

            Act = Ell + np.multiply(Delta / 2, np.tanh(Theta - np.matmul(Connection_matrix, x)) + 1)
            y = Act - np.multiply(Gamma, x)
            return y

        y = np.array([scipy.integrate.odeint(toggleswitch, X[i, :], [0, deltat])[-1, :] for i in range(batch_size)])
        X = torch.abs(X + noise * torch.randn(X.shape))

        # plot of ICs (blue) and final time points (red)
        plt.scatter(X[:, 0], X[:, 1], color='dodgerblue')
        plt.scatter(y[:, 0], y[:, 1], color='red')
        plt.show()

    else:
        print('datatype not supported')
        return None, None

    if label == 'vector':
        if data_type == 'TS' or data_type == 'repr':
            print('No change applied to TS or repr data')
            # y = np.array([(1., 0.) if label == 1 else (0., 1.) for label in y])
        else:
            y = np.array([(2., 0.) if label == 1 else (-2., 0.) for label in y])

    g = torch.Generator()
    g.manual_seed(random_state)

    if data_type != 'TS' and data_type != 'repr':
        X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_state, shuffle=shuffle)

    X_train = torch.Tensor(X_train)  # transform to torch tensor for dataloader
    y_train = torch.Tensor(y_train)  # transform to torch tensor for dataloader

    X_test = torch.Tensor(X_test)  # transform to torch tensor for dataloader
    y_test = torch.Tensor(y_test)  # transform to torch tensor for dataloader

    X_train = X_train.type(torch.float32)  # type of orginial pickle.load data
    y_train = y_train.type(torch.float32)  # dtype of original picle.load data

    X_test = X_test.type(torch.float32)  # type of orginial pickle.load data
    y_test = y_test.type(torch.float32)  # dtype of original picle.load data

    train_data = TensorDataset(X_train, y_train)  # create your datset
    test_data = TensorDataset(X_test, y_test)

    train = DataLoader(train_data, batch_size=64, shuffle=shuffle, generator=g)
    test = DataLoader(test_data, batch_size=256, shuffle=shuffle, generator=g)  # 128 before
    if label == 'scalar':
        data_0 = X_train[y_train == 0]
        data_1 = X_train[y_train == 1]
    else:
        data_0 = X_train[y_train[:, 0] > 0]
        data_1 = X_train[y_train[:, 0] < 0]
    # fig = plt.figure(figsize = (5,5), dpi = 100)
    # plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333",  alpha = 0.5)
    # plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", alpha = 0.5)
    # plt.xlim(plotlim)
    # plt.ylim(plotlim)
    # ax = plt.gca()
    # ax.set_aspect('equal')
    # plt.savefig('trainingset.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
    # plt.show()

    return train, test
