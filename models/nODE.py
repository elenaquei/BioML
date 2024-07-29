#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
further adapted by Elena Queirolo
"""
##------------#
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from warnings import warn
import numpy as np

# from adjoint_neural_ode import adj_Dynamics

# odeint Returns:
#         y: Tensor, where the first dimension corresponds to different
#             time points. Contains the solved value of y for each desired time point in
#             `t`, with the initial value `y0` being the first element along the first
#             dimension.


MAX_NUM_STEPS = 1000


def tanh_prime(input):
    # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
    return 1 - torch.tanh(input) * torch.tanh(
        input)


# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class Tanh_Prime(nn.Module):
    '''
    Applies tanh'(x) function element-wise:

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()  # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return tanh_prime(input)  # simply apply already implemented SiLU


# Useful dicos:
activations = {'tanh': nn.Tanh(),
               'relu': nn.ReLU(),
               'sigmoid': nn.Sigmoid(),
               'leakyrelu': nn.LeakyReLU(negative_slope=0.25, inplace=True),
               'tanh_prime': tanh_prime
               }
derivatives_activations = {'tanh': tanh_prime  # at the moment, we only care about tanh non-linearity
                           }
architectures = {'inside_weights': -1, 'outside_weights': 0, 'both': 1}


class nODE(nn.Module):

    def __init__(self, ODE_dim, architecture='inside_weights', time_interval=None, non_linearity='tanh',
                 gamma_layer_bool=True):
        super(nODE, self).__init__()
        if time_interval is None:
            time_interval = [0, 1]
        self.ODE_dim = ODE_dim
        self.n_layers = 1
        self.architecture = architecture
        self.time_interval = time_interval
        self.non_linearity = activations[non_linearity]
        self.non_linear_derivative = derivatives_activations[non_linearity]
        self.gamma_layer_bool = gamma_layer_bool
        self.gamma_layer = None
        self.inside_weights = None
        self.outside_weights = None
        self.setup_weights()
        return

    def setup_weights(self):
        self.inside_weights = nn.Linear(self.ODE_dim, self.ODE_dim)
        self.outside_weights = nn.Linear(self.ODE_dim, self.ODE_dim)
        self.gamma_layer = nn.Linear(self.ODE_dim, self.ODE_dim).bias
        return

    def set_weights(self, Gamma, Wout, bout, Win, bin):
        self.gamma_layer = Gamma
        self.inside_weights.weight = Win
        self.inside_weights.bias = bin
        self.outside_weights.weight = Wout
        self.outside_weights.bias = bout
        return

    def set_nonlinearity(self, sigma, der_sigma):
        self.non_linearity = sigma
        self.non_linear_derivative = der_sigma
        return

    def set_time(self, T, t0=0):
        self.time_interval = [t0, T]
        return

    def right_hand_side(self, t, x):
        if architectures[self.architecture] == 1:  # outside architecture ahs no inside layer
            out = x
        else:
            w1_t = self.inside_weights.weight
            b1_t = self.inside_weights.bias
            out = x.matmul(w1_t.t()) + b1_t
        out = self.non_linearity(out)
        if architectures[self.architecture] == 0:  # inside architecture has no outside layer
            out = out
        else:
            w2_t = self.outside_weights.weight
            b2_t = self.outside_weights.bias
            out = out.matmul(w2_t.t()) + b2_t
        Gamma = torch.diagonal(self.gamma_layer)
        out = x.matmul(Gamma) + out
        return out

    def derivative(self, t, x):
        """
        The output of the class -> D_xf(x(t), u(t))
        """
        def rowKronecker(x_vector, y_matrix):
            temp = [x_vector[i].detach() * y_matrix[i, :].detach() for i in range(len(x))]
            result = torch.Tensor()
            result = torch.cat(temp, out=result).reshape(y_matrix.shape)
            return result

        k = self.layer_selection(t)
        if architectures[self.architecture] == 0:
            w_t = self.outside_weights[k].weight
            b_t = self.outside_weights[k].bias
            # w(t)\sigma(x(t))+b(t)  inner
            # # #     -> derivative is w(t)\sigma'(x(t))
            out = w_t.matmul(torch.diag(self.non_linear_derivative(x)))
        elif architectures[self.architecture] == -1:
            w_t = self.inside_weights[k].weight
            b_t = self.inside_weights[k].bias
            out = rowKronecker(self.non_linear_derivative(w_t.matmul(x) + b_t), w_t)
        else:
            # w1(t)\sigma(w2(t)x(t)+b2(t))+b1(t) bottle-neck
            # # #     -> derivative is w1(t)\sigma'(w2(t)x(t)+b2(t))\row dy row kronecked product w2(t)
            w1_t = self.inside_weights[k].weight
            b1_t = self.inside_weights[k].bias
            w2_t = self.outside_weights[k].weight
            # b2_t = self.fc3_time[k].bias
            out = rowKronecker(self.non_linear_derivative(w1_t.matmul(x) + b1_t), w1_t)
            out = w2_t.matmulå(out)

            # x.matmul(w1_t.t()) is the same as torch.matmul(w1_t,x) simple matrix-vector multiplication
        return out

    def compute_dt(self):
        dt = (self.time_interval[1] - self.time_interval[0]) / (50 * self.n_layers)
        return dt

    def forward(self, x, return_features=False):
        if return_features:
            time_intervals = torch.linspace(self.time_interval[0], self.time_interval[1], 300)
            integration_interval = torch.tensor(time_intervals).float().type_as(x)
        else:
            integration_interval = torch.tensor(self.time_interval).float().type_as(x)
        if self.first_layer_bool:
            x_in = self.first_layer(x)
        else:
            x_in = x
        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x_in, integration_interval, method='euler', options={'step_size': dt})
        out = out[1, :, :]
        if self.last_layer_bool:
            x_out = self.last_layer(out)
        else:
            x_out = out
        return x_out

    def forward_integration(self, x, integration_time=None):
        if integration_time is None:
            time_intervals = torch.tensor([self.time_interval[0], self.time_interval[1]])
        else:
            time_intervals = torch.tensor([integration_time[0], integration_time[1]])
        integration_interval = torch.tensor(time_intervals).float().type_as(x)
        if self.first_layer_bool:
            x_in = self.first_layer(x)
        else:
            x_in = x
        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x_in, integration_interval, method='euler', options={'step_size': dt})
        out = out[1, :, :]
        if self.last_layer_bool:
            x_out = self.last_layer(out)
        else:
            x_out = out
        return x_out

    def __str__(self):
        """a __str__ function for readability with print statements"""
        activation_string = [i for i in activations if activations[i] == self.non_linearity][0]
        string = str()
        if architectures[self.architecture] < 1:
            if architectures[self.architecture] == -1:
                string += str(
                    'w(t)' + activation_string + '(x(t))+b(t)    over the interval ' + str(self.time_interval) + ',\n')
                layer = self.inside_weights
            else:
                string += str(
                    activation_string + '(w(t)x(t)+b(t))    over the interval ' + str(self.time_interval) + '\n')
                layer = self.outside_weights

            string += str(
                'W = ' + str(layer.weight.detach().numpy()) + ',      b = ' + str(layer.bias.detach().numpy()) + '\n\n')
        else:
            string += str(
                'w1(t)' + activation_string + '(w2(t)x(t)+b2(t))+b1(t)    over the interval ' + str(self.time_interval) + '\n')
            for k in range(self.n_layers):
                string += str(
                    'W1 = ' + str(self.outside_weights.weight.detach().numpy()) + ',        b1 = ' +
                    str(self.outside_weights.bias.detach().numpy()) + '\n\n')
                string += str(
                    'W2= ' + str(self.inside_weights.weight.detach().numpy()) + ',        b2 = ' +
                    str(self.inside_weights.bias.detach().numpy()) + '\n\n')
        return string

    def info(self):
        print('Time interval: ', self.time_interval, '\n')
        print('Non-linearity: ', self.non_linearity, '\n')
        print('Architecture: ', self.architecture, '\n')
        print('ODE dimension: ', self.ODE_dim, '\n')
        return

    def set_inside_weigths(self, linear_layer):
        self.inside_weights = linear_layer
        return

    def set_outside_weights(self, linear_layer):
        self.outside_weights = linear_layer
        return

    def trajectory(self, x, n_evals=100):
        time_intervals = torch.linspace(self.time_interval[0], self.time_interval[1], n_evals)
        integration_interval = time_intervals.clone().detach().float().type_as(x)

        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x, integration_interval, method='euler', options={'step_size': dt})

        return out


def grad_loss_inputs(model, data_inputs, data_labels, loss_module):
    data_inputs.requires_grad = True

    data_inputs_grad = torch.tensor(0.)

    preds, _ = model(data_inputs)

    loss = loss_module(preds, data_labels)

    data_inputs_grad = torch.autograd.grad(loss, data_inputs)[0]
    data_inputs.requires_grad = False
    return data_inputs_grad
