#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
further adapted by Elena Queirolo
"""
##------------#
import torch
import torch.nn as nn
from torchdiffeq import odeint
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

    def set_weights(self, Gamma, Wout=None, bout=None, Win=None, bin=None):
        self.gamma_layer = torch.nn.Parameter(torch.from_numpy(Gamma))
        if Win is None:
            self.inside_weights = None
        else:
            self.inside_weights.weight = torch.nn.Parameter(torch.from_numpy(Win))
            self.inside_weights.bias = torch.nn.Parameter(torch.from_numpy(bin))

        if Wout is None:
            self.outside_weights = None
        else:
            self.outside_weights.weight = torch.nn.Parameter(torch.from_numpy(Wout))
            self.outside_weights.bias = torch.nn.Parameter(torch.from_numpy(bout))

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
        Gamma = torch.diag(self.gamma_layer)
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
            out = w2_t.matmul(out)

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
        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x, integration_interval, method='euler', options={'step_size': dt})
        out = out[1, :, :]
        return out

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
                    'Gx + w' + activation_string + '(x(t))+b    over the interval ' + str(self.time_interval) + ',\n')
                layer = self.inside_weights
            else:
                string += str(
                    'Gx + ' + activation_string + '(wx(t)+b)    over the interval ' + str(self.time_interval) + '\n')
                layer = self.outside_weights

            string += str('G = ' + str(self.gamma_layer.detach().numpy()) +
                          ', W = ' + str(layer.weight.detach().numpy()) + ',      b = ' + str(
                layer.bias.detach().numpy()) + '\n\n')
        else:
            string += str(
                'Gx + w1' + activation_string + '(w2x(t)+b2)+b1    over the interval ' + str(self.time_interval) + '\n')
            string += str('G = ' + str(self.gamma_layer.detach().numpy()) +
                          ', W1 = ' + str(self.outside_weights.weight.detach().numpy()) + ',        b1 = ' +
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

    def plot(self):
        print('Warning: this is an experimental feature')
        if self.architecture == 'both':
            W = self.outside_weights.weight.matmul(self.inside_weights.weight)
        elif self.architecture == 'inside_weights':
            W = self.inside_weights.weight
        else:
            W = self.outside_weights.weight
        values_log_W = torch.sort(torch.log(torch.abs(W.flatten())))[0]
        treshold = values_log_W[torch.sort(values_log_W[1:] - values_log_W[:-1])[1][-1] + 1]
        W_bool_activation = (W >= torch.exp(treshold))
        activation_edges = [('x' + str(i), 'x' + str(j)) for i in range(self.ODE_dim) for j in range(self.ODE_dim) if
                            W_bool_activation[i][j] == True]
        W_bool_repr = (W <= -torch.exp(treshold))
        repression_edges = [('x' + str(i), 'x' + str(j)) for i in range(self.ODE_dim) for j in range(self.ODE_dim) if
                            W_bool_repr[i][j] == True]
        G = nx.Graph()
        G.add_edges_from(activation_edges)
        G.add_edges_from(repression_edges)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=activation_edges, edge_color='g', arrows=True,
                               connectionstyle="arc3,rad=0.2")
        nx.draw_networkx_edges(G, pos, edgelist=repression_edges, edge_color='r', arrows=True,
                               connectionstyle="arc3,rad=0.2")
        plt.show()
        return


def grad_loss_inputs(model, data_inputs, data_labels, loss_module):
    data_inputs.requires_grad = True

    data_inputs_grad = torch.tensor(0.)

    preds, _ = model(data_inputs)

    loss = loss_module(preds, data_labels)

    data_inputs_grad = torch.autograd.grad(loss, data_inputs)[0]
    data_inputs.requires_grad = False
    return data_inputs_grad


def make_nODE_from_parameters(Gamma, Win=None, bin=None, Wout=None, bout=None):
    dim = np.max(Gamma.shape)
    if Win is None:
        if Wout is None:
            architecture = 'both'
        else:
            architecture = 'outside_weights'
    elif Wout is None:
        architecture = 'inside_weights'
    else:
        architecture = 'both'
    node = nODE(dim, architecture=architecture)
    node.set_weights(Gamma, Wout=Wout, bout=bout, Win=Win, bin=bin)
    return node
