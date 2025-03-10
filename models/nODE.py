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
import scipy

# from adjoint_neural_ode import adj_Dynamics

# odeint Returns:
#         y: Tensor, where the first dimension corresponds to different
#             time points. Contains the solved value of y for each desired time point in
#             `t`, with the initial value `y0` being the first element along the first
#             dimension.


def become_torch(input):
    if isinstance(input, (np.ndarray, np.generic)):
        input = torch.from_numpy(input).float()
    return input

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
        self.gamma_layer = torch.nn.Parameter(become_torch(Gamma), requires_grad=True)
        if Win is not None:
            self.inside_weights.weight = torch.nn.Parameter(become_torch(Win), requires_grad=True)
            self.inside_weights.bias = torch.nn.Parameter(become_torch(bin), requires_grad=True)
        if Wout is not None:
            self.outside_weights.weight = torch.nn.Parameter(become_torch(Wout), requires_grad=True)
            self.outside_weights.bias = torch.nn.Parameter(become_torch(bout), requires_grad=True)

        return

    def set_vec_weights(self, vec_weights):
        dim_vec = torch.tensor(list(vec_weights.flatten().size())[0])
        dim_float = (-3 + torch.sqrt(9 + 8 * dim_vec)) / 4
        if dim_float - dim_float.int() != 0:
            raise ValueError('The given vector cannot have the right parameters')
        dim = int(dim_float)

        # Decompose the vector into the individual weights and biases
        gamma = vec_weights[0:dim]
        vec_weights = vec_weights[dim:]
        Win = torch.reshape(vec_weights[0:dim**2], [dim, dim])
        vec_weights = vec_weights[dim**2:]
        bin = vec_weights[:dim]
        vec_weights = vec_weights[dim:]
        Wout = torch.reshape(vec_weights[0:dim**2], [dim, dim])
        vec_weights = vec_weights[dim**2:]
        bout = vec_weights

        # Call set_weights without breaking the gradient flow
        self.set_weights(gamma, Wout=Wout, bout=bout, Win=Win, bin=bin)
        return

    def get_weights(self):
        if self.architecture != "outside":
            W1 = self.inside_weights.weight
            b1 = self.inside_weights.bias
        else:
            W1 = torch.from_numpy(np.identity(self.ODE_dim))
            b1 = torch.from_numpy(np.zeros(self.ODE_dim))

        if self.architecture != "inside":
            W2 = self.outside_weights.weight
            b2 = self.outside_weights.bias
        else:
            W2 = torch.from_numpy(np.identity(self.ODE_dim))
            b2 = torch.from_numpy(np.zeros(self.ODE_dim))
        return self.gamma_layer, W1, b1, W2, b2

    def set_nonlinearity(self, sigma, der_sigma):
        self.non_linearity = sigma
        self.non_linear_derivative = der_sigma
        return

    def set_time(self, T, t0=0):
        self.time_interval = [t0, T]
        return

    def right_hand_side(self, t, x):
        if architectures[self.architecture] == 0:  # outside architecture ahs no inside layer
            out = x
        else:
            w1_t = self.inside_weights.weight
            b1_t = self.inside_weights.bias
            out = x.matmul(w1_t.t()) + b1_t.t()
        out = self.non_linearity(out)
        if architectures[self.architecture] == -1:  # inside architecture has no outside layer
            out = out
        else:
            w2_t = self.outside_weights.weight
            b2_t = self.outside_weights.bias
            # print(out)
            # print(w2_t.t())
            # print(out.matmul(w2_t.t()))
            out = out.matmul(w2_t.t()) + b2_t.t()
            # print(out)
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

        if architectures[self.architecture] == 0:
            w_t = self.outside_weights.weight
            b_t = self.outside_weights.bias
            # w(t)\sigma(x(t))+b(t)  inner
            # # #     -> derivative is w(t)\sigma'(x(t))
            out = w_t.matmul(torch.diag(self.non_linear_derivative(x)))
        elif architectures[self.architecture] == -1:
            w_t = self.inside_weights.weight
            b_t = self.inside_weights.bias
            out = rowKronecker(self.non_linear_derivative(x.matmul(w_t.t()) + b_t.t()), w_t)
        else:
            # w1(t)\sigma(w2(t)x(t)+b2(t))+b1(t) bottle-neck
            # # #     -> derivative is w1(t)\sigma'(w2(t)x(t)+b2(t))\row dy row kronecked product w2(t)
            w1_t = self.inside_weights.weight
            b1_t = self.inside_weights.bias
            w2_t = self.outside_weights.weight
            # b2_t = self.fc3_time[k].bias
            out = torch.matmul(torch.diagflat(self.non_linear_derivative(x.matmul(w1_t.t()) + b1_t.t())), w1_t)
            out = w2_t.matmul(out)

            # x.matmul(w1_t.t()) is the same as torch.matmul(w1_t,x) simple matrix-vector multiplication
        out += torch.diag(self.gamma_layer)
        return out

    def compute_dt(self):
        dt = (self.time_interval[1] - self.time_interval[0]) / (50 * self.n_layers)
        return dt

    def forward(self, x, return_features=False, time_intervals=None):
        if return_features:
            if time_intervals is None:
                time_intervals = torch.linspace(self.time_interval[0], self.time_interval[1], 300)
            integration_interval = torch.tensor(time_intervals).float().type_as(x)
        else:
            integration_interval = torch.tensor(self.time_interval).float().type_as(x)
        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x, integration_interval, method='euler', options={'step_size': dt})
        if return_features:
            return out
        if len(out.size()) == 3:
            out = out[1, :, :]
        else:
            out = out[1, :]
        return out
    
    def integrate_timepoints(self, x, time_points):
        out = odeint(self.right_hand_side, x, time_points)
        
        return out[1:, :, :]

    def forward_integration(self, x, integration_time=None):
        if integration_time is None:
            time_intervals = torch.tensor([self.time_interval[0], self.time_interval[1]])
        else:
            time_intervals = torch.tensor(integration_time)
        integration_interval = torch.tensor(time_intervals).float().type_as(x)
        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x, integration_interval, method='euler', options={'step_size': dt})
        return out

    def __str__(self):
        """a __str__ function for readability with print statements"""
        # print(self.architecture)
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

    def trajectory(self, x, n_evals=100, time_interval=None):
        if time_interval is None:
            time_intervals = torch.linspace(self.time_interval[0], self.time_interval[1], n_evals)
        else:
            time_intervals = torch.linspace(time_interval[0], time_interval[1], n_evals)
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

    def phase_portrait(self, dim1=0, dim2=1, dim3=None, range1=[0, 5], range2=[0, 5], range3=[0, 5], gridpoints=10,
                       time_interval=None):
        print('Plot phase portrait based on current nODE parameters..')
        x0 = torch.zeros(self.ODE_dim)

        xv = np.arange(range1[0], range1[1], (range1[1] - range1[0]) / gridpoints)
        xv = np.append(xv, range1[1])

        yv = np.arange(range2[0], range2[1], (range2[1] - range2[0]) / gridpoints)
        yv = np.append(yv, range2[1])

        if dim3 != None:
            zv = np.arange(range3[0], range3[1], (range3[1] - range3[0]) / gridpoints)
            zv = np.append(zv, range3[1])

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            for x in xv:
                for y in yv:
                    for z in zv:
                        x0[dim1] = x
                        x0[dim2] = y
                        x0[dim3] = z

                        traj = self.trajectory(x0, time_interval=time_interval).detach().numpy()

                        ax.plot(traj[:, dim1], traj[:, dim2], traj[:, dim3])
            plt.show()
            return

        for x in xv:
            for y in yv:
                x0[dim1] = x
                x0[dim2] = y
                traj = self.trajectory(x0, time_interval=time_interval).detach().numpy()
                plt.plot(traj[:, dim1], traj[:, dim2])

        plt.show()
        return

    def adjacency_matrix(self):
        if self.architecture == 'both':
            W = self.outside_weights.weight.matmul(self.inside_weights.weight)
        elif self.architecture == 'inside_weights':
            W = self.inside_weights.weight
        else:
            W = self.outside_weights.weight
        values_log_W = torch.sort(torch.log(torch.abs(W.flatten())))[0]
        threshold = values_log_W[torch.sort(values_log_W[1:] - values_log_W[:-1])[1][-1] + 1]
        W_bool_activation = (W >= torch.exp(threshold))
        W_bool_repr = (W <= -torch.exp(threshold))
        adjacency_mat = W_bool_activation.float() - W_bool_repr.float()
        return adjacency_mat


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


def repressilator(dim=3):
    gamma = - np.ones([dim])
    Wout = 2 * np.eye(dim)
    Win = np.zeros([dim, dim])
    for i in range(dim - 1):
        Win[i, i + 1] = -1
    Win[dim - 1, 0] = -1
    bin, bout = 2 * np.ones([dim, 1]), 2 * np.ones([dim, 1])

    node_repr = make_nODE_from_parameters(gamma, Win=Win, bin=bin, Wout=Wout, bout=bout)
    return node_repr
