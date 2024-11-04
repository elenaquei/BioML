import torch
import numpy as np
from models.training import easyTrainer, weights_to_dataset
from models.nODE import nODE, make_nODE_from_parameters
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchdiffeq import odeint
from data_creation import create_dataset, torch_parameter_structure
from models.ranking import fixed_point_ranking, parameter_ranking, network_ranking

x_train, x_noise, y_train, param = create_dataset(2,1,100)


class param_classifier(torch.nn.Module):
    def __init__(self, input_size, ode_dim, layers_size=[10, 10],
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()

        self.layers_size = layers_size
        self.input_size = input_size
        self.output_size = 2 * ode_dim ** 2 + 3 * ode_dim
        # print(self.output_size)
        self.ode_dim = ode_dim

        self.num_layers = len(layers_size) + 1
        self.device = device

        # initialise linear layers for classification block
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, layers_size[0])])
        self.linears.extend(
            [torch.nn.Linear(layers_size[i - 1], layers_size[i]) for i in range(1, self.num_layers - 1)])
        self.linears.append(torch.nn.Linear(layers_size[-1], self.output_size))
        return

    def forward_integration(self, x, parameter, integration_time=None):
        if integration_time is None:
            time_intervals = torch.tensor([0., 1.])
        else:
            time_intervals = torch.tensor(integration_time)
        integration_interval = torch.tensor([0, 1.]).float().type_as(x)

        par_struct = torch_parameter_structure(self.ode_dim)
        par_struct.set_vec_par(parameter)
        gamma, Win, bin, Wout, bout = par_struct.get_parameters()

        dt = 0.01
        out = odeint(lambda t, x: self.right_hand_side(t, x, Win, Wout, bin, bout, gamma), x, integration_interval,
                     method='euler', options={'step_size': dt})
        return out[-1, :]

    def right_hand_side(self, t, x, Win, Wout, bin, bout, gamma):
        out = x.matmul(Win.t()) + bin.t()
        out = out.matmul(Wout.t()) + bout.t()
        out = x.matmul(gamma) + out
        return out

    def get_adjacency(self, parameter):
        par_struct = torch_parameter_structure(self.ode_dim)
        par_struct.set_vec_par(parameter)
        gamma, Win, bin, Wout, bout = par_struct.get_parameters()

        A = Wout.matmul(Win).flatten()
        return A

    # forward pass of NN (both classifier and neural ODE)
    def forward(self, data):
        x = data
        for i in range(0, self.num_layers):
            x = self.linears[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        # here x denote the estimated parameters for the ODE
        # x = self.linears[len(self.layers_size) - 1](x)

        return x


def stat_tests(classifier, data_size=300):
    """
    def stat_tests(classifier, data_size = 300)

    tests the classifier by generating a new dataset with size data_size and testing it w.r.t. the accuracy of
    the dynamics, the network and the parameters found by the classifier for different parameters and network structures

    INPUT:
    classifier
    data_size

    OUTPUT:
    percentage_accurate_dynamics, percentage_ac_networks, percentage_acc_par  : 3 votes for the accuracy of the classifier
    """
    ode_dim = classifier.ode_dim

    found_node = nODE(ode_dim, architecture='both')
    correct_node = nODE(ode_dim, architecture='both')

    x_exact, x_noisy, y, p = create_dataset(ode_dim, 1, data_size)

    n_accurate_dynamics, n_ac_networks, n_acc_par = 0, 0, 0
    for i in range(data_size):
        data_noisy = x_noisy[i][0]
        pars = classifier(data_noisy)
        found_node.set_vec_weights(pars)

        correct_pars = p[i].get_vec_par()
        correct_node.set_vec_weights(torch.from_numpy(correct_pars).float())

        rank_dynamics = fixed_point_ranking(correct_node, found_node)
        n_accurate_dynamics += rank_dynamics

        ranking_network = network_ranking(correct_node, found_node)
        n_ac_networks += ranking_network

        ranking_parameter, scaling = parameter_ranking(correct_node, found_node)
        n_acc_par += ranking_parameter

    percentage_accurate_dynamics, percentage_ac_networks, percentage_acc_par = n_accurate_dynamics / data_size, n_ac_networks / data_size, n_acc_par / data_size
    return percentage_accurate_dynamics, percentage_ac_networks, percentage_acc_par


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier = param_classifier(8, 2).to(device)

perc_dym, perc_net, perc_par = stat_tests(classifier)
print(perc_dym, perc_net, perc_par)
