import torch
import numpy as np
from models.training import easyTrainer, weights_to_dataset
from models.nODE import nODE, make_nODE_from_parameters
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchdiffeq import odeint
from data_creation import create_dataset

x_train, x_noise, y_train, param = create_dataset(2,1,100)


class param_classifier(torch.nn.Module):
    def __init__(self, input_size, ode_dim, layers_size=[10, 10]):
        super().__init__()

        self.layers_size = layers_size
        self.input_size = input_size
        self.output_size = 2 * ode_dim ** 2 + 3 * ode_dim
        # print(self.output_size)
        self.ode_dim = ode_dim

        self.num_layers = len(layers_size) + 1

        # initialise linear layers for classification block
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, layers_size[0])])
        self.linears.extend(
            [torch.nn.Linear(layers_size[i - 1], layers_size[i]) for i in range(1, self.num_layers - 1)])
        self.linears.append(torch.nn.Linear(layers_size[-1], self.output_size))
        # self.neuralODE = nODE(ode_dim, architecture='both', time_interval=[0, 1])
        return

    def forward_integration(self, x, parameter, integration_time=None):
        if integration_time is None:
            time_intervals = torch.tensor([0., 1.])
        else:
            time_intervals = torch.tensor(integration_time)
        integration_interval = torch.tensor(time_intervals).float().type_as(x)
        dt = 0.01
        out = odeint(lambda t, x : self.right_hand_side(t, x, parameter), x, integration_interval, method='euler', options={'step_size': dt})
        return out[-1,:]

    def right_hand_side(self, t, x, parameter):
        """
        w1_t = self.inside_weights.weight
        b1_t = self.inside_weights.bias
        out = x.matmul(w1_t.t()) + b1_t.t()
        out = torch.nn.Tanh(out)
        w2_t = self.outside_weights.weight
        b2_t = self.outside_weights.bias
        out = out.matmul(w2_t.t()) + b2_t.t()
        Gamma = torch.diag(self.gamma_layer)
        out = x.matmul(Gamma) + out
        """
        out = - torch.sum(parameter) * x + torch.tanh(x)
        return out

    # forward pass of NN (both classifier and neural ODE)
    def forward(self, data):
        x = data
        for i in range(0, self.num_layers):
            x = self.linears[i](x)
            if i < self.num_layers:
                x = F.relu(x)

        # here x denote the estimated parameters for the ODE
        # x = self.linears[len(self.layers_size) - 1](x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier = param_classifier(8, 2).to(device)
ode_dim = 2
integration_time = 1
# node = nODE(ode_dim, architecture='both', time_interval=[0, integration_time]).to(device)
# # replaced by classifier.neuralODE
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
"""
for name, param in classifier.named_parameters():
    print(name, param.requires_grad)

for name, param in node.named_parameters():
    print(name, param.requires_grad)
"""
for epoch in range(100):
    optimizer.zero_grad()
    loss = 0

    for i in range(0, len(x_train)):
        x = x_train[i][0].to(device)
        y = y_train[i][0].to(device)

        # get necessary inputs from data
        classifier_inp = x

        # get u_0, u_T from data
        u0 = classifier_inp[:2]
        ut = y[:2]

        # get true adjacency from data
        A = y[2:]

        # estimate parameters using classifier network
        p = classifier(classifier_inp)

        # define ODE with found weights
        # classifier.neuralODE.set_vec_weights(p)

        # integrate ODE
        ut_hat = classifier.forward_integration(u0, p)

        """
        # estimate adjacency matrix from output of classifier network
        Ahat = torch.eye(ode_dim) # node.adjacency_matrix()

        # add to loss: error between found adjacency and true adjacency
        loss = loss + loss_fn(A.flatten().float(), Ahat.flatten().float())
        """
        # add to loss: error between found solution at time t and true solution
        loss = loss + loss_fn(ut_hat.float(), ut.float())

        loss += loss_fn(p, 0*p)
        # print('Epoch ' + str(epoch))
        # print(loss)

    # backward propagation
    loss.backward(retain_graph=True)
    for name, param in classifier.named_parameters():
        if param.grad is None:
            print(f"Gradient for {name} is None")
        #else:
        #    # print(f"Gradient for {name}: {param.grad}")
    optimizer.step()
    # print(list(classifier.parameters())[0].grad)

print(classifier)