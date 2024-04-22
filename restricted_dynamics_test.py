# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import numpy as np
from models.training import create_dataloader


# Juptyer magic: For export. Makes the plots size right for the screen 
# %matplotlib inline
# # %config InlineBackend.figure_format = 'retina'

# %config InlineBackend.figure_formats = ['svg'] 

torch.backends.cudnn.deterministic = True
seed = np.random.randint(1,200)
seed = 56
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(seed)
g = torch.Generator()
g.manual_seed(seed)

# design choices 
chosen_problem = 'restricted_TS'
data_noise = 0.
n_different_weights = 1
if n_different_weights == 1:
    print('This choice will generate autonomous dynamics')
else:
    print('This choice generates non-autonomous dynamics, letting the weights depend on time')

possible_problem = {'moons':'moons', 'ToggleSwitch':'TS', 'repressilator':'repr', 'restricted_TS': 'restrictedTS'} 
# this choices determine the data set that we build and subsequent choices on the construction of the neural ODE 
# - in particular, it determines the dimensions 
problem = possible_problem[chosen_problem]

plotlim = [0, 5]

dataloader, dataloader_viz = create_dataloader(problem, batch_size = 2, noise = data_noise, 
                                               plotlim = plotlim, random_state = seed, label = 'vector')

# %%
import matplotlib.pyplot as plt
if problem == 'repr':
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

for X_viz, y_viz in dataloader:
    x = X_viz.numpy()
    y = y_viz.numpy()
    plt.scatter(x[:,0], x[:,1], marker = '*')
    plt.scatter(y[:,0], y[:,1], marker = '^')
    plt.show()
    break

# %%
#Import of the model dynamics that describe the neural ODE
#The dynamics are based on the torchdiffeq package, that implements ODE solvers in the pytorch setting
from models.neural_odes import NeuralODE

#T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
T, num_steps = 1, n_different_weights
bound = 0.
fp = False
cross_entropy = False
turnpike = False

# choice of model: what nonlinearity is used and if the nonlinearity is applied before (inside) or after (outside) the linear weights
# another choice is bottleneck, but I don't understand it
# non_linearity = 'tanh' # OR 'relu' 'sigmoid' 'leakyrelu' 'tanh_prime'
# architecture = 'inside' 'outside'
non_linearity = 'tanh'
architecture = 'restricted'
architectures = {'inside': -1, 'outside': 0, 'bottleneck': 1, 'restricted': 2}
# number of optimization runs in which the dataset is used for gradient decent
num_epochs = 50
if problem == 'moons' or problem == 'TS' or problem == "restrictedTS":
    hidden_dim, data_dim = 2, 2 
else:
    hidden_dim, data_dim = 3, 3 
augment_dim = 0


# %%
# resets the seed - allows for coherent runs in the gradient descent as well
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
anode = NeuralODE(device, data_dim, hidden_dim, output_dim=data_dim, augment_dim=augment_dim, non_linearity=non_linearity, 
                    architecture=architecture, T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-1) 

# %%
anode.flow.dynamics.fc1_time[0].weight

# %%
from models.training import doublebackTrainer

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
trainer_anode = doublebackTrainer(anode, optimizer_anode, device, cross_entropy=cross_entropy, turnpike = turnpike,
                         bound=bound, fixed_projector=fp, verbose = True, eps_comp = 0.2) 

for i, (x_batch, y_batch) in enumerate(dataloader):
    print(x_batch)
    y_pred, traj = trainer_anode.model(x_batch)
    print(y_pred, traj)


# %%
for i, (x_batch, y_batch) in enumerate(dataloader):
    print(x_batch, y_batch)

# %%
y_pred, traj = anode(x_batch)
print(y_pred, traj)

# %%
anode.flow.trajectory(torch.Tensor([[2.,3],[7.,9.]]), anode.time_steps) # still don't understand!

# %%
times = torch.linspace(0,1,100)
x = 10*torch.rand([2,20])
x = torch.Tensor([[1., 4.],[2., 3.]]) #x_batch
for x_i in x.t():
    y = anode.flow.forward(x_i, times)
    plt.plot(y[:,0].detach().numpy(),y[:,1].detach().numpy())
    print(y[-1,:]-anode(x_i)[0])

# %%
loss = 0.
loss_trainer = 0.
for i in range(2):
    x_i = x_batch[i]
    y_i = y_batch[i]
    # print('starting point=', x_i, 'wanted end point = ',y_i)
    y_pred = anode(x_i)[0]
    y_test = anode.flow.forward(x_i, torch.Tensor([0,1]))[-1]
    print('prediction = ',y_pred)
    loss += torch.sum((y_pred - y_i)**2)
    loss_trainer += trainer_anode.loss_func(y_pred, y_i)

pred_all = anode(x_batch)[0]
test_loss = torch.sum( (pred_all - y_batch))**2
print(pred_all)
print('loss in one go:',trainer_anode.loss_func(y_batch, pred_all),'   VS tested in one go:', test_loss/2)
print(loss/2, loss_trainer)

# %%
trainer_anode.train(dataloader, 500)

# %%
anode.flow.dynamics.fc1_time[0].weight

# %%
