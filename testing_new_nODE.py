import torch
import numpy as np
from models.training import easyTrainer, weights_to_dataset
from models.nODE import nODE, make_nODE_from_parameters
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')''



ODE_dim = 2
Gamma = np.array([-1., -1.])
integration_time = 3
Win = np.array([[0, -1.], [-2., 0]])
bin = np.array([[2.], [2.]])

node2 = make_nODE_from_parameters(Gamma, Win=Win, bin=bin)
node2.plot()
print('real Toggle Switch')
print(node2)


train_data, test_data = weights_to_dataset(integration_time, Gamma, Win=Win, bin=bin)

node = nODE(ODE_dim, architecture='inside_weights', time_interval=[0, integration_time])

optimizer_node = torch.optim.Adam(node.parameters(), lr=1e-1)

trainer = easyTrainer(node, optimizer_node, device)

trainer.train(train_data, 50)

node.plot()
print(node)

