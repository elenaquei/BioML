import torch
import matplotlib.pyplot as plt
import numpy as np
from models.training import create_dataloader, maskedTrainer, doublebackTrainer
import scipy.io as io
from models.neural_odes import NeuralODE

device = 'cpu'

torch.backends.cudnn.deterministic = True
# seed = np.random.randint(1, 200)
seed = 56
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(seed)
g = torch.Generator()
g.manual_seed(seed)

# design choices
data_noise = 0.
n_different_weights = 1
if n_different_weights == 1:
    print('This choice will generate autonomous dynamics')
else:
    print('This choice generates non-autonomous dynamics, letting the weights depend on time')

possible_problem = {'moons': 'moons', 'ToggleSwitch': 'TS', 'repressilator': 'repr', 'restricted_TS': 'restrictedTS',
                    'repr_alt': 'repr_alt'}
# this choices determine the data set that we build and subsequent choices on the construction of the neural ODE
# - in particular, it determines the dimensions
chosen_problem = 'repressilator'
problem = possible_problem[chosen_problem]

plotlim = [0, 5]

# T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
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
architecture = 'inside'  # architecture grn doesn seem to work as expected
architectures = {'inside': -1, 'outside': 0, 'bottleneck': 1, 'restricted': 2, 'grn': 3}
# number of optimization runs in which the dataset is used for gradient decent
num_epochs = 50
hidden_dim, data_dim = 3, 3
augment_dim = 0
# explicit mask for the repressilator
mask = torch.Tensor([[0, 0., 1.0], [1.0, 0, 0.], [0, 1.0, 0]])


def start_node(architecture='inside'):
    node = NeuralODE(device, data_dim, hidden_dim, output_dim=data_dim, augment_dim=augment_dim,
                     non_linearity=non_linearity,
                     architecture=architecture, T=T, time_steps=num_steps, fixed_projector=fp,
                     cross_entropy=cross_entropy)
    return node


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


def correct_network(W):
    if W[0, 2] < -0.5 and W[1, 0] < -0.5 and W[2, 1] < -0.5:
        return True
    else:
        return False


def masked_repr(batch_size, seed_loc):
    chosen_problem = 'repr_alt'
    problem = possible_problem[chosen_problem]

    set_seed(seed_loc)
    dataloader, dataloader_viz = create_dataloader(problem, batch_size=batch_size, noise=data_noise,
                                                   plotlim=plotlim, random_state=seed, label='vector', Figure=False)
    set_seed(seed_loc)
    anode = start_node()
    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-1)

    set_seed(seed_loc)
    trainer_anode = maskedTrainer(anode, optimizer_anode, device, mask, cross_entropy=cross_entropy,
                                  bound=bound, verbose=False)

    trainer_anode.train(dataloader, 400)
    return anode, trainer_anode


def restricted_repr(batch_size, seed_loc):
    chosen_problem = 'repr_alt'
    problem = possible_problem[chosen_problem]

    set_seed(seed_loc)
    dataloader, dataloader_viz = create_dataloader(problem, batch_size=batch_size, noise=data_noise,
                                                   plotlim=plotlim, random_state=seed, label='vector', Figure=False)
    set_seed(seed_loc)
    architecture = 'restr_repr'
    anode = start_node(architecture)
    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-1)

    set_seed(seed_loc)
    trainer_anode = maskedTrainer(anode, optimizer_anode, device, mask, cross_entropy=cross_entropy,
                                  bound=bound, verbose=False)

    trainer_anode.train(dataloader, 400)
    return anode, trainer_anode


def unknown_repr(batch_size, seed_loc):
    chosen_problem = 'repr_alt'
    problem = possible_problem[chosen_problem]

    set_seed(seed_loc)
    dataloader, dataloader_viz = create_dataloader(problem, batch_size=batch_size, noise=data_noise,
                                                   plotlim=plotlim, random_state=seed, label='vector', Figure=False)
    set_seed(seed_loc)
    architecture = 'restr_repr'
    anode = start_node(architecture)
    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-1)

    set_seed(seed_loc)
    trainer_anode = doublebackTrainer(anode, optimizer_anode, device,
                                      verbose=False)

    trainer_anode.train(dataloader, 400)
    return anode, trainer_anode


def extract_results(node, trainer):
    k = 0
    if node.flow.dynamics.architecture == 1:
        weights = node.flow.dynamics.fc1_time[k].weight.matmul(
            node.flow.dynamics.fc3_time[k].weight)
    elif node.flow.dynamics.architecture == 0:
        weights = node.flow.dynamics.fc2_time[k].weight
    elif node.flow.dynamics.architecture == 3:
        weights = node.flow.dynamics.fc2_time[k].weight
    else:
        weights = node.flow.dynamics.fc2_time[k].weight
    unexpected_connections = weights * (1 - mask)

    masked_loss = unexpected_connections.norm(1)

    final_loss = trainer.histories["loss_history"][-1]
    return weights, masked_loss.detach().numpy().min(), final_loss


# initial test to check convergence with large datasets
initial_test = False
if initial_test:
    anode, trainer_anode = masked_repr(3000, 56)
    final_loss = trainer_anode.histories["loss_history"][-1]
    print('Final loss with 3K training set is ', final_loss)

    ax = plt.figure().add_subplot(projection='3d')
    start = torch.Tensor([500, 20, 3.])
    for i in range(400):
        traj = anode.flow.trajectory(start, 20)
        ax.plot(traj.detach().numpy()[:, 0], traj.detach().numpy()[:, 1], traj.detach().numpy()[:, 2], color='b')
        start = traj[-1, :]
    plt.show()
    print('As shown, appropriate convergence is reached with low loss function.')

masked_maskedloss_vec = []
masked_lossvec = []
masked_success =  np.empty([0,2])

restricted_maskedloss_vec = []
restricted_lossvec = []
restricted_success =  np.empty([0,2])

unknown_maskedloss_vec = []
unknown_lossvec = []
unknown_success =  np.empty([0,2])

#iters = 2
iters = 10
for sz in [20, 10, 9, 8, 7, 6, 5, 4, 3]:
#for sz in [4, 3]:
    masked_maskedloss_vectemp = []
    masked_lossvectemp = []
    masked_W1vec = []
    masked_n_correct_size = 0

    restricted_maskedloss_vectemp = []
    restricted_lossvectemp = []
    restricted_W1vec = []
    restricted_n_correct_size = 0

    unknown_maskedloss_vectemp = []
    unknown_lossvectemp = []
    unknown_W1vec = []
    unknown_n_correct_size = 0

    for seed_k in range(0, iters):
        # MASKED TRAINER
        masked_node, masked_trainer = masked_repr(sz, seed_k)
        weights, masked_loss, final_loss = extract_results(masked_node, masked_trainer)

        masked_maskedloss_vectemp.append(masked_loss)
        masked_lossvectemp.append(final_loss)
        masked_W1vec.append(weights)
        masked_n_correct_size += correct_network(weights)

        # RESTRICTED (NO MASK)
        node, restricted_trainer = restricted_repr(sz, seed_k)
        weights, masked_loss, final_loss = extract_results(node, restricted_trainer)

        restricted_maskedloss_vectemp.append(masked_loss)
        restricted_lossvectemp.append(final_loss)
        restricted_W1vec.append(weights)
        restricted_n_correct_size += correct_network(weights)

        # UNKNOWN
        node, unknown_trainer = unknown_repr(sz, seed_k)
        weights, masked_loss, final_loss = extract_results(node, unknown_trainer)

        unknown_maskedloss_vectemp.append(masked_loss)
        unknown_lossvectemp.append(final_loss)
        unknown_W1vec.append(weights)
        unknown_n_correct_size += correct_network(weights)

    masked_success_rate = masked_n_correct_size / iters
    restricted_success_rate = restricted_n_correct_size / iters
    unknown_success_rate = unknown_n_correct_size / iters

    mdic = {"l": masked_lossvectemp, "masked_loss": masked_maskedloss_vectemp, "W1": masked_W1vec, "correct_perc": [sz, masked_success_rate]}
    # io.savemat("inside_repr_mask_n" + str(sz) + ".mat", mdic)
    masked_maskedloss_vec.append(np.mean(masked_maskedloss_vectemp))
    masked_lossvec.append(np.mean(masked_lossvectemp))
    masked_success = np.append(masked_success, np.array([[sz, masked_success_rate]]), axis=0)

    restricted_maskedloss_vec.append(np.mean(restricted_maskedloss_vectemp))
    restricted_lossvec.append(np.mean(restricted_lossvectemp))
    restricted_success = np.append(restricted_success, np.array([[sz, restricted_success_rate]]), axis=0)

    unknown_maskedloss_vec.append(np.mean(unknown_maskedloss_vectemp))
    unknown_lossvec.append(np.mean(unknown_lossvectemp))
    unknown_success = np.append(unknown_success, np.array([[sz, unknown_success_rate]]), axis=0)

    print('for ', sz, ' number of points, the average loss is ', np.mean(masked_lossvectemp))
    print('and the success rate is ', masked_success_rate)


np.savez('success_rates_repr', masked_success=masked_success, restricted_success=restricted_success,unknown_success=unknown_success)
plt.figure()
plt.plot(masked_success[:, 0], masked_success[:, 1])
plt.plot(restricted_success[:, 0], restricted_success[:, 1])
plt.plot(unknown_success[:, 0], unknown_success[:, 1])
plt.show()

