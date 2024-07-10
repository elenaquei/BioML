import torch
import matplotlib.pyplot as plt
import numpy as np
from models.training import create_dataloader
import scipy.io as io
device = 'cpu'

torch.backends.cudnn.deterministic = True
#seed = np.random.randint(1, 200)
seed = 56
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(seed)
g = torch.Generator()
g.manual_seed(seed)

# design choices
chosen_problem = 'repressilator'
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
problem = possible_problem[chosen_problem]

plotlim = [0, 5]

ind = 0
m1vec = []
m2vec = []
lossvec = []
var1vec = []
var2vec = []
for sz in range(2, 21):

    m1vectemp = []
    m2vectemp = []
    lossvectemp = []
    Xtempvec = []
    W1vec = []

    for k in range(0, 100):

        ind = ind + 1
        seed = ind

        dataloader, dataloader_viz, X_train = create_dataloader(problem, batch_size=sz, noise=data_noise,
                                                                plotlim=plotlim, random_state=seed, label='vector')

        print(k)
        # print(X_train.detach().numpy())

        # Import of the model dynamics that describe the neural ODE
        # The dynamics are based on the torchdiffeq package, that implements ODE solvers in the pytorch setting
        from models.neural_odes import NeuralODE

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
        architecture = 'inside'
        architectures = {'inside': -1, 'outside': 0, 'bottleneck': 1, 'restricted': 2, 'restr_repr': 3}
        # number of optimization runs in which the dataset is used for gradient decent
        num_epochs = 50
        if problem == 'moons' or problem == 'TS' or problem == "restrictedTS":
            hidden_dim, data_dim = 2, 2
        else:
            hidden_dim, data_dim = 3, 3
        augment_dim = 0

        # resets the seed - allows for coherent runs in the gradient descent as well
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        anode = NeuralODE(device, data_dim, hidden_dim, output_dim=data_dim, augment_dim=augment_dim,
                          non_linearity=non_linearity,
                          architecture=architecture, T=T, time_steps=num_steps, fixed_projector=fp,
                          cross_entropy=cross_entropy)
        optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-1)

        from models.training import maskedTrainer

        mask = torch.Tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        trainer_anode = maskedTrainer(anode, optimizer_anode, device, mask, cross_entropy=cross_entropy,
                                      bound=bound, verbose=False)

        trainer_anode.train(dataloader, 400)

        W1 = anode.flow.dynamics.fc2_time[0].weight
        W1 = W1.detach().numpy()
        m1 = abs(W1[0][1] - W1[1][0])
        m2 = abs(W1[0][0]) + abs(W1[1][1])

        lv = trainer_anode.histories["loss_history"]
        l = lv[-1]

        m1vectemp.append(m1)
        m2vectemp.append(m2)
        lossvectemp.append(l)
        Xtempvec.append(X_train.detach().numpy())
        W1vec.append(W1)

    mdic = {"X": Xtempvec, "l": lossvectemp, "symm": m1vectemp, "offdiag": m2vectemp, "W1": W1vec}
    io.savemat("repr_mask_n" + str(sz) + ".mat", mdic)
    m1vec.append(np.mean(m1vectemp))
    m2vec.append(np.mean(m2vectemp))
    lossvec.append(np.mean(lossvectemp))
    var1vec.append(np.var(m1vectemp))
    var2vec.append(np.var(m2vectemp))