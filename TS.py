# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Late robustness
#
# During the model training, we activate the robustness term only after the model expresses the basic topology of the data
# In the robustness phase the decision boundaries are then diffused and robustified

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import numpy as np

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
print("done")

# %% [markdown]
#
# # Data preparation

# %%
data_noise = 0.2
plotlim = [-3, 3]
subfolder = 'traj_moons'


from models.training import create_dataloader
dataloader, dataloader_viz = create_dataloader('moons', noise = data_noise, plotlim = plotlim, random_state = seed, label = 'vector')
print("done")

# %% [markdown]
# ## Model dynamics

# %%
#Import of the model dynamics that describe the neural ODE
#The dynamics are based on the torchdiffeq package, that implements ODE solvers in the pytorch setting
from models.neural_odes import NeuralODE

#for neural ODE based networks the network width is constant. In this example the input is 2 dimensional
hidden_dim, data_dim = 2, 2 
augment_dim = 0

#T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
T, num_steps = 20, 200
bound = 0.
fp = False #this recent change made things not work anymore
cross_entropy = False
turnpike = False

non_linearity = 'tanh' #'relu' #
architecture = 'inside' #outside

# %% [markdown]
# ## Training and generating level sets

# %%

num_epochs = 80 #number of optimization runs in which the dataset is used for gradient decent
eps = 0.2

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=augment_dim, non_linearity=non_linearity, 
                    architecture=architecture, T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3) 

# %%
from models.training import doublebackTrainer

trainer_anode = doublebackTrainer(anode, optimizer_anode, device, cross_entropy=cross_entropy, turnpike = turnpike,
                         bound=bound, fixed_projector=fp, verbose = True, eps_comp = 0.2) 
trainer_anode.train(dataloader, num_epochs)

# %% [markdown]
# ### Check the basins of attraction

# %%
from plots.plots import classification_levelsets

classification_levelsets(anode, 'classification_levelsets')

from IPython.display import Image

img1 = Image(filename = 'classification_levelsets' + '.png', width = 400)
display(img1)

# %%
for X_viz, y_viz in dataloader_viz:
    X_viz = X_viz[0:5]
    break

trajectories = anode.flow.trajectory(X_viz, 10).detach()
print(X_viz)
print(trajectories.size()) #time is in the first coordinate

# %% [markdown]
# ## Computation of the Maximal Lyapunov Exponent by integration 

# %%
'''
In this section, the set of function definitions needed to compute the Maximal Lyapunov Exponent by direct integration
(i.e. the analytical way) are coded.
'''

import numpy as np
import scipy
import torch
from models.neural_odes import NeuralODE


def linear_dynamics(node, derivative):
    '''
    Takes as input a nODE and return the augmented (flattened) linear system (f(x), Df(x)Y) as a 
    function of (x,Y) and t (flattened)
    '''
    if isinstance(node, NeuralODE):
        f_x = node.flow.dynamics.forward
        isnode = True
    else:
        f_x = node
        isnode = False
    
    def composed_lin_ODE(x_and_Y,t):
        '''
        Build the extended right hand side of the augmented (flattened) linear system (f(x), Df(x)Y)
        '''
        size_x = int((-1 + np.sqrt(1 + 4 * x_and_Y.size))/2)
        x = x_and_Y[:size_x]
        Y = np.reshape(x_and_Y[size_x:], [size_x, size_x])
        # print('Y = ', Y)
        if isnode:
            x_torch = torch.from_numpy(x).type(torch.float32)
            x_torch.requires_grad=True
        else:
            x_torch = x
        if isnode:
            f_x_torch = f_x(t, x_torch)
            f_x_torch = f_x_torch.detach().numpy()
        else:
            f_x_torch = f_x(x_torch, t)
        
        Df_x = derivative(x_torch, t)
        
        if isnode: 
            Df_x = Df_x.numpy()
        
        Dfx_Y = np.matmul(Df_x, Y).flatten()
        
        rhs = np.concatenate((f_x_torch, Dfx_Y))
        return rhs
    
    return composed_lin_ODE


def local_FTLE(node, x, integration_time, dt = 0.5, der = torch.autograd.functional.jacobian):
    '''
    Full Maximal Lyapunov exponent computation, based on the lynear_dynamics function
    '''
    def numpy_rhs_from_node():
        def func(x,t):
            x_torch = torch.from_numpy(x).type(torch.float32)
            x_torch.requires_grad=True
            f_x_torch = node.flow.dynamics.forward(t, x_torch)
            f_x_torch = f_x_torch.detach().numpy()
            return f_x_torch
        def der(x,t):
            x_torch = torch.from_numpy(x).type(torch.float32)
            x_torch.requires_grad=True
            f_x_t = lambda x: node.flow.dynamics.forward(t, x)
            Df_x = torch.autograd.functional.jacobian(f_x_t, x_torch) # symbolical (I think)
            Dfx = Df_x.numpy()
            return Dfx
        return func, der
    
    if isinstance(node, NeuralODE):
        func, der = numpy_rhs_from_node()
        #der = torch.autograd.functional.jacobian
    else:
        func = node
    
    # start with evolving x forward for "a bit"
    # theory, but not in our case! 
    # x_t = scipy.integrate.odeint(func, x, [0, integration_time])
    # x = x_t[-1]
    
    # set up of initial values
    Jac = np.identity(x.size)
    x_and_Jac_t = np.concatenate((x,Jac.flatten()))
    L = np.zeros(x.size)

    #selection functionalities
    select_Jac_int = lambda mat: np.reshape(mat[-1,x.size:],[x.size,x.size])
    select_x_int = lambda mat: mat[-1,:x.size]
    
    # first flow forward
    #x_and_Jac_t = scipy.integrate.odeint(linear_dynamics(node, der), x_and_Jac_t, [0, integration_time])
    #Jac_t = select_Jac_int(x_and_Jac_t)
    #Q_t, R_t = np.linalg.qr(Jac_t)
    #x_and_Jac_t = np.concatenate((select_x_int(x_and_Jac_t),Q_t.flatten()))
    
    # set up of time values 
    start_time = 0
    
    iters = 20  # number of iterations done
    n_start_iter = 0 # number of integration time used to get in position
    length_iter = integration_time/iters # length of each iteration
    used_time = (iters - n_start_iter) * length_iter
    
    for i in range(iters):
        time_array_iter = start_time + np.array([0, length_iter])
        start_time = time_array_iter[-1]
        x_and_Jac_t = scipy.integrate.odeint(linear_dynamics(func, der), x_and_Jac_t, time_array_iter)
        
        Jac_t = select_Jac_int(x_and_Jac_t)
        
        Q_t, R_t = np.linalg.qr(Jac_t)
        x_and_Jac_t = np.concatenate((select_x_int(x_and_Jac_t),Q_t.flatten())) # for numerical stability

        # if the system has stabilised, the lyapunov computation takes place
        if i >= n_start_iter:
            L += np.log(np.abs((np.diagonal(R_t)))) # the QR decomposition seems to, at times, give negative Rs, why??
        
    return (L)/length_iter/(iters-n_start_iter)



# %%
import time

# testing ODE with different MLE in phase space
def multiple_exponents_ODE2D():
    def func(x, t):
        if t<1/2 and np.linalg.norm(x-np.array([1,1]))<1/2:
            return -x
        else:
            return 2*x
    def der(x,t):
        if t<1/2 and np.linalg.norm(x-np.array([1,1]))<1/2:
            return -np.eye(2)
        else:
            return 2*np.eye(2)
    return func, der


func, der = multiple_exponents_ODE2D()

x_amount = 5
integration_time = T 
dt = 0.1

x = np.linspace(-2,2,x_amount)
y = np.linspace(-2,2,x_amount)
X, Y = np.meshgrid(x, y)
XY = np.array([X.flatten(), Y.flatten()])

lyap_small_exp = np.zeros(x_amount**2)
start_time = time.time()
for i in range(x_amount**2):
    lyap_small_exp[i] = np.max(local_FTLE(func, XY[:,i], 1, 0.1, der))
    iteration_time = time.time() - start_time
    if np.mod(i, 10) == 0:
        print(i+1,' out of ', x_amount**2, ' after ', iteration_time)
    # print(lyap[i])
    #break
lyap_small_exp = np.reshape(lyap_small_exp, (x_amount,x_amount))
print(lyap_small_exp)

# Create heatmap using imshow
from IPython.display import Image

file_name = 'MLE_analytic_2DExp.png'

anodeimg = plt.imshow(np.rot90(lyap_small_exp), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()  # Show color scale
plt.savefig(file_name, bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

img1 = Image(file_name, width = 400)
display(img1)


# %%
## the simplest example: 1D linear 

def linear1D():
    def func(x_func, t):
        #print(t)
        return x_func
    def derivative(x_func, t):
        return np.array([[1.0]])
    return func, derivative


# %%
## a little harder test case : the Lorentz system
def lorenz():
    sigma = 10
    beta = 8/3
    rho = 28
    def func(x_func, t):
        #print(t)
        x = x_func[0]
        y = x_func[1]
        z = x_func[2]
        lor = np.array([sigma*(y-x), x*(rho - z)-y, x*y - beta*z])
        return lor
    def derivative(x_func, t):
        x = x_func[0]
        y = x_func[1]
        z = x_func[2]
        return np.array([[-sigma, sigma, 0],[rho-z, -1, -x],[y, x, -beta]])
    return func, derivative


# %%
## a little harder test case: the Roessler attractor
def rossler():
    a = 0.1
    b = 0.1
    c = 14
    def func(x_func, t):
        #print(t)
        x = x_func[0]
        y = x_func[1]
        z = x_func[2]
        ross = np.array([-y-z, x+a*y, b+z*(x-c)])
        return ross
    def derivative(x_func, t):
        x = x_func[0]
        y = x_func[1]
        z = x_func[2]
        return np.array([[0, -1, -1],[1, a, 0],[z, 0, x-c]])
    return func, derivative


# %%
import numpy as np
import scipy

print('\nResults for 1D linear')
lin1D_func, lin1D_der = linear1D()
print('Refined implementation \n', local_FTLE(lin1D_func, np.array([1]), 60, 0.1, lin1D_der), 'VS 1\n\n') 

test = True
if test:
    lor_func, lor_der = lorenz()
    print('Results for Lorentz')
    print('Refined implementation \n', local_FTLE(lor_func, np.array([1,2,3]), 100, 0.1, lor_der), 'VS 0.90')
    
    print('\nResults for Rossler show convergence over time')
    ross_func, ross_der = rossler()
    print('Refined implementation t=100\n', local_FTLE(ross_func, np.array([1,2,3]), 100, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')
    print('Refined implementation t=500\n', local_FTLE(ross_func, np.array([1,2,3]), 500, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')
    print('Refined implementation t=1500\n', local_FTLE(ross_func, np.array([1,2,3]), 1500, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')


# %%
import time

# sanity check:
l = local_FTLE(anode, np.array([0.1,-0.5]), 19, 0.1)
print(l)

x_amount = 100
integration_time = T 
dt = 0.1

x = np.linspace(-2,2,x_amount)
y = np.linspace(-2,2,x_amount)
X, Y = np.meshgrid(x, y)
XY = np.array([X.flatten(), Y.flatten()])

lyap2 = np.zeros(x_amount**2)
start_time = time.time()
for i in range(x_amount**2):
    lyap2[i] = np.max(local_FTLE(anode, XY[:,i], 20, 0.1))
    iteration_time = time.time() - start_time
    if np.mod(i, 10) == 0:
        print(i+1,' out of ', x_amount**2, ' after ', iteration_time)
    # print(lyap[i])
    #break
lyap2 = np.reshape(lyap2, (x_amount,x_amount))
print(lyap2)

# %%
lyap2 = np.reshape(lyap2, (x_amount,x_amount))


# %%
# Create heatmap using imshow
from IPython.display import Image

file_name = 'MLE_analytic_7.png'

anodeimg = plt.imshow(np.rot90(lyap2), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()  # Show color scale
plt.savefig(file_name, bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

img1 = Image(file_name, width = 400)
display(img1)

img1 = Image(filename = 'classification_levelsets' + '.png', width = 400)
display(img1)


# %%
def func(x,t):
    x_torch = torch.from_numpy(x).type(torch.float32)
    x_torch.requires_grad=True
    f_x_torch = anode.flow.dynamics.forward(t, x_torch)
    f_x_torch = f_x_torch.detach().numpy()
    return f_x_torch


x = np.array([0.6, -0.4])

print(local_FTLE(anode, x, 20, 0.1))

x_torch = torch.from_numpy(x).type(torch.float32)

trajectories = anode.flow.trajectory(x_torch, 5000).detach()
x_t = scipy.integrate.odeint(func, x, np.linspace(0,20,500))

#plt.plot(trajectories[:,0],trajectories[:,1])
plt.plot(x_t[:,0],x_t[:,1])
#plt.show()

x = x - np.array([0.06, 0.02])
x_torch = torch.from_numpy(x).type(torch.float32)

#trajectories = anode.flow.trajectory(x_torch, 200).detach()

#plt.plot(trajectories[:,0],trajectories[:,1])
x_t = scipy.integrate.odeint(func, x, np.linspace(0,20,500))

#plt.plot(trajectories[:,0],trajectories[:,1])
plt.plot(x_t[:,0],x_t[:,1])

plt.show()

# %%
with open('Lyap2_pic_data_for_RNN.npy', 'wb') as f:
    np.save(f, lyap2)

# %%
with open('Lyap_pic_data_for_RNN.npy', 'rb') as f:
    lyap_load = np.load(f)

# Create heatmap using imshow
from IPython.display import Image

file_name = 'MLE_analytic_6.png'

anodeimg = plt.imshow(np.rot90(lyap_load), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()  # Show color scale
plt.savefig(file_name, bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

img1 = Image(file_name, width = 400)
display(img1)

# %%
import time

# testing ODE with different MLE in phase space
def weirdODE1D():
    def func(x, t):
        if x<0:
            return -x # contracting
        else:
            return x # expanding
    def der(x,t):
        if x<0:
            return np.array([[-1]]) # contracting
        else:
            return np.array([[1]]) # expanding
    return func, der


def weirdODE2D():
    def func(x, t):
        return np.abs(x) # expanding
    def der(x,t):
        return np.diagflat(np.sign(x)) # expanding
    return func, der



weird1D_func, weird1D_der = weirdODE1D()
local_FTLE(weird1D_func, np.array([1]), 60, 0.1, weird1D_der)

x_amount = 10
integration_time = T 
dt = 0.1

x = np.linspace(-2,2,x_amount)

lyap_weird = np.zeros(x_amount)
start_time = time.time()
for i in range(x_amount):
    lyap_weird[i] = np.max(local_FTLE(weird1D_func, x[i], 20, 0.1, weird1D_der))
    iteration_time = time.time() - start_time
    if np.mod(i, 10) == 0:
        print(i+1,' out of ', x_amount, ' after ', iteration_time)

print(lyap_weird)

# %%
# Create heatmap using imshow
from IPython.display import Image

file_name = 'weirdODE_MLE_5.png'

anodeimg = plt.imshow(np.rot90(lyap_weird), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()  # Show color scale
plt.savefig(file_name, bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

img1 = Image(file_name, width = 400)
display(img1)

# %%
import time

def weirdODE2D():
    def func(x, t):
        return np.abs(x) # expanding
    def der(x,t):
        return np.diagflat(np.sign(x)) # expanding
    return func, der



weird1D_func, weird1D_der = weirdODE2D()
local_FTLE(weird1D_func, np.array([1]), 60, 0.1, weird1D_der)

x_amount = 6
integration_time = T 
dt = 0.1

x = np.linspace(-2,2,x_amount)
y = np.linspace(-2,2,x_amount)
X, Y = np.meshgrid(x, y)
XY = np.array([X.flatten(), Y.flatten()])

lyap_abs2D = np.zeros(x_amount**2)
start_time = time.time()
for i in range(x_amount**2):
    lyap_abs2D[i] = np.max(local_FTLE(weird1D_func, XY[:,i], 20, 0.1,weird1D_der))
    iteration_time = time.time() - start_time
    if np.mod(i, 10) == 0:
        print(i+1,' out of ', x_amount**2, ' after ', iteration_time)
    # print(lyap[i])
    #break
lyap_abs2D = np.reshape(lyap_abs2D, (x_amount,x_amount))
print(lyap_abs2D)


# %%
# Create heatmap using imshow
from IPython.display import Image

file_name = 'weirdODE2D_MLE_5.png'

anodeimg = plt.imshow(np.rot90(lyap_abs2D), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()  # Show color scale
plt.savefig(file_name, bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

img1 = Image(file_name, width = 400)
display(img1)

# %%
# testing the auto grad function 
testing = False
if testing:
    w = torch.rand(2,2)
    b = torch.rand(2)
    print(w, b)
    def exp_reducer(x):
         return torch.matmul(w, x) + b

    inputs = torch.rand(2)

    print(torch.autograd.functional.jacobian(exp_reducer, inputs))

# %%
# TESTING 
testing = False
if testing:
    x = np.array([1., 2.]) # point where we compute the MLE
    Jac = np.identity(2)     # vector with appropriate size
    x_and_Jac = np.concatenate((x,Jac.flatten())).astype(np. double)
    dt = 0.1
    integration_time = 1.
    time_array = np.arange(0, integration_time, dt)
    debug_MLE = False
    f_x = anode.flow.dynamics.forward
    f_x(integration_time, torch.from_numpy(x).type(torch.float32)) # this now works
    CLO = linear_dynamics(anode)
    CLO(x_and_Jac, integration_time)

    out = scipy.integrate.odeint(linear_dynamics(anode), x_and_Jac, time_array)
    # out has one entry for each time

    # we are only interested in the last time integration, that we reshape as a matrix
    L = np.reshape(out[-1,2:],[2,2])

    print('\nL = ', L)
    print('\nlog( L * L.T )= ', np.log(np.matmul(L, L.transpose())))
    
    lyap_exp = np.linalg.eig( 1/(2*integration_time) * np.log(np.matmul(L, L.transpose())))
    
    print('\nLyap exps = ', lyap_exp.eigenvalues)
    
    # using the full function
    l = local_FTLE(anode, np.array([0.1,-0.5]), 1)
    print('Local FTLE = ', l)

# %%
from plots.gifs import trajectory_gif

#the function trajectory_gif creates the evolution trajectories of the input data through the network
#the passing time corresponds to the depth of the network

for X_viz, y_viz in dataloader_viz:
    trajectory_gif(anode, X_viz[0:50], y_viz[0:50], timesteps=num_steps, filename = 'trajectory.gif', axlim = 8, dpi = 100)
    trajectory_gif(rnode, X_viz[0:50], y_viz[0:50], timesteps=num_steps, filename = 'trajectory_db.gif', axlim = 8, dpi = 100)
    break

#Display of the generated gif
traj = Image(filename="trajectory.gif")
rtraj = Image(filename="trajectory_db.gif")
display(traj, rtraj)

# %%
from IPython.display import Image
from plots.plots import comparison_plot
# traj = Image(filename="trajectory19.png", retina = True)
# rtraj = Image(filename="trajectory_db19.png", retina = True)
# display(traj, rtraj)

comparison_plot("trajectory39.png", 'standard training', "trajectory_db39.png", 'robust training', 'traj_comp.png')
display(Image('traj_comp.png', width = 800))

# %% [markdown]
# I want to visualize the separation boundary in the final linear layer.

# %%
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from matplotlib.colors import to_rgb
import imageio

from matplotlib.colors import LinearSegmentedColormap
import os


@torch.no_grad()
def linlayer_levelsets(model, fig_name=None, footnote=None, contour = True, plotlim = [-2, 2]):
    
    
    x1lower, x1upper = plotlim
    x2lower, x2upper = plotlim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig = plt.figure(figsize=(5, 5), dpi=100)
    
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)

    
   
    model.to(device)

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    
    preds = model.linear_layer(model_inputs)
    
    # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
    m = nn.Softmax(dim=2)
    # softmax normalizes the model predictions to probabilities
    preds = m(preds)

    #we only need the probability for being in class1 (as prob for class2 is then 1- class1)
    preds = preds[:, :, 0]
    preds = preds.unsqueeze(2)  # adds a tensor dimension at position 2
    
    plt.grid(False)
    plt.xlim([x1lower, x1upper])
    plt.ylim([x2lower, x2upper])

    ax = plt.gca()
    ax.set_aspect('equal') 
    
    if contour:
        colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is orange, last is blue
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=40)
        z = np.array(preds).reshape(xx1.shape)
        
        levels = np.linspace(0.,1.,20).tolist()
        
        cont = plt.contourf(xx1, xx2, z, levels, alpha=1, cmap=cm, zorder = 0, extent=(x1lower, x1upper, x2lower, x2upper)) #plt.get_cmap('coolwarm')
        cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('prediction prob.')
    

    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
    # else: plt.show()


# %%
linlayer_levelsets(anode)


# %%
@torch.no_grad()
def trajectory_gif_new(model, inputs, targets, timesteps, dpi=200, alpha=0.9,
                   alpha_line=1, filename='trajectory.gif', axlim = 0):
    
    from matplotlib import rc
    from scipy.interpolate import interp1d
    rc("text", usetex = False)
    font = {'size'   : 18}
    rc('font', **font)

    if not filename.endswith(".gif"):
        raise RuntimeError("Name must end in with .gif, but ends with {}".format(filename))
    base_filename = filename[:-4]

    ## We focus on 3 colors at most
    if False in (t < 2 for t in targets): 
        color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
    else:
        #color = ['crimson' if targets[i, 0] > 0.0 else 'dodgerblue' for i in range(len(targets))]
        color = ['C1' if targets[i] > 0.0 else 'dimgrey' for i in range(len(targets))]

    trajectories = model.flow.trajectory(inputs, timesteps).detach()
    num_dims = trajectories.shape[2]

    if axlim == 0:        
        x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
        y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    else: 
        x_min, x_max = -axlim, axlim  #to normalize for rob and standard nODE
        y_min, y_max = -axlim, axlim   #
        
    if num_dims == 3:
        z_min, z_max = trajectories[:, :, 2].min(), trajectories[:, :, 2].max()
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    if num_dims == 3:
        z_range = z_max - z_min
        z_min -= margin * z_range
        z_max += margin * z_range
        
    T = model.T 
    integration_time = torch.linspace(0.0, T, timesteps)
    
    interp_x = []
    interp_y = []
    interp_z = []
    for i in range(inputs.shape[0]):
        interp_x.append(interp1d(integration_time, trajectories[:, i, 0], kind='cubic', fill_value='extrapolate'))
        interp_y.append(interp1d(integration_time, trajectories[:, i, 1], kind='cubic', fill_value='extrapolate'))
        if num_dims == 3:
            interp_z.append(interp1d(integration_time, trajectories[:, i, 2], kind='cubic', fill_value='extrapolate'))
    
    interp_time = 20
    # interp_time = 3 #this was 5 before
    _time = torch.linspace(0., T, interp_time)

    plt.rc('grid', linestyle="dotted", color='lightgray')
    for t in range(interp_time):
        if num_dims == 2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            label_size = 13
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size 
            ax.set_axisbelow(True)
            ax.xaxis.grid(color='lightgray', linestyle='dotted')
            ax.yaxis.grid(color='lightgray', linestyle='dotted')
            ax.set_facecolor('whitesmoke')
            
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.rc('text', usetex=False)
            plt.rc('font', family='serif')
            plt.xlabel(r'$x_1$', fontsize=12)
            plt.ylabel(r'$x_2$', fontsize=12)
            
            x1 = torch.arange(x_min, x_max, step=0.01, device=device)
            x2 = torch.arange(y_min, y_max, step=0.01, device=device)
            xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
            model_inputs = torch.stack([xx1, xx2], dim=-1)
            
            preds = model.linear_layer(model_inputs)
            
            # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
            m = nn.Softmax(dim=2)
            # softmax normalizes the model predictions to probabilities
            preds = m(preds)

            #we only need the probability for being in class1 (as prob for class2 is then 1- class1)
            preds = preds[:, :, 0]
            preds = preds.unsqueeze(2)  # adds a tensor dimension at position 2
            
            plt.grid(False)
    

            ax = plt.gca()
            ax.set_aspect('equal') 
            
            
            colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is orange, last is blue
            cm = LinearSegmentedColormap.from_list(
                "Custom", colors, N=40)
            z = np.array(preds).reshape(xx1.shape)
            
            levels = np.linspace(0.,1.,15).tolist()
            
            cont = plt.contourf(xx1, xx2, z, levels, alpha=0.5, cmap=cm, zorder = 0, extent=(x_min, x_max, y_min, y_max)) #plt.get_cmap('coolwarm')
            
            
            
            
            plt.scatter([x(_time)[t] for x in interp_x], 
                         [y(_time)[t] for y in interp_y], 
                         c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black', zorder=3)

            if t > 0:
                for i in range(inputs.shape[0]):
                    x_traj = interp_x[i](_time)[:t+1]
                    y_traj = interp_y[i](_time)[:t+1]
                    plt.plot(x_traj, y_traj, c=color[i], alpha=alpha_line, linewidth = 0.75, zorder=1)
            
        
        ax.set_aspect('equal')

        plt.savefig(base_filename + "{}.png".format(t),
                    format='png', dpi=dpi, bbox_inches='tight', facecolor = 'white')
        # Save only 3 frames (.pdf for paper)
        # if t in [0, interp_time//5, interp_time//2, interp_time-1]:
        #     plt.savefig(base_filename + "{}.pdf".format(t), format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

    imgs = []
    for i in range(interp_time):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        if i not in [0, interp_time//5, interp_time//2, interp_time-1]: os.remove(img_file) 
    imageio.mimwrite(filename, imgs, duration = 500)

# %%
from plots.gifs import trajectory_gif

#the function trajectory_gif creates the evolution trajectories of the input data through the network
#the passing time corresponds to the depth of the network

data_iter = iter(dataloader)
X,y = next(data_iter)
batch_size = 40

X_viz = torch.zeros_like(X[0:batch_size])
y_viz = torch.zeros(batch_size)
print(y_viz.size())
print(X_viz.size())
X_viz[:,1] = torch.linspace(-5,5,batch_size)
X_viz[:,0] = torch.tensor([0])
print(X_viz)

# plt.scatter(X_viz[:,0],X_viz[:,1])

trajectory_gif_new(anode, X_viz[0:batch_size], y_viz[0:batch_size], timesteps=num_steps, filename = 'trajectory.gif', axlim = 8, dpi = 100)
trajectory_gif_new(rnode, X_viz[0:batch_size], y_viz[0:batch_size], timesteps=num_steps, filename = 'trajectory_db.gif', axlim = 8, dpi = 100)


# %% [markdown]
# ### comparison with other Lyapunov computation

# %%
'''
input: two trajectories x(t), y(t) where each has size (x_dim,len_t)
output: maximum lyapunov exponent for each time t, size (1,len_t)'''
def le(x,y,t):
    d = x - y
    d = torch.linalg.norm(x - y, dim = 0)
    # print(f'{d.shape = }')
    d = d/d[0]
    d = np.log(d)
    

    d = d/t
    return d



'''
Compute the maximum Lyapunov exponent with initial value tolerance eps
input: 
trajectories of size (x_amount,x_amount,x_dim,time_dim)
each trajectory has initial value x_0 = traj[i,j,:,0]

output:
MLE for each trajectory as averaged LE for all initial values that is eps close to the initial value
output size (x_amount,x_amount,t_dim)
'''
def MLE(traj, t, eps = 0.1):
    x_amount = traj.size(0)
    x_amount, y_amount, x_dim, t_dim = traj.shape
    le_val = torch.zeros((x_amount, y_amount, t_dim)) 
    print(f'{le_val.shape = }')
    for i in range(x_amount):
        for j in range(y_amount):
            count = 0
            for i_comp in range(x_amount):
                for j_comp in range(y_amount):
                    if (torch.norm(traj[i,j,:,0] - traj[i_comp,j_comp,:,0]) < eps and not(i == i_comp and j == j_comp)):
                        count += 1
                        le_val[i,j] += le(traj[i,j,:,:],traj[i_comp,j_comp,:,:],t)
                        print('avg le update with count ',count,' and value ',le_val[i,j,-1])
            if count > 1:
                le_val[i,j,:] = le_val[i,j,:]/count
    return le_val



# %%

x_amount = 100
eps = 0.08
t = torch.linspace(0,T,2) #only take initial and final time for MLE


x = torch.linspace(-2,2,x_amount)
y = torch.linspace(-2,2,x_amount)
X, Y = torch.meshgrid(x, y)

inputs = torch.stack([X,Y], dim=-1)
print(f'{inputs.size() = }')
# print(inputs)

trajectories = anode.flow.trajectory(inputs, num_steps).detach()
t_indices = torch.tensor([0,-1])

trajectories = trajectories[t_indices] #only take the initial and final time of the trajectory
print('shape of trajectories of grid', trajectories.shape) #MLE works with the time in the last dimension
trajectories = trajectories.permute(1,2,3,0)
print('shape of permuted traj of grid', trajectories.shape) #this fits with the MLE function

output = MLE(trajectories,t,eps)
print(output)
output = output[:,:,-1] #reduce to last time instance T



# %%
# Create heatmap using imshow
anodeimg = plt.imshow(np.rot90(output), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()  # Show color scale
plt.savefig('MLE.png',bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

img = Image(filename = 'MLE.png', width = 400)

display(img)

# %% [markdown]
# # compare output (from rough computation) and lyap2 (from more refined computation)

# %%
from matplotlib import colors


vmin = min([min(output.flatten()), min(lyap2.flatten())])
vmax = max([max(output.flatten()), max(lyap2.flatten())])
norm = colors.Normalize(vmin=vmin, vmax=vmax)

anodeimg = plt.imshow(np.rot90(output), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
anodeimg.set_norm(norm)
plt.colorbar()
plt.savefig('MLE_rough.png',bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

lyapimg = plt.imshow(np.rot90(lyap2), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
lyapimg.set_norm(norm)
plt.colorbar()
plt.savefig('MLE_integration.png',bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

img1 = Image(filename = 'MLE_rough.png', width = 400)
img2 = Image(filename = 'MLE_integration.png', width = 400)

display(img1,img2)

# %%
