import numpy as np
import matplotlib.pyplot as plt
from models.training import easyTrainer, weights_to_dataset
from models.nODE import nODE, make_nODE_from_parameters
import scipy
import torch


rng1 = np.random.default_rng()


def rand_dim(dim):
    return rng1.standard_normal(dim)


def adjacency_matrix(W):
    if isinstance(W, np.ndarray):
        W = torch.from_numpy(W)
    values_log_W = torch.sort(torch.log(torch.abs(W.flatten())))[0]
    threshold = values_log_W[torch.sort(values_log_W[1:] - values_log_W[:-1])[1][-1] + 1]
    W_bool_activation = (W >= torch.exp(threshold))
    W_bool_repr = (W <= -torch.exp(threshold))
    adjacency_mat = W_bool_activation.float() - W_bool_repr.float()
    return adjacency_mat


def random_zeros(mat, threshold = 0.7):
    rand_mat = np.random.uniform(size = np.shape(mat))
    bool_mask = rand_mat > threshold
    return np.multiply(mat, bool_mask)


def randomize_adjacency(adjacency, threshold=0.8):
    noisy_adjacency = np.abs(adjacency)
    flipped_adjacency = 1 - noisy_adjacency

    # randomly remove  connections
    rand_mat = np.random.uniform(size=np.shape(adjacency))
    bool_mask = rand_mat > (1 - threshold)
    print(bool_mask)
    noisy_adjacency = np.multiply(noisy_adjacency, bool_mask)
    print(noisy_adjacency)

    # randomly add unwanted connections
    rand_mat = np.random.uniform(size=np.shape(adjacency))
    bool_mask = rand_mat > threshold
    print(bool_mask)
    noisy_adjacency += np.multiply(flipped_adjacency, bool_mask)

    return noisy_adjacency


class np_parameter_structure():
    def __init__(self, gamma, Win=None, Wout=None, bin=None, bout=None):
        self.gamma = gamma
        self.dim = np.size(gamma)
        if Win is None:
            self.Win = np.eye(dim)
        else:
            self.Win = Win
        if bin is None:
            self.bin = np.zeros(dim)
        else:
            self.bin = bin
        if Wout is None:
            self.Wout = np.eye(dim)
        else:
            self.Wout = Wout
        if bout is None:
            self.bout = np.zeros(dim)
        else:
            self.bout = bout
        self.check_dim()
        return

    def check_dim(self):
        def check_dim_vec(vec,dim):
            if np.max(np.shape(vec)) != np.size(vec):
                raise ValueError('Expected vector, received %f instead', vec)
            if np.size(vec)!=dim:
                raise ValueError('Expected vector of length %i, got length %i instead', self.dim, np.size(vec))
            return

        def check_dim_mat(mat, dim):
            if len(np.shape(mat))!=2 or any([x!=np.shape(mat)[0] for x in np.shape(mat)]):
                raise ValueError('Expected matrix, received %f instead', mat)
            if np.shape(mat)[0] != dim:
                raise ValueError('Expected square matrix of size %i, got length %i instead', self.dim, np.size(vec))
            return
        check_dim_mat(self.Win, self.dim)
        check_dim_mat(self.Wout, self.dim)
        check_dim_vec(self.gamma, self.dim)
        check_dim_vec(self.bin, self.dim)
        check_dim_vec(self.bout, self.dim)
        return

    def set_parameters(self, gamma=None, Win=None, Wout=None, bin=None, bout=None):
        if gamma is not None:
            self.gamma = gamma
            self.dim = np.size(gamma)
        if Win is None:
            self.Win = np.eye(dim)
        else:
            self.Win = Win
        if bin is None:
            self.bin = np.zeros(dim)
        else:
            self.bin = bin
        if Wout is None:
            self.Wout = np.eye(dim)
        else:
            self.Wout = Wout
        if bout is None:
            self.bout = np.zeros(dim)
        else:
            self.bout = bout
        return

    def get_parameters(self):
        return self.gamma, self.Win, self.bin, self.Wout, self.bout

    def get_vec_par(self):
        vec_par = np.append(np.append(self.gamma.flatten(), self.Win.flatten()),
                            np.append(np.append(self.bin.flatten(), self.Wout.flatten()),
                            self.bout.flatten()))
        return vec_par

    def set_vec_par(self, vec_par):
        dim_vec = np.shape(vec_par.flatten())[0]
        dim_float = (- 3 + np.sqrt(9 + 8 * dim_vec))/4
        if np.remainder(dim_float, 1) != 0:
            raise ValueError('The given vector cannot have the right parameters')
        dim = int(dim_float)
        self.gamma = vec_par[0:dim]
        vec_par = vec_par[dim:]
        Win = vec_par[0:dim**2]
        self.Win = np.reshape(Win, [dim,dim])
        vec_par = vec_par[dim**2:]
        self.bin = vec_par[0:dim]
        vec_par = vec_par[dim:]
        Wout = vec_par[0:dim**2]
        self.Wout = np.reshape(Wout, [dim,dim])
        vec_par = vec_par[dim**2:]
        self.bout = vec_par[0:dim]
        return


class torch_parameter_structure():
    def __init__(self, gamma, Win=None, Wout=None, bin=None, bout=None):
        self.gamma = gamma
        self.dim = torch.size(gamma)
        if Win is None:
            self.Win = torch.eye(dim)
        else:
            self.Win = Win
        if bin is None:
            self.bin = torch.zeros(dim)
        else:
            self.bin = bin
        if Wout is None:
            self.Wout = torch.eye(dim)
        else:
            self.Wout = Wout
        if bout is None:
            self.bout = torch.zeros(dim)
        else:
            self.bout = bout
        self.check_dim()
        return

    def check_dim(self):
        def check_dim_vec(vec, dim):
            if torch.max(torch.tensor(vec.size())) != torch.prod(torch.tensor(vec.size())):
                raise ValueError('Expected vector, received %f instead', vec)
            if torch.max(torch.tensor(vec.size())) !=dim:
                raise ValueError('Expected vector of length %i, got length %i instead', self.dim, torch.size(vec))
            return

        def check_dim_mat(mat, dim):
            if len(mat.size())!=2 or any(mat.size()!=mat.size()[0]):
                raise ValueError('Expected matrix, received %f instead', mat)
            if mat.size()[0] != dim:
                raise ValueError('Expected square matrix of size %i, got length %i instead', self.dim, np.size(vec))
            return
        check_dim_mat(self.Win, self.dim)
        check_dim_mat(self.Wout, self.dim)
        check_dim_vec(self.gamma, self.dim)
        check_dim_vec(self.bin, self.dim)
        check_dim_vec(self.bout, self.dim)
        return

    def set_parameters(self, gamma=None, Win=None, Wout=None, bin=None, bout=None):
        if gamma is not None:
            self.gamma = gamma
            self.dim = torch.size(gamma)
        if Win is None:
            self.Win = torch.eye(dim)
        else:
            self.Win = Win
        if bin is None:
            self.bin = torch.zeros(dim)
        else:
            self.bin = bin
        if Wout is None:
            self.Wout = torch.eye(dim)
        else:
            self.Wout = Wout
        if bout is None:
            self.bout = torch.zeros(dim)
        else:
            self.bout = bout
        return

    def get_parameters(self):
        return self.gamma, self.Win, self.bin, self.Wout, self.bout

    def get_vec_par(self):
        vec_par = np.array([self.gamma.flatten(), self.Win.flatten(), self.bin.flatten(), self.Wout.flatten(),
                            self.bout.flatten()])
        return vec_par

    def set_vec_par(self, vec_par):
        dim_vec = (vec_par.flatten().size())[0]
        dim_float = (- 3 + torch.sqrt(torch.tensor(9 + 8 * dim_vec)))/4
        if dim_float-dim_float.int() != 0:
            raise ValueError('The given vector cannot have the right parameters')
        dim = int(dim_float)
        self.gamma = vec_par[0:dim]
        vec_par = vec_par[dim:]
        Win = vec_par[0:dim**2]
        self.Win = np.reshape(Win, [dim,dim])
        vec_par = vec_par[dim**2:]
        self.bin = vec_par[0:dim]
        vec_par = vec_par[dim:]
        Wout = vec_par[0:dim**2]
        self.Wout = np.reshape(Wout, [dim,dim])
        vec_par = vec_par[dim**2:]
        self.bout = vec_par[0:dim]
        return


def create_random_network(dim):
    gamma = - np.abs(rand_dim(dim))
    Wout = rand_dim([dim, dim])
    Win = rand_dim([dim, dim])
    bin, bout = rand_dim(dim), rand_dim(dim)

    Win = random_zeros(Win, threshold=0.5)
    Wout = random_zeros(Wout)

    par_struct = np_parameter_structure(gamma, Win=Win, bin=bin, Wout=Wout, bout=bout)
    adjacency = adjacency_matrix(np.matmul(Wout, Win))
    return par_struct, adjacency


def from_network_to_data(par_struct, n_data):
    gamma, Win, bin, Wout, bout = par_struct.get_parameters()
    node_2D = make_nODE_from_parameters(gamma, Win=Win, bin=bin, Wout=Wout, bout=bout)

    u0 = rand_dim([n_data, dim])
    uT = node_2D.trajectory(torch.from_numpy(u0).float()).detach().numpy()[-1, :, :]
    return u0, uT



if __name__ == "__main__":
    dim = 2
    n_data = 50

    gamma = - np.abs(rand_dim(dim))
    Wout = rand_dim([dim, dim])
    Win = rand_dim([dim, dim])
    bin, bout = rand_dim(dim), rand_dim(dim)

    Win = random_zeros(Win, threshold=0.5)
    Wout = random_zeros(Wout)

    adjacency = adjacency_matrix(np.matmul(Wout, Win))

    node_2D = make_nODE_from_parameters(gamma, Win=Win, bin=bin, Wout=Wout, bout=bout)

    u0 = rand_dim([n_data, dim])
    uT = node_2D.trajectory(torch.from_numpy(u0).float()).detach().numpy()[-1, :, :]

    approx_adjacency = randomize_adjacency(adjacency)

    struct = np_parameter_structure(gamma)
    struct.set_parameters(Win=Win,Wout=Wout)
    print(struct.get_parameters())
    struct.set_parameters(bin=bin)
    a = struct.get_vec_par()
    print(a)

