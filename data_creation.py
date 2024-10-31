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
    # print(bool_mask)
    noisy_adjacency = np.multiply(noisy_adjacency, bool_mask)

    # print(noisy_adjacency)

    # randomly add unwanted connections
    rand_mat = np.random.uniform(size=np.shape(adjacency))
    bool_mask = rand_mat > threshold
    #print(bool_mask)
    noisy_adjacency += np.multiply(flipped_adjacency, bool_mask)

    return noisy_adjacency


def noisy_data(*data_list):
    new_list = []
    for i in range(len(data_list)):
        new_list.append(data_list[i] + 0.01*torch.rand(data_list[i].size()))
    return data_list


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


def from_network_to_data(par_struct, n_data, dim):
    gamma, Win, bin, Wout, bout = par_struct.get_parameters()
    node_2D = make_nODE_from_parameters(gamma, Win=Win, bin=bin, Wout=Wout, bout=bout)

    u0 = torch.rand([n_data, dim])
    uT = node_2D.forward(u0)
    return u0, uT


def create_dataset(dim, n_data, n_networks=1):
    def x_squish_data(u0, uT, Atilde):
        x = torch.cat((u0, uT, Atilde.flatten()))
        return x

    def y_squish_data(uT, A):
        y = torch.cat((uT, A.flatten()))
        return y

    x_exact = list()  # x_i = u0i, uTi, A.flatten
    x_noisy = list()  # x_i = u0i_tilde, uTi_tilde, Atilde.flatten
    y = list()  # y_i = uTi, A.flatten
    p = list()  # p_i = torch_parameter_structure
    for i in range(n_networks):
        par_struct, adjacency = create_random_network(dim)
        data_u0, data_uT = from_network_to_data(par_struct, n_data, dim)
        noisy_data_u0, noisy_data_uT = noisy_data(data_u0, data_uT)
        noisy_adjacency = randomize_adjacency(adjacency)
        x_noisy.append([x_squish_data(noisy_data_u0[i, :], noisy_data_uT[i, :], noisy_adjacency) for i in range(n_data)])
        x_exact.append([x_squish_data(data_u0[i, :], data_uT[i, :], adjacency) for i in range(n_data)])
        y.append([y_squish_data(data_uT[i, :], adjacency) for i in range(n_data)])
        p.append(par_struct)
    return x_exact, x_noisy, y, p


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

    data = create_dataset(2, 5, 2)
    print(data)