import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

if __name__ == "__main__":
    from graph_plotting import plot_graph
from models.nODE import nODE, make_nODE_from_parameters, become_torch
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


def random_zeros(mat, threshold=0.7):
    # edges in the given matrix are set to zero with independent probability 1 - threshold
    rand_mat = np.random.uniform(size=np.shape(mat))
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
    # print(bool_mask)
    noisy_adjacency += np.multiply(flipped_adjacency, bool_mask)

    return noisy_adjacency


def noisy_data(*data_list):
    new_list = []
    for i in range(len(data_list)):
        new_list.append(data_list[i] + 0.01 * torch.rand(data_list[i].size()))
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
        def check_dim_vec(vec, dim):
            if np.max(np.shape(vec)) != np.size(vec):
                raise ValueError('Expected vector, received %f instead', vec)
            if np.size(vec) != dim:
                raise ValueError('Expected vector of length %i, got length %i instead', self.dim, np.size(vec))
            return

        def check_dim_mat(mat, dim):
            if len(np.shape(mat)) != 2 or any([x != np.shape(mat)[0] for x in np.shape(mat)]):
                raise ValueError('Expected matrix, received %f instead', mat)
            if np.shape(mat)[0] != dim:
                raise ValueError('Expected square matrix of size %i, got length %i instead', self.dim, np.size(mat))
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
        dim_float = (- 3 + np.sqrt(9 + 8 * dim_vec)) / 4
        if np.remainder(dim_float, 1) != 0:
            raise ValueError('The given vector cannot have the right parameters')
        dim = int(dim_float)
        self.gamma = vec_par[0:dim]
        vec_par = vec_par[dim:]
        Win = vec_par[0:dim ** 2]
        self.Win = np.reshape(Win, [dim, dim])
        vec_par = vec_par[dim ** 2:]
        self.bin = vec_par[0:dim]
        vec_par = vec_par[dim:]
        Wout = vec_par[0:dim ** 2]
        self.Wout = np.reshape(Wout, [dim, dim])
        vec_par = vec_par[dim ** 2:]
        self.bout = vec_par[0:dim]
        return


def create_torch_par(gamma, Win, bin, Wout, bout):
    return torch_parameter_structure(len(gamma), gamma=become_torch(gamma), Win=become_torch(Win),
                                     bin=become_torch(bin), Wout=become_torch(Wout),
                                     bout=become_torch(bout))


class torch_parameter_structure():
    def __init__(self, dim, gamma=None, Win=None, Wout=None, bin=None, bout=None):
        self.dim = dim
        if gamma is None:
            self.gamma = torch.ones(dim)
        else:
            self.gamma = gamma
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
            if torch.max(torch.tensor(vec.size())) != dim:
                raise ValueError('Expected vector of length %i, got length %i instead', self.dim, torch.size(vec))
            return

        def check_dim_mat(mat, dim):
            if len(mat.size()) != 2 or any([mat.size()[i] != mat.size()[0] for i in range(2)]):
                raise ValueError('Expected matrix, received %f instead', mat)
            if mat.size()[0] != dim:
                raise ValueError('Expected square matrix of size %i, got length %i instead', self.dim, np.size(mat))
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
        dim_float = (- 3 + torch.sqrt(torch.tensor(9 + 8 * dim_vec))) / 4
        if dim_float - dim_float.int() != 0:
            raise ValueError('The given vector cannot have the right parameters')
        dim = int(dim_float)
        self.gamma = vec_par[0:dim]
        vec_par = vec_par[dim:]
        Win = vec_par[0:dim ** 2]
        self.Win = torch.reshape(Win, [dim, dim])
        vec_par = vec_par[dim ** 2:]
        self.bin = vec_par[0:dim]
        vec_par = vec_par[dim:]
        Wout = vec_par[0:dim ** 2]
        self.Wout = torch.reshape(Wout, [dim, dim])
        vec_par = vec_par[dim ** 2:]
        self.bout = vec_par[0:dim]
        return


def create_random_network(dim):
    def hub_struct(adjacency):
        def draw_connection():
            random_vector = np.abs(np.random.gamma(3, 0.5, size=[dim]))
            random_vector[np.random.uniform(size=[dim]) > 0.25] = 0
            return random_vector

        adjacency = random_zeros(adjacency, threshold=0.7)
        # add hubs
        n_hubs = int(np.floor(dim ** (1 / 3)))
        selected_hubs = np.random.choice(np.array(range(dim)), size=n_hubs, replace=False)
        for i in selected_hubs:
            random_vector = draw_connection()
            while all(random_vector == 0):
                random_vector = draw_connection()
            adjacency[:, i] = random_vector
        # ensure one connected component with directionality, i.e. every row has one outgoing edge
        for i in range(dim):
            if all(adjacency[i, :] == 0):
                index = np.random.choice(range(dim))
                adjacency[i, index] = np.random.gamma(3, 0.5)
        return adjacency

    gamma = - np.abs(np.random.gamma(3, 0.5, size=[dim]))
    bin, bout = np.random.gamma(3, 0.5, size=[dim]), np.random.gamma(3, 0.5, size=[dim])

    adjacency = np.random.gamma(3, 0.5, size=[dim, dim])
    Wout = np.random.gamma(3, 0.5, size=[dim, dim])
    Wout = random_zeros(Wout, threshold=0.45)
    # remove the diagonal
    Wout = np.multiply(Wout, (1 - np.diag(np.ones(dim))))
    # add random non-zero diagonal to enforce non-singularity
    Wout = np.diag(np.random.gamma(3, 0.5, size=[dim])) + Wout
    adjacency = hub_struct(adjacency)

    try:
        Win = np.linalg.solve(Wout, adjacency)
    except ValueError:
        Win = np.eye([dim])
        print(99)

    par_struct = create_torch_par(gamma, Win, bin, Wout, bout)
    return par_struct, become_torch(adjacency)


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
    for j in range(n_networks):
        par_struct, adjacency = create_random_network(dim)
        data_u0, data_uT = from_network_to_data(par_struct, n_data, dim)
        noisy_data_u0, noisy_data_uT = noisy_data(data_u0, data_uT)
        noisy_adjacency = randomize_adjacency(adjacency)
        x_noisy.append(x_squish_data(noisy_data_u0[:, :].flatten().detach(), noisy_data_uT[:, :].flatten().detach(),
                                     noisy_adjacency.detach()))
        W = torch.matmul(par_struct.Wout, par_struct.Win).flatten().detach()
        x_exact.append(x_squish_data(data_u0[:, :].flatten().detach(), data_uT[:, :].flatten().detach(), W))
        # y.append(y_squish_data(data_uT[:, :].flatten(), torch.matmul(torch.tensor(par_struct.Wout),torch.tensor(
        # par_struct.Win)).flatten()))
        y.append(torch.matmul(par_struct.Wout, par_struct.Win).flatten().detach())
        p.append(par_struct)
    return x_exact, x_noisy, y, p


def extract_adjacency(x, n_nodes):
    # x is a list of items of the form u01, u02, .. u0n, uT1, uT2... uTn, adjacency.flatten()
    adj = []
    for x_iter in x:
        flat_adj = x_iter[-n_nodes ** 2:]
        adj.append(torch.reshape(flat_adj, [n_nodes, n_nodes]).detach())
    return adj


def connectivity_test(test_dim, n_data):
    data = create_dataset(test_dim, 1, n_data)
    all_adj = extract_adjacency(data[0], test_dim)
    perc_nonzero_el = torch.zeros(test_dim ** 2 + 1)
    for adj in all_adj:
        perc_nonzero_el[torch.sum(adj != 0)] += 1
    perc_nonzero_el /= n_data
    plt.stairs(perc_nonzero_el)
    plt.xlabel('Number of non-zero elements in the adjacency matrix')
    plt.ylabel('Frequency')
    plt.show()
    return perc_nonzero_el

def to_pyg_data(x_train, y_train, ode_dim, n_data):
    # distribute x_train values as node features
    x = torch.zeros([ode_dim, 2*n_data])

    k = 0
    for j in range(0,2*n_data):
        for i in range(0,ode_dim):
            x[i,j] = x_train[k]
            k += 1

    # build edge indices (assuming fully connected network)
    adj_matrix = torch.ones((ode_dim, ode_dim))
    edge_index, _ = dense_to_sparse(adj_matrix)

    # distribute weights as edge attributes 
    edge_attr = torch.zeros([len(edge_index.t()),1])
    weights  = torch.reshape(x_train[k:],[ode_dim,ode_dim])

    k = 0
    for i in range(0,ode_dim):
        for j in range(0,ode_dim):
                edge_attr[k,0] = weights[i,j]
                k += 1

    y = y_train

    data = Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y)

    return data


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
    struct.set_parameters(Win=Win, Wout=Wout)
    # print(struct.get_parameters())
    struct.set_parameters(bin=bin)
    a = struct.get_vec_par()

    # test_dim = 2
    # data = create_dataset(test_dim, 5, 2)
    # print(data)

    test_dim = 4
    n_data = 100
    perc_nonzero_el = connectivity_test(test_dim, n_data)
    print('Distribution of types of newtorks:', perc_nonzero_el)
    print('Newtorks with no connections : ', perc_nonzero_el[0])
    # for i in range(1, test_dim ** 2 + 1):
    #     print('Newtorks with ', i, ' connections : ', perc_nonzero_el[i])

    pars, adj = create_random_network(10)
    plot_graph(adj, linewidth=1.)
    plt.show()


