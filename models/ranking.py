import torch
import torch.nn as nn
from torchdiffeq import odeint
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from interval import interval

from models.nODE import nODE, make_nODE_from_parameters


def infty_bound(model: nODE):
    def prod_loc(gamma, bout, Wout, interval_x):
        x_bound = np.empty([model.ODE_dim, 2])
        for i in range(model.ODE_dim):
            sum_w = interval(0)
            for j in range(model.ODE_dim):
                sum_w += interval(Wout[i, j]) * interval_x
            x_int = interval(-1 / gamma[i]) * (interval(bout[i]) + sum_w)
            x_bound[i, 0] = x_int[0].inf
            x_bound[i, 1] = x_int[0].sup
        return x_bound

    tanh_bound = interval([-1, 1])
    gamma, Win, bin, Wout, bout = model.get_weights()
    infty_bound = prod_loc(gamma.detach().numpy(), bout.detach().numpy(), Wout.detach().numpy(), tanh_bound)
    return infty_bound



def Newton(t, x_loc, F, DF=None, *parameters):
    def F_newton(x_whatever):
        return F(t, x_whatever, *parameters)

    def DF_newton(x_whatever):
        if DF is None:
            return DF_num(x_whatever)
        return DF(t, x_whatever, *parameters)

    def DF_num(x_w):
        dim = np.size(x_w)
        df = np.zeros([dim, dim])
        eps = 10 ** -5
        for i in range(dim):
            pert_i = np.zeros(dim)
            pert_i[i] = eps
            df[:, i] = (F_newton(x_w + pert_i) - F_newton(x_w)) / eps
        return df

    x_old, x_new = x_loc, x_loc
    for i in range(40):
        df = DF_newton(x_old)
        f = F_newton(x_old)
        try:
            x_new = x_old - np.linalg.solve(df, f)
        except:
            return x_old
        if np.linalg.norm(x_new - x_old) < 10 ** -5:
            return x_new
        else:
            x_old = x_new
    return x_new


class Fitting:
    def __init__(self, node: nODE):
        self.dimODE = node.ODE_dim

        def F_node(t, x):
            x_torch = torch.from_numpy(x).float()
            y = node.right_hand_side(t, x_torch).detach().numpy()
            return y

        def DF_node(t, x):
            x_torch = torch.from_numpy(x).float()
            y = node.derivative(t, x_torch).detach().numpy()
            return y

        self.F = lambda t, x: F_node(t, x)
        self.DF = lambda t, x: DF_node(t, x)

        if isinstance(node, nODE):
            eqBound = infty_bound(node)
        else:
            eqBound = np.array([[0, 100] for i in range(self.dimODE)])
        self.infty_bound = eqBound


def fixed_points(model, *parameters, gridDensity=3):
    if isinstance(model, nODE):
        model = Fitting(model)
    if not isinstance(model, Fitting):
        ValueError("Wrong class, only nODE or Fitting are accepted")
    eqBound = model.infty_bound
    coordinateIntervals = [np.linspace(*bound, num=gridDensity) for bound in eqBound]
    evalGrid = np.meshgrid(*coordinateIntervals)
    X = np.column_stack([G_i.flatten() for G_i in evalGrid])
    end_points = [Newton(0, x, model.F, model.DF, *parameters) for x in X]
    end_points = np.unique(np.round(end_points, 4), axis=0)
    return end_points


def num_fixed_points(model, *parameters, gridDensity=4):
    fp = fixed_points(model, *parameters, gridDensity=gridDensity)
    return fp.shape[0]


def fixed_point_ranking(true_model, found_model):
    true_num_fp = num_fixed_points(true_model)
    found_num_fp = num_fixed_points(found_model)
    grade = 1 - np.abs(true_num_fp - found_num_fp) / true_num_fp
    return np.max([0, grade])


def network_ranking(true_nODE: nODE, found_nODE: nODE):
    true_network = true_nODE.adjacency_matrix()
    found_network = found_nODE.adjacency_matrix()

    ranking_network = 1 - np.linalg.norm(true_network - found_network) / np.max([1., np.linalg.norm(true_network)])
    return np.max([0, ranking_network])


def numpify(a):
    return [x.detach().numpy() for x in a]


def parameter_ranking(true_nODE: nODE, found_nODE: nODE):
    def norm(vecORmat):
        return np.linalg.norm(vecORmat)
    def append_local(gamma, W1, b1, W2, b2):
        return np.append(gamma, np.append(W1.flatten(), np.append(b1, np.append(W2.flatten(), b2))))

    gamma_T, W1_T, b1_T, W2_T, b2_T = true_nODE.get_weights()
    gamma_T, W1_T, b1_T, W2_T, b2_T = numpify([gamma_T, W1_T, b1_T, W2_T, b2_T])
    gamma_F, W1_F, b1_F, W2_F, b2_F = found_nODE.get_weights()
    gamma_F, W1_F, b1_F, W2_F, b2_F = numpify([gamma_F, W1_F, b1_F, W2_F, b2_F])
    selected_pars_T = append_local(gamma_T, norm(W1_T), norm(b1_T), norm(W2_T), b2_T)
    selected_pars_F = append_local(gamma_F, norm(W1_F), norm(b1_F), norm(W2_F), b2_F)
    scaling = norm(selected_pars_T) / norm(selected_pars_F)
    ranking = max(1 - norm(+scaling * selected_pars_F - selected_pars_T) / norm(selected_pars_T),
                  1 - norm(-scaling * selected_pars_F - selected_pars_T) / norm(selected_pars_T))
    ranking = max(0, ranking)

    # ranking = max([1 - norm(np.array([(scaling * selected_pars_F[i] - selected_pars_T[i]) / max(1, selected_pars_T[i])
    #                                       for i in range(np.size(selected_pars_T))])) for scaling in [-1, 1]])
    return ranking, scaling


def rank(true_model, found_model):
    """ all parameters are supposed to be fixed in this case"""
    rank_on_parameters = parameter_ranking(true_model, found_model)[0]
    rank_on_network = network_ranking(true_model, found_model)
    rank_on_fixed_points = fixed_point_ranking(true_model, found_model)
    return [rank_on_network, rank_on_parameters, rank_on_fixed_points]


if __name__ == "__main__":
    Gamma = np.array([-1., -1.])
    Win = np.array([[0, -1], [-1, 0]]).astype(float)
    Wout = np.array([[2, 0], [0, 2]]).astype(float)
    bin = np.array([[2.], [2.]]).astype(float)
    bout = np.array([[2.], [2.]]).astype(float)
    TS_bistable = make_nODE_from_parameters(Gamma, Win=Win, bin=bin, Wout=Wout, bout=bout)
    true_model = Fitting(TS_bistable)

    ODE_dim = 2
    Gamma = np.array([-1., -1.])
    integration_time = 1
    Win = np.array([[0, -1], [1, 0]]).astype(float)
    Wout = np.array([[2, 0], [0, -2]]).astype(float)
    bin = np.array([[2.], [-2.]]).astype(float)
    bout = np.array([[2.], [2.]]).astype(float)
    node2 = make_nODE_from_parameters(Gamma, Win=Win, bin=bin, Wout=Wout, bout=bout)
    found_model = Fitting(node2)

    fp = num_fixed_points(found_model)
    print("found model number of fixed points : ", fp)

    rank_dynamics = fixed_point_ranking(true_model, found_model)
    print("Ranking of found model : ", rank_dynamics)

    A = node2.adjacency_matrix()
    B = TS_bistable.adjacency_matrix()

    ranking_network = 1 - np.linalg.norm(A - B) / np.linalg.norm(B)
    print('network raking : ', ranking_network)

    pr, scaling = parameter_ranking(TS_bistable, node2)
    print("parameter ranking : ", pr, ' with rescaling : ', scaling)

    full_rank = rank(TS_bistable, node2)
    print('full ranking : ', full_rank)

    ODE_dim = 2
    Gamma = np.array([-1., -1.])
    integration_time = 1
    Win = 6*np.array([[0, -1], [1, 0]]).astype(float)
    Wout = 6*np.array([[2, 0], [0, -2]]).astype(float)
    bin = 6*np.array([[2.], [-2.]]).astype(float)
    bout = 6*np.array([[2.], [2.]]).astype(float)
    node3 = make_nODE_from_parameters(Gamma, Win=Win, bin=bin, Wout=Wout, bout=bout)
    found_model2 = Fitting(node3)
    full_rank = rank(TS_bistable, node3)
    print('full ranking for double parameters : ', full_rank) # the scaling of the parameters mellows the effect of the rescaling
    pr, scaling = parameter_ranking(TS_bistable, node3)
    print("parameter ranking : ", pr, ' with rescaling : ', scaling)