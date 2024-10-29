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


def fixed_points(model, *parameters, gridDensity=3):
    eqBound = model.infty_bound
    coordinateIntervals = [np.linspace(*bound, num=gridDensity) for bound in eqBound]
    evalGrid = np.meshgrid(*coordinateIntervals)
    X = np.column_stack([G_i.flatten() for G_i in evalGrid])
    end_points = [Newton(0, x, model.F, model.DF, *parameters) for x in X]
    end_points = np.unique(np.round(end_points, 5), axis=0)
    return end_points


def num_fixed_points(model, *parameters, gridDensity=3):
    fp = fixed_points(model, *parameters, gridDensity=3)
    return fp.shape[0]


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
            y = node.right_hand_side(t, x_torch).detach().numpy()[0]
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


def fixed_point_ranking(true_model, found_model):
    true_num_fp = num_fixed_points(true_model)
    found_num_fp = num_fixed_points(found_model)
    grade = 1 - np.abs(true_num_fp - found_num_fp) / true_num_fp
    return grade


def network_ranking(true_nODE: nODE, found_nODE: nODE):
    true_network = true_nODE.adjacency_matrix()
    found_network = found_nODE.adjacency_matrix()

    ranking_network = 1 - np.linalg.norm(true_network - found_network) / np.linalg.norm(true_network)
    return ranking_network


def numpify(a):
    return [x.detach().numpy() for x in a]


def parameter_ranking(true_nODE: nODE, found_nODE: nODE):
    def norm(vecORmat):
        return np.linalg.norm(vecORmat)

    gamma_T, W1_T, b1_T, W2_T, b2_T = true_nODE.get_weights()
    gamma_T, W1_T, b1_T, W2_T, b2_T = numpify([gamma_T, W1_T, b1_T, W2_T, b2_T])
    gamma_F, W1_F, b1_F, W2_F, b2_F = found_nODE.get_weights()
    gamma_F, W1_F, b1_F, W2_F, b2_F = numpify([gamma_F, W1_F, b1_F, W2_F, b2_F])
    selected_pars_T = np.append(np.append(gamma_T, np.array([norm(W1_T), norm(b1_T), norm(W2_T)])), b2_T)
    selected_pars_F = np.append(np.append(gamma_F, np.array([norm(W1_F), norm(b1_F), norm(W2_F)])), b2_F)
    scaling = norm(selected_pars_T) / norm(selected_pars_F)
    ranking = max(1 - norm(+scaling * selected_pars_F - selected_pars_T) / norm(selected_pars_T),
                  1 - norm(-scaling * selected_pars_F - selected_pars_T) / norm(selected_pars_T))
    return ranking, scaling


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
