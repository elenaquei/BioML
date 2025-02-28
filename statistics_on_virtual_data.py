import numpy as np
import matplotlib.pyplot as plt

from data_creation import create_dataset, extract_adjacency, create_random_network
from graph_plotting import plot_graph
from models.nODE import nODE, make_nODE_from_parameters, become_torch
import torch
from models.ranking import num_fixed_points

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

def distribution_n_fp(dim, n_tests = 100):
    dict_fp = {0 : 0, 1 : 0, 2 : 0, 3 : 0}
    for i in range(n_tests):
        par_struct, adj = create_random_network(dim)
        model = par_struct.make_nODE_from_parstruct()
        fp = num_fixed_points(model)
        if fp in dict_fp:
            dict_fp[fp] += 1
        else:
            dict_fp[fp] = 1
    return dict_fp


print(distribution_n_fp(3))
