from sklearn.manifold import TSNE
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import csv
import pandas as pd


def make_adj_from_df(datadir, df, name):

    gene_names = df.index.tolist()

    adj = torch.zeros(len(gene_names),len(gene_names))

    with open(datadir + name + '/refNetwork.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for ln in reader:
            i1 = gene_names.index(ln[0])
            i2 = gene_names.index(ln[1])
            if ln[2] == '+':
                #adj[i1,i2] = 1
                adj[i2,i1] = 1
            if ln[2] == '-':
                #adj[i1,i2] = 1
                adj[i2,i1] = 1

    edge_index, _ = dense_to_sparse(adj)
    return edge_index, adj


def to_pyg_data(x_in, ode_dim, n_data, edge_index=None):
    # distribute x_train values as node features
    x = torch.zeros([ode_dim, n_data])

    k = 0
    for j in range(0,n_data):
        for i in range(0,ode_dim):
            x[i,j] = x_in[i,j]
            k += 1

    # build edge indices (assuming fully connected network)
    if edge_index is None:
        adj_matrix = torch.ones((ode_dim, ode_dim))
        edge_index, _ = dense_to_sparse(adj_matrix)

    data = Data(x=x,edge_index=edge_index)

    return data