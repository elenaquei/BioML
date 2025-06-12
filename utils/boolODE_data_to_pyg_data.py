from sklearn.manifold import TSNE
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import csv
import pandas as pd

# method to construct and adjacency matrix + edge_index based on given expression level data and a reference network (loaded here)
def make_adj_from_df(datadir, df, name):

    # read out row names of the gene expression matrix to get gene names
    gene_names = df.index.tolist()

    adj = torch.zeros(len(gene_names),len(gene_names))

    with open(datadir + name + '/refNetwork.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for ln in reader:
            # check that both gene names are in the gene list 
            if ln[0] in gene_names and ln[1] in gene_names:

                # fill in one entry of the adjacency matrix
                i1 = gene_names.index(ln[0])
                i2 = gene_names.index(ln[1])

                # if sign of connection is available, we use it, otherwise we put everything as activating
                if len(ln) > 2:
                    if ln[2] == '+':
                        adj[i2,i1] = 1
                    if ln[2] == '-':
                        adj[i2,i1] = 1
                else:
                    adj[i2,i1] = 1

    # check if adjacency matrix should be transposed (in the case that header structure is Gene1 Gene2, rather than Gene2 Gene1)
    if header[0] == 'Gene1':
        adj = adj.t()
    
    # construct edge index for the given adjacency matrix (used in pytorch geometric)
    edge_index, _ = dense_to_sparse(adj)
    return edge_index, adj

# method to convert a gene expression matrix + edge_index to a Pytorch Geometric data object to be used in a graph neural network
def to_pyg_data(x_in, ode_dim, n_data, edge_index=None):
    # distribute x_train values as node features
    x = torch.zeros([ode_dim, n_data])

    k = 0
    for j in range(0,n_data):
        for i in range(0,ode_dim):
            x[i,j] = x_in[i,j]
            k += 1

    # build edge indices (assuming fully connected network in case no edge_index is given as input)
    if edge_index is None:
        adj_matrix = torch.ones((ode_dim, ode_dim))
        edge_index, _ = dense_to_sparse(adj_matrix)

    data = Data(x=x,edge_index=edge_index)

    return data