import torch
from torch_geometric.nn.conv import GATConv
from torch.nn import Linear, Sequential, ReLU, LeakyReLU
from data_creation import torch_parameter_structure
from torchdiffeq import odeint
import math

class EdgeNet(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,edge_attr_dim):
        super(EdgeNet, self).__init__()

        self.conv_out = int(in_channels)
        self.conv_out2 = int(in_channels/2)
        self.conv_out3 = 4
        
        self.gat_conv = GATConv(2*in_channels,self.conv_out)
        self.gat_conv2 = GATConv(self.conv_out,self.conv_out2)
        self.gat_conv3 = GATConv(self.conv_out2,self.conv_out3)

        self.edge_mlp = Sequential(
            Linear(2 * self.conv_out3 + edge_attr_dim, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, edge_attr_dim)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.gat_conv(x,edge_index = edge_index)
        x = self.gat_conv2(x,edge_index = edge_index)
        x = self.gat_conv3(x,edge_index = edge_index)

        edge_attr = edge_attr + self.update_attributes(x,edge_index,edge_attr, self.edge_mlp)

        return edge_attr
    
    def update_attributes(self, x, edge_index, edge_attr, mlp):

        row, col = edge_index
        src, tgt = x[row], x[col]

        # Concatenate node features with edge attributes
        edge_features = torch.cat([src, tgt, edge_attr], dim=-1)

        # Update edge attributes using an MLP
        edge_attr = mlp(edge_features)
        
        return edge_attr
    
class EdgeODENet(torch.nn.Module):
    def __init__(self,n_data,ode_dim,edge_attr_dim,gat_out=4,edge_hidden=10, bias_hidden = 10):
        super(EdgeODENet, self).__init__()
        
        self.n_data = n_data
        self.ode_dim = ode_dim
        self.edge_attr_dim = edge_attr_dim
        self.gat_out = gat_out
        self.edge_hidden = edge_hidden
        self.bias_hidden = bias_hidden
        self.integration_interval = torch.tensor([0., 1.])

        # a graph attention network transforms the node features using an attention mechanism in order to take into account global features
        self.gat = GATConv(2*n_data, self.gat_out)
        self.gat2 = GATConv(self.gat_out,self.gat_out)
        self.gat3 = GATConv(self.gat_out,self.gat_out)
        
        # an MLP is used to transform the initial guess for the edge features to better edge features
        self.edge_mlp = Sequential(
            Linear(2 * self.gat_out + 1, self.edge_hidden),
            LeakyReLU(),
            Linear(self.edge_hidden, self.edge_hidden),
            LeakyReLU(),
            Linear(self.edge_hidden, self.edge_hidden),
            LeakyReLU(),
            Linear(self.edge_hidden, edge_attr_dim)
        )

        # another MLP is used to transform the node features to node-wise parameters
        self.bias_mlp = Sequential(
            Linear(self.gat_out, self.bias_hidden),
            LeakyReLU(),
            Linear(self.bias_hidden, self.bias_hidden),
            LeakyReLU(),
            Linear(self.bias_hidden, self.bias_hidden),
            LeakyReLU(),
            Linear(self.bias_hidden, 3*ode_dim)
        )

    def forward(self,data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x_init = x[0]

        x = self.gat(x,edge_index = edge_index)

        x = self.gat2(x,edge_index = edge_index)

        x = self.gat3(x,edge_index = edge_index)

        bias = self.bias_mlp(x)

        adjacencies = self.update_attributes(x,edge_index,edge_attr)

        Win,Wout,bin,bout,gamma = self.get_params(bias,adjacencies)

        dt = 0.01
        x_hat = odeint(lambda t, x : self.grn_ode(t, x, Win, Wout, bin, bout, gamma), x_init, self.integration_interval, method='euler', options={'step_size': dt})[-1]
        
        return torch.cat(adjacencies,x_hat)
    
    def get_params(self, bias, adjacencies):

        num_adj = adjacencies.size(0) // 2*(self.ode_dim * self.ode_dim)

        w_1, w_2 = adjacencies[:adjacencies.size(0) // 2], adjacencies[adjacencies.size(0) // 2:]

        Win = w_1.view(num_adj, self.ode_dim, self.ode_dim)
        Wout = w_2.view(num_adj, self.ode_dim, self.ode_dim)

        Win = torch.block_diag(*Win)
        Wout = torch.block_diag(*Wout)

        bin = bias[:self.ode_dim]
        bout = bias[self.ode_dim:2*self.ode_dim]
        gamma = bias[2*self.ode_dim:]

        return Win,Wout,bin,bout,gamma
        

    def update_attributes(self, x, edge_index, edge_attr):
        row, col = edge_index
        src, tgt = x[row], x[col]

        print(src.shape)
        print(tgt.shape)
        print(edge_attr.shape)

        edge_features = torch.cat([src, tgt, edge_attr], dim=-1)

        print(edge_features.shape)

        edge_attr = self.edge_mlp(edge_features)

        print(edge_attr.shape)
        
        return edge_attr

    def grn_ode(self, t, x, Win,Wout,bin,bout,gamma):
        out = x.matmul(Win.t()) + bin.t()
        out = out.matmul(Wout.t()) + bout.t()
        out = x.matmul(gamma) + out
        return out
    
# idea: train both W1, W2 and multiply out at the end
class W1W2Net(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,edge_attr_dim):
        super(W1W2Net, self).__init__()

        self.conv_out = int(in_channels)
        
        self.gat_conv = GATConv(2*in_channels,self.conv_out)

        self.ode_dim = int(math.sqrt(edge_attr_dim))

        self.edge_mlp = Sequential(
            Linear(2 * self.conv_out + edge_attr_dim, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, 2*edge_attr_dim)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.gat_conv(x,edge_index = edge_index)

        print(edge_attr)

        edge_attr = self.update_attributes(x,edge_index,edge_attr, self.edge_mlp)

        print(edge_attr)
        W1 = edge_attr[:self.ode_dim*self.ode_dim].reshape(self.ode_dim, self.ode_dim)  # First nxn matrix
        W2 = edge_attr[self.ode_dim*self.ode_dim:].reshape(self.ode_dim, self.ode_dim)  # Second nxn matrix

        edge_attr = torch.mm(W1, W2).flatten()

        return edge_attr
    
    def update_attributes(self, x, edge_index, edge_attr, mlp):

        row, col = edge_index
        src, tgt = x[row], x[col]

        # Concatenate node features with edge attributes
        edge_features = torch.cat([src, tgt, edge_attr], dim=-1)

        # Update edge attributes using an MLP
        edge_attr = mlp(edge_features)
        
        return edge_attr

# idea: separate out (x(0),x(T)) pairs for finding additional information to put into edge_mlp      
class NodeNet(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,edge_attr_dim):
        super(NodeNet, self).__init__()

        self.ode_dim = int(math.sqrt(out_channels))

        self.n_data = in_channels
        

        self.edge_mlp = Sequential(
            Linear(self.ode_dim + edge_attr_dim, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, edge_attr_dim)
        )

        self.node_mlp = Sequential(
            Linear(2 * self.ode_dim, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, self.ode_dim)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for graph_id in batch.unique(sorted=True):
            graph_mask = (batch == graph_id)
            graph_features = x[graph_mask,:]

            feature_enc = torch.zeros(self.ode_dim)

            for i in range(0,self.n_data):
                datapoint = graph_features[:,[i,i+self.n_data]].flatten()

                feature_enc += self.node_mlp(datapoint)

            edge_mask = graph_mask[edge_index[0]] & graph_mask[edge_index[1]]

            # Concatenate feature_enc with edge attributes for edges in the graph
            attr = torch.cat([feature_enc.expand(edge_attr[edge_mask].size(0), -1), edge_attr[edge_mask]], dim=-1)

            # Update edge attributes for the edges in this graph
            edge_attr[edge_mask] = self.edge_mlp(attr)
        
        return edge_attr

# idea: update edge weights by propagating through a neural ODE, rather than updating once   
class EdgeNODE(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,edge_attr_dim):
        super(EdgeNODE, self).__init__()

        self.conv_out = 4

        self.gat_conv = GATConv(2*in_channels,self.conv_out)

        self.edge_mlp = Sequential(
            Linear(2 * self.conv_out + edge_attr_dim, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, edge_attr_dim)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.gat_conv(x,edge_index = edge_index)

        edge_attr = odeint(lambda t,edge_attr: self.rhs(t,edge_attr,edge_index,x,self.edge_mlp),edge_attr,torch.tensor([0.0, 1.0]))[-1]

        return edge_attr
    
    def rhs(self, t, edge_attr, edge_index, x, mlp):

        row, col = edge_index
        src, tgt = x[row], x[col]

        # Concatenate node features with edge attributes
        edge_features = torch.cat([src, tgt, edge_attr], dim=-1)

        # Update edge attributes using an MLP
        edge_attr = mlp(edge_features)
        
        return edge_attr    