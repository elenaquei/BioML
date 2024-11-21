import torch
from torch_geometric.nn.conv import GATConv
from torch.nn import Linear, Sequential, ReLU
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
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
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
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
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