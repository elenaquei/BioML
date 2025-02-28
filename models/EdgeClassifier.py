import torch
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import aggr
from torch.nn import Linear, Sequential, ReLU, LeakyReLU, Softmax
from data_creation import torch_parameter_structure
from torchdiffeq import odeint
import torch.nn.functional as F
import math

class EdgeClassNet(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,edge_attr_dim):
        super(EdgeClassNet, self).__init__()

        self.conv_out = int(in_channels)
        # self.conv_out2 = int(in_channels)
        #self.conv_out3 = 4
        
        self.gat_conv = GATConv(in_channels,in_channels)
        # self.gat_conv2 = GATConv(self.conv_out,self.conv_out2)
        # self.gat_conv3 = GATConv(self.conv_out2,self.conv_out3)

        self.x_mlp = Linear(2*in_channels, in_channels)
        self.act = LeakyReLU()


        self.edge_mlp = Sequential(
            Linear(2* self.conv_out + edge_attr_dim, hidden_channels),
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
            Linear(hidden_channels, 3),
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # x = self.gat_conv(x,edge_index = edge_index)
        # x = self.gat_conv2(x,edge_index = edge_index)
        # x = self.gat_conv3(x,edge_index = edge_index)
        x = self.x_mlp(x)
        x = self.act(x)

        # x = self.gat_conv(x, edge_index=edge_index)

        edge_attr = self.update_attributes(x,edge_index,edge_attr, self.edge_mlp)

        return edge_attr
    
    def update_attributes(self, x, edge_index, edge_attr, mlp):

        row, col = edge_index
        src, tgt = x[row], x[col]

        # Concatenate node features with edge attributes
        edge_features = torch.cat([src, tgt, edge_attr], dim=-1)

        # Update edge attributes using an MLP
        edge_attr = mlp(edge_features)

        edge_attr = F.log_softmax(edge_attr, dim=1)

        return edge_attr
    

class EdgeClassNetBin(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,edge_attr_dim):
        super(EdgeClassNetBin, self).__init__()

        self.conv_out = int(in_channels)
        # self.conv_out2 = int(in_channels)
        #self.conv_out3 = 4
        
        self.gat_conv = GATConv(in_channels,in_channels)
        # self.gat_conv2 = GATConv(self.conv_out,self.conv_out2)
        # self.gat_conv3 = GATConv(self.conv_out2,self.conv_out3)

        self.x_mlp = Linear(2*in_channels, in_channels)
        self.act = LeakyReLU()


        self.edge_mlp = Sequential(
            Linear(2* self.conv_out, hidden_channels),
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
            Linear(hidden_channels, 2),
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # x = self.gat_conv(x,edge_index = edge_index)
        # x = self.gat_conv2(x,edge_index = edge_index)
        # x = self.gat_conv3(x,edge_index = edge_index)
        x = self.x_mlp(x)
        x = self.act(x)

        x = self.gat_conv(x, edge_index=edge_index)

        print(edge_attr)
        edge_attr = self.update_attributes(x,edge_index,edge_attr, self.edge_mlp)

        print(edge_attr)

        return edge_attr
    
    def update_attributes(self, x, edge_index, edge_attr, mlp):

        row, col = edge_index
        src, tgt = x[row], x[col]

        # Concatenate node features with edge attributes
        edge_features = torch.cat([src, tgt], dim=-1)

        # Update edge attributes using an MLP
        edge_attr = mlp(edge_features)

        edge_attr = F.log_softmax(edge_attr, dim=1)

        return edge_attr