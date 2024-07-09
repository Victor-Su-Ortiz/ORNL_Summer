import os
import torch
import csv
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch_geometric.nn import GraphConv, global_mean_pool

device = torch.device('mps')

class GCNNet_mse(torch.nn.Module):
    def __init__(self, num_node_features, hp):
        super(GCNNet_mse, self).__init__()

        self.use_batch_norm = hp['use_batch_norm']
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if self.use_batch_norm else None

        hidden_dims = [hp[f'hidden_dim{i+1}'] for i in range(6)]
        num_layers = [hp[f'num_layers{i+1}'] for i in range(6)]

        # Define the Graph Convolution layers
        for i in range(len(hidden_dims)):
            for j in range(num_layers[i]):
                if i == 0 and j == 0:
                    in_features = num_node_features
                elif j == 0:
                    in_features = hidden_dims[i-1]
                else:
                    in_features = hidden_dims[i]
                out_features = hidden_dims[i]
                self.convs.append(GraphConv(in_features, out_features))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(out_features))

        # Define the dense layer
        self.dense = torch.nn.Linear(hidden_dims[-1], hidden_dims[-1])

        # Define the output layer
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph Convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

        # Aggregate node features into a single graph-level representation
        x = global_mean_pool(x, data.batch)

        # Dense layer
        x = self.dense(x)
        x = torch.nn.functional.relu(x)

        # Output layer
        mu = self.fc_mu(x).squeeze(-1)

        return mu