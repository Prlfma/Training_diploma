import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MLP(nn.Module):
    """Стандартний MLP з LayerNorm для MeshGraphNet."""
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        super().__init__()
        modules = []
        current_dim = in_dim
        for _ in range(layers - 1):
            modules.append(nn.Linear(current_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.01))
            current_dim = hidden_dim
            
        modules.append(nn.Linear(current_dim, out_dim))
        modules.append(nn.LayerNorm(out_dim)) 
        
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


def cauchy_activation(x, l1, l2, d):
    d_sq = d * d + 1e-5
    x2_d2 = (x * x) + d_sq
    numerator = torch.addcmul(l2, l1, x)
    return numerator / x2_d2

class CauchyLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.l1 = nn.Parameter(torch.ones(out_dim))
        self.l2 = nn.Parameter(torch.zeros(out_dim))
        self.d = nn.Parameter(torch.ones(out_dim))
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        a = cauchy_activation(x, self.l1, self.l2, self.d)
        x = self.linear2(a)
        return x

class CauchyMLP(nn.Module):
    """Повний аналог стандартного MLP на базі Cauchy XNet."""
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        super().__init__()
        modules = []
        current_dim = in_dim
        for _ in range(layers - 1):
            modules.append(CauchyLayer(current_dim, hidden_dim))
            current_dim = hidden_dim
            
        modules.append(CauchyLayer(current_dim, out_dim))
        modules.append(nn.LayerNorm(out_dim)) 
        
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MeshGraphNetProcessorLayer(MessagePassing):
    def __init__(self, hidden_dim, mlp_class=MLP):
        super().__init__(aggr='add')
        
        self.edge_mlp = mlp_class(hidden_dim * 3, hidden_dim, hidden_dim)
        self.node_mlp = mlp_class(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        out_x, out_edge_attr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out_x, out_edge_attr

    def message(self, x_i, x_j, edge_attr):
        edge_inputs = torch.cat([edge_attr, x_i, x_j], dim=-1)
        updated_edge_attr = self.edge_mlp(edge_inputs)
        self._updated_edge_attr = updated_edge_attr
        return updated_edge_attr

    def update(self, aggr_out, x):
        node_inputs = torch.cat([x, aggr_out], dim=-1)
        dx = self.node_mlp(node_inputs)
        return x + dx, self._updated_edge_attr

class CustomMeshGraphNet(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, output_dim=3, hidden_dim=128, num_processor_layers=15, mlp_class=MLP):
        super().__init__()
        
        self.node_encoder = mlp_class(node_in_dim, hidden_dim, hidden_dim)
        self.edge_encoder = mlp_class(edge_in_dim, hidden_dim, hidden_dim)
        
        self.processor = nn.ModuleList([
            MeshGraphNetProcessorLayer(hidden_dim, mlp_class) 
            for _ in range(num_processor_layers)
        ])
        
        self.decoder = nn.Sequential(
            mlp_class(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            
        out_accel = self.decoder(x)
        return out_accel