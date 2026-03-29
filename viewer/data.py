import os
import torch
import numpy as np
import scipy.sparse as sp
from physicsnemo.nn.functional import radius_search as nvidia_radius_search
from viewer.core import BaseSimulationData

class SimulationData(BaseSimulationData):
    def __init__(self, folder_path, stats_path, device):
        self.device = device
        self.stats = torch.load(stats_path, map_location='cpu')
        
        static = torch.load(os.path.join(folder_path, "static_graph.pt"), map_location='cpu')
        traj = torch.load(os.path.join(folder_path, "trajectory.pt"), map_location='cpu')
        
        self.dt = static['dt'].item()
        self.node_type = static['node_type'].squeeze().to(device)
        self.materials = static['materials'].to(device)
        self.dyn_mask = (self.node_type == 0)
        
        self.pos = traj['pos'][0].clone().to(device)
        self.vel = traj['vel'][0].clone().to(device)
        
        self.edge_index_static = static['edge_index'].to(device)
        self.num_nodes = self.pos.size(0)
        self.num_static = self.edge_index_static.shape[1]
        
        faces = static.get('faces', None)
        if faces is not None:
            self.faces = faces.to(device)
            if self.faces.shape[0] == 3:
                self.faces = self.faces.T
            self.faces_np = self.faces.cpu().numpy()
            print(f"✅ Завантажено {self.faces_np.shape[0]} трикутників сітки.")
        else:
            self.faces = None
            self.faces_np = None
            print("No Faces")
        
        row_stat = self.edge_index_static[0].cpu().numpy()
        col_stat = self.edge_index_static[1].cpu().numpy()
        adj = sp.coo_matrix((np.ones_like(row_stat), (row_stat, col_stat)), shape=(self.num_nodes, self.num_nodes))
        _, comp_labels = sp.csgraph.connected_components(adj, directed=False)
        self.comp_labels = torch.from_numpy(comp_labels).to(device)
        
        self.edge_index_dynamic = torch.empty((2, 0), dtype=torch.long, device=device)

    def find_dynamic_edges(self, radius_mult: float):
        contact_radius = self.stats['avg_edge_len'].item() * radius_mult
        pos_lookahead = self.pos + self.vel * self.dt
        MAX_DYNAMIC_POINTS = 32
        
        neighbors = nvidia_radius_search(pos_lookahead, pos_lookahead, radius=contact_radius, max_points=MAX_DYNAMIC_POINTS)
        row_idx = torch.arange(self.num_nodes, device=self.device).view(-1, 1).repeat(1, MAX_DYNAMIC_POINTS).view(-1)
        col_idx = neighbors.view(-1)
        
        mask = col_idx >= 0
        edge_idx = torch.stack([row_idx[mask], col_idx[mask]], dim=0).long()
        
        if edge_idx.numel() == 0:
            self.edge_index_dynamic = torch.empty((2, 0), dtype=torch.long, device=self.device)
            return

        src, dst = edge_idx[0], edge_idx[1]
        
        cross_mask = self.comp_labels[src] != self.comp_labels[dst]
        d_ij = pos_lookahead[src] - pos_lookahead[dst]
        dist_mask = torch.norm(d_ij, dim=1) <= contact_radius
        loop_mask = src != dst
        
        final_mask = cross_mask & dist_mask & loop_mask
        self.edge_index_dynamic = edge_idx[:, final_mask]
        
    def step_physics(self, accel: torch.Tensor):
        self.vel[self.dyn_mask] = self.vel[self.dyn_mask] + accel[self.dyn_mask] * self.dt
        self.pos[self.dyn_mask] = self.pos[self.dyn_mask] + self.vel[self.dyn_mask] * self.dt