import os
import torch
import torch_cluster
from torch_geometric.data import Data, Dataset
import scipy.sparse as sp
import numpy as np
from physicsnemo.nn.functional import radius_search as nvidia_radius_search

class IPCDataset(Dataset):
    def __init__(self, processed_dir, mode='train', noise_scale=0.03):
        super().__init__(None) 
        
        self.data_dir = processed_dir
        self.mode = mode
        
        self.stats = torch.load(os.path.join(self.data_dir, "global_stats.pt"))
        
        self.contact_radius = self.stats['avg_edge_len'].item() * 4
        print(f"[{mode}] Dataset-driven Contact Radius: {self.contact_radius:.6f}")
        
        self.noise_std_vel = self.stats['vel_std'] * noise_scale
        print(f"[{mode}] Dataset-driven Velocity Noise: {self.noise_std_vel.tolist()}")
        
        sim_folders = sorted([f.path for f in os.scandir(self.data_dir) if f.is_dir()])
        
        split_idx = int(len(sim_folders) * 0.8)
        self.sim_folders = sim_folders[:split_idx] if mode == 'train' else sim_folders[split_idx:]
        
        self.index_map = []
        for sim_idx, folder in enumerate(self.sim_folders):
            traj = torch.load(os.path.join(folder, "trajectory.pt"), map_location='cpu')
            num_frames = traj['pos'].shape[0]
            for t in range(num_frames - 1):
                self.index_map.append((folder, t))

    def len(self):
        return len(self.index_map)

    def get(self, idx):
        sim_folder, t = self.index_map[idx]
        
        static = torch.load(os.path.join(sim_folder, "static_graph.pt"), map_location='cpu')
        traj = torch.load(os.path.join(sim_folder, "trajectory.pt"), map_location='cpu')
        
        dt = static['dt'].item()
        pos = traj['pos'][t]
        vel = traj['vel'][t]
        accel = traj['accel'][t]
        
        edge_index_static = static['edge_index']
        node_type = static['node_type'].squeeze()
        materials = static['materials']
        
        if self.mode == 'train':
            is_dynamic = (node_type == 0).unsqueeze(-1)
            noise_v = torch.randn_like(vel) * self.noise_std_vel * is_dynamic
            noise_p = noise_v * dt 
            
            pos_noisy = pos + noise_p
            vel_noisy = vel + noise_v
            
            target_accel = accel - (noise_v / dt)
        else:
            pos_noisy = pos
            vel_noisy = vel
            target_accel = accel

        pos_lookahead = pos_noisy + vel_noisy * dt
        
        MAX_DYNAMIC_POINTS = 32
        neighbors = nvidia_radius_search(
            pos_lookahead, 
            pos_lookahead, 
            radius=self.contact_radius,
            max_points=MAX_DYNAMIC_POINTS
        )

        num_nodes = pos_lookahead.size(0)
        row = torch.arange(num_nodes, device=pos_lookahead.device).view(-1, 1).repeat(1, MAX_DYNAMIC_POINTS).view(-1)
        col = neighbors.view(-1)

        mask = col >= 0
        edge_index_dynamic = torch.stack([row[mask], col[mask]], dim=0).long()
        
        src_dyn = edge_index_dynamic[0]
        dst_dyn = edge_index_dynamic[1]
        
        row_stat = edge_index_static[0].cpu().numpy()
        col_stat = edge_index_static[1].cpu().numpy()
        adj = sp.coo_matrix((np.ones_like(row_stat), (row_stat, col_stat)), shape=(num_nodes, num_nodes))
        _, comp_labels = sp.csgraph.connected_components(adj, directed=False)
        comp_labels = torch.from_numpy(comp_labels).to(edge_index_dynamic.device)

        cross_mask = comp_labels[src_dyn] != comp_labels[dst_dyn]
        
        d_ij_dyn = pos_lookahead[src_dyn] - pos_lookahead[dst_dyn]
        dist_dyn = torch.norm(d_ij_dyn, dim=1)
        dist_mask = dist_dyn <= self.contact_radius
        
        loop_mask = src_dyn != dst_dyn
        
        final_mask = cross_mask & dist_mask & loop_mask
        edge_index_dynamic = edge_index_dynamic[:, final_mask]

        full_edge_index = torch.cat([edge_index_static, edge_index_dynamic], dim=1)

        src, dst = full_edge_index
        d_ij_current = pos_noisy[src] - pos_noisy[dst]
        d_ij_lookahead = pos_lookahead[src] - pos_lookahead[dst]
        
        d_ij_curr_norm = torch.norm(d_ij_current, dim=1, keepdim=True)
        d_ij_look_norm = torch.norm(d_ij_lookahead, dim=1, keepdim=True)
        
        num_static = edge_index_static.shape[1]
        num_dynamic = edge_index_dynamic.shape[1]
        edge_type = torch.cat([
            torch.tensor([[1.0, 0.0]]).repeat(num_static, 1),
            torch.tensor([[0.0, 1.0]]).repeat(num_dynamic, 1)
        ], dim=0)

        edge_attr = torch.cat([d_ij_current, d_ij_curr_norm, d_ij_look_norm, edge_type], dim=1)
        
        vel_norm = (vel_noisy - self.stats['vel_mean']) / self.stats['vel_std']
        accel_norm = (target_accel - self.stats['accel_mean']) / self.stats['accel_std']
        
        edge_attr[:, :3] = edge_attr[:, :3] / self.contact_radius
        edge_attr[:, 3:5] = edge_attr[:, 3:5] / self.contact_radius
        
        x = torch.cat([vel_norm, materials, node_type.unsqueeze(1)], dim=1)
        
        data = Data(
            x=x,
            edge_index=full_edge_index,
            edge_attr=edge_attr,
            y=accel_norm,           
            pos=pos_noisy,          
            node_type=node_type,    
            dt=torch.tensor([dt], dtype=torch.float)
        )
        
        return data