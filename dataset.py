import os
import torch
import torch_cluster
from torch_geometric.data import Data, Dataset

# РОЗКОМЕНТУЙ, ЯКЩО ВСТАНОВИВ PHYSICS NEMO
from physicsnemo.nn.functional import radius_search as nvidia_radius_search

class IPCDataset(Dataset):
    def __init__(self, processed_dir, mode='train', noise_scale=0.03):
        super().__init__(None) 
        
        self.data_dir = processed_dir
        self.mode = mode
        
        self.stats = torch.load(os.path.join(self.data_dir, "global_stats.pt"))
        
        self.contact_radius = self.stats['avg_edge_len'].item() * 1.5
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
        
        # ==========================================
        # 1. ІН'ЄКЦІЯ ШУМУ (Лише для train і лише для динаміки)
        # ==========================================
        if self.mode == 'train':
            is_dynamic = (node_type == 0).unsqueeze(-1)
            
            # Швидкісний шум беремо напряму з нашого датасет-вектора
            noise_v = torch.randn_like(vel) * self.noise_std_vel * is_dynamic
            
            # Позиційний шум вираховуємо динамічно через поточний dt
            noise_p = noise_v * dt 
            
            pos_noisy = pos + noise_p
            vel_noisy = vel + noise_v
            
            # Коригуємо таргет
            target_accel = accel - (noise_v / dt)
        else:
            pos_noisy = pos
            vel_noisy = vel
            target_accel = accel # Наш Ground Truth таргет

        # ==========================================
        # 2. РОЗРАХУНОК LOOKAHEAD ТА RADIUS SEARCH
        # ==========================================
        pos_lookahead = pos_noisy + vel_noisy * dt
        
        # --- БЕКЕНД ПОШУКУ (NVIDIA PhysicsNeMo або PyG) ---
        # Якщо ти імпортував nvidia_radius_search, використовуй його:
        # Використовуємо NVIDIA PhysicsNeMo:
        edge_index_dynamic = nvidia_radius_search(
            pos_lookahead, 
            pos_lookahead, 
            radius=self.contact_radius
        )
        
        # Якщо вони раптом повертають кортеж
        if isinstance(edge_index_dynamic, tuple):
            edge_index_dynamic = torch.stack(edge_index_dynamic, dim=0)
            
        # Якщо вони повертають тензор у форматі [E, 2] замість нашого [2, E]
        if isinstance(edge_index_dynamic, torch.Tensor) and edge_index_dynamic.shape[0] != 2:
            edge_index_dynamic = edge_index_dynamic.t().contiguous()
        
        # Fallback (Torch Cluster - те, що NeMo використовує під капотом):
        #edge_index_dynamic = torch_cluster.radius_graph(
         #   pos_lookahead, 
          #  r=self.contact_radius, 
           # max_num_neighbors=20, 
            #loop=False
        #)
        
        # ==========================================
        # 3. ФІЛЬТРАЦІЯ (Topological Mask)
        # ==========================================
        MAX_NODES = pos_noisy.shape[0]
        static_hashed = edge_index_static[0] * MAX_NODES + edge_index_static[1]
        dynamic_hashed = edge_index_dynamic[0] * MAX_NODES + edge_index_dynamic[1]
        
        mask = ~torch.isin(dynamic_hashed, static_hashed)
        edge_index_dynamic = edge_index_dynamic[:, mask]
        
        full_edge_index = torch.cat([edge_index_static, edge_index_dynamic], dim=1)
        
        # ==========================================
        # 4. ФІЧІ РЕБЕР
        # ==========================================
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
        
        # ==========================================
        # 5. НОРМАЛІЗАЦІЯ І ЗБІРКА DATA
        # ==========================================
        vel_norm = (vel_noisy - self.stats['vel_mean']) / self.stats['vel_std']
        accel_norm = (target_accel - self.stats['accel_mean']) / self.stats['accel_std']
        
        # Нормалізація відносних векторів d_ij_current (за бажанням можна додати в stats)
        # Для стабільності їх часто просто ділять на радіус контакту
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