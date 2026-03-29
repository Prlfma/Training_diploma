import os
import torch
from train.model import CustomMeshGraphNet, MLP
from viewer.core import BasePredictor, BaseSimulationData

class MLPredictor(BasePredictor):
    def __init__(self, config, device):
        self.device = device
        self.model = CustomMeshGraphNet(
            node_in_dim=7, edge_in_dim=7, output_dim=3, 
            hidden_dim=config["hidden_dim"], 
            num_processor_layers=config["model_layers"], 
            mlp_class=MLP
        ).to(device)
        weights_path = config.get("weights_path", "")
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            raise FileNotFoundError(f"❌ КРИТИЧНА ПОМИЛКА: Файл ваг не знайдено за шляхом: '{weights_path}'! Перевір назву експерименту або директорію запуску.")
        self.model.eval()

    def predict(self, sim_data: BaseSimulationData, radius_mult: float) -> torch.Tensor:
        contact_radius = sim_data.stats['avg_edge_len'].item() * radius_mult
        
        with torch.no_grad():
            full_edge_index = torch.cat([sim_data.edge_index_static, sim_data.edge_index_dynamic], dim=1)
            src, dst = full_edge_index
            
            pos_lookahead = sim_data.pos + sim_data.vel * sim_data.dt
            d_ij_curr = sim_data.pos[src] - sim_data.pos[dst]
            d_ij_look = pos_lookahead[src] - pos_lookahead[dst]
            
            d_ij_curr_norm = torch.norm(d_ij_curr, dim=1, keepdim=True)
            d_ij_look_norm = torch.norm(d_ij_look, dim=1, keepdim=True)
            
            num_dyn = sim_data.edge_index_dynamic.shape[1]
            edge_type = torch.cat([
                torch.tensor([[1.0, 0.0]], device=self.device).repeat(sim_data.num_static, 1),
                torch.tensor([[0.0, 1.0]], device=self.device).repeat(num_dyn, 1)
            ], dim=0)
            
            full_edge_attr = torch.cat([d_ij_curr, d_ij_curr_norm, d_ij_look_norm, edge_type], dim=1)
            full_edge_attr[:, :3] /= contact_radius
            full_edge_attr[:, 3:5] /= contact_radius
            
            vel_norm = (sim_data.vel - sim_data.stats['vel_mean'].to(self.device)) / sim_data.stats['vel_std'].to(self.device)
            node_type_tensor = sim_data.node_type.unsqueeze(1)
            x = torch.cat([vel_norm, sim_data.materials, node_type_tensor], dim=1)
            
            class FakeBatch:
                def __init__(b):
                    b.x = x
                    b.edge_index = full_edge_index
                    b.edge_attr = full_edge_attr
                    b.batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                    b.node_type = node_type_tensor
            
            pred_accel_norm = self.model(FakeBatch())
            pred_accel_phys = pred_accel_norm * sim_data.stats['accel_std'].to(self.device) + sim_data.stats['accel_mean'].to(self.device)
            return pred_accel_phys