import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from viewer.core import BaseVisualizer, BasePredictor
from viewer.data import SimulationData

class PolyscopeVisualizer(BaseVisualizer):
    def __init__(self, config, sim_folders, predictor: BasePredictor):
        self.config = config
        self.sim_folders = sim_folders
        self.predictor = predictor
        self.device = predictor.device
        
        self.current_folder_idx = 0
        self.target_frames = 100
        self.calc_progress = 0
        self.is_calculating = False
        
        self.radius_mult = 4.0        
        self.wireframe_opacity = 0.2 
        
        self.show_floor_grid = False     
        self.show_cloth_surface = True   
        
        self.frames_pos = []
        self.frames_dyn_edges = []
        
        self.edges_dyn_only_np = None
        self.edges_stat_only_np = None
        
        self.current_frame = 0
        self.is_playing = False
        self.sim_data = None 
        
        ps.init()
        ps.set_program_name("Modular ML Physics")
        ps.set_up_dir("y_up")
        ps.set_autocenter_structures(False)
        ps.set_autoscale_structures(False)
        
        ps.set_ground_plane_mode("shadow_only")
        
        ps.set_user_callback(self.ui_callback)

    def load_scenario(self, idx: int):
        folder_path = self.sim_folders[idx]
        stats_path = os.path.join(self.config["dataset_path"], "global_stats.pt")
        
        self.sim_data = SimulationData(folder_path, stats_path, self.device)
        
        edge_index_all_np = self.sim_data.edge_index_static.T.cpu().numpy()
        
        node_type_np = self.sim_data.node_type.cpu().numpy()
        
        def is_edge_dynamic(u, v):
            return node_type_np[u] == 0 and node_type_np[v] == 0
        
        mask_dyn_edges = np.array([is_edge_dynamic(u, v) for u, v in edge_index_all_np])
        
        self.edges_dyn_only_np = edge_index_all_np[mask_dyn_edges]
        self.edges_stat_only_np = edge_index_all_np[~mask_dyn_edges]
        
        self.frames_pos = [self.sim_data.pos.cpu().numpy()]
        self.frames_dyn_edges = [np.empty((0, 2), dtype=np.int32)]
        self.current_frame = 0
        self.update_renderer()
        ps.look_at_dir((0, 2, 3), (0, 0, 0), (0, 1, 0))

    def generate_step(self):
        self.sim_data.find_dynamic_edges(self.radius_mult)
        accel = self.predictor.predict(self.sim_data, self.radius_mult)
        self.sim_data.step_physics(accel)
        
        self.frames_pos.append(self.sim_data.pos.cpu().numpy())
        self.frames_dyn_edges.append(self.sim_data.edge_index_dynamic.T.cpu().numpy())
        self.calc_progress += 1
        
        self.current_frame = len(self.frames_pos) - 1
        self.update_renderer()

    def update_renderer(self):
        """Побудова сцени згідно з твоїми вимогами"""
        pos = self.frames_pos[self.current_frame]
        
        can_show_surface = self.sim_data.faces_np is not None
        
        if self.show_cloth_surface and can_show_surface:
            ps_mesh = ps.register_surface_mesh("Cloth Surface", pos, self.sim_data.faces_np, color=(0.3, 0.6, 1.0))
            ps_mesh.set_material('shading') 
        else:
            if ps.has_surface_mesh("Cloth Surface"):
                ps.remove_surface_mesh("Cloth Surface")

        opacity = self.wireframe_opacity if (self.show_cloth_surface and can_show_surface) else 1.0
        
        net_static = ps.register_curve_network("Cloth Structure Wires", pos, self.edges_dyn_only_np, color=(0.2, 0.5, 0.8))
        net_static.set_radius(0.009, relative=False)
        
        net_env = ps.register_curve_network("Static Environment", pos, self.edges_stat_only_np, color=(0.6, 0.6, 0.6))
        net_env.set_radius(0.005, relative=False) 

        dyn_edges = self.frames_dyn_edges[self.current_frame]
        if len(dyn_edges) > 0:
            net_dyn = ps.register_curve_network("World Contacts (Red)", pos, dyn_edges, color=(0.8, 0.2, 0.2))
            net_dyn.set_radius(0.0020, relative=False)
        else:
            if ps.has_curve_network("World Contacts (Red)"):
                ps.remove_curve_network("World Contacts (Red)")

    def ui_callback(self):
        psim.PushItemWidth(150)
        
        psim.TextUnformatted("--- Налаштування Сцени ---")
        
        changed_grid, self.show_floor_grid = psim.Checkbox("Сітка підлоги", self.show_floor_grid)
        if changed_grid:
            ps.set_ground_plane_enabled(self.show_floor_grid)
        data_has_faces = self.sim_data.faces_np is not None
        if not data_has_faces:
            psim.PushStyleVar(psim.ImGuiStyleVar_Alpha, 0.5) 
            
        changed_surf, self.show_cloth_surface = psim.Checkbox("Поверхня тканини (Planes)", self.show_cloth_surface)
        if not data_has_faces:
            psim.PopStyleVar()
            if psim.IsItemHovered():
                psim.SetTooltip("Помилка: У static_graph.pt не знайдено faces сітки.")
        
        if changed_surf:
            self.update_renderer() 
            
        psim.Separator()
        
        psim.TextUnformatted("--- Налаштування Симуляції ---")
        folder_names = [os.path.basename(f) for f in self.sim_folders]
        changed, self.current_folder_idx = psim.Combo("Сценарій", self.current_folder_idx, folder_names)
        if changed and not self.is_calculating:
            self.load_scenario(self.current_folder_idx)

        _, self.target_frames = psim.InputInt("Кількість кадрів", self.target_frames)
        _, self.radius_mult = psim.SliderFloat("Множник радіуса", self.radius_mult, 0.01, 20.0)
        
        psim.Separator()
        
        if not self.is_calculating:
            if psim.Button("Прорахувати!"):
                self.load_scenario(self.current_folder_idx)
                self.is_calculating = True
                self.calc_progress = 0
        else:
            psim.ProgressBar(self.calc_progress / max(1, self.target_frames), (-1, 0), f"{self.calc_progress}/{self.target_frames}")
            if self.calc_progress < self.target_frames:
                self.generate_step()
            else:
                self.is_calculating = False
                self.current_frame = 0

        psim.Separator()
        if not self.is_calculating and len(self.frames_pos) > 1:
            if psim.Button("Play/Pause"):
                self.is_playing = not self.is_playing
            
            changed, self.current_frame = psim.SliderInt("Frame", self.current_frame, 0, len(self.frames_pos) - 1)
            if changed:
                self.update_renderer()

            if self.is_playing:
                self.current_frame = (self.current_frame + 1) % len(self.frames_pos)
                self.update_renderer()
                
        psim.PopItemWidth()

    def run(self):
        self.load_scenario(self.current_folder_idx)
        ps.show()