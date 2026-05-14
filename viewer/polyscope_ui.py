import os
import time
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from viewer.core import BaseVisualizer, BasePredictor
from viewer.data import SimulationData

MODE_DATASET_FPS = 0
MODE_DATASET_TIME = 1
MODE_ROLLOUT_FPS = 2
MODE_ROLLOUT_TIME = 3

MODE_LABELS = [
    "Dataset (Fixed FPS)",
    "Dataset (Calc Time)",
    "Rollout (Fixed FPS)",
    "Rollout (Calc Time)",
]


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

        self.radius_mult = 0.6
        self.wireframe_opacity = 0.2

        self.show_floor_grid = False
        self.show_cloth_surface = True

        self.mode = MODE_DATASET_FPS
        self.playback_fps = config.get("playback_fps", 30)

        self.frames_pos = []
        self.frames_dyn_edges = []

        self._dataset_frames_pos = None
        self._dataset_frames_dyn_edges = None
        self._rollout_frames_pos = None
        self._rollout_frames_dyn_edges = None
        self._cached_scenario_idx = -1

        self.edges_dyn_only_np = None
        self.edges_stat_only_np = None

        self.current_frame = 0
        self.is_playing = False
        self.sim_data = None

        self._play_start_wall = 0.0
        self._play_start_frame = 0
        self._total_time = 1.0

        self._dataset_total_time = 0.0
        self._rollout_start_time = 0.0
        self._rollout_total_time = 0.0

        # Export status
        self._export_msg = ""
        self._export_msg_time = 0.0

        ps.init()
        ps.set_program_name("Modular ML Physics")
        ps.set_up_dir("y_up")
        ps.set_autocenter_structures(False)
        ps.set_autoscale_structures(False)

        ps.set_ground_plane_mode("shadow_only")

        ps.set_user_callback(self.ui_callback)


    def load_scenario(self, idx: int, force_reload=False):
        if idx == self._cached_scenario_idx and not force_reload and self.sim_data is not None:
            self._switch_frame_source()
            return

        if self.sim_data is not None:
            self.sim_data.unload_trajectory()

        folder_path = self.sim_folders[idx]
        stats_path = os.path.join(self.config["dataset_path"], "global_stats.pt")

        self.sim_data = SimulationData(folder_path, stats_path, self.device)

        edge_index_all_np = self.sim_data.edge_index_static.T.cpu().numpy()
        node_type_np = self.sim_data.node_type.cpu().numpy()

        mask_dyn_edges = np.array([node_type_np[u] == 0 and node_type_np[v] == 0
                                   for u, v in edge_index_all_np])
        self.edges_dyn_only_np = edge_index_all_np[mask_dyn_edges]
        self.edges_stat_only_np = edge_index_all_np[~mask_dyn_edges]

        self._dataset_total_time = self.sim_data.traj_total_time

        self._dataset_frames_pos = None
        self._dataset_frames_dyn_edges = None
        self._rollout_frames_pos = None
        self._rollout_frames_dyn_edges = None
        self._rollout_total_time = 0.0
        self._cached_scenario_idx = idx

        if self.mode in (MODE_DATASET_FPS, MODE_DATASET_TIME):
            self._load_dataset_frames()
        else:
            self._init_rollout_frames()

        self._update_total_time()
        self.current_frame = 0
        self.is_playing = False
        self.update_renderer()
        ps.look_at_dir((0, 2, 3), (0, 0, 0), (0, 1, 0))

    def _switch_frame_source(self):
        """Switch between cached dataset/rollout frames without reloading."""
        if self.mode in (MODE_DATASET_FPS, MODE_DATASET_TIME):
            if self._dataset_frames_pos is not None:
                self.frames_pos = self._dataset_frames_pos
                self.frames_dyn_edges = self._dataset_frames_dyn_edges
            else:
                self._load_dataset_frames()
        else:
            if self._rollout_frames_pos is not None and len(self._rollout_frames_pos) > 1:
                self.frames_pos = self._rollout_frames_pos
                self.frames_dyn_edges = self._rollout_frames_dyn_edges
            else:
                self._init_rollout_frames()

        self._update_total_time()
        self.current_frame = 0
        self.is_playing = False
        self.update_renderer()

    def _update_total_time(self):
        """Set _total_time based on current mode."""
        if self.mode in (MODE_DATASET_FPS, MODE_DATASET_TIME):
            self._total_time = self._dataset_total_time
        else:
            self._total_time = self._rollout_total_time if self._rollout_total_time > 0 else self._dataset_total_time

    def _load_dataset_frames(self):
        """Load all ground-truth frames from dataset (lazy trajectory load)."""
        n = self.sim_data.traj_num_frames
        self.frames_pos = []
        self.frames_dyn_edges = []
        for i in range(n):
            pos_np, _, _ = self.sim_data.get_dataset_frame(i)
            self.frames_pos.append(pos_np)
            self.frames_dyn_edges.append(np.empty((0, 2), dtype=np.int32))
        self._dataset_frames_pos = self.frames_pos
        self._dataset_frames_dyn_edges = self.frames_dyn_edges

    def _init_rollout_frames(self):
        """Reset to frame 0 for model rollout."""
        self.sim_data.reset_to_frame(0)
        self.frames_pos = [self.sim_data.pos.cpu().numpy()]
        self.frames_dyn_edges = [np.empty((0, 2), dtype=np.int32)]


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
        pos = self.frames_pos[self.current_frame]

        can_show_surface = self.sim_data.faces_np is not None

        if self.show_cloth_surface and can_show_surface:
            ps_mesh = ps.register_surface_mesh("Cloth Surface", pos, self.sim_data.faces_np, color=(0.3, 0.6, 1.0))
            ps_mesh.set_material('clay')
        else:
            if ps.has_surface_mesh("Cloth Surface"):
                ps.remove_surface_mesh("Cloth Surface")

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


    def _get_time_based_frame(self):
        """Map wall-clock elapsed time to a frame index using total_time."""
        elapsed = time.time() - self._play_start_wall
        total_frames = len(self.frames_pos)
        if self._total_time <= 0 or total_frames <= 1:
            return 0
        frac = elapsed / self._total_time
        frame = int(frac * (total_frames - 1))
        if frame >= total_frames:
            self._play_start_wall = time.time()
            return 0
        return frame


    def ui_callback(self):
        psim.PushItemWidth(150)

        psim.TextUnformatted("--- Scene ---")

        changed_grid, self.show_floor_grid = psim.Checkbox("Floor Grid", self.show_floor_grid)
        if changed_grid:
            ps.set_ground_plane_mode("shadow_only" if self.show_floor_grid else "none")

        data_has_faces = self.sim_data is not None and self.sim_data.faces_np is not None
        if not data_has_faces:
            psim.PushStyleVar(psim.ImGuiStyleVar_Alpha, 0.5)
        changed_surf, self.show_cloth_surface = psim.Checkbox("Surface Mesh", self.show_cloth_surface)
        if not data_has_faces:
            psim.PopStyleVar()
        if changed_surf:
            self.update_renderer()

        psim.Separator()

        psim.TextUnformatted("--- Mode ---")
        for m_idx, m_label in enumerate(MODE_LABELS):
            if psim.RadioButton(m_label, self.mode == m_idx):
                if self.mode != m_idx and not self.is_calculating:
                    self.mode = m_idx
                    self.is_playing = False
                    self.load_scenario(self.current_folder_idx)

        psim.Separator()

        psim.TextUnformatted("--- Simulation ---")
        folder_names = [os.path.basename(f) for f in self.sim_folders]
        changed, self.current_folder_idx = psim.Combo("Scenario", self.current_folder_idx, folder_names)
        if changed and not self.is_calculating:
            self.load_scenario(self.current_folder_idx)

        if self.mode in (MODE_ROLLOUT_FPS, MODE_ROLLOUT_TIME):
            _, self.target_frames = psim.InputInt("Rollout Frames", self.target_frames)
            _, self.radius_mult = psim.SliderFloat("Radius Mult", self.radius_mult, 0.01, 20.0)

        if self.mode in (MODE_DATASET_FPS, MODE_ROLLOUT_FPS):
            _, self.playback_fps = psim.InputInt("Playback FPS", self.playback_fps)
            self.playback_fps = max(1, self.playback_fps)

        if self._dataset_total_time > 0:
            psim.TextUnformatted(f"IPC sim time: {self._dataset_total_time:.3f}s")
        if self._rollout_total_time > 0:
            psim.TextUnformatted(f"Model calc time: {self._rollout_total_time:.3f}s")

        psim.Separator()

        if self.mode in (MODE_ROLLOUT_FPS, MODE_ROLLOUT_TIME):
            if not self.is_calculating:
                if psim.Button("Calculate Rollout"):
                    self._rollout_frames_pos = None
                    self._rollout_frames_dyn_edges = None
                    self._rollout_total_time = 0.0
                    self.load_scenario(self.current_folder_idx, force_reload=True)
                    self.is_calculating = True
                    self.calc_progress = 0
                    self._rollout_start_time = time.time()
            else:
                psim.ProgressBar(self.calc_progress / max(1, self.target_frames), (-1, 0),
                                 f"{self.calc_progress}/{self.target_frames}")
                if self.calc_progress < self.target_frames:
                    self.generate_step()
                else:
                    self._rollout_total_time = time.time() - self._rollout_start_time
                    self._total_time = self._rollout_total_time
                    self._rollout_frames_pos = list(self.frames_pos)
                    self._rollout_frames_dyn_edges = list(self.frames_dyn_edges)
                    self.is_calculating = False
                    self.current_frame = 0

            if self._rollout_total_time > 0:
                psim.TextUnformatted(f"Rollout calc time: {self._rollout_total_time:.3f}s")

        psim.Separator()

        if not self.is_calculating and len(self.frames_pos) > 1:
            if psim.Button("Play / Pause"):
                self.is_playing = not self.is_playing
                if self.is_playing:
                    self._play_start_wall = time.time()
                    self._play_start_frame = self.current_frame

            psim.SameLine()
            if psim.Button("Reset"):
                self.is_playing = False
                self.current_frame = 0
                self.update_renderer()

            changed, self.current_frame = psim.SliderInt("Frame", self.current_frame, 0, len(self.frames_pos) - 1)
            if changed:
                self.is_playing = False
                self.update_renderer()

            total_f = len(self.frames_pos)
            if self.mode in (MODE_DATASET_TIME, MODE_ROLLOUT_TIME) and self._total_time > 0:
                cur_time = (self.current_frame / max(1, total_f - 1)) * self._total_time
                psim.TextUnformatted(f"Frame {self.current_frame}/{total_f - 1}  |  t = {cur_time:.3f}s / {self._total_time:.3f}s")
            else:
                psim.TextUnformatted(f"Frame {self.current_frame}/{total_f - 1}")

            if self.is_playing:
                if self.mode in (MODE_DATASET_FPS, MODE_ROLLOUT_FPS):
                    elapsed = time.time() - self._play_start_wall
                    target_frame = self._play_start_frame + int(elapsed * self.playback_fps)
                    if target_frame >= len(self.frames_pos):
                        self._play_start_wall = time.time()
                        self._play_start_frame = 0
                        target_frame = 0
                    self.current_frame = target_frame
                else:
                    self.current_frame = self._get_time_based_frame()

                self.update_renderer()

        psim.Separator()

        # --- EXPORT OBJ ---
        if self.mode in (MODE_ROLLOUT_FPS, MODE_ROLLOUT_TIME) and self._rollout_frames_pos is not None and len(self._rollout_frames_pos) > 1 and not self.is_calculating:
            if psim.Button("Save Frames to OBJ"):
                self._export_msg = "Saving..."
                self._export_frames_obj()

        if self._export_msg:
            if time.time() - self._export_msg_time > 3.0:
                self._export_msg = ""
            else:
                psim.TextUnformatted(self._export_msg)

        psim.PopItemWidth()

    def _export_frames_obj(self):
        """Export all current frames as OBJ files."""
        sim_name = os.path.basename(self.sim_folders[self.current_folder_idx])
        mode_name = MODE_LABELS[self.mode].replace(" ", "_").replace("(", "").replace(")", "")
        out_dir = os.path.join("export_obj", f"{sim_name}_{mode_name}")
        os.makedirs(out_dir, exist_ok=True)

        faces = self.sim_data.faces_np if self.sim_data.faces_np is not None else None

        for i, pos in enumerate(self.frames_pos):
            filepath = os.path.join(out_dir, f"frame_{i:04d}.obj")
            with open(filepath, 'w') as f:
                for v in pos:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                if faces is not None:
                    for face in faces:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        self._export_msg = f"Saved {len(self.frames_pos)} frames to {out_dir}/"
        self._export_msg_time = time.time()

    def run(self):
        self.load_scenario(self.current_folder_idx)
        ps.show()