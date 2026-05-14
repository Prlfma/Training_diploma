import os
import glob
import re
import torch
import meshio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

INPUT_DATASET_DIR = "./cube_dataset"
OUTPUT_PROCESSED_DIR = "./cube_dataset_processed"
CUBE_MSH_PATH = os.path.abspath("cube.msh")

FLOOR_RES = 0.5
FLOOR_PADDING = 2.0

os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)

_cube_mesh = meshio.read(CUBE_MSH_PATH)
CUBE_NUM_NODES = len(_cube_mesh.points)

_cube_cells = None
for cb in _cube_mesh.cells:
    if cb.type in ['tetra', 'triangle']:
        _cube_cells = cb.data
        break

if _cube_cells is None:
    raise RuntimeError(f"No tetra/triangle cells found in {CUBE_MSH_PATH}")

_c = torch.tensor(_cube_cells, dtype=torch.long)
if _c.shape[1] == 4:  
    _cube_edges = torch.cat([
        _c[:, [0, 1]], _c[:, [0, 2]], _c[:, [0, 3]],
        _c[:, [1, 2]], _c[:, [1, 3]], _c[:, [2, 3]]
    ], dim=0)
else:  
    _cube_edges = torch.cat([
        _c[:, [0, 1]], _c[:, [1, 2]], _c[:, [2, 0]]
    ], dim=0)

CUBE_EDGES_LOCAL = torch.unique(torch.sort(_cube_edges, dim=1)[0], dim=0)

_cube_faces = None
for cb in _cube_mesh.cells:
    if cb.type == 'triangle':
        _cube_faces = torch.tensor(cb.data, dtype=torch.long)
        break

print(f"Cube mesh: {CUBE_NUM_NODES} nodes, {CUBE_EDGES_LOCAL.shape[0]} edges, "
      f"{_cube_faces.shape[0] if _cube_faces is not None else 0} surface faces")


def generate_adaptive_floor(min_x, max_x, min_z, max_z, friction):
    """Generate a graph-based floor grid adapted to object spread."""
    min_x = np.floor((min_x - FLOOR_PADDING) / FLOOR_RES) * FLOOR_RES
    max_x = np.ceil((max_x + FLOOR_PADDING) / FLOOR_RES) * FLOOR_RES
    min_z = np.floor((min_z - FLOOR_PADDING) / FLOOR_RES) * FLOOR_RES
    max_z = np.ceil((max_z + FLOOR_PADDING) / FLOOR_RES) * FLOOR_RES

    steps_x = int((max_x - min_x) / FLOOR_RES) + 1
    steps_z = int((max_z - min_z) / FLOOR_RES) + 1

    x = np.linspace(min_x, max_x, steps_x)
    z = np.linspace(min_z, max_z, steps_z)
    xx, zz = np.meshgrid(x, z)
    yy = np.zeros_like(xx)

    vertices = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    faces = []
    for i in range(steps_z - 1):
        for j in range(steps_x - 1):
            tl = i * steps_x + j
            tr = tl + 1
            bl = (i + 1) * steps_x + j
            br = bl + 1
            faces.append([tl, bl, tr])
            faces.append([tr, bl, br])

    faces = np.array(faces)
    if len(faces) > 0:
        src = torch.tensor(faces[:, [0, 1, 2]].flatten(), dtype=torch.long)
        dst = torch.tensor(faces[:, [1, 2, 0]].flatten(), dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        edge_index = torch.unique(torch.sort(edge_index, dim=0)[0], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    floor_faces = torch.tensor(faces, dtype=torch.long) if len(faces) > 0 else None

    pos_tensor = torch.tensor(vertices, dtype=torch.float)

    materials = torch.tensor([10.0, 10000.0, friction], dtype=torch.float)
    materials = materials.unsqueeze(0).repeat(len(pos_tensor), 1)

    return pos_tensor, edge_index, materials, floor_faces


def parse_config(config_path):
    """Extract dt, ground friction, and object info from config.txt."""
    dt = 0.025
    ground_friction = 0.0
    num_objects = 0

    with open(config_path, 'r') as f:
        content = f.read()

        t_match = re.search(r'time\s+[\d\.]+\s+([\d\.eE\+\-]+)', content)
        if t_match:
            dt = float(t_match.group(1))

        g_match = re.search(r'ground\s+([\d\.]+)', content)
        if g_match:
            ground_friction = float(g_match.group(1))

        objects_info = []
        for line in content.split('\n'):
            if 'cube.msh' in line:
                mat_match = re.search(
                    r'material\s+([\d\.eE\+\-]+)\s+([\d\.eE\+\-]+)', line
                )
                if mat_match:
                    density = float(mat_match.group(1))
                    stiffness = float(mat_match.group(2))
                else:
                    density, stiffness = 1000.0, 1e6

                objects_info.append({
                    'density': density,
                    'stiffness': stiffness
                })

    return dt, ground_friction, objects_info


def parse_status_file(filepath):
    """Read a single IPC status frame file. Returns (timestep_index, pos, vel, accel)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    iterator = iter(lines)
    pos, vel, accel = [], [], []
    timestep_idx = None

    try:
        for line in iterator:
            line = line.strip()
            if line.startswith("timestep"):
                timestep_idx = int(line.split()[1])
            elif line.startswith("position"):
                num_nodes = int(line.split()[1])
                for _ in range(num_nodes):
                    pos.append([float(c) for c in next(iterator).strip().split()])
            elif line.startswith("velocity"):
                num_vals = int(line.split()[1])
                for _ in range(num_vals):
                    vel.append(float(next(iterator).strip()))
            elif line.startswith("acceleration"):
                num_nodes_acc = int(line.split()[1])
                for _ in range(num_nodes_acc):
                    accel.append([float(c) for c in next(iterator).strip().split()])
    except Exception:
        return None, None, None, None

    if not pos:
        return None, None, None, None

    t_pos = torch.tensor(pos, dtype=torch.float)
    t_vel = torch.tensor(vel, dtype=torch.float).view(-1, 3) if vel else torch.zeros_like(t_pos)
    t_acc = torch.tensor(accel, dtype=torch.float) if accel else torch.zeros_like(t_pos)

    return timestep_idx, t_pos, t_vel, t_acc


def parse_ipc_wallclock(sim_folder):
    """Parse IPC log.txt to get wall-clock computation time in seconds."""
    log_path = os.path.join(sim_folder, "log.txt")
    if not os.path.exists(log_path):
        return 0.0

    from datetime import datetime
    timestamps = []
    with open(log_path, 'r') as f:
        for line in f:
            m = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]', line)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S.%f')
                    timestamps.append(ts)
                except ValueError:
                    pass

    if len(timestamps) >= 2:
        delta = (timestamps[-1] - timestamps[0]).total_seconds()
        return delta
    return 0.0


def process_simulation(sim_folder):
    sim_name = os.path.basename(sim_folder)
    out_dir = os.path.join(OUTPUT_PROCESSED_DIR, sim_name)
    config_path = os.path.join(sim_folder, "config.txt")

    if not os.path.exists(config_path):
        return f"Skipped {sim_name}: No config"

    dt, ground_friction, objects_info = parse_config(config_path)
    num_cubes = len(objects_info)
    if num_cubes == 0:
        return f"Skipped {sim_name}: No objects in config"

    output_subfolder = os.path.join(sim_folder, "output")
    if os.path.isdir(output_subfolder):
        subdirs = [d for d in glob.glob(os.path.join(output_subfolder, "*"))
                   if os.path.isdir(d)]
        data_dir = subdirs[0] if subdirs else output_subfolder
    else:
        data_dir = sim_folder

    valid_files = []
    for f in glob.glob(os.path.join(data_dir, "status*")):
        match = re.search(r'status(\d+)$', os.path.basename(f))
        if match:
            valid_files.append((int(match.group(1)), f))
    valid_files.sort(key=lambda x: x[0])

    if len(valid_files) < 2:
        return f"Skipped {sim_name}: Not enough frames ({len(valid_files)})"

    dyn_traj_pos, dyn_traj_vel, dyn_traj_acc = [], [], []
    timestep_indices = []
    for _, f_path in valid_files:
        ts_idx, p, v, a = parse_status_file(f_path)
        if p is not None:
            dyn_traj_pos.append(p)
            dyn_traj_vel.append(v)
            dyn_traj_acc.append(a)
            timestep_indices.append(ts_idx if ts_idx is not None else len(dyn_traj_pos) - 1)

    if not dyn_traj_pos:
        return f"Skipped {sim_name}: Empty status files"

    total_dyn_nodes = CUBE_NUM_NODES * num_cubes
    if dyn_traj_pos[0].shape[0] != total_dyn_nodes:
        return (f"Skipped {sim_name}: Node count mismatch "
                f"(Expected {total_dyn_nodes}, Got {dyn_traj_pos[0].shape[0]})")

    all_pos_tensor = torch.stack(dyn_traj_pos)  
    min_x = all_pos_tensor[:, :, 0].min().item()
    max_x = all_pos_tensor[:, :, 0].max().item()
    min_z = all_pos_tensor[:, :, 2].min().item()
    max_z = all_pos_tensor[:, :, 2].max().item()

    dyn_edge_indices = []
    dyn_materials = []
    dyn_faces = []

    for i, obj in enumerate(objects_info):
        offset = i * CUBE_NUM_NODES

        src, dst = (CUBE_EDGES_LOCAL + offset).t()
        dyn_edge_indices.append(torch.stack([src, dst], dim=0))

        mat_vec = torch.tensor(
            [np.log10(obj['stiffness']), obj['density'], ground_friction],
            dtype=torch.float
        )
        dyn_materials.append(mat_vec.unsqueeze(0).repeat(CUBE_NUM_NODES, 1))

        if _cube_faces is not None:
            dyn_faces.append(_cube_faces + offset)

    edge_index_dyn = torch.cat(dyn_edge_indices, dim=1)
    materials_dyn = torch.cat(dyn_materials, dim=0)

    floor_pos, floor_edges, floor_materials, floor_faces = generate_adaptive_floor(
        min_x, max_x, min_z, max_z, ground_friction
    )
    floor_num_nodes = len(floor_pos)

    full_edge_index = torch.cat([edge_index_dyn, floor_edges + total_dyn_nodes], dim=1)
    full_materials = torch.cat([materials_dyn, floor_materials], dim=0)

    node_type = torch.cat([
        torch.zeros((total_dyn_nodes, 1), dtype=torch.float),
        torch.ones((floor_num_nodes, 1), dtype=torch.float)
    ], dim=0)

    all_faces = []
    if dyn_faces:
        all_faces.extend(dyn_faces)
    if floor_faces is not None:
        all_faces.append(floor_faces + total_dyn_nodes)
    combined_faces = torch.cat(all_faces, dim=0) if all_faces else None

    final_traj_pos, final_traj_vel, final_traj_acc = [], [], []
    floor_vel = torch.zeros_like(floor_pos)
    floor_acc = torch.zeros_like(floor_pos)

    num_frames = len(dyn_traj_pos)
    total_time = parse_ipc_wallclock(sim_folder)   
    if total_time <= 0:
        total_time = timestep_indices[-1] * dt      

    for i in range(num_frames):
        final_traj_pos.append(torch.cat([dyn_traj_pos[i], floor_pos], dim=0))
        final_traj_vel.append(torch.cat([dyn_traj_vel[i], floor_vel], dim=0))
        final_traj_acc.append(torch.cat([dyn_traj_acc[i], floor_acc], dim=0))

    os.makedirs(out_dir, exist_ok=True)

    static_dict = {
        'edge_index': full_edge_index,
        'materials': full_materials,
        'node_type': node_type,
        'dt': torch.tensor(dt, dtype=torch.float),
    }
    if combined_faces is not None:
        static_dict['faces'] = combined_faces

    torch.save(static_dict, os.path.join(out_dir, "static_graph.pt"))

    torch.save({
        'pos': torch.stack(final_traj_pos),
        'vel': torch.stack(final_traj_vel),
        'accel': torch.stack(final_traj_acc),
        'total_time': torch.tensor(total_time, dtype=torch.float),
    }, os.path.join(out_dir, "trajectory.pt"))

    return f"Processed {sim_name}: {len(final_traj_pos)} frames, {num_cubes} cubes (Floor: {floor_num_nodes} nodes)"


def compute_global_stats():
    """Compute vel/accel mean/std and avg edge length over all processed sims."""
    sim_folders = sorted(glob.glob(os.path.join(OUTPUT_PROCESSED_DIR, "sim_*")))

    vel_sum = torch.zeros(3, dtype=torch.float64)
    vel_sq_sum = torch.zeros(3, dtype=torch.float64)
    accel_sum = torch.zeros(3, dtype=torch.float64)
    accel_sq_sum = torch.zeros(3, dtype=torch.float64)
    total_dynamic_elements = 0

    total_edge_len_sum = 0.0
    total_edges = 0

    print(f"\nComputing global stats from {len(sim_folders)} processed simulations...")

    for sim_folder in tqdm(sim_folders):
        traj_path = os.path.join(sim_folder, "trajectory.pt")
        static_path = os.path.join(sim_folder, "static_graph.pt")

        if not os.path.exists(traj_path) or not os.path.exists(static_path):
            continue

        traj = torch.load(traj_path, map_location='cpu')
        static = torch.load(static_path, map_location='cpu')

        mask = (static['node_type'].squeeze() == 0)

        vel = traj['vel'][:, mask, :]
        accel = traj['accel'][:, mask, :]

        num_elements = vel.shape[0] * vel.shape[1]
        total_dynamic_elements += num_elements

        vel_sum += vel.sum(dim=(0, 1)).double()
        vel_sq_sum += (vel ** 2).sum(dim=(0, 1)).double()

        accel_sum += accel.sum(dim=(0, 1)).double()
        accel_sq_sum += (accel ** 2).sum(dim=(0, 1)).double()

        edge_index = static['edge_index']
        src, dst = edge_index
        is_dyn_edge = mask[src] & mask[dst]
        dyn_src = src[is_dyn_edge]
        dyn_dst = dst[is_dyn_edge]

        p0 = traj['pos'][0]
        d_ij = p0[dyn_src] - p0[dyn_dst]
        edge_lengths = torch.norm(d_ij, dim=1)

        total_edge_len_sum += edge_lengths.sum().double().item()
        total_edges += edge_lengths.shape[0]

    if total_dynamic_elements == 0:
        print("ERROR: No dynamic vertices found for stats computation!")
        return

    vel_mean = (vel_sum / total_dynamic_elements).float()
    vel_var = (vel_sq_sum / total_dynamic_elements).float() - (vel_mean ** 2)
    vel_std = torch.sqrt(torch.clamp(vel_var, min=1e-8))

    accel_mean = (accel_sum / total_dynamic_elements).float()
    accel_var = (accel_sq_sum / total_dynamic_elements).float() - (accel_mean ** 2)
    accel_std = torch.sqrt(torch.clamp(accel_var, min=1e-8))

    avg_edge_len = torch.tensor(total_edge_len_sum / total_edges, dtype=torch.float)

    stats = {
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        'accel_mean': accel_mean,
        'accel_std': accel_std,
        'avg_edge_len': avg_edge_len,
    }

    out_path = os.path.join(OUTPUT_PROCESSED_DIR, "global_stats.pt")
    torch.save(stats, out_path)

    print(f"\nglobal_stats.pt saved to {out_path}")
    print(f"  Velocity Mean: {vel_mean.tolist()}")
    print(f"  Velocity Std:  {vel_std.tolist()}")
    print(f"  Accel Mean:    {accel_mean.tolist()}")
    print(f"  Accel Std:     {accel_std.tolist()}")
    print(f"  Avg Edge Len:  {avg_edge_len.item():.6f}")


def main():
    sim_folders = sorted(glob.glob(os.path.join(INPUT_DATASET_DIR, "sim_*")))
    print(f"Processing {len(sim_folders)} simulations from {INPUT_DATASET_DIR}...")

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = list(tqdm(
            executor.map(process_simulation, sim_folders),
            total=len(sim_folders)
        ))

    success = sum(1 for r in results if "Processed" in r)
    skipped = [r for r in results if "Skipped" in r]

    print(f"\nProcessed: {success}/{len(sim_folders)}")
    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for s in skipped[:20]:
            print(f"  - {s}")

    if success > 0:
        compute_global_stats()
    else:
        print("No simulations processed, skipping stats computation.")


if __name__ == "__main__":
    main()
