import os
import random
import shutil
import subprocess
import concurrent.futures
import gc
import math
import numpy as np

# --- 1. CONFIGURATION ---
IPC_BIN = "./build/IPC_bin"
CUBE_MSH = os.path.abspath("cube.msh")
OUTPUT_DATASET_DIR = "./cube_dataset"

NUM_SIMULATIONS = 120
MAX_WORKERS = 1
SIMULATION_TIMEOUT = 2000

TIME_DURATION = 3.75  # 3.75 / 0.025 = 150 frames
DT = 0.025  # Fixed across all configs

MIN_OBJECTS = 1
MAX_OBJECTS = 3

# Unit cube [0,1]^3 bounding sphere radius = sqrt(3) ≈ 1.732
CUBE_RADIUS = math.sqrt(3.0)
# Gap between stacked objects to guarantee no interpenetration
STACK_GAP = 0.1
# Floor clearance gap
FLOOR_GAP = 0.05

# --- 2. SCENARIO GENERATOR ---
def generate_scenario(sim_id, output_dir):
    """Generate a single IPC config for cube-only collision scenario."""
    try:
        num_cubes = random.randint(MIN_OBJECTS, MAX_OBJECTS)

        # Per-scenario physics
        ground_friction = round(random.uniform(0.1, 0.8), 2)
        ground_restitution = round(random.uniform(0.0, 0.5), 2)
        self_friction = round(random.uniform(0.3, 0.9), 2)

        shapes_lines = []
        stack_top_y = 0.0  # top of the current stack (floor level)

        for i in range(num_cubes):
            # --- HEIGHT (Y) ---
            # Safety from floor: center must be at least radius + gap above Y=0
            floor_safe_y = CUBE_RADIUS + FLOOR_GAP
            # Safety from previous object on the stack
            stack_safe_y = stack_top_y + CUBE_RADIUS + STACK_GAP
            pos_y = max(floor_safe_y, stack_safe_y)
            # Update stack top for next cube
            stack_top_y = pos_y + CUBE_RADIUS

            # --- X, Z jitter (encourage off-axis collisions) ---
            pos_x = round(random.uniform(-0.3, 0.3), 3)
            pos_z = round(random.uniform(-0.3, 0.3), 3)

            # --- ROTATION (degrees) ---
            rot_x = round(random.uniform(0, 360), 1)
            rot_y = round(random.uniform(0, 360), 1)
            rot_z = round(random.uniform(0, 360), 1)

            # --- SCALE (uniform, keep unit cube) ---
            scale = round(random.uniform(0.8, 1.2), 2)

            # --- MATERIAL ---
            density = round(random.uniform(900, 1500), 1)
            youngs = 10 ** random.uniform(5.5, 6.5)
            poisson = 0.4

            # --- INITIAL VELOCITY ---
            vel_x = round(random.uniform(-0.5, 0.5), 3)
            vel_y = round(random.uniform(-2.5, -0.5), 3)  # downward (clamped to avoid solver divergence)
            vel_z = round(random.uniform(-0.5, 0.5), 3)

            # --- ANGULAR VELOCITY ---
            ang_x = round(random.uniform(-2.0, 2.0), 3)
            ang_y = round(random.uniform(-2.0, 2.0), 3)
            ang_z = round(random.uniform(-2.0, 2.0), 3)

            line = (
                f"{CUBE_MSH} {pos_x:.3f} {pos_y:.3f} {pos_z:.3f} "
                f"{rot_x:.1f} {rot_y:.1f} {rot_z:.1f} "
                f"{scale} {scale} {scale} "
                f"material {density} {youngs:.6e} {poisson} "
                f"initVel {vel_x:.3f} {vel_y:.3f} {vel_z:.3f} "
                f"{ang_x:.3f} {ang_y:.3f} {ang_z:.3f}"
            )
            shapes_lines.append(line)

        if not shapes_lines:
            return None

        config_content = (
            f"energy NH\n"
            f"time {TIME_DURATION} {DT}\n"
            f"density 1000\n"
            f"stiffness 1e6 0.4\n"
            f"selfFric {self_friction}\n"
            f"ground {ground_friction} {ground_restitution}\n"
            f"dHat 1e-3\n"
            f"constraintSolver interiorPoint\n"
            f"\n"
            f"halfSpace 0 -0.01 0  0 1 0  {ground_friction} {ground_restitution}\n"
            f"\n"
            f"shapes input {len(shapes_lines)}\n"
        )
        for sl in shapes_lines:
            config_content += sl + "\n"

        path = os.path.join(output_dir, "config.txt")
        with open(path, "w") as f:
            f.write(config_content)
        return path

    except Exception as e:
        print(f"[Sim {sim_id}] Config generation error: {e}")
        return None


MAX_RETRIES = 3


def _count_status_files(sim_dir):
    """Return number of status files produced by IPC in sim_dir."""
    output_dir = os.path.join(sim_dir, "output")
    if os.path.isdir(output_dir):
        subdirs = [d for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d))]
        data_dir = os.path.join(output_dir, subdirs[0]) if subdirs else output_dir
        return len([f for f in os.listdir(data_dir) if f.startswith("status")]) if os.path.isdir(data_dir) else 0
    return len([f for f in os.listdir(sim_dir) if f.startswith("status")])


def _wipe_sim_dir(sim_dir):
    """Remove all contents of sim_dir but keep the directory."""
    if os.path.isdir(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir, exist_ok=True)


# --- 3. WORKER ---
def run_simulation_task(sim_id):
    sim_dir = os.path.join(OUTPUT_DATASET_DIR, f"sim_{sim_id:04d}")
    os.makedirs(sim_dir, exist_ok=True)

    # Skip already-completed sims
    if _count_status_files(sim_dir) >= 2:
        return f"Sim {sim_id}: SKIPPED (already done, {_count_status_files(sim_dir)} frames)"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    for attempt in range(1, MAX_RETRIES + 1):
        # Wipe previous failed output and regenerate config with new random seed
        _wipe_sim_dir(sim_dir)
        random.seed(sim_id * 1000 + attempt)
        config_path = generate_scenario(sim_id, sim_dir)

        if not config_path:
            continue

        cmd = [os.path.abspath(IPC_BIN), "100", os.path.abspath(config_path)]

        try:
            with open(os.path.join(sim_dir, "log.txt"), "w") as log:
                subprocess.run(
                    cmd, cwd=sim_dir, stdout=log, stderr=log,
                    check=True, env=env, timeout=SIMULATION_TIMEOUT
                )

            n_frames = _count_status_files(sim_dir)
            if n_frames >= 2:
                return f"Sim {sim_id}: OK ({n_frames} frames, attempt {attempt})"
            else:
                print(f"  Sim {sim_id}: only {n_frames} frames on attempt {attempt}, retrying...")
                continue

        except subprocess.TimeoutExpired:
            print(f"  Sim {sim_id}: timeout on attempt {attempt}, retrying...")
            continue
        except subprocess.CalledProcessError as e:
            print(f"  Sim {sim_id}: IPC error rc={e.returncode} on attempt {attempt}, retrying...")
            continue
        except Exception as e:
            print(f"  Sim {sim_id}: error '{e}' on attempt {attempt}, retrying...")
            continue
        finally:
            # Flush OS disk cache to prevent RAM buildup on slow HDD
            subprocess.run(["sync"], timeout=30)
            gc.collect()

    return f"Sim {sim_id}: FAILED after {MAX_RETRIES} attempts"


# --- 4. MAIN ---
def main():
    if not os.path.exists(IPC_BIN):
        print(f"IPC binary not found at {IPC_BIN}")
        return

    if not os.path.exists(CUBE_MSH):
        print(f"cube.msh not found at {CUBE_MSH}")
        return

    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    print(f"Generating {NUM_SIMULATIONS} cube-only scenarios (1-{MAX_OBJECTS} cubes each, dt={DT}, T={TIME_DURATION}s)...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_simulation_task, i): i
            for i in range(NUM_SIMULATIONS)
        }

        done = 0
        ok = 0
        for f in concurrent.futures.as_completed(futures):
            done += 1
            result = f.result()
            if "OK" in result:
                ok += 1
            print(f"[{done}/{NUM_SIMULATIONS}] {result}")
            if done % 10 == 0:
                gc.collect()

    print(f"\nFinished: {ok}/{NUM_SIMULATIONS} simulations completed successfully.")


if __name__ == "__main__":
    main()
