import os
import json
import argparse
import torch
from viewer.predictor import MLPredictor
from viewer.polyscope_ui import PolyscopeVisualizer

def main():
    parser = argparse.ArgumentParser(description="Modular ML Physics Viewer")
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--exp', type=str, default='run_standard')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sim_folders = sorted([f.path for f in os.scandir(config["dataset_path"]) if f.is_dir()])
    
    predictor = MLPredictor(config, device)
    app = PolyscopeVisualizer(config, sim_folders, predictor)
    
    app.run()

if __name__ == "__main__":
    main()