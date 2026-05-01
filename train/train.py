import os
import argparse
import requests
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import traceback
import io
import matplotlib.pyplot as plt

from dataset import IPCDataset
from model import CustomMeshGraphNet, CauchyMLP, MLP

PROCESSED_DIR = "./dataset"
BATCH_SIZE = 8 
EPOCHS = 200
LR = 1e-4
BARRIER_MARGIN = 0.001
BARRIER_WEIGHT = 1000.0

def send_tg_message(token, chat_id, text):
    """Відправляє текстове повідомлення."""
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=5)
    except Exception as e: print(f"⚠️ Помилка TG: {e}")

def send_tg_image(token, chat_id, image_buf, caption=""):
    """Відправляє фото (графік) у Telegram."""
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        image_buf.seek(0)
        files = {'photo': ('graph.png', image_buf, 'image/png')}
        requests.post(url, data={'chat_id': chat_id, 'caption': caption}, files=files, timeout=10)
    except Exception as e: print(f"⚠️ Помилка відправки фото: {e}")

def run_epoch(model, loader, optimizer, device, stats, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_mse = 0.0     
    total_barrier = 0.0
    
    with torch.set_grad_enabled(is_train):
        desc = "Train" if is_train else "Test"
        pbar = tqdm(loader, desc=f"[{desc}]")
        
        for batch in pbar:
            batch = batch.to(device)
            if is_train: optimizer.zero_grad()
            
            pred_accel_norm = model(batch)
            
            dynamic_mask = (batch.node_type.squeeze() == 0)
            pred_accel_dyn = pred_accel_norm[dynamic_mask]
            target_accel_dyn = batch.y[dynamic_mask]
            loss_mse = F.mse_loss(pred_accel_dyn, target_accel_dyn)
            
            dt = batch.dt[0].item()
            pred_accel_phys = pred_accel_norm * stats['accel_std'].to(device) + stats['accel_mean'].to(device)
            vel_phys = batch.x[:, :3] * stats['vel_std'].to(device) + stats['vel_mean'].to(device) 
            
            pred_v_phys = vel_phys + pred_accel_phys * dt
            pred_pos = batch.pos + pred_v_phys * dt
            
            is_dynamic_edge = batch.edge_attr[:, 4] == 1.0
            dyn_edges = batch.edge_index[:, is_dynamic_edge]
            
            if dyn_edges.shape[1] > 0:
                src, dst = dyn_edges
                d_ij = pred_pos[src] - pred_pos[dst]
                dist_future = torch.norm(d_ij, dim=1)
                penetration = F.relu(BARRIER_MARGIN - dist_future)
                loss_barrier = penetration.mean() * BARRIER_WEIGHT
            else:
                loss_barrier = torch.tensor(0.0, device=device)
            
            loss = loss_mse + loss_barrier
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item()
            total_mse += loss_mse.item()        
            total_barrier += loss_barrier.item()
            pbar.set_postfix({'MSE': f"{loss_mse.item():.4f}", 'Bar': f"{loss_barrier.item():.4f}"})
            
    return total_loss / len(loader), total_mse / len(loader), total_barrier / len(loader)
    
def run_epoch_multistep(model, loader, optimizer, device, stats, is_train=True, n_steps=5):
    model.train() if is_train else model.eval()
    total_loss, total_mse, total_barrier = 0.0, 0.0, 0.0
    
    with torch.set_grad_enabled(is_train):
        desc = "Train (5-Step)" if is_train else "Test (5-Step)"
        pbar = tqdm(loader, desc=f"[{desc}]")
        
        for batch in pbar:
            batch = batch.to(device)
            if is_train: optimizer.zero_grad()
            
            loss_mse = 0.0
            loss_barrier = 0.0
            
            current_x = batch.x
            current_pos = batch.pos
            current_edge_attr = batch.edge_attr
            
            current_v_phys = batch.x[:, :3] * stats['vel_std'].to(device) + stats['vel_mean'].to(device)
            dt = batch.dt[0].item()
            cr = batch.contact_radius[0].item() # Contact radius
            dynamic_mask = (batch.node_type.squeeze() == 0)
            
            for step in range(n_steps):
                step_data = batch.clone()
                step_data.x = current_x
                step_data.pos = current_pos
                step_data.edge_attr = current_edge_attr
                
                pred_accel_norm = model(step_data)
                
                pred_accel_dyn = pred_accel_norm[dynamic_mask]
                target_accel_dyn = batch.y[dynamic_mask, step, :] # [N_dyn, 3]
                
                step_loss_mse = F.mse_loss(pred_accel_dyn, target_accel_dyn)
                loss_mse += step_loss_mse
                
                pred_accel_phys = pred_accel_norm * stats['accel_std'].to(device) + stats['accel_mean'].to(device)
                
                accel_phys_dyn = pred_accel_phys * dynamic_mask.unsqueeze(-1)
                current_v_phys = current_v_phys + accel_phys_dyn * dt
                current_pos = current_pos + current_v_phys * dt
                
                is_dynamic_edge = current_edge_attr[:, 4] == 1.0
                dyn_edges = batch.edge_index[:, is_dynamic_edge]
                if dyn_edges.shape[1] > 0:
                    src, dst = dyn_edges
                    d_ij = current_pos[src] - current_pos[dst]
                    dist_future = torch.norm(d_ij, dim=1)
                    penetration = F.relu(BARRIER_MARGIN - dist_future)
                    loss_barrier += penetration.mean() * BARRIER_WEIGHT
                
                if step < n_steps - 1:
                    new_vel_norm = (current_v_phys - stats['vel_mean'].to(device)) / stats['vel_std'].to(device)
                    current_x = torch.cat([new_vel_norm, current_x[:, 3:]], dim=1)
                    
                    src, dst = batch.edge_index
                    d_ij_curr = current_pos[src] - current_pos[dst]
                    pos_lookahead = current_pos + current_v_phys * dt
                    d_ij_look = pos_lookahead[src] - pos_lookahead[dst]
                    
                    current_edge_attr = torch.cat([
                        d_ij_curr / cr,                                          
                        (torch.norm(d_ij_curr, dim=1, keepdim=True) / cr),       
                        (torch.norm(d_ij_look, dim=1, keepdim=True) / cr),          
                        current_edge_attr[:, 5:7]                                
                    ], dim=1)
                    
            loss_mse = loss_mse / n_steps
            loss_barrier = loss_barrier / n_steps
            loss = loss_mse + loss_barrier
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item()
            total_mse += loss_mse.item()        
            total_barrier += loss_barrier.item()
            pbar.set_postfix({'MSE(5s)': f"{loss_mse.item():.4f}", 'Bar': f"{loss_barrier.item():.4f}"})
            
    return total_loss / len(loader), total_mse / len(loader), total_barrier / len(loader)
def train(args):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Запуск експерименту '{args.exp_name}' на: {device}")
        send_tg_message(args.tg_token, args.tg_chat, f"🚀 *Старт тренування: {args.exp_name}*\nАрхітектура: {args.arch}\nDevice: {device}")

        train_dataset = IPCDataset(processed_dir=PROCESSED_DIR, mode='train', noise_scale=0.0)
        test_dataset = IPCDataset(processed_dir=PROCESSED_DIR, mode='test',noise_scale=0.0)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        if args.arch == 'standard':
            model = CustomMeshGraphNet(node_in_dim=7, edge_in_dim=7, output_dim=3, hidden_dim=128, num_processor_layers=15, mlp_class=MLP)
        elif args.arch == 'cauchy':
            model = CustomMeshGraphNet(node_in_dim=7, edge_in_dim=7, output_dim=3, hidden_dim=128, num_processor_layers=5, mlp_class=CauchyMLP)
        else:
            raise ValueError("Невідома архітектура!")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        stats = train_dataset.stats
        
        save_dir = f"checkpoints/{args.exp_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        history = {'train_mse': [], 'test_mse': []}

        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
            
            train_loss, train_mse, train_bar = run_epoch_multistep(model, train_loader, optimizer, device, stats, is_train=True)
            test_loss, test_mse, test_bar = run_epoch_multistep(model, test_loader, optimizer, device, stats, is_train=False)
            
            history['train_mse'].append(train_mse)
            history['test_mse'].append(test_mse)
            
            print(f"📈 [{args.exp_name}] Epoch {epoch+1}")
            print(f"   Train | MSE: {train_mse:.6f} | Barrier: {train_bar:.6f}")
            print(f"   Test  | MSE: {test_mse:.6f} | Barrier: {test_bar:.6f}")
            
            if test_mse < best_loss:
                best_loss = test_mse
                patience_counter = 0
                torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
            else:
                patience_counter += 1

            if (epoch + 1) % args.notify_every == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(history['train_mse'], label='Train MSE', color='blue')
                plt.plot(history['test_mse'], label='Test MSE', color='orange')
                plt.yscale('log') 
                plt.title(f"MSE Progress: {args.exp_name}")
                plt.xlabel("Epoch")
                plt.ylabel("MSE (Log Scale)")
                plt.legend()
                plt.grid(True)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                caption = (f"📊 *{args.exp_name}* | Епоха: {epoch+1}\n"
                           f"📉 Train MSE: {train_mse:.6f} (Bar: {train_bar:.4f})\n"
                           f"📈 Test MSE: {test_mse:.6f} (Bar: {test_bar:.4f})")
                
                send_tg_image(args.tg_token, args.tg_chat, buf, caption)
                plt.close()

            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch+1}.pt")

            if patience_counter >= args.patience:
                print(f"🛑 Early Stopping! Тренування зупинено на епосі {epoch+1}.")
                send_tg_message(args.tg_token, args.tg_chat, f"🛑 *Early Stopping: {args.exp_name}*\nЗупинено на епосі {epoch+1}.\nНайкращий Test Loss: {best_loss:.6f}")
                break
                
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"КРАШ:\n{error_trace}")
        
        safe_trace = error_trace[-800:].replace('`', "'") 
        msg = f"❌ *КРАШ ЕКСПЕРИМЕНТУ: {args.exp_name}*\n\n*Тип помилки:* `{type(e).__name__}`\n*Деталі:* `{str(e)}`\n\n```python\n{safe_trace}\n```"
        send_tg_message(args.tg_token, args.tg_chat, msg)
        raise e 

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Запуск тренування MeshGraphNet з Early Stopping та TG")
    parser.add_argument('--arch', type=str, required=True, choices=['standard', 'cauchy'])
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--tg_token', type=str, default="")
    parser.add_argument('--tg_chat', type=str, default="")
    parser.add_argument('--notify_every', type=int, default=5)
    args = parser.parse_args()
    
    train(args)