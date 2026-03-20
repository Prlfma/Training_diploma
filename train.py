import os
import argparse
import requests
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import IPCDataset
from model import CustomMeshGraphNet, CauchyMLP, MLP

# --- НАЛАШТУВАННЯ ---
PROCESSED_DIR = "./dataset_processed_final"
BATCH_SIZE = 2
EPOCHS = 200
LR = 1e-4
BARRIER_MARGIN = 0.001
BARRIER_WEIGHT = 1000.0

# --- ФУНКЦІЯ TELEGRAM ---
def send_tg_message(token, chat_id, text):
    """Відправляє повідомлення в Telegram, якщо задані токен та ID."""
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"⚠️ Помилка відправки в Telegram: {e}")

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Запуск експерименту '{args.exp_name}' на: {device}")
    
    send_tg_message(args.tg_token, args.tg_chat, f"🚀 *Старт тренування: {args.exp_name}*\nАрхітектура: {args.arch}\nDevice: {device}")

    train_dataset = IPCDataset(processed_dir=PROCESSED_DIR, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    if args.arch == 'standard':
        model = CustomMeshGraphNet(
            node_in_dim=7, edge_in_dim=7, output_dim=3, 
            hidden_dim=128, num_processor_layers=15, 
            mlp_class=MLP 
        )
    elif args.arch == 'cauchy':
        model = CustomMeshGraphNet(
            node_in_dim=7, edge_in_dim=7, output_dim=3, 
            hidden_dim=128, num_processor_layers=15, 
            mlp_class=CauchyMLP 
        )
    else:
        raise ValueError("Невідома архітектура!")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    stats = train_dataset.stats
    
    save_dir = f"checkpoints/{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # --- ЗМІННІ ДЛЯ EARLY STOPPING ---
    best_loss = float('inf')
    patience_counter = 0

    # --- ЦИКЛ ТРЕНУВАННЯ ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"[{args.exp_name}] Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred_accel_norm = model(batch)
            
            # MSE
            dynamic_mask = (batch.node_type.squeeze() == 0)
            pred_accel_dyn = pred_accel_norm[dynamic_mask]
            target_accel_dyn = batch.y[dynamic_mask]
            loss_mse = F.mse_loss(pred_accel_dyn, target_accel_dyn)
            
            # Barrier
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'MSE': f"{loss_mse.item():.4f}", 'Barrier': f"{loss_barrier.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"📈 [{args.exp_name}] Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")
        
        # --- EARLY STOPPING ЛОГІКА ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Завжди зберігаємо найкращу модель!
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
        else:
            patience_counter += 1
            print(f"⚠️ Loss не покращується {patience_counter}/{args.patience} епох.")

        # --- ТЕЛЕГРАМ СПОВІЩЕННЯ КОЖНІ N ЕПОХ ---
        if (epoch + 1) % args.notify_every == 0:
            msg = f"📊 *Експеримент:* {args.exp_name}\n" \
                  f"🔄 *Епоха:* {epoch+1}/{EPOCHS}\n" \
                  f"📉 *Loss:* {avg_loss:.6f}\n" \
                  f"🏆 *Best Loss:* {best_loss:.6f}"
            send_tg_message(args.tg_token, args.tg_chat, msg)

        # Регулярне збереження про всяк випадок
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/epoch_{epoch+1}.pt")

        # --- ПЕРЕВІРКА EARLY STOPPING ---
        if patience_counter >= args.patience:
            print(f"🛑 Early Stopping! Тренування зупинено на епосі {epoch+1}.")
            send_tg_message(args.tg_token, args.tg_chat, f"🛑 *Early Stopping: {args.exp_name}*\nЗупинено на епосі {epoch+1}.\nНайкращий Loss: {best_loss:.6f}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск тренування MeshGraphNet з Early Stopping та TG")
    parser.add_argument('--arch', type=str, required=True, choices=['standard', 'smooth', 'attention'], help="Яку архітектуру використовувати")
    parser.add_argument('--exp_name', type=str, required=True, help="Назва експерименту (для папки збереження)")
    
    # Нові аргументи для TG та Early Stopping
    parser.add_argument('--patience', type=int, default=15, help="Кількість епох без покращення до зупинки")
    parser.add_argument('--tg_token', type=str, default="", help="Telegram Bot Token")
    parser.add_argument('--tg_chat', type=str, default="", help="Telegram Chat ID")
    parser.add_argument('--notify_every', type=int, default=5, help="Відправляти повідомлення в TG кожні N епох")
    
    args = parser.parse_args()
    
    # Оскільки requests потрібен, гарантуємо, що він є
    try:
        import requests
    except ImportError:
        print("Встанови модуль requests: uv pip install requests")
        exit(1)
        
    train(args)