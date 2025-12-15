import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
import json
import time
from pathlib import Path
from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import csv

# ==========================================
# 1. UTILS
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ==========================================
# 2. MODEL (Updated for dynamic Dropout)
# ==========================================
class SimpleTrafficCNN(nn.Module):
    def __init__(self, num_classes=43, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate), # Tu wchodzi parametr z Optuny
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 3. TRAIN & EVAL LOOPS
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

# ==========================================
# 4. LOGGING
# ==========================================
class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        first = not os.path.exists(path)
        self.f = open(path, 'a', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if first:
            self.writer.writeheader()
    def log(self, row: dict):
        self.writer.writerow(row)
        self.f.flush()
    def close(self):
        self.f.close()

def plot_training_curves(csv_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acc_curves.png'))
    plt.close()

    print(f"Saved training plots to: {save_dir}")

# ==========================================
# 5. MAIN
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='path to data folder')
    parser.add_argument('--workdir', type=str, default='workdir/final_run', help='output folder')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    
    # --- KONFIGURACJA: BEST PARAMS FROM OPTUNA ---
    # Tutaj wstawilem Twoje wyniki na sztywno, zeby "ładnie spięło"
    cfg = {
        'lr': 0.003633534722941957,
        'batch_size': 64,
        'dropout': 0.4463433172678807,
        'weight_decay': 3.7785732030838055e-06,
        'epochs': 30,         # Standard dla finalnego treningu
        'img_size': 64,
        'num_classes': 43,
        'save_every': 5,
        'lr_step': 15,
        'lr_gamma': 0.1,
        'data_root': args.data_root
    }
    
    print(">>> Starting Final Training with Optuna Best Params:")
    print(json.dumps(cfg, indent=2))

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'checkpoints').mkdir(exist_ok=True)
    (workdir / 'tb').mkdir(exist_ok=True)
    
    save_json(cfg, workdir / 'hyperparams.json')
    tb_writer = SummaryWriter(log_dir=str(workdir / 'tb'))
    device = torch.device(args.device)

    # Transformacje
    img_size = cfg['img_size']
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8,1.0)),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Dane
    data_root = Path(cfg['data_root'])
    train_ds = ImageFolder(data_root / 'train' / 'GTSRB', transform=train_tf)
    val_ds = ImageFolder(data_root / 'val' / 'GTSRB', transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model (z dynamicznym dropoutem)
    model = SimpleTrafficCNN(num_classes=cfg['num_classes'], dropout_rate=cfg['dropout'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['lr_step'], gamma=cfg['lr_gamma'])

    csv_logger = CSVLogger(str(workdir / 'train_log.csv'),
                           ['epoch','train_loss','train_acc','val_loss','val_acc','lr','time'])

    best_val_loss = float('inf')
    best_val_acc = 0.0
    start_time = time.time()
    epochs = cfg['epochs']

    for epoch in range(1, epochs+1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - ep_start

        tb_writer.add_scalar('loss/train', train_loss, epoch)
        tb_writer.add_scalar('loss/val', val_loss, epoch)
        tb_writer.add_scalar('acc/train', train_acc, epoch)
        tb_writer.add_scalar('acc/val', val_acc, epoch)
        tb_writer.add_scalar('lr', lr, epoch)

        print(f"Epoch {epoch}/{epochs} | T.Loss: {train_loss:.4f} Acc: {train_acc:.4f} | V.Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")

        csv_logger.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr,
            'time': time.time() - start_time
        })

        # Zapisywanie Checkpointow
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'cfg': cfg
        }
        
        # Zapisz "Best Model" na podstawie accuracy (lub loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, workdir / 'checkpoints' / 'best_model.pth')
            print(f"  --> New Best Model (Acc: {val_acc:.4f}) saved!")
        
        # Zapisz okresowy
        if epoch % cfg['save_every'] == 0:
             torch.save(ckpt, workdir / 'checkpoints' / f'ckpt_epoch{epoch}.pth')

        scheduler.step()

    csv_logger.close()
    tb_writer.close()
    
    # Podsumowanie i wykresy
    summary = {
        'final_epoch': epoch,
        'best_val_acc': best_val_acc,
        'total_time_s': time.time() - start_time
    }
    save_json(summary, workdir / 'summary.json')
    
    plot_dir = workdir / 'plots'
    plot_training_curves(str(workdir / 'train_log.csv'), str(plot_dir))

    print("\nTraining Finished!")
    print(f"Best Model saved to: {workdir / 'checkpoints' / 'best_model.pth'}")
    print(f"Plots saved to: {plot_dir}")

if __name__ == '__main__':
    main()
