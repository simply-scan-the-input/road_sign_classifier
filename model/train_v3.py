"""
Training script with:
- config YAML or CLI
- TensorBoard logging (persistent to disk)
- CSV logging of epoch metrics
- checkpointing (periodic + best)
- --overfit-batch N mode for sanity check
"""

import argparse
import os
import yaml
import json
import time
from pathlib import Path
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
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------
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

# ---------------------------
# Simple CNN model
# ---------------------------
class SimpleTrafficCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------
# Training / Eval helpers
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

# ---------------------------
# CSV logger
# ---------------------------
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

# ---------------------------
# Main
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config yaml')
    parser.add_argument('--workdir', type=str, default='workdir', help='output folder for logs/checkpoints')
    parser.add_argument('--overfit-batch', type=int, default=0, help='if >0 use single batch for sanity check')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-classes', type=int, default=43)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['_cli'] = vars(args)

    # Directories
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'checkpoints').mkdir(exist_ok=True)
    (workdir / 'tb').mkdir(exist_ok=True)
    save_yaml(cfg, workdir / 'used_config.yaml')
    save_json(cfg, workdir / 'hyperparams.json')

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=str(workdir / 'tb'))

    # Device
    device = torch.device(args.device)

    # Transforms
    img_size = cfg.get('img_size', 64)
    train_tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((img_size,img_size), scale=(0.8,1.0)),
        transforms.ToTensor()
    ])
    val_tf = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])

    # Dataset
    data_root = Path(cfg['data_root'])
    train_folder = data_root / 'train/GTSRB'
    val_folder = data_root / 'val/GTSRB'

    if args.overfit_batch > 0:
        ds = ImageFolder(train_folder, transform=train_tf)
        if len(ds) == 0:
            raise RuntimeError("Train folder empty; cannot overfit.")
        loader_tmp = DataLoader(ds, batch_size=args.overfit_batch, shuffle=True, num_workers=2)
        batch_imgs, batch_targets = next(iter(loader_tmp))

        class SingleBatchDataset(Dataset):
            def __init__(self, imgs, targets):
                self.imgs = imgs
                self.targets = targets
            def __len__(self):
                return self.imgs.size(0)
            def __getitem__(self, idx):
                return self.imgs[idx], self.targets[idx]

        train_ds = SingleBatchDataset(batch_imgs, batch_targets)
        val_ds = SingleBatchDataset(batch_imgs, batch_targets)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    else:
        train_ds = ImageFolder(train_folder, transform=train_tf)
        val_ds = ImageFolder(val_folder, transform=val_tf)
        train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size',32), shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=cfg.get('batch_size',32), shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model, loss, optimizer, scheduler
    model = SimpleTrafficCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr',1e-3), weight_decay=cfg.get('weight_decay',1e-6))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get('lr_step',10), gamma=cfg.get('lr_gamma',0.1))

    # CSV logger
    csv_logger = CSVLogger(workdir / 'train_log.csv',
                           ['epoch','train_loss','train_acc','val_loss','val_acc','lr','time'])

    # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    start_time = time.time()
    epochs = cfg.get('epochs', 30 if args.overfit_batch==0 else 50)
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        # TensorBoard
        tb_writer.add_scalar('loss/train', train_loss, epoch)
        tb_writer.add_scalar('loss/val', val_loss, epoch)
        tb_writer.add_scalar('acc/train', train_acc, epoch)
        tb_writer.add_scalar('acc/val', val_acc, epoch)
        tb_writer.add_scalar('lr', lr, epoch)

        # CSV log
        csv_logger.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr,
            'time': epoch_time
        })

        # Checkpointing
        if epoch % cfg.get('save_every',5)==0 or val_loss < best_val_loss:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'cfg': cfg
            }
            ckpt_name = workdir / 'checkpoints' / f'ckpt_epoch{epoch}.pth'
            torch.save(ckpt, ckpt_name)
            print(f"Saved checkpoint {ckpt_name}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_name = workdir / 'checkpoints' / 'best.pth'
            torch.save(ckpt, best_name)
            print(f"Saved best checkpoint {best_name}")

        scheduler.step()

        if args.overfit_batch > 0 and epoch >= 20:
            break

    csv_logger.close()
    tb_writer.close()

    # Save curves
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(train_losses,label='Train Loss'); axes[0].plot(val_losses,label='Val Loss')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(train_accs,label='Train Acc'); axes[1].plot(val_accs,label='Val Acc')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(workdir / 'training_curves.png')
    plt.show()

    # Summary
    summary = {
        'final_epoch': epoch,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_loss_at_best': train_losses[best_epoch-1],
        'train_acc_at_best': train_accs[best_epoch-1],
        'best_val_acc': val_accs[best_epoch-1],
        'total_time_s': time.time()-start_time,
        'hparams': {
            'lr': cfg.get('lr',1e-3),
            'batch_size': cfg.get('batch_size',32),
            'weight_decay': cfg.get('weight_decay',1e-6),
            'img_size': cfg.get('img_size',64),
            'epochs': epochs
        }
    }
    save_json(summary, workdir / 'summary.json')
    print(f"Training finished. Summary saved to {workdir / 'summary.json'}")

if __name__ == '__main__':
    main()
