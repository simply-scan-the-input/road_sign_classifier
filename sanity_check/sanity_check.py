"""
sanity_check.py

Quick overfitting (sanity) test for SimpleTrafficCNN.
Verifies the model can memorize a small batch.
Saves TensorBoard, CSV, and plots under --workdir.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import yaml
import argparse
from pathlib import Path
import time
import random
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model definition (same as train.py)
# -----------------------------
class SimpleTrafficCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
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


# -----------------------------
# Training helpers
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to YAML config')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num-classes', type=int, default=43)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workdir', type=str, default='./workdir/overfit')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'checkpoints').mkdir(exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(workdir / 'tb'))

    data_root = Path(cfg['data_root'])
    train_folder = data_root / 'train/GTSRB'
    img_size = cfg.get('img_size', 64)

    # deterministic transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # sample one batch
    full_ds = ImageFolder(train_folder, transform=transform)
    loader_tmp = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)
    batch_imgs, batch_targets = next(iter(loader_tmp))

    # single-batch dataset
    class SingleBatchDataset(Dataset):
        def __init__(self, imgs, targets):
            self.imgs, self.targets = imgs, targets
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            return self.imgs[idx], self.targets[idx]

    train_ds = SingleBatchDataset(batch_imgs, batch_targets)
    val_ds = SingleBatchDataset(batch_imgs, batch_targets)
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    device = torch.device(args.device)
    model = SimpleTrafficCNN(num_classes=args.num_classes).to(device)

    # disable dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    csv_path = workdir / 'overfit_log.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    writer.writeheader()

    print(f"Running sanity check on {args.batch_size} samples for {args.epochs} epochs...")
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        best_acc = max(best_acc, val_acc)

        writer.writerow({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                         'val_loss': val_loss, 'val_acc': val_acc})
        csv_file.flush()

        tb_writer.add_scalar('loss/train', train_loss, epoch)
        tb_writer.add_scalar('loss/val', val_loss, epoch)
        tb_writer.add_scalar('acc/train', train_acc, epoch)
        tb_writer.add_scalar('acc/val', val_acc, epoch)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, workdir / 'checkpoints' / f'ckpt_{epoch:03d}.pth')

        # if val_acc > 0.999:
        #     print(f"Overfit achieved (val_acc={val_acc:.4f}) at epoch {epoch}")
        #     break

    tb_writer.close()
    csv_file.close()

    # save plot
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(workdir / 'loss_curve.png')

    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(workdir / 'acc_curve.png')

    # summary
    summary = {
        'best_val_acc': best_acc,
        'epochs_run': len(history['train_loss']),
        'total_time_s': round(time.time() - start_time, 2)
    }
    with open(workdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Finished sanity check. Logs, CSV, plots, and TensorBoard saved to: {workdir}")


if __name__ == "__main__":
    main()