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

# Utilities
def set_seed(seed=42): #ustawianie ziarna losowści
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_yaml(obj, path): #zapisywanie do pliku Yaml
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def save_json(obj, path): #zapisywanie obiektu do JSON
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

class SimpleTrafficCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
        # konwolucja 3 kanały->32 filtry, kernel 3x3,normalizacja, nieliniowość, pooling 2x2
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            #64 i 128 kanały
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #warstwa gęsta, nielinowość, dropout 40%
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x): #przepływ danych
        x = self.features(x)
        x = self.classifier(x)
        return x

#trenowanie jednej epoki
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

#ocena jednej epoki
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

#logowanie wyników do pliku
class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        first = not os.path.exists(path)
        self.f = open(path, 'a', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if first:
            self.writer.writeheader()
    #dopisywanie jednego wiersza, po tym flush
    def log(self, row: dict):
        self.writer.writerow(row)
        self.f.flush()
    #zamkniecie pliku
    def close(self):
        self.f.close()

#argumenty
def parse_args():
    parser = argparse.ArgumentParser()
    #ścieżka do pliku Yaml
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config yaml')
    #miejsce zapisu wyniku treningu
    parser.add_argument('--workdir', type=str, default='workdir', help='output folder for logs/checkpoints')
    #sanity check
    parser.add_argument('--overfit-batch', type=int, default=0, help='if >0 use single batch of given size for both train and val (sanity)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    #jakbyśmy chcieli więcej klas
    parser.add_argument('--num-classes', type=int, default=43)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    #ładowanie configu
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg['_cli'] = vars(args)
    workdir = Path(args.workdir)

    import shutil
    if workdir.exists():
        print(f"Cleaning previous results in: {workdir}")
        for item in workdir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"Could not remove {item}: {e}")
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    #tworzenie folderów
    (workdir / 'checkpoints').mkdir(exist_ok=True)
    (workdir / 'tb').mkdir(exist_ok=True)
    #zapisuje konfig
    cfg_path = workdir / 'used_config.yaml'
    save_yaml(cfg, cfg_path)

    #zapisywanie do jsona
    save_json(cfg, workdir / 'hyperparams.json')

    #writer logów
    tb_writer = SummaryWriter(log_dir=str(workdir / 'tb'))

    device = torch.device(args.device)

    #transformacje danych
    img_size = cfg.get('img_size', 64)
    train_tf = transforms.Compose([ #transform do treningu
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8,1.0)),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([ #transform do walidacji
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    data_root = Path(cfg['data_root'])
    train_folder = data_root / 'train' / 'GTSRB'
    val_folder = data_root / 'val' / 'GTSRB'

    if args.overfit_batch > 0: #jeżeli overfit batch(tryb sanity)
        #do każdej epoki ta sama próbka danych by przeuczyć model
        ds = ImageFolder(train_folder, transform=train_tf)
        if len(ds) == 0:
            raise RuntimeError('Train folder empty; cannot overfit.')
        loader_tmp = DataLoader(ds, batch_size=args.overfit_batch, shuffle=True, num_workers=2)
        ds = ImageFolder(train_folder, transform=train_tf)
        print("Original ds len:", len(ds))

        from collections import Counter
        labels = [y for _, y in ds.samples]
        print("Label counts (sample):", Counter(labels))

        loader_tmp = DataLoader(ds, batch_size=args.overfit_batch, shuffle=True, num_workers=0)
        batch_imgs, batch_targets = next(iter(loader_tmp))
        print("batch_imgs.shape:", batch_imgs.shape)  
        print("batch_targets.shape:", batch_targets.shape)
        print("unique labels in batch:", batch_targets.unique().tolist())
        print("batch_targets:", batch_targets.tolist())

        batch_imgs, batch_targets = next(iter(loader_tmp))

        class SingleBatchDataset(Dataset):
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return batch_imgs, batch_targets
        train_ds = SingleBatchDataset()
        val_ds = SingleBatchDataset()
        train_loader = DataLoader(train_ds, batch_size=None, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=None, shuffle=False)
    else:
        train_ds = ImageFolder(train_folder, transform=train_tf)
        val_ds = ImageFolder(val_folder, transform=val_tf)
        train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size',32), shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=cfg.get('batch_size',32), shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    # model
    model = SimpleTrafficCNN(num_classes=args.num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3), weight_decay=cfg.get('weight_decay',1e-6))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get('lr_step',10), gamma=cfg.get('lr_gamma',0.1))

    #dobranie specjalnych parametrów do sanity check
    if args.overfit_batch > 0:
        print(">>> Overfit mode detected: applying overfit-friendly hyperparams")
        for g in optimizer.param_groups:
            g['weight_decay'] = 0.0       #bez regularyzacji
            g['lr'] = 1e-2                # lepszy  lr
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0                 # bez dropoutu
        scheduler = None                  
        epochs = 200                      #dużo epok

    #zapisywanie
    csv_logger = CSVLogger(str(workdir / 'train_log.csv'),
                           ['epoch','train_loss','train_acc','val_loss','val_acc','lr','time'])

    best_val_loss = float('inf')
    start_time = time.time()

    # jeśli nie overfit, to bierz z config.yaml, inaczej z patcha
    if args.overfit_batch > 0:
        epochs = 200   # sanity test-dłuższy, żeby dobrze zapamiętać batch
    else:
        epochs = cfg.get('epochs', 30)

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        tb_writer.add_scalar('loss/train', train_loss, epoch)
        tb_writer.add_scalar('loss/val', val_loss, epoch)
        tb_writer.add_scalar('acc/train', train_acc, epoch)
        tb_writer.add_scalar('acc/val', val_acc, epoch)
        tb_writer.add_scalar('lr', lr, epoch)

        csv_logger.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr,
            'time': epoch_time
        })

        # mechanizm checkpointowy
        if epoch % cfg.get('save_every', 5) == 0 or val_loss < best_val_loss:
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

        # szukamy najlepszego checka
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_name = workdir / 'checkpoints' / 'best.pth'
            torch.save(ckpt, best_name)
            print(f"Saved best checkpoint {best_name}")

        if scheduler is not None:
          scheduler.step()

    csv_logger.close()
    tb_writer.close()
    #na koniec krotkie podsumowanie
    summary = {
        'final_epoch': epoch,
        'best_val_loss': best_val_loss,
        'total_time_s': time.time() - start_time
    }
    save_json(summary, workdir / 'summary.json')

    # ploty
    plot_dir = workdir / 'plots'
    plot_training_curves(str(workdir / 'train_log.csv'), str(plot_dir))

    print("Training finished. Summary saved to", workdir / 'summary.json')

def plot_training_curves(csv_path, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Plot 1: Loss
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

    # Plot 2: Accuracy
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

    # Plot 3: Learning Rate
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['lr'], label='Learning Rate', marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_curve.png'))
    plt.close()

    print(f"Saved training plots to: {save_dir}")


if __name__ == '__main__':
    main()
