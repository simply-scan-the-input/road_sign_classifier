# pip install optuna torch torchvision tensorboard pyyaml pandas matplotlib


from pathlib import Path
import time
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# =========================
# Reuse: model + train utils
# =========================

class SimpleTrafficCNN(nn.Module):
    def __init__(self, num_classes=43, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


# =========================
# Data
# =========================

def get_loaders(data_root: Path, batch_size: int, img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),    # augumentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolder(data_root / "train" / "GTSRB", transform=train_tf)
    val_ds = ImageFolder(data_root / "val" / "GTSRB", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


# =========================
# Early Stopping (manual)
# =========================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, value):
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# =========================
# Optuna objective (V2)
# =========================

def objective(trial: optuna.Trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    train_loader, val_loader = get_loaders(
        Path("data"), batch_size=batch_size, img_size=64
    )

    model = SimpleTrafficCNN(num_classes=43, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(f"logs/optuna/trial_{trial.number}")
    early_stop = EarlyStopping(patience=5)

    max_epochs = 30
    best_val_acc = 0.0

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        trial.report(val_acc, epoch)

        if trial.should_prune():  # Optuna PRUNER (wbudowany)
            raise optuna.TrialPruned()

        if early_stop.step(val_loss):  # manual early stopping
            break

        best_val_acc = max(best_val_acc, val_acc)

    writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "dropout": dropout, "weight_decay": weight_decay},
        {"val_acc": best_val_acc},
    )
    writer.close()

    return best_val_acc


# =========================
# Main
# =========================

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    study.optimize(objective, n_trials=25)

    print("Best params:")
    print(study.best_params)
