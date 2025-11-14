# test_model.py

import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image

# -----------------------------------------------------------
# MODEL DEFINITION (must match train.py)
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# CUSTOM TEST DATASET (skips empty class folders)
# -----------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        root = Path(root)
        for idx, class_dir in enumerate(sorted(root.iterdir())):
            if class_dir.is_dir():
                imgs = [img for img in class_dir.glob("*") if img.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.ppm','.pgm','.tif','.tiff','.webp']]
                if len(imgs) == 0:
                    print(f"Skipping empty class folder: {class_dir.name}")
                    continue
                self.samples.extend([(img, idx) for img in imgs])
                self.class_to_idx[class_dir.name] = idx
                self.classes.append(class_dir.name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

# -----------------------------------------------------------
# EVALUATION FUNCTION
# -----------------------------------------------------------
def eval_model(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

# -----------------------------------------------------------
# PLOTS
# -----------------------------------------------------------
def plot_per_class_accuracy(targets, preds, class_names, save_path):
    correct = (preds == targets)
    per_class_acc = [
        correct[targets == i].mean() if np.any(targets == i) else 0
        for i in range(len(class_names))
    ]
    plt.figure(figsize=(12,6))
    plt.bar(range(len(class_names)), per_class_acc)
    plt.xlabel("Class ID")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return per_class_acc

def plot_confusion_matrix(targets, preds, class_names, save_path):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(14,12))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--num_classes", type=int, default=43)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # output directories
    workdir = Path(args.ckpt).parents[1]       # assumes checkpoint in workdir/checkpoints/
    test_dir = workdir / "test"
    plot_dir = test_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # transforms
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # test dataset
    test_folder = Path(args.data_root) / "test" / "private"
    test_ds = TestDataset(test_folder, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    class_names = test_ds.classes

    # model
    model = SimpleTrafficCNN(num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, targets = eval_model(model, test_loader, criterion, device)

    # plots
    per_class_acc = plot_per_class_accuracy(targets, preds, class_names, plot_dir / "per_class_accuracy.png")
    plot_confusion_matrix(targets, preds, class_names, plot_dir / "confusion_matrix.png")

    # save json results
    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "per_class_accuracy": {class_names[i]: float(per_class_acc[i]) for i in range(len(class_names))}
    }
    with open(test_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved test results to: {test_dir}")
    print(f"Saved plots to: {plot_dir}")

if __name__ == "__main__":
    main()
