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
import pandas as pd
import sys

# -----------------------------------------------------------
# MODEL DEFINITION
# -----------------------------------------------------------
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
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------
# DATASET (NAPRAWIONA OBSŁUGA CSV - IGNORUJE ZŁE PLIKI)
# -----------------------------------------------------------
class GTSRBTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.targets = []
        self.classes = [str(i) for i in range(43)] 

        # 1. SZUKANIE WŁAŚCIWEGO PLIKU CSV
        # Lista miejsc, gdzie szukamy pliku "GT-final_test.csv" (tego dobrego)
        potential_files = [
            self.root_dir / "GT-final_test.csv",           # W folderze Images
            self.root_dir.parent / "GT-final_test.csv"     # W folderze wyżej (Final_Test)
        ]

        csv_path = None
        for p in potential_files:
            if p.exists():
                csv_path = p
                break
        
        # Jeśli nie znaleziono po nazwie, szukamy jakiegokolwiek CSV, ALE ignorujemy te z ".test." w nazwie
        if csv_path is None:
            all_csvs = list(self.root_dir.glob("*.csv")) + list(self.root_dir.parent.glob("*.csv"))
            # Filtrujemy śmieciowe pliki (np. GT-final_test.test.csv)
            valid_csvs = [f for f in all_csvs if ".test." not in f.name]
            
            if valid_csvs:
                csv_path = valid_csvs[0]

        if csv_path is None:
            raise RuntimeError(
                f"❌ Nie znaleziono pliku 'GT-final_test.csv'!\n"
                f"Szukano w: {self.root_dir} oraz katalogu wyżej.\n"
                f"Upewnij się, że masz plik z odpowiedziami (nie ten pusty .test.csv!)."
            )
        
        print(f"✅ Loading labels from CORRECT file: {csv_path}")

        # 2. WCZYTYWANIE CSV
        try:
            # Próba wczytania średnikiem (GTSRB standard)
            df = pd.read_csv(csv_path, sep=';')
            df.columns = df.columns.str.strip() # Usuń spacje z nazw kolumn
            
            # Fallback na przecinek
            if 'ClassId' not in df.columns:
                df = pd.read_csv(csv_path, sep=',')
                df.columns = df.columns.str.strip()

            if 'ClassId' not in df.columns:
                 raise KeyError(f"Nadal brak kolumny 'ClassId'. Dostępne kolumny: {list(df.columns)}")

        except Exception as e:
            print(f"❌ Critical CSV Error w pliku {csv_path.name}: {e}")
            sys.exit(1)
        
        # 3. Parsowanie
        valid_count = 0
        for index, row in df.iterrows():
            img_filename = row['Filename']
            class_id = int(row['ClassId'])
            
            # Szukamy obrazka w folderze root lub obok CSV
            potential_img_paths = [
                self.root_dir / img_filename,
                csv_path.parent / "Images" / img_filename, # Jeśli CSV jest w Final_Test, a zdjęcia w Images
                csv_path.parent / img_filename
            ]

            found = False
            for p in potential_img_paths:
                if p.exists():
                    self.images.append(p)
                    self.targets.append(class_id)
                    found = True
                    valid_count += 1
                    break
        
        if valid_count == 0:
             raise RuntimeError(f"CSV wczytany, ale nie znaleziono obrazków! Sprawdź ścieżki.")
        
        print(f"✅ Successfully loaded {valid_count} images mapped to classes.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target
# -----------------------------------------------------------
# EVALUATION & PLOTS
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

    if total == 0:
        return 0.0, 0.0, np.array([]), np.array([])

    avg_loss = total_loss / total
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

def plot_per_class_accuracy(targets, preds, class_names, save_path):
    if len(targets) == 0: return []
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
    if len(targets) == 0: return
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
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to folder with .ppm images")
    parser.add_argument("--num_classes", type=int, default=43)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    
    # Obsługa ścieżki wyjściowej
    if ckpt_path.parent.name == 'checkpoints':
        workdir = ckpt_path.parent.parent
    else:
        workdir = ckpt_path.parent
    
    test_results_dir = workdir / "test_results"
    plot_dir = test_results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- GTSRB Evaluation ---")
    
    # Transforms
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # Dataset
    try:
        test_ds = GTSRBTestDataset(args.test_dir, transform=tf)
    except Exception as e:
        print(f"\n❌ ERROR initializing dataset: {e}")
        return

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    class_names = test_ds.classes

    # Model
    print(f"Loading model from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    saved_cfg = checkpoint.get('cfg', {})
    dropout_val = saved_cfg.get('dropout', 0.5)
    
    model = SimpleTrafficCNN(num_classes=args.num_classes, dropout_rate=dropout_val).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Eval
    criterion = nn.CrossEntropyLoss()
    print("Starting evaluation (this may take a moment)...")
    test_loss, test_acc, preds, targets = eval_model(model, test_loader, criterion, device)

    print(f"\n" + "="*30)
    print(f"🎯 RESULTS")
    print(f"="*30)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"="*30)

    # Save
    per_class_acc = plot_per_class_accuracy(targets, preds, class_names, plot_dir / "per_class_accuracy.png")
    plot_confusion_matrix(targets, preds, class_names, plot_dir / "confusion_matrix.png")

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "per_class_accuracy": {class_names[i]: float(per_class_acc[i]) for i in range(len(class_names))}
    }
    with open(test_results_dir / "final_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {test_results_dir}")

if __name__ == "__main__":
    main()