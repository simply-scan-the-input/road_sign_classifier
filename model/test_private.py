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

# ==========================================
# 1. MODEL DEFINITION (Musi pasować do treningu)
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
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. DATASET (Dla struktury folderów klasowych)
# ==========================================
class PrivateFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = [str(i) for i in range(43)] # Zakładamy 43 klasy jak w GTSRB

        if not self.root_dir.exists():
            raise RuntimeError(f"❌ Folder nie istnieje: {self.root_dir}")

        print(f"🔍 Skanowanie folderu: {self.root_dir}")
        
        # Skanujemy tylko foldery, które wyglądają jak numery klas (00000, 1, 42 itp.)
        count = 0
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                # Próba sparsowania nazwy folderu jako ID klasy
                try:
                    class_id = int(class_dir.name)
                    if class_id < 0 or class_id > 42:
                        continue # Ignorujemy foldery spoza zakresu GTSRB
                except ValueError:
                    continue # Ignorujemy foldery, które nie są liczbami

                # Zbieranie zdjęć
                imgs = [x for x in class_dir.glob("*") if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp']]
                
                for img_path in imgs:
                    self.samples.append((img_path, class_id))
                    count += 1

        if count == 0:
            raise RuntimeError(f"❌ Nie znaleziono żadnych zdjęć w podfolderach (00000, 00001...) w {self.root_dir}")
        
        print(f"✅ Znaleziono łącznie {count} zdjęć w Twoim prywatnym zbiorze.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

# ==========================================
# 3. EVALUATION & PLOTS
# ==========================================
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

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if total > 0 else 0
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

def plot_confusion_matrix(targets, preds, class_names, save_path):
    if len(targets) == 0: return
    # Wymuszamy macierz 43x43 nawet jak w teście nie ma wszystkich klas
    cm = confusion_matrix(targets, preds, labels=range(43))
    plt.figure(figsize=(16,14))
    sns.heatmap(cm, cmap="Blues", annot=False) # Annot=False bo przy 43 klasach będzie nieczytelne
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix (Private Dataset)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_per_class_accuracy(targets, preds, class_names, save_path):
    if len(targets) == 0: return {}
    
    # Liczymy dokładność tylko dla klas, które faktycznie wystąpiły w zbiorze prywatnym
    unique_targets = np.unique(targets)
    per_class_stats = {}
    
    accuracies = []
    labels = []

    for cls_id in range(43):
        if cls_id in unique_targets:
            mask = (targets == cls_id)
            acc = (preds[mask] == targets[mask]).mean()
            per_class_stats[class_names[cls_id]] = float(acc)
            accuracies.append(acc)
            labels.append(cls_id)
        else:
            # Klasy nieobecne w teście oznaczamy jako -1 lub pomijamy
            pass

    if not accuracies: return {}

    plt.figure(figsize=(12, 6))
    plt.bar(labels, accuracies, color='green')
    plt.xlabel("Class ID")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy (Only classes present in Private Set)")
    plt.xticks(labels)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return per_class_stats

# ==========================================
# 4. MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to 'fotki_znaków' root folder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Ustalanie folderu wyjściowego
    ckpt_path = Path(args.ckpt)
    # Zapiszemy wyniki w workdir/final_run/test_private
    if ckpt_path.parent.name == 'checkpoints':
        base_dir = ckpt_path.parent.parent
    else:
        base_dir = ckpt_path.parent
        
    out_dir = base_dir / "test_private"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Private Dataset Evaluation ---")
    print(f"Data Source: {args.data_dir}")
    print(f"Output Dir:  {out_dir}")

    # Transforms (Tylko resize i toTensor)
    img_size = 64 # Zakładamy 64, bo tak trenowaliśmy
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # Dataset
    try:
        test_ds = PrivateFolderDataset(args.data_dir, transform=tf)
    except Exception as e:
        print(str(e))
        return

    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Model Load
    print(f"Loading model: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    saved_cfg = checkpoint.get('cfg', {})
    dropout_val = saved_cfg.get('dropout', 0.5)

    model = SimpleTrafficCNN(num_classes=43, dropout_rate=dropout_val).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Run Eval
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, targets = eval_model(model, loader, criterion, device)

    print(f"\n" + "="*30)
    print(f"🎯 PRIVATE TEST RESULTS")
    print(f"="*30)
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(f"Loss:     {test_loss:.4f}")
    print(f"="*30)

    # Plots & Save
    per_class_stats = plot_per_class_accuracy(targets, preds, test_ds.classes, plot_dir / "private_accuracy_per_class.png")
    plot_confusion_matrix(targets, preds, test_ds.classes, plot_dir / "private_confusion_matrix.png")

    results = {
        "private_test_accuracy": float(test_acc),
        "private_test_loss": float(test_loss),
        "per_class_accuracy": per_class_stats
    }
    
    with open(out_dir / "private_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {out_dir}")

if __name__ == "__main__":
    main()