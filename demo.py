import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# ==========================================
# 1. SŁOWNIK NAZW ZNAKÓW (GTSRB)
# ==========================================
# To tłumaczy numerek (np. 14) na ludzką nazwę (np. Stop)
GTSRB_CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# ==========================================
# 2. MODEL DEFINITION 
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
# 3. FUNKCJA PRZEWIDUJĄCA
# ==========================================
def predict_image(model, image_path, device, img_size=64):
    # 1. Wczytanie obrazu
    if not os.path.exists(image_path):
        print(f"Błąd: Nie znaleziono pliku: {image_path}")
        sys.exit(1)
        
    image = Image.open(image_path).convert('RGB')

    # 2. Transformacje (takie same jak w treningu, ale bez augmentacji)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # 3. Przygotowanie tensora (dodanie wymiaru batcha: [1, 3, 64, 64])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 4. Predykcja
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        # Używamy Softmax, żeby dostać prawdopodobieństwa (pewność)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return predicted_idx.item(), confidence.item()

# ==========================================
# 4. MAIN DEMO APP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Proste demo klasyfikatora znaków drogowych")
    parser.add_argument("--image", type=str, required=True, help="Ścieżka do pliku obrazka (np. .jpg, .ppm)")
    parser.add_argument("--ckpt", type=str, required=True, help="Ścieżka do modelu best_model.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("\n" + "="*40)
    print("Ładowanie DEMO...")
    print("="*40)

    # 1. Ładowanie modelu
    try:
        checkpoint = torch.load(args.ckpt, map_location=device)
        # Wyciągamy dropout z configu jeśli jest, żeby nie było błędu
        saved_cfg = checkpoint.get('cfg', {})
        dropout_val = saved_cfg.get('dropout', 0.5)
        
        model = SimpleTrafficCNN(num_classes=43, dropout_rate=dropout_val)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        print(f"Model załadowany pomyślnie z: {args.ckpt}")
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        sys.exit(1)

    print(f"Analiza obrazu: {args.image}...")
    
    # 2. Predykcja
    class_id, confidence = predict_image(model, args.image, device)
    
    # 3. Wyniki
    class_name = GTSRB_CLASSES.get(class_id, "Nieznany znak")
    
    print("\n" + "="*40)
    print("WYNIK PREDYKCJI MODELU")
    print("="*40)
    print(f" ID Klasy:   {class_id}")
    print(f" Nazwa Znaku: \033[1m{class_name}\033[0m") # \033[1m to pogrubienie w terminalu
    print(f" Pewność:     {confidence*100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":

    main()


