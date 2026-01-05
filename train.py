import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================
# Configuración
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "Train")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "emotion_classifier.pth")

IMG_SIZE = (224, 224)  # usar tamano nativo de ResNet
NUM_CLASSES = 8
EMOTIONS = ['ira', 'desprecio', 'asco', 'miedo', 'felicidad', 'neutralidad', 'tristeza', 'sorpresa']
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalización ImageNet (RGB)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Rebalanceo y fine-tuning
USE_WEIGHTED_SAMPLER = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 1.5
BASE_LR = 0.001
FINETUNE_LR = 0.0001
WEIGHT_DECAY = 0.01
UNFREEZE_AT_EPOCH = 5  # descongelar backbone para ajuste fino
MINORITY_RATIO = 0.75  # <75% del max se considera minoritaria

# =============================================
# Preprocesamiento RGB (3 canales)
# =============================================
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        return image

    except Exception as e:
        print(f"Error procesando {image_path}: {str(e)}")
        return None

# =============================================
# Dataset con Data Augmentation mejorado
# =============================================
class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False, class_counts=None):
        # Guardar rutas y etiquetas; se cargan en __getitem__ para ahorrar RAM
        self.file_paths = file_paths
        self.y = torch.tensor(labels).long()
        self.augment = augment
        counts = class_counts if class_counts is not None else np.bincount(labels, minlength=NUM_CLASSES)
        self.class_counts = counts
        self.max_count = max(self.class_counts) if len(self.class_counts) > 0 else 0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        y = self.y[idx]

        img = preprocess_image(path)
        if img is None:
            # Imagen corrupta: devolver tensor negro para no romper el batch
            x = torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1])
        else:
            x = torch.tensor(img).permute(2, 0, 1).float()

        minority_ratio = (self.class_counts[y.item()] / (self.max_count + 1e-6)) if self.max_count else 1.0
        is_minority = minority_ratio < MINORITY_RATIO

        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.3:
                x = torch.flip(x, dims=[2])

            # Random rotation (-12 a 12 grados, un poco mas para minoritarias)
            if torch.rand(1) > (0.4 if is_minority else 0.6):
                angle = (torch.rand(1) * 24 - 12).item()
                x = self._rotate(x, angle)

            # Color jitter suave (mas frecuente en minoritarias)
            if torch.rand(1) > (0.35 if is_minority else 0.65):
                x = self._color_jitter(x, strength=0.2 if is_minority else 0.1)

            # Ruido gaussiano ligero para minoritarias
            if is_minority and torch.rand(1) > 0.7:
                x = self._add_gaussian_noise(x, std=0.02)

            # Random erasing (ocluir parte aleatoria)
            if torch.rand(1) > (0.5 if is_minority else 0.7):
                x = self._random_erasing(x)

        return x, y

    def _rotate(self, img, angle):
        angle_rad = angle * np.pi / 180
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)

        grid = torch.nn.functional.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
        rotated = torch.nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=False, padding_mode='border')
        return rotated.squeeze(0)

    def _random_erasing(self, img):
        # Ocultar un rectangulo aleatorio (simula oclusion)
        c, h, w = img.shape
        area = h * w
        target_area = torch.rand(1).item() * 0.2 * area  # max 20% del area

        aspect_ratio = torch.rand(1).item() * 0.5 + 0.5  # 0.5 a 1.0
        eh = int(np.sqrt(target_area * aspect_ratio))
        ew = int(np.sqrt(target_area / aspect_ratio))

        if eh < h and ew < w:
            x1 = torch.randint(0, w - ew, (1,)).item()
            y1 = torch.randint(0, h - eh, (1,)).item()
            img[:, y1:y1+eh, x1:x1+ew] = 0

        return img

    def _color_jitter(self, img, strength=0.1):
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
        img = img * factor
        return torch.clamp(img, -3.0, 3.0)

    def _add_gaussian_noise(self, img, std=0.02):
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, -3.0, 3.0)

# =============================================
# Carga de datos
# =============================================
def load_dataset(dataset_path):
    file_paths = []
    labels = []
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}

    print("\n" + "="*50)
    print("CARGANDO DATOS (RGB)")
    print("="*50)

    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(dataset_path, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"Advertencia: {emotion_dir} no encontrado")
            continue

        print(f"Procesando: {emotion}")
        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            file_paths.append(img_path)
            labels.append(emotion_idx)
            emotion_counts[emotion] += 1

    print("\n" + "="*50)
    print("ESTADÍSTICAS")
    print("="*50)
    total = 0
    for emotion in EMOTIONS:
        count = emotion_counts[emotion]
        total += count
        print(f"{emotion:15}: {count}")
    print(f"TOTAL: {total}")

    return file_paths, np.array(labels)

# =============================================
# Calcular pesos de clase
# =============================================
def calculate_class_weights(y):
    class_counts = np.bincount(y, minlength=NUM_CLASSES)
    total = len(y)
    weights = total / (NUM_CLASSES * class_counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(y):
    class_counts = np.bincount(y, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y]
    return WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        ce_loss = nn.functional.nll_loss(log_probs, targets, weight=self.alpha, reduction='none')
        probs = torch.exp(-ce_loss)
        loss = (1 - probs) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

# =============================================
# Modelo ResNet-18 con capas congeladas
# =============================================
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, freeze_layers=True):
        super(EmotionClassifier, self).__init__()

        # Cargar ResNet-18 preentrenado (3 canales RGB)
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')

        # Congelar solo primeras capas (conv1, bn1, layer1)
        if freeze_layers:
            # Congelar todo excepto layer2, layer3, layer4 y fc
            for name, param in self.resnet.named_parameters():
                if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

        # Reemplazar capa final con más regularización
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def unfreeze_backbone(model: EmotionClassifier):
    for param in model.parameters():
        param.requires_grad = True

# =============================================
# Entrenamiento
# =============================================
def train_model(train_paths, y_train, val_paths, y_val, class_weights, class_counts):
    train_dataset = EmotionDataset(train_paths, y_train, augment=True, class_counts=class_counts)
    val_dataset = EmotionDataset(val_paths, y_val, augment=False, class_counts=class_counts)

    train_sampler = make_weighted_sampler(y_train) if USE_WEIGHTED_SAMPLER else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # Modelo con capas congeladas
    model = EmotionClassifier(NUM_CLASSES, freeze_layers=True).to(DEVICE)

    # Contar parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros entrenables: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    if USE_WEIGHTED_SAMPLER:
        print("Sampler: WeightedRandomSampler activado")
    print(f"Loss: {'Focal' if USE_FOCAL_LOSS else 'CrossEntropy'}")

    # Loss con pesos de clase
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights.to(DEVICE), gamma=FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Optimizador con weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience = 10
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_acc': []}
    unfrozen = False

    print("\n" + "="*50)
    print("ENTRENAMIENTO (Transfer Learning Optimizado)")
    print("="*50)
    print(f"Dispositivo: {DEVICE}")
    print(f"Modelo: ResNet-18 (layer1 congelado, layer2-4 entrenables)")
    print(f"Entrenamiento: {len(train_paths)} | Validación: {len(val_paths)}")

    for epoch in range(EPOCHS):
        if not unfrozen and epoch + 1 == UNFREEZE_AT_EPOCH:
            print(f"\n>> Descongelando backbone en epoch {epoch+1} y reduciendo LR para fine-tuning")
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=FINETUNE_LR,
                weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6
            )
            unfrozen = True

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
              f"Loss: {val_loss:.4f} | LR: {lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >> Guardado ({val_acc:.2f}%)")

        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping en epoch {epoch+1}")
            break

    # Guardar último modelo
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "last_model.pth"))

    # Gráficas
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Perdida')
    plt.xlabel('Epoca')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Precision')
    plt.xlabel('Epoca')
    plt.ylabel('%')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history['val_acc'], 'g-')
    plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Mejor: {best_val_acc:.2f}%')
    plt.title('Validacion')
    plt.xlabel('Epoca')
    plt.ylabel('%')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
    plt.show()

    print("\n" + "="*50)
    print(f"COMPLETADO - Mejor: {best_val_acc:.2f}% (epoch {best_epoch+1})")
    print("="*50)

    return model

# =============================================
# Distribución de clases
# =============================================
def plot_class_distribution(y):
    class_counts = [np.sum(y == i) for i in range(len(EMOTIONS))]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(EMOTIONS, class_counts, color='skyblue')
    plt.title('Distribución de clases')
    plt.xlabel('Emoción')
    plt.ylabel('Imágenes')
    plt.xticks(rotation=45)

    max_count = max(class_counts)
    for bar, count in zip(bars, class_counts):
        if count / max_count < 0.5:
            bar.set_color('salmon')
        elif count / max_count < 0.7:
            bar.set_color('lightyellow')
        plt.text(bar.get_x() + bar.get_width()/2, count + 5, str(count), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'class_distribution.png'))
    plt.show()

# =============================================
# Main
# =============================================
if __name__ == "__main__":
    print("="*50)
    print("CLASIFICADOR DE EMOCIONES - ResNet-18 RGB")
    print("="*50)

    paths, y = load_dataset(TRAIN_PATH)

    if len(paths) == 0:
        print("Error: No se encontraron imágenes")
        exit()

    plot_class_distribution(y)

    class_weights = calculate_class_weights(y)
    print("\nPesos de clase:")
    for e, w in zip(EMOTIONS, class_weights):
        print(f"  {e}: {w:.3f}")

    train_paths, val_paths, y_train, y_val = train_test_split(
        paths, y, test_size=0.2, random_state=42, stratify=y
    )

    train_class_counts = np.bincount(y_train, minlength=NUM_CLASSES)

    model = train_model(train_paths, y_train, val_paths, y_val, class_weights, train_class_counts)

    print(f"\nModelo guardado en: {MODEL_SAVE_PATH}")
    print("Ejecuta: python test.py")
