import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================
# Configuración
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "Train")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "emotion_classifier.pth")

IMG_SIZE = (96, 96)
NUM_CLASSES = 8
EMOTIONS = ['ira', 'desprecio', 'asco', 'miedo', 'felicidad', 'neutralidad', 'tristeza', 'sorpresa']
BATCH_SIZE = 32
EPOCHS = 100  # Aumentado, early stopping detendrá si es necesario
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================
# Preprocesamiento optimizado para 96x96
# =============================================
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Advertencia: No se pudo cargar {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        normalized = enhanced.astype('float32') / 255.0

        return np.expand_dims(normalized, axis=-1)

    except Exception as e:
        print(f"Error procesando {image_path}: {str(e)}")
        return None

# =============================================
# Dataset con Data Augmentation
# =============================================
class EmotionDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X).permute(0, 3, 1, 2).float()
        self.y = torch.tensor(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])

            # Random rotation (-15 to 15 degrees)
            if torch.rand(1) > 0.5:
                angle = (torch.rand(1) * 30 - 15).item()
                x = self._rotate(x, angle)

            # Random brightness adjustment
            if torch.rand(1) > 0.5:
                factor = 0.8 + torch.rand(1).item() * 0.4  # 0.8 to 1.2
                x = torch.clamp(x * factor, 0, 1)

        return x, y

    def _rotate(self, img, angle):
        # Simple rotation using grid_sample
        angle_rad = angle * np.pi / 180
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)

        grid = torch.nn.functional.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
        rotated = torch.nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=False, padding_mode='border')
        return rotated.squeeze(0)

# =============================================
# Carga de datos con estadísticas
# =============================================
def load_dataset(dataset_path):
    X = []
    y = []
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    problematic_images = []

    print("\n" + "="*50)
    print("INICIO DE CARGA DE DATOS")
    print("="*50)

    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(dataset_path, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"Advertencia: Directorio no encontrado - {emotion_dir}")
            continue

        print(f"Procesando: {emotion}")
        image_files = os.listdir(emotion_dir)

        for img_file in image_files:
            img_path = os.path.join(emotion_dir, img_file)
            processed_img = preprocess_image(img_path)

            if processed_img is not None:
                X.append(processed_img)
                y.append(emotion_idx)
                emotion_counts[emotion] += 1
            else:
                problematic_images.append(img_path)

    print("\n" + "="*50)
    print("ESTADÍSTICAS DE CARGA")
    print("="*50)
    total_loaded = 0
    for emotion in EMOTIONS:
        count = emotion_counts[emotion]
        total_loaded += count
        print(f"{emotion:15}: {count} imágenes")

    print("\n" + "="*50)
    print(f"TOTAL IMÁGENES CARGADAS: {total_loaded}")

    if problematic_images:
        print(f"\nIMÁGENES PROBLEMÁTICAS ({len(problematic_images)}):")
        for img in problematic_images[:5]:
            print(f" - {img}")

    return np.array(X), np.array(y)

# =============================================
# Calcular pesos de clase para desbalance
# =============================================
def calculate_class_weights(y):
    class_counts = np.bincount(y, minlength=NUM_CLASSES)
    total = len(y)
    weights = total / (NUM_CLASSES * class_counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES  # Normalizar
    return torch.tensor(weights, dtype=torch.float32)

# =============================================
# Modelo CNN Mejorado
# =============================================
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()

        # Bloque 1 (64 filtros)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Bloque 2 (128 filtros)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Bloque 3 (256 filtros)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Bloque 4 (512 filtros)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Capas fully connected
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc_layers(x)
        return x

# =============================================
# Entrenamiento con Early Stopping
# =============================================
def train_model(X_train, y_train, X_val, y_val, class_weights):
    # Crear datasets con augmentation solo para entrenamiento
    train_dataset = EmotionDataset(X_train, y_train, augment=True)
    val_dataset = EmotionDataset(X_val, y_val, augment=False)

    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Inicializar modelo
    model = EmotionClassifier(NUM_CLASSES).to(DEVICE)

    # Función de pérdida con pesos de clase
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Optimizador con learning rate reducido
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Scheduler mejorado
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Variables para early stopping
    best_val_acc = 0.0
    best_epoch = 0
    patience = 15  # Aumentado
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_acc': []}

    print("\n" + "="*50)
    print("INICIANDO ENTRENAMIENTO")
    print("="*50)
    print(f"Dispositivo: {DEVICE}")
    print(f"Imágenes de entrenamiento: {len(X_train)}")
    print(f"Imágenes de validación: {len(X_val)}")
    print(f"Data Augmentation: Activado")
    print(f"Class Weights: {class_weights.numpy().round(2)}")

    # Bucle de entrenamiento
    for epoch in range(EPOCHS):
        # Fase de entrenamiento
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

        # Fase de validación
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

        # Calcular métricas
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total

        # Actualizar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Actualizar scheduler
        scheduler.step(val_acc)

        # Obtener learning rate actual
        current_lr = optimizer.param_groups[0]['lr']

        # Mostrar progreso
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >> Mejor modelo guardado (Precisión: {val_acc:.2f}%)")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping en epoch {epoch+1}")
            break

    # Guardar el último modelo
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "last_model.pth"))

    # Graficar historial de entrenamiento
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history['val_acc'], 'g-', label='Val Accuracy')
    plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Mejor: {best_val_acc:.2f}%')
    plt.title('Precisión de validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
    plt.show()

    print("\n" + "="*50)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*50)
    print(f"Mejor precisión: {best_val_acc:.2f}% en epoch {best_epoch+1}")
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")

    return model

# =============================================
# Función para ver balance de clases
# =============================================
def plot_class_distribution(y):
    class_counts = [np.sum(y == i) for i in range(len(EMOTIONS))]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(EMOTIONS, class_counts, color='skyblue')
    plt.title('Distribución de clases')
    plt.xlabel('Emoción')
    plt.ylabel('Número de imágenes')
    plt.xticks(rotation=45)

    # Colorear barras según cantidad (rojo = pocas muestras)
    max_count = max(class_counts)
    for bar, count in zip(bars, class_counts):
        ratio = count / max_count
        if ratio < 0.5:
            bar.set_color('salmon')
        elif ratio < 0.7:
            bar.set_color('lightyellow')

    for i, count in enumerate(class_counts):
        plt.text(i, count + 5, str(count), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'class_distribution.png'))
    plt.show()

# =============================================
# Flujo principal
# =============================================
if __name__ == "__main__":
    print("="*50)
    print("PROCESAMIENTO DE DATOS")
    print("="*50)
    X, y = load_dataset(TRAIN_PATH)

    if len(X) == 0:
        print("\nError: No se encontraron imágenes válidas")
        exit()

    # Mostrar distribución de clases
    plot_class_distribution(y)

    # Calcular pesos de clase
    class_weights = calculate_class_weights(y)
    print("\n" + "="*50)
    print("PESOS DE CLASE (para balanceo)")
    print("="*50)
    for emotion, weight in zip(EMOTIONS, class_weights):
        print(f"{emotion:15}: {weight:.4f}")

    # Dividir datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n" + "="*50)
    print("DIVISIÓN DE DATOS")
    print("="*50)
    print(f"Imágenes de entrenamiento: {len(X_train)}")
    print(f"Imágenes de validación: {len(X_val)}")

    # Entrenar modelo
    model = train_model(X_train, y_train, X_val, y_val, class_weights)

    print("\n" + "="*50)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*50)
    print(f"Modelo final guardado en: {MODEL_SAVE_PATH}")
    print("Puedes usar test.py para evaluar el modelo con datos de prueba")
