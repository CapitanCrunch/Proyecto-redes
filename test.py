import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# =============================================
# Configuración
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "Test")
MODEL_PATH = os.path.join(BASE_DIR, "emotion_classifier.pth")
IMG_SIZE = (224, 224)  # igual que train.py
NUM_CLASSES = 8
EMOTIONS = ['ira', 'desprecio', 'asco', 'miedo', 'felicidad', 'neutralidad', 'tristeza', 'sorpresa']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
USE_TTA = True

# Normalización ImageNet (RGB)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================
# Preprocesamiento RGB (igual que train.py)
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


class TestDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        y = self.labels[idx]
        img = preprocess_image(path)
        if img is None:
            x = torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1])
        else:
            x = torch.tensor(img).permute(2, 0, 1).float()
        return x, y

# =============================================
# Carga de datos
# =============================================
def load_dataset(dataset_path):
    file_paths = []
    labels = []
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}

    print("\n" + "="*50)
    print("CARGANDO DATOS DE PRUEBA (RGB)")
    print("="*50)

    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Advertencia: {emotion_dir} no encontrado")
            continue

        print(f"Procesando {emotion}...")
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
# Modelo ResNet-18 (IDÉNTICO a train.py)
# =============================================
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, freeze_layers=True):
        super(EmotionClassifier, self).__init__()

        # ResNet-18 (sin pesos para inferencia, se cargan del archivo)
        self.resnet = models.resnet18(weights=None)

        # Misma estructura de capa final que train.py
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


def predict_with_tta(model, batch, use_tta=True):
    outputs = model(batch)
    if not use_tta:
        return outputs

    flipped = torch.flip(batch, dims=[3])
    outputs += model(flipped)
    return outputs / 2


# =============================================
# Búsqueda del modelo
# =============================================
def find_model():
    search_paths = [
        MODEL_PATH,
        os.path.join(BASE_DIR, "last_model.pth"),
        os.path.join(BASE_DIR, "models", "emotion_classifier.pth"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            print(f"Modelo encontrado: {path}")
            return path

    print("ERROR: Modelo no encontrado")
    print("Ejecuta primero: python train.py")
    exit(1)

# =============================================
# Evaluación
# =============================================
def evaluate_model(model, X_test, y_test):
    dataset = TestDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    y_pred = []
    y_true = []
    print(f"TTA {'activado' if USE_TTA else 'desactivado'}")

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = predict_with_tta(model, inputs, use_tta=USE_TTA)
            _, batch_pred = torch.max(outputs, 1)
            y_pred.extend(batch_pred.cpu().numpy())
            y_true.extend(labels.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Reporte
    print("\n" + "="*50)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4))

    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Matriz de Confusión (ResNet-18 RGB)')
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Verdaderas')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_test.png'))
    print("Guardado: confusion_matrix_test.png")

    # Precisión por clase
    class_acc = {}
    class_counts = {}
    print("\n" + "="*50)
    print("PRECISIÓN POR CLASE")
    print("="*50)
    for i, emotion in enumerate(EMOTIONS):
        indices = np.where(y_true == i)[0]
        class_counts[emotion] = len(indices)
        if len(indices) > 0:
            correct = np.sum(y_pred[indices] == i)
            acc = 100 * correct / len(indices)
            class_acc[emotion] = acc
            print(f"{emotion:15}: {acc:.2f}% ({correct}/{len(indices)})")
        else:
            class_acc[emotion] = 0

    # Precisión global
    global_acc = 100 * np.mean(y_true == y_pred)
    print("\n" + "="*50)
    print(f"PRECISIÓN GLOBAL: {global_acc:.2f}%")
    print("="*50)

    # Gráfico de precisión
    plt.figure(figsize=(12, 6))
    colors = ['salmon' if class_acc[e] < 50 else 'lightyellow' if class_acc[e] < 70 else 'lightgreen' for e in EMOTIONS]
    bars = plt.bar(EMOTIONS, [class_acc[e] for e in EMOTIONS], color=colors)
    plt.title('Precisión por Emoción - ResNet-18 RGB')
    plt.xlabel('Emoción')
    plt.ylabel('Precisión (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=global_acc, color='blue', linestyle='--', label=f'Promedio: {global_acc:.1f}%')
    plt.legend()

    for i, emotion in enumerate(EMOTIONS):
        plt.text(i, class_acc[emotion] + 2, f"{class_counts[emotion]}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'class_accuracy_test.png'))
    print("Guardado: class_accuracy_test.png")

# =============================================
# Main
# =============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    model_path = args.model if args.model else find_model()

    X_test, y_test = load_dataset(TEST_PATH)

    if len(X_test) == 0:
        print("ERROR: No se encontraron imágenes de prueba")
        exit(1)

    print(f"\nCargando modelo: {model_path}")
    model = EmotionClassifier(NUM_CLASSES).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Vuelve a entrenar con: python train.py")
        exit(1)

    print("\nEvaluando...")
    evaluate_model(model, X_test, y_test)

    print("\nCompletado!")
