import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# =============================================
# Configuración con rutas relativas
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "Test")
MODEL_PATH = os.path.join(BASE_DIR, "emotion_classifier.pth")
IMG_SIZE = (96, 96)
NUM_CLASSES = 8
EMOTIONS = ['ira', 'desprecio', 'asco', 'miedo', 'felicidad', 'neutralidad', 'tristeza', 'sorpresa']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

# =============================================
# Preprocesamiento consistente con train.py
# =============================================
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Advertencia: No se pudo cargar {image_path}")
            return None

        if image.shape[:2] != IMG_SIZE:
            image = cv2.resize(image, IMG_SIZE)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        normalized = enhanced.astype('float32') / 255.0

        return np.expand_dims(normalized, axis=-1)

    except Exception as e:
        print(f"Error procesando {image_path}: {str(e)}")
        return None

# =============================================
# Carga de datos
# =============================================
def load_dataset(dataset_path):
    X = []
    y = []
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    problematic_images = []

    print("\n" + "="*50)
    print("CARGANDO DATOS DE PRUEBA")
    print("="*50)

    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Advertencia: {emotion_dir} no encontrado")
            continue

        print(f"Procesando {emotion}...")
        for img_file in os.listdir(emotion_dir):
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
# Modelo CNN Mejorado (idéntico a train.py)
# =============================================
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()

        # Bloque 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Bloque 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Bloque 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Bloque 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Capas fully connected
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
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
# Búsqueda inteligente del modelo
# =============================================
def find_model():
    search_paths = [
        MODEL_PATH,
        os.path.join(BASE_DIR, "models", "emotion_classifier.pth"),
        os.path.join(BASE_DIR, "saved_models", "emotion_classifier.pth"),
        os.path.expanduser("~/emotion_classifier.pth"),
        os.path.join(BASE_DIR, "last_model.pth"),
        os.path.join(BASE_DIR, "backup_model.pth")
    ]

    for path in search_paths:
        if os.path.exists(path):
            print(f"Modelo encontrado en: {path}")
            return path

    print("\nERROR: Modelo no encontrado. Buscó en:")
    for path in search_paths:
        print(f" - {path}")

    print("\nSoluciones posibles:")
    print("1. Ejecuta primero train.py para generar el modelo")
    print("2. Copia manualmente el modelo a una de las rutas anteriores")
    print("3. Especifica la ruta manualmente con: python test.py --model ruta/al/modelo.pth")
    exit(1)

# =============================================
# Evaluación del modelo mejorada
# =============================================
def evaluate_model(model, X_test, y_test):
    test_tensor = torch.tensor(X_test).permute(0, 3, 1, 2).float().to(DEVICE)
    y_pred = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_tensor), BATCH_SIZE):
            batch = test_tensor[i:i+BATCH_SIZE]
            outputs = model(batch)
            _, batch_pred = torch.max(outputs, 1)
            y_pred.extend(batch_pred.cpu().numpy())

    y_pred = np.array(y_pred)

    # Reporte de clasificación
    print("\n" + "="*50)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=EMOTIONS, digits=4))

    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Verdaderas')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_test.png'))
    print("Matriz de confusión guardada como confusion_matrix_test.png")

    # Precisión por clase
    class_acc = {}
    class_counts = {}
    print("\n" + "="*50)
    print("PRECISIÓN POR CLASE")
    print("="*50)
    for i, emotion in enumerate(EMOTIONS):
        indices = np.where(y_test == i)[0]
        class_counts[emotion] = len(indices)
        if len(indices) > 0:
            correct = np.sum(y_pred[indices] == i)
            acc = 100 * correct / len(indices)
            class_acc[emotion] = acc
            print(f"{emotion:15}: {acc:.2f}% ({correct}/{len(indices)})")
        else:
            class_acc[emotion] = 0
            print(f"{emotion:15}: Sin muestras")

    # Precisión global
    global_acc = 100 * np.mean(y_test == y_pred)
    print("\n" + "="*50)
    print(f"PRECISIÓN GLOBAL: {global_acc:.2f}%")
    print("="*50)

    # Gráfico de precisión por clase
    plt.figure(figsize=(12, 6))
    colors = ['salmon' if class_acc[e] < 50 else 'lightyellow' if class_acc[e] < 70 else 'lightgreen' for e in EMOTIONS]
    plt.bar(EMOTIONS, [class_acc[e] for e in EMOTIONS], color=colors)
    plt.title('Precisión por Emoción (Conjunto de Test)')
    plt.xlabel('Emoción')
    plt.ylabel('Precisión (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Línea de referencia
    plt.axhline(y=global_acc, color='blue', linestyle='--', label=f'Promedio: {global_acc:.1f}%')
    plt.legend()

    for i, emotion in enumerate(EMOTIONS):
        plt.text(i, class_acc[emotion] + 2, f"{class_counts[emotion]}\nmuestras",
                ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'class_accuracy_test.png'))
    print("Gráfico de precisión por clase guardado como class_accuracy_test.png")

# =============================================
# Flujo principal
# =============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluación del modelo de reconocimiento de emociones')
    parser.add_argument('--model', type=str, default=None,
                        help='Ruta personalizada al modelo entrenado')
    args = parser.parse_args()

    model_path = args.model if args.model else find_model()

    print("\nCargando datos de prueba...")
    X_test, y_test = load_dataset(TEST_PATH)

    if len(X_test) == 0:
        print("\nERROR: No se encontraron imágenes válidas en el directorio de prueba")
        exit(1)

    print(f"\nCargando modelo desde: {model_path}")
    model = EmotionClassifier(NUM_CLASSES).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Modelo cargado exitosamente!")
    except Exception as e:
        print(f"\nERROR al cargar el modelo: {str(e)}")
        print("Posibles causas:")
        print("- La arquitectura no coincide con la definición actual")
        print("- El archivo está corrupto")
        print("- Fue entrenado con una versión diferente del código")
        print("\nSolución: Vuelve a entrenar el modelo con train.py")
        exit(1)

    print("\nEvaluando modelo...")
    evaluate_model(model, X_test, y_test)

    print("\nEvaluación completada exitosamente!")
