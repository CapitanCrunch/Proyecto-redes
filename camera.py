import os
import glob
import cv2
import torch
import torch.nn as nn

# =============================================
# Configuración
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_classifier.pth")
IMG_SIZE = (96, 96)
NUM_CLASSES = 8
EMOTIONS = ['ira', 'desprecio', 'asco', 'miedo', 'felicidad', 'neutralidad', 'tristeza', 'sorpresa']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Colores para cada emoción (BGR)
EMOTION_COLORS = {
    'ira': (0, 0, 255),        # Rojo
    'desprecio': (128, 0, 128), # Morado
    'asco': (0, 128, 0),       # Verde oscuro
    'miedo': (128, 128, 128),  # Gris
    'felicidad': (0, 255, 255), # Amarillo
    'neutralidad': (255, 255, 255), # Blanco
    'tristeza': (255, 0, 0),   # Azul
    'sorpresa': (0, 165, 255)  # Naranja
}

# =============================================
# Modelo CNN (idéntico a train.py y test.py)
# =============================================
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()

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
# Preprocesamiento (igual que train.py)
# =============================================
def preprocess_face(face_img):
    """Preprocesa una cara para el modelo"""
    # Resize a 96x96
    face_resized = cv2.resize(face_img, IMG_SIZE)

    # Convertir a escala de grises si es necesario
    if len(face_resized.shape) == 3:
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_resized

    # Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Normalizar
    normalized = enhanced.astype('float32') / 255.0

    # Convertir a tensor
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0).float()
    return tensor

# =============================================
# Predicción con TTA
# =============================================
def predict_emotion(model, face_tensor):
    """Predice emoción con TTA (flip horizontal)"""
    model.eval()
    with torch.no_grad():
        face_tensor = face_tensor.to(DEVICE)

        # Predicción original
        output1 = model(face_tensor)

        # Predicción con flip
        flipped = torch.flip(face_tensor, dims=[3])
        output2 = model(flipped)

        # Promediar
        output = (output1 + output2) / 2

        # Obtener probabilidades
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

        emotion_idx = predicted.item()
        confidence_val = confidence.item() * 100

        return EMOTIONS[emotion_idx], confidence_val, probs[0].cpu().numpy()


def resolve_cascade_path():
    """Encuentra la ruta del cascade haar frontal en ubicaciones comunes (cv2.data, conda)."""
    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    pkg_dir = os.path.dirname(cv2.__file__)
    env_root = os.path.abspath(os.path.join(pkg_dir, "..", "..", ".."))
    candidates.append(os.path.join(pkg_dir, "data", "haarcascade_frontalface_default.xml"))
    candidates.append(os.path.join(pkg_dir, "haarcascade_frontalface_default.xml"))
    # Rutas típicas en conda-forge (Library/etc/haarcascades o share/opencv4/haarcascades)
    candidates.append(os.path.join(env_root, "Library", "etc", "haarcascades", "haarcascade_frontalface_default.xml"))
    candidates.append(os.path.join(env_root, "Library", "share", "opencv4", "haarcascades", "haarcascade_frontalface_default.xml"))
    candidates.extend(glob.glob(os.path.join(env_root, "Library", "**", "haarcascade_frontalface_default.xml"), recursive=True))

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None

# =============================================
# Programa principal
# =============================================
def main():
    print("="*50)
    print("DETECTOR DE EMOCIONES EN TIEMPO REAL")
    print("="*50)

    # Cargar modelo
    print(f"\nCargando modelo desde: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Modelo no encontrado. Ejecuta train.py primero.")
        return

    model = EmotionClassifier(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Modelo cargado en: {DEVICE}")

    # Cargar detector de rostros Haar Cascade con búsqueda defensiva
    cascade_path = resolve_cascade_path()
    if not cascade_path:
        print("ERROR: No se encontro el cascade haarcascades. Reinstala opencv con datos o apunta manualmente al xml.")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"ERROR: No se pudo cargar el cascade en {cascade_path}")
        return

    # Iniciar captura de video
    print("\nIniciando camara...")
    print("Presiona 'q' para salir")
    print("Presiona 's' para guardar captura")
    print("Presiona '0-9' para cambiar camara")

    camera_id = 0  # Camara por defecto, cambiar si es externa
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la camara {camera_id}")
        print("Intentando con camara 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("ERROR: No se encontro ninguna camara")
            return

    # Configurar resolucion
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar frame")
            break

        # Convertir a escala de grises para deteccion
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )

        # Procesar cada rostro
        for (x, y, w, h) in faces:
            # Extraer region del rostro
            face_roi = gray_frame[y:y+h, x:x+w]

            # Preprocesar y predecir
            face_tensor = preprocess_face(face_roi)
            emotion, confidence, probs = predict_emotion(model, face_tensor)

            # Color segun emocion
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

            # Dibujar rectangulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Texto con emocion y confianza
            label = f"{emotion}: {confidence:.1f}%"

            # Fondo para el texto
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-30), (x + text_w + 10, y), color, -1)
            cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Barra de confianza
            bar_width = int(w * confidence / 100)
            cv2.rectangle(frame, (x, y+h+5), (x + bar_width, y+h+15), color, -1)
            cv2.rectangle(frame, (x, y+h+5), (x + w, y+h+15), color, 1)

        # Mostrar info en pantalla
        cv2.putText(frame, f"Dispositivo: {DEVICE}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Rostros: {len(faces)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Q: Salir | S: Captura | 0-9: Cambiar cam", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mostrar frame
        cv2.imshow('Detector de Emociones', frame)

        # Controles
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"captura_{frame_count}.png"
            cv2.imwrite(os.path.join(BASE_DIR, filename), frame)
            print(f"Captura guardada: {filename}")
            frame_count += 1
        elif key >= ord('0') and key <= ord('9'):
            new_cam = key - ord('0')
            print(f"Cambiando a camara {new_cam}...")
            cap.release()
            cap = cv2.VideoCapture(new_cam)
            if not cap.isOpened():
                print(f"Camara {new_cam} no disponible, volviendo a 0")
                cap = cv2.VideoCapture(0)

    cap.release()
    cv2.destroyAllWindows()
    print("\nCamara cerrada.")

if __name__ == "__main__":
    main()
