# CLAUDE.md

Este archivo proporciona orientación a Claude Code (claude.ai/code) al trabajar con código en este repositorio.

## Descripción del Proyecto

Modelo CNN basado en PyTorch para clasificación de emociones faciales. Clasifica imágenes en 8 categorías de emociones: ira, desprecio, asco, miedo, felicidad, neutralidad, tristeza, sorpresa.

## Comandos Comunes

### Configuración Inicial
```bash
pip install -r requirements.txt
```

### Entrenamiento
```bash
python train.py
```
Entrena el modelo con las imágenes del directorio `Train/`. Genera:
- `emotion_classifier.pth` - mejor modelo según la precisión de validación
- `last_model.pth` - modelo de la última época
- `training_history.png` - gráficas de pérdida y precisión de entrenamiento/validación
- `class_distribution.png` - visualización del balance de clases

### Pruebas
```bash
python test.py
```
Evalúa el modelo entrenado con las imágenes del directorio `Test/`. Genera:
- `confusion_matrix_test.png` - mapa de calor de la matriz de confusión
- `class_accuracy_test.png` - visualización de precisión por clase
- Reporte de clasificación con precisión, recall, F1-score

Para probar un modelo específico:
```bash
python test.py --model ruta/al/modelo.pth
```

## Arquitectura

### Modelo: EmotionClassifier (Transfer Learning - ResNet-18 RGB)
- **Entrada**: Imágenes RGB de 224x224
- **Preprocesamiento**: Normalización ImageNet (mean/std)
- **Data Augmentation**: Flip horizontal, rotación (±10°), brillo, Random Erasing
- **Arquitectura base**: ResNet-18 preentrenado en ImageNet
  - Capas congeladas: conv1, bn1, layer1, layer2 (features genéricas)
  - Capas entrenables: layer3, layer4 (features específicas)
  - Capa final: Dropout(0.5) → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 8)
- **Regularización**: Weight decay (L2), Dropout múltiple, Random Erasing
- **Balanceo de clases**: Pesos inversamente proporcionales a frecuencia
- **Salida**: Predicciones de 8 clases de emociones

### Restricción Crítica
La definición de la clase `EmotionClassifier` en [train.py](train.py) y [test.py](test.py) DEBE ser idéntica. Cualquier cambio arquitectónico requiere actualizar ambos archivos, de lo contrario fallará la carga del modelo.

### Detalles del Entrenamiento
- Optimizador AdamW con weight decay=0.01 (L2 regularization)
- Learning rate: 0.001 (solo capas entrenables)
- Función de pérdida CrossEntropyLoss con pesos de clase
- Scheduler ReduceLROnPlateau (factor=0.5, patience=4)
- Early stopping (paciencia de 10 épocas)
- División entrenamiento/validación: 80/20 estratificada
- Tamaño de batch: 32
- Máximo 50 épocas
- Dispositivo: Detección automática CUDA/CPU

### Organización de Datos
Los directorios `Train/` y `Test/` contienen subdirectorios por emoción:
```
Train/
├── ira/
├── desprecio/
├── asco/
├── miedo/
├── felicidad/
├── neutralidad/
├── tristeza/
└── sorpresa/
```

Las imágenes se cargan y preprocesan automáticamente desde estos subdirectorios. El script reporta:
- Total de imágenes cargadas por emoción
- Imágenes problemáticas que fallaron al cargar
- Estadísticas de distribución de clases

## Notas de Desarrollo

- Los archivos del modelo (`.pth`) están en gitignore pero deben preservarse localmente
- El preprocesamiento de imágenes usa CLAHE de OpenCV para mejorar el contraste
- El entrenamiento incluye monitoreo de validación y guarda el mejor modelo según la precisión de validación
- El script de pruebas incluye búsqueda inteligente del modelo en múltiples rutas posibles
