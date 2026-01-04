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

### Modelo: EmotionClassifier (CNN Mejorada)
- **Entrada**: Imágenes en escala de grises de 96x96
- **Preprocesamiento**: Mejora de contraste CLAHE, normalización a [0,1]
- **Data Augmentation**: Flip horizontal, rotación (±15°), ajuste de brillo
- **Arquitectura**:
  - 4 bloques convolucionales (32→64→128→256 filtros)
  - Cada bloque: 2x(Conv2d → BatchNorm → ReLU) → MaxPool → Dropout(0.25)
  - Capas completamente conectadas: 256×6×6 → 512 → 256 → NUM_CLASSES
  - Dropout(0.5) en capas FC
- **Balanceo de clases**: Pesos inversamente proporcionales a frecuencia
- **Salida**: Predicciones de 8 clases de emociones

### Restricción Crítica
La definición de la clase `EmotionClassifier` en [train.py](train.py) y [test.py](test.py) DEBE ser idéntica. Cualquier cambio arquitectónico requiere actualizar ambos archivos, de lo contrario fallará la carga del modelo.

### Detalles del Entrenamiento
- Usa optimizador Adam (lr=0.0005) con scheduler ReduceLROnPlateau
- Función de pérdida CrossEntropyLoss con pesos de clase
- Early stopping (paciencia de 15 épocas)
- División entrenamiento/validación: 80/20 estratificada
- Tamaño de batch: 32
- Máximo 100 épocas (early stopping detiene antes)
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
