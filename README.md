# Reconocimiento de Emociones Faciales con CNN

Sistema de reconocimiento de emociones faciales en tiempo real utilizando redes neuronales convolucionales (CNN) con PyTorch.

## Emociones Detectadas

El modelo clasifica imágenes en 8 categorías:
- Ira
- Desprecio
- Asco
- Miedo
- Felicidad
- Neutralidad
- Tristeza
- Sorpresa

## Requisitos

- Python 3.8+
- GPU con CUDA (opcional, pero recomendado para entrenamiento)

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/CapitanCrunch/Proyecto-redes.git
cd Proyecto-redes
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
Proyecto-redes/
├── data/
│   ├── Train/                   # Datos de entrenamiento
│   │   ├── ira/
│   │   ├── desprecio/
│   │   ├── asco/
│   │   ├── miedo/
│   │   ├── felicidad/
│   │   ├── neutralidad/
│   │   ├── tristeza/
│   │   └── sorpresa/
│   └── Test/                   # Datos de prueba
│       ├── ira/
│       ├── desprecio/
│       ├── asco/
│       ├── miedo/
│       ├── felicidad/
│       ├── neutralidad/
│       ├── tristeza/
│       └── sorpresa/
├── train.py                   # Script de entrenamiento
├── test.py                    # Script de evaluación
├── camera.py                  # Detección en tiempo real
├── emotion_classifier.pth     # Modelo entrenado
└── requirements.txt           # Dependencias
```

## Uso

### Entrenamiento

Entrena el modelo desde cero con las imágenes en `data/Train/`:

```bash
python train.py
```

Genera:
- `emotion_classifier.pth` - Mejor modelo según precisión de validación
- `training_history.png` - Gráficas de pérdida y precisión
- `class_distribution.png` - Distribución de clases

### Evaluación

Evalúa el modelo con las imágenes en `data/Test/`:

```bash
python test.py
```

Para evaluar un modelo específico:
```bash
python test.py --model ruta/al/modelo.pth
```

Genera:
- `confusion_matrix_test.png` - Matriz de confusión
- `class_accuracy_test.png` - Precisión por clase
- Reporte de clasificación en consola

### Detección en Tiempo Real

Ejecuta la detección de emociones con webcam:

```bash
python camera.py
```

Controles:
- `ESC` o `q` - Salir

## Arquitectura del Modelo

CNN personalizada con 4 bloques convolucionales:

```
Entrada: Imagen 96x96 escala de grises
    ↓
Bloque 1: Conv(64) → BatchNorm → ReLU → Conv(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Bloque 2: Conv(128) → BatchNorm → ReLU → Conv(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Bloque 3: Conv(256) → BatchNorm → ReLU → Conv(256) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Bloque 4: Conv(512) → BatchNorm → ReLU → Conv(512) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Flatten → FC(512) → BatchNorm → ReLU → Dropout(0.5)
    ↓
FC(256) → BatchNorm → ReLU → Dropout(0.5)
    ↓
FC(8) → Salida (8 clases)
```

## Técnicas Implementadas

- **Class Weights**: Balanceo de clases minoritarias
- **Data Augmentation**: Flip horizontal, rotación, ajuste de brillo
- **Mixup**: Regularización para reducir overfitting (alpha=0.2)
- **TTA**: Test-Time Augmentation para mejorar predicciones
- **CLAHE**: Mejora de contraste en preprocesamiento
- **Early Stopping**: Detiene entrenamiento cuando no mejora

## Resultados

- **Precisión global**: 70.20%
- **Mejor clase**: Neutralidad (94.84%)
- **Gap de overfitting**: ~19%

### Precisión por Clase

| Emoción | Precisión |
|---------|-----------|
| Felicidad | 91.21% |
| Neutralidad | 94.84% |
| Tristeza | 89.69% |
| Desprecio | 67.93% |
| Ira | 57.39% |
| Asco | 55.56% |
| Sorpresa | 54.64% |
| Miedo | 51.96% |

## Datasets Utilizados

- [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) - Dataset principal
- [AffectNet](https://www.kaggle.com/datasets/mstjebashazida/affectnet) - Complementario
- RAF-DB - Expresiones reales
- CK+ - Expresiones reales
