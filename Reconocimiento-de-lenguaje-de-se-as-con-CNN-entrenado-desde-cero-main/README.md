# Reconocimiento de senas estaticas con CNN entrenada desde cero

Sistema de vision por computadora para reconocer senas estaticas usando camara, segmentacion HSV, bordes y una red neuronal convolucional pequena entrenada desde cero. La configuracion recomendada fusiona la CNN con descriptores geometricos de la silueta. No reconoce lenguaje de senas continuo ni movimientos.

El proyecto no usa modelos preentrenados para clasificar las senas.

## Inicio rapido

En Windows, haz doble clic en:

```text
iniciar_app.cmd
```

O ejecuta desde PowerShell:

```powershell
.\iniciar_app.cmd
```

La primera ejecucion crea `.venv`, instala las dependencias necesarias y abre la interfaz con el modelo hibrido incluido. Requiere Internet solamente para instalar las dependencias. Las siguientes ejecuciones abren la aplicacion directamente.

Requisito previo: Python 3.11, 3.12 o 3.13.

## Instalacion manual

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Captura de muestras por sesion

Para obtener una evaluacion realista, cada persona y sesion debe tener un identificador distinto:

```text
data/raw/A/persona01_sesion01/
data/raw/A/persona02_sesion01/
data/raw/NONE/persona01_sesion01/
```

Capturar una clase:

```powershell
python scripts\capture_dataset.py --label A --session persona01_sesion01 --count 200 --backend dshow
```

Captura tambien la clase `NONE` con mano relajada, transiciones, posturas desconocidas y falsos positivos:

```powershell
python scripts\capture_dataset.py --label NONE --session persona01_sesion01 --count 200 --backend dshow
```

Durante la captura veras:

- video con recuadro central;
- mascara HSV;
- entrada final que vera la CNN.

Presiona:

- `c` para capturar;
- `q` para salir.

## Revisar imagenes procesadas

Antes de entrenar puedes generar vistas de diagnostico:

```powershell
python scripts\preview_preprocessing.py --data data\raw --out models\previews --per-class 20
```

Las imagenes se guardan en:

```text
models/previews/
```

Cada preview muestra:

```text
Original | Mascara HSV | ROI detectado | Bordes | Entrada CNN
```

## Entrenar CNN

```powershell
python scripts\train_cnn.py --data data\raw --model-out models\hybrid\sign_hybrid.pt --epochs 50 --model-type hybrid
```

El entrenamiento usa:

- division aproximada 70/15/15 para entrenamiento, validacion y prueba;
- separacion por persona/sesion, sin mezclar una sesion entre conjuntos;
- bloques consecutivos para mantener compatibilidad con capturas antiguas sin carpetas de sesion;
- aumentacion de datos;
- fusion de caracteristicas locales de la CNN con momentos de Hu y medidas geometricas;
- scheduler de learning rate y early stopping;
- calibracion de rechazo usando confianza y margen entre las dos clases mas probables;
- matriz de confusion normal y matriz con columna `NR` (no reconocida).

Salidas:

```text
models/hybrid/sign_hybrid.pt
models/hybrid/cnn_confusion_matrix.png
models/hybrid/cnn_rejection_matrix.png
models/hybrid/cnn_training_curves.png
models/hybrid/cnn_metrics.json
```

El conjunto de prueba se evalua una sola vez al final. La seleccion del mejor modelo y el umbral de confianza usan validacion.

## Interfaz en vivo

```powershell
python scripts\live_gui.py --cnn-model models\hybrid\sign_hybrid.pt --backend dshow
```

La interfaz permite:

- ver la camara en tiempo real;
- ver la mascara HSV;
- superponer contorno, envolvente convexa, caja rotada, centroide y ejes geometricos;
- consultar aspecto, extension, solidez y circularidad en tiempo real;
- calibrar HSV desde menu;
- tomar una muestra de piel del centro;
- predecir continuamente con suavizado por historial.

## Flujo del sistema

```text
Camara
-> recuadro central
-> conversion HSV
-> mascara de piel
-> limpieza morfologica
-> contorno de mano
-> bordes
-> entrada 64x64 de 2 canales
-> momentos de Hu y geometria de la silueta
-> fusion CNN + forma
-> rechazo de lecturas ambiguas
-> prediccion
```

## Resultado de referencia

Con las 1,600 imagenes actuales y semilla 42, el modelo hibrido obtuvo:

- exactitud de prueba: 96.7%;
- F1 macro: 96.6%;
- recall de G: 93.3%, frente a 80% de la CNN anterior;
- precision selectiva: 98.7% aceptando 93.3% de las muestras;
- umbrales calibrados: confianza 64% y margen entre clases 28%.

Estos resultados todavia deben confirmarse con personas y sesiones realmente independientes.

## Archivos principales

```text
scripts/capture_dataset.py       Captura muestras con vista HSV.
scripts/preview_preprocessing.py Revisa como se procesan las imagenes.
scripts/train_cnn.py             Entrena la CNN.
scripts/live_gui.py              Interfaz PySide6 en vivo.
sign_recognition/preprocess.py   HSV, mascara, ROI, bordes.
sign_recognition/cnn_model.py    Arquitectura CNN y guardado/carga.
sign_recognition/camera.py       Apertura robusta de camara.
sign_recognition/dataset.py      Recorrido de imagenes por clase.
```
