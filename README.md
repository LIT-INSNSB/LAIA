## LAIA-Net

Pipeline para construir un dataset de keypoints de manos con etiquetas de consenso para tareas de reconocimiento de acciones de lavado de manos.

### Estructura de pasos

- Paso 1: `1.LabelAlignment/builder.py`
  - Lee anotaciones múltiples por video (PSKUS) y construye el consenso por frame:
    - `is_washing_cons` ∈ {0,1} (−1 = indeterminado)
    - `movement_cons` ∈ {0..7} (−1 = indeterminado o no washing)
    - `transition_mask` ∈ {0,1} (1 alrededor de cambios de movimiento)
  - Genera:
    - CSV por video en `1.LabelAlignment/Consensus/<DataSetX>/<video>.csv`
    - Manifest en `1.LabelAlignment/Manifests/videos.csv`

- Paso 2: `2.Keypoints/build_h5.py`
  - Extrae keypoints de manos (42×3) con el estimador seleccionado (por defecto MediaPipe)
  - Combina keypoints con las etiquetas de consenso del Paso 1
  - Escribe un HDF5 con un grupo por video:
    - `data(T,42,3)`, `lw(T,)`, `rw(T,)`, `is_washing(T,)`, `movement(T,)`, `transition(T,)`
    - Atributos: `dataset_id`, `camera_id`, `fps`, `width`, `height`

### Requisitos

Instala dependencias con:

```bash
python -m pip install -r requirements.txt
```

Archivo `requirements.txt` incluye: numpy, pandas, opencv-python, mediapipe, h5py.

Si trabajas en servidores sin GUI: considera `opencv-python-headless` en vez de `opencv-python`.

### Datos esperados en disco

```
PSKUS_dataset/
  DataSetX/
    Videos/<video>.mp4
    Annotations/Annotator1/<video>.{json|csv}
    Annotations/Annotator2/<video>.{json|csv}
    ...
```

### Paso 1: Generar consenso (LabelAlignment)

Desde `1.LabelAlignment/` ejecuta:

```bash
python builder.py \
  --datasets-root <ruta a PSKUS_dataset> \
  --out-root . \
  --fps 30 \
  --transition-frames 15 \
  --map7to0 \
  --min-annotators 1
```

Salida relevante:
- `1.LabelAlignment/Consensus/DataSetX/<video>.csv`
- `1.LabelAlignment/Manifests/videos.csv`

Notas:
- La función `majority_vote_int` exige mayoría absoluta; si no existe devuelve −1.
- `apply_transition_mask(labels, radius)` marca con 1 los frames en ±`radius` alrededor de cada cambio de movimiento.

### Paso 2: Construir HDF5 de frames con keypoints

Desde `2.Keypoints/` ejecuta:

```bash
python build_h5.py \
  --manifest ../1.LabelAlignment/Manifests/videos.csv \
  --estimator mediapipe \
  --out-h5 ../prepared/h5/frames_mediapipe.h5 \
  --coord-space image \
  --stride 1 \
  --limit 0
```

Parámetros clave:
- `--coord-space`: `image` (x,y normalizados + z) o `world` (si el estimador lo soporta).
- `--stride`: procesa 1 de cada N frames (1 = todos los frames). Se aplica también al CSV de consenso para mantener alineación.
- `--limit`: procesa solo los primeros N videos (0 = todos).

Estructura del HDF5 resultante:

```
<out>.h5
├── meta
│   ├── estimator: "mediapipe"
│   ├── coord_space: "image"
│   ├── C: 3
│   └── K: 42
└── videos/
    └── <video_id>/
        ├── data: float32 (T,42,3)
        ├── lw: uint8 (T,)
        ├── rw: uint8 (T,)
        ├── is_washing: int8 (T,)
        ├── movement: int8 (T,)
        └── transition: uint8 (T,)
```

### Consideraciones

- Si `path_consensus` no existe, las etiquetas se rellenan con −1 y `transition` con 0.
- Los frames y etiquetas se recortan a la longitud mínima común para evitar desalineación.
- `mediapipe_funcs.py` expone `model_init`, `frame_process` y `close_model`.

### Troubleshooting

- OpenCV no abre el video:
  - Verifica `path_video` en el manifest
  - Prueba `opencv-python-headless` en entornos sin GUI
- Conflictos de `mediapipe` y `protobuf`:
  - Actualiza pip y reinstala dependencias
- H5 ya existe:
  - El script hace backup automático `*_old.h5` y lo sobrescribe


