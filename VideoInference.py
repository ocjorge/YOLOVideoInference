import os
import cv2
from ultralytics import YOLO
import shutil
from base64 import b64encode

print("Paso 0: Librerías importadas correctamente.")

# ==============================================================================
# PASO 1: CONFIGURACIÓN DE RUTAS
# ==============================================================================

print("\nPaso 1: Configurando rutas...")

# Ruta a tu modelo .pt entrenado
MODEL_PATH = '/Users/jorgeoc/PycharmProjects/YoloModel/best.pt'

# Ruta a tu video de entrada
VIDEO_INPUT_PATH = '/Users/jorgeoc/PycharmProjects/YoloModel/GH012372.mp4'

# Carpeta local donde se guardarán los resultados
OUTPUT_PROJECT_DIR = "./runs/detect_video"
OUTPUT_RUN_NAME = "video_processed_output"

# Crear directorio si no existe
os.makedirs(OUTPUT_PROJECT_DIR, exist_ok=True)

print(f"Ruta del modelo: {MODEL_PATH}")
print(f"Ruta del video de entrada: {VIDEO_INPUT_PATH}")
print(f"Resultados en: {os.path.abspath(os.path.join(OUTPUT_PROJECT_DIR, OUTPUT_RUN_NAME))}")

# ==============================================================================
# PASO 2: CARGAR EL MODELO
# ==============================================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("Modelo cargado correctamente.")

# ==============================================================================
# PASO 3: VERIFICAR VIDEO
# ==============================================================================

if not os.path.exists(VIDEO_INPUT_PATH):
    raise FileNotFoundError(f"Video no encontrado en: {VIDEO_INPUT_PATH}")

cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video verificado: {width}x{height} @ {fps:.2f} FPS")
cap.release()

# ==============================================================================
# PASO 4: PROCESAR VIDEO
# ==============================================================================

print("\nProcesando video con YOLO...")

results_generator = model.predict(
    source=VIDEO_INPUT_PATH,
    stream=True,
    save=True,
    project=OUTPUT_PROJECT_DIR,
    name=OUTPUT_RUN_NAME,
    exist_ok=True,
    conf=0.3,
    iou=0.5
)

processed_frames = 0
for i, result in enumerate(results_generator):
    processed_frames += 1
    if (i + 1) % 100 == 0:
        print(f"  Procesado frame {i+1}...")

print(f"Video procesado exitosamente. Total de frames: {processed_frames}")

# ==============================================================================
# PASO 5: MOSTRAR RUTA DEL VIDEO DE SALIDA
# ==============================================================================

expected_output_filename = os.path.basename(VIDEO_INPUT_PATH)
video_output_path = os.path.join(OUTPUT_PROJECT_DIR, OUTPUT_RUN_NAME, expected_output_filename)

if os.path.exists(video_output_path):
    print(f"\n✅ Video procesado guardado en: {video_output_path}")
    print("Ábrelo desde tu explorador de archivos o un reproductor multimedia.")
else:
    print("⚠️ No se encontró el archivo de salida. Verifica si hubo errores en el procesamiento.")
