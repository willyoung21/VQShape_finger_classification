import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sktime.datasets import load_from_tsfile

# 1) Configuramos las rutas de los pesos preentrenados

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VQSHAPE_DIR = os.path.join(BASE_DIR, "VQShape")

# Ruta al checkpoint original de VQShape
CHECKPOINT = os.path.join(
    VQSHAPE_DIR,
    "checkpoints",
    "uea_dim256_codebook512",
    "VQShape.ckpt"
)

# Ruta al clasificador entrenado sobre histogramas VQShape
CLASSIFIER_WEIGHTS = os.path.join(BASE_DIR, "Modelo/best_eeg_classifier.pt")

# Añadir carpeta VQShape al sistema para poder importar módulos
sys.path.append(VQSHAPE_DIR)

try:
    from vqshape.pretrain import LitVQShape
except Exception as e:
    st.error(f"Error importando VQShape: {e}")

DEVICE = "cpu"

# 2) Cargamos el modelo VQShape (tokenizador de series temporales)

@st.cache_resource
def load_vqshape():
    """
    Cargamos el modelo VQShape desde el checkpoint y lo deja congelado.
    Este modelo solo genera representaciones discretas (histogramas).
    """
    lit = LitVQShape.load_from_checkpoint(
        CHECKPOINT,
        map_location=DEVICE
    )
    base_model = lit.model
    base_model.eval()
    
    # Congelamos parámetros porque no lo vamos a entrenar aquí
    for p in base_model.parameters():
        p.requires_grad = False
    return base_model

vqshape = load_vqshape()


# 3) Clasificador simple (Linear 512 → 2)
class SimpleClassifier(nn.Module):
    """
    Clasificador lineal entrenado sobre vectores de 512 dimensiones
    generados por VQShape.
    """    
    def __init__(self, hist_dim=512):
        super().__init__()
        self.net = nn.Linear(hist_dim, 2)

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_classifier():
    """
    Carga el clasificador entrenado desde disco y lo deja en modo evaluación.
    """
    clf = SimpleClassifier(hist_dim=512)
    clf.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE))
    clf.eval()
    return clf


clf = load_classifier()

# 4) Funciones de preprocesamiento y extracción de histogramas
def preprocess_signal(x_np):
    """
    Preprocesa una muestra EEG (.ts) para que coincida con el formato
    utilizado en el entrenamiento de VQShape.
    - Entrada : (28 canales × 50 timesteps)
    - Salida  : (28 canales × 512 timesteps)
    """
    x = torch.tensor(x_np).float()   # (28,50)
    x = F.interpolate(x.unsqueeze(0), size=512, mode="linear").squeeze(0)
    return x  # (28,512)

def get_histogram(x):
    """
    Obtiene el histograma del codebook generado por VQShape.
    Promedia los histogramas de los 28 canales.
    """
    with torch.no_grad():
        reps, _ = vqshape(x, mode="tokenize")
        hist = reps["histogram"]    # (28, 512)
        hist = hist.float().mean(dim=0)  # (512,)
    return hist


# 5) Interfaz Streamlit

st.title("Clasificación EEG (Left / Right) — VQShape + Linear")
st.write("Sube un archivo `.ts`")

uploaded = st.file_uploader("Archivo .ts", type=["ts"])

if uploaded is not None:
    try:
        st.write("Cargando archivo .ts completo...")

        # # Guardamos temporalmente el archivo subido
        temp_path = "temp_input.ts"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Cargar todas las muestras
        X, y = load_from_tsfile(temp_path, return_data_type="numpy3D")

        st.success(f"Archivo cargado correctamente. Total de muestras: {X.shape[0]}")

        # Slider para elegir una muestra
        idx = st.slider("Selecciona la muestra a clasificar", 0, X.shape[0] - 1, 0)

        sample = X[idx]     # (28, 50)
        real_label = y[idx] # string ("left"/"right")


        # 6) Visualización de la señal EEG

        st.subheader(f"Señal EEG — Trial {idx}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(sample.T, linewidth=0.7)
        ax.set_title(f"Trial {idx} — Etiqueta real: {real_label.upper()}")
        ax.set_yticks([]); ax.set_xticks([])
        st.pyplot(fig)

        # 7) Extracción del histograma con VQShape
        
        x_proc = preprocess_signal(sample)
        hist = get_histogram(x_proc)


        # 8) Clasificamos con el modelo entrenado
        
        with torch.no_grad():
            logits = clf(hist.unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_label = "right" if pred_idx == 1 else "left"
        

        # 9) Mostramos la comparación entre la etiqueta real y la prediccion de el modelo
        
        st.subheader("🔍 Resultado de Clasificación")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Etiqueta real", real_label.upper())

        with col2:
            st.metric("Predicción modelo", pred_label.upper())

        st.write("### Probabilidades:")
        st.write(f"Left  : {probs[0]:.4f}")
        st.write(f"Right : {probs[1]:.4f}")

    except Exception as e:
        st.error(f"Error procesando archivo: {e}")
