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
st.set_page_config(
    page_title="Clasificación EEG — VQShape + Linear",
    page_icon="🧠",
    layout="wide"
)

# ---------- Estilos globales ligeros ----------
st.markdown(
    """
    <style>
    /* Fondo más limpio para los bloques principales */
    .main > div {
        padding-top: 1rem;
    }
    /* Botones más grandes y elegantes */
    div.stButton > button:first-child {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #357ABD;
        color: white;
    }
    /* Métricas más compactas */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- CABECERA ----------
col_title, col_info = st.columns([2, 1])

with col_title:
    st.title("🧠 Clasificación EEG (Left / Right) — VQShape + Linear")
    st.markdown(
        """
        Esta aplicación toma señales EEG del dataset **FingerMovements** y las 
        tokeniza con **VQShape** para luego clasificarlas como movimiento de 
        **mano izquierda** o **mano derecha** usando un clasificador lineal.
        """
    )

with col_info:
    st.markdown("### 📌 Resumen rápido")
    st.markdown(
        """
        - Modelo base: **VQShape** (preentrenado UEA)  
        - Clasificador: **Linear layer**  
        - Datos: **EEG multicanal (28 canales)**  
        - Formato entrada: archivo **`.ts`**  
        """
    )

st.markdown("---")

# ---------- SIDEBAR: CARGA Y OPCIONES ----------
st.sidebar.header("⚙️ Configuración")

st.sidebar.markdown("### 1. Cargar archivo `.ts`")
uploaded = st.sidebar.file_uploader(
    "Selecciona o arrastra un archivo FingerMovements_*.ts",
    type=["ts"]
)

view_hist = st.sidebar.checkbox("Mostrar histograma de tokens", value=True)
view_probs = st.sidebar.checkbox("Mostrar detalle de probabilidades", value=True)

# ---------- CUERPO PRINCIPAL ----------
if uploaded is None:
    st.info(
        "Sube un archivo `.ts` desde la barra lateral para comenzar. "
        "Idealmente, utiliza **FingerMovements_TEST.ts**."
    )
else:
    try:
        with st.spinner("Cargando archivo .ts completo..."):
            # Guardamos temporalmente el archivo subido
            temp_path = "temp_input.ts"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # Cargar todas las muestras
            X, y = load_from_tsfile(temp_path, return_data_type="numpy3D")

        st.success(f"✔ Archivo cargado correctamente. Total de muestras: **{X.shape[0]}**")

        # ---------- Selección de muestra ----------
        st.markdown("### 🎚 Selección de muestra")

        col_idx, col_hint = st.columns([2, 1])
        with col_idx:
            idx = st.slider(
                "Selecciona la muestra (trial) a visualizar y clasificar:",
                0, X.shape[0] - 1, 0
            )
        with col_hint:
            st.caption(
                "Cada trial corresponde a un intento de movimiento capturado en EEG "
                "(etiqueta `left` o `right`)."
            )

        sample = X[idx]     # (28, 50)
        real_label = y[idx] # string ("left"/"right")

        # ---------- TABS PRINCIPALES ----------
        tab_signal, tab_tokens, tab_results = st.tabs(
            ["📡 Señal EEG", "🧩 Tokens VQShape", "🎯 Clasificación"]
        )

        # ===== TAB 1: SEÑAL EEG =====
        with tab_signal:
            st.markdown(f"#### Señal EEG — Trial {idx}")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(sample.T, linewidth=0.7, alpha=0.9)
            ax.set_title(f"Trial {idx} — Etiqueta real: {real_label.upper()}", fontsize=13)
            ax.set_yticks([])
            ax.set_xticks([])
            st.pyplot(fig)

            st.caption(
                "Cada línea corresponde a un canal EEG. La señal se representa cruda, "
                "antes de la tokenización por VQShape."
            )

        # ===== Procesamiento compartido (tokens + clasificación) =====
        x_proc = preprocess_signal(sample)   # NO se cambia esta función
        hist = get_histogram(x_proc)         # NO se cambia esta función

        # ===== TAB 2: TOKENS =====
        with tab_tokens:
            st.markdown("#### Histograma de Tokens (Codebook 512D)")

            if view_hist:
                hist_np = hist.numpy() if hasattr(hist, "numpy") else hist
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.bar(range(len(hist_np)), hist_np, color="skyblue")
                ax2.set_title("Distribución de tokens usados para este trial")
                ax2.set_xlabel("Índice de Token")
                ax2.set_ylabel("Frecuencia")
                st.pyplot(fig2)

                st.caption(
                    "Cada barra indica cuántas veces un token del codebook de VQShape "
                    "fue asignado a las subsecuencias del EEG."
                )
            else:
                st.info("La visualización del histograma está desactivada desde la barra lateral.")

        # ===== TAB 3: CLASIFICACIÓN =====
        with tab_results:
            st.markdown("#### Resultado de Clasificación")

            # --- FUNCIONES VISUALES ---
            def get_hand_icon(label):
                label = label.lower()
                return "👈" if label == "left" else "👉"

            def get_color(pred, real):
                return "green" if pred == real else "red"

            def get_status_text(pred, real):
                return "✔ ACIERTO" if pred == real else "✘ ERROR"

            def box_style(color):
                return f"""
                    padding: 15px;
                    border-radius: 12px;
                    border: 2px solid {color};
                    background-color: rgba(0, 0, 0, 0.05);
                    text-align: center;
                    margin-bottom: 15px;
                """

            # ------------------------------------------------------------
            # Clasificación usando el clasificador lineal
            # ------------------------------------------------------------
            with torch.no_grad():
                logits = clf(hist.unsqueeze(0))
                probs = torch.softmax(logits, dim=1).numpy()[0]
                pred_idx = int(np.argmax(probs))

            pred_label = "right" if pred_idx == 1 else "left"

            hand_real  = get_hand_icon(real_label)
            hand_pred  = get_hand_icon(pred_label)
            color_eval = get_color(pred_label, real_label)
            status     = get_status_text(pred_label, real_label)

            col1, col2, col3 = st.columns([1, 1, 1])

            # -------- ETIQUETA REAL --------
            with col1:
                st.markdown(
                    f"""
                    <div style="{box_style('black')}">
                        <h5>Etiqueta real</h5>
                        <div style="font-size:38px;">{hand_real}</div>
                        <div style="font-size:20px; font-weight:bold;">{real_label.upper()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -------- PREDICCIÓN --------
            with col2:
                st.markdown(
                    f"""
                    <div style="{box_style(color_eval)}">
                        <h5>Predicción modelo</h5>
                        <div style="font-size:38px; color:{color_eval};">{hand_pred}</div>
                        <div style="font-size:20px; font-weight:bold; color:{color_eval};">
                            {pred_label.upper()}
                        </div>
                        <div style="color:{color_eval}; margin-top:4px; font-weight:bold;">
                            {status}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -------- CONFIANZA --------
            with col3:
                diff = probs[pred_idx] - probs[1 - pred_idx]
                st.metric(
                    "Confianza",
                    f"{probs[pred_idx]*100:.2f}%",
                    delta=f"{diff*100:.2f}%"
                )

            # --------PROBABILIDADES -------
            if view_probs:
                st.markdown("##### Probabilidades por clase")

                st.write("**👈 LEFT**")
                st.progress(float(probs[0]))

                st.write("**👉 RIGHT**")
                st.progress(float(probs[1]))

                st.write(
                    f"""
                    **Left:**  {probs[0]:.4f}  
                    **Right:** {probs[1]:.4f}
                    """
                )
            else:
                st.caption("Las probabilidades detalladas están ocultas (ver barra lateral).")


            st.markdown("---")
            if pred_label == real_label:
                st.success(
                    "✅ La predicción coincide con la etiqueta real. "
                    "Para este trial, el patrón de tokens fue suficientemente distintivo."
                )
            else:
                st.warning(
                    "⚠️ La predicción NO coincide con la etiqueta real. "
                    "Esto refleja la dificultad de clasificar EEG solo a partir de formas discretas."
                )


    except Exception as e:
        st.error(f"❌ Error procesando archivo: {e}")