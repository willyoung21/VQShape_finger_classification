<div align="center">
    <h1>Finger Movements Classification Using VQShape Architecture</h1>
    <h3>Universidad Autónoma de Occidente 2025</h3>
    <p><strong>Lopez Juan Manuel, Botero William, Salamanca Danna</strong></p>
</div>

---

## 📋 Guía de Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/willyoung21/VQShape_finger_classification.git
cd VQShape_finger_classification
```

### 2. Preparar los checkpoints de VQShape (⚠️ IMPORTANTE)

El repositorio **no contiene directamente** la carpeta `/checkpoints`, pero sí incluye un archivo comprimido:
 
**`uea_dim256_codebook512.zip`**

**Debes:**
- Descomprimirlo dentro de la carpeta `VQShape`
- Esto generará automáticamente la ruta necesaria: `VQShape/checkpoints/uea_dim256_codebook512/VQShape.ckpt`

**La estructura final debe quedar así:**
```
VQShape/
 └── checkpoints/
      └── uea_dim256_codebook512/
           └── VQShape.ckpt
```

> **Nota:** El modelo de clasificación EEG (`best_eeg_classifier.pt`) **sí está incluido** en el repositorio, dentro de:
> ```
> Modelo/best_eeg_classifier.pt
> ```
> Por lo tanto, no requiere instalación adicional.

---

### 3. Ejecutar la aplicación con Docker 🐳

El proyecto incluye un `Dockerfile` y un `docker-compose.yml`, por lo que solo necesitas ejecutar:
```bash
docker compose up --build
```

Esto construirá la imagen automáticamente:

✅ Instala Python 3.11  
✅ Instala dependencias desde `app/requirements.txt`  
✅ Copia el código dentro del contenedor  
✅ Configura el `PYTHONPATH` para VQShape  

Cuando termine el build, la app de **Streamlit** quedará disponible en:

🌐 **[http://localhost:8501](http://localhost:8501)**

---

### 4. ¿Necesito instalar `requirements.txt` localmente?

**❌ No.**

Debido a Docker:

- Todas las dependencias se instalan **dentro del contenedor**
- Tu máquina local **no necesita instalar nada** (ni Python ni pip)

**Solo si quisieras ejecutar la app sin Docker**, entonces sí tendrías que instalar:
```bash
pip install -r app/requirements.txt
```

Pero **no es necesario** para el flujo principal, ya que Docker gestiona todo.

---

## 🔧 ¿Cómo se cargan los pesos?

El archivo `app.py` carga dos modelos:

### 1️⃣ VQShape (tokenizador pretrained)

**Ruta generada después de descomprimir el ZIP:**
```
VQShape/checkpoints/uea_dim256_codebook512/VQShape.ckpt
```

**El modelo se carga así:**
```python
lit = LitVQShape.load_from_checkpoint(CHECKPOINT, map_location="cpu")
base_model = lit.model
```

> **Nota:** Luego se congela porque solo se utiliza para **inferencia**, no para entrenamiento.

---

### 2️⃣ Clasificador EEG (linear head)

**Este sí viene dentro del repo:**
```
Modelo/best_eeg_classifier.pt
```

**Y se carga con:**
```python
clf.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location="cpu"))
```

---

## 🧠 ¿Cómo funciona la inferencia?

### 📂 Datos de ejemplo

En el repositorio, dentro de la carpeta `VQShape/datos/FingerMovements`, están dos archivos `.ts`:

- `FingerMovements_TEST.ts` ✅ **(úsalo para pruebas)**
- `FingerMovements_TRAIN.ts` ⚠️ **(no usar para inferencia)**

---

### 🔄 Flujo de procesamiento

1. **Subes un archivo `.ts`** con señales EEG (por ejemplo, `FingerMovements_TEST.ts`)

2. **Se toma una muestra (trial)** del archivo

3. **Interpolación:** Se ajusta de **50 → 512 timesteps** para mantener compatibilidad con VQShape

4. **Tokenización:** VQShape convierte cada canal EEG en un **histograma de códigos** (512 dimensiones)

5. **Promediado:** Se promedian los histogramas de todos los canales

6. **Clasificación:** Ese vector de 512 valores entra al clasificador lineal

7. **Predicción final:** El modelo predice:
   - 👈 **LEFT** 
   - 👉 **RIGHT**

---

<div align="center">
    <p>Made with ❤️ by the UAO Team</p>
</div>