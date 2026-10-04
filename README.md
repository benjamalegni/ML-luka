# 🧠 Machine Learning & Deep Learning Portfolio
### Diplomatura Universitaria en Inteligencia Artificial (DUIA) — UNICEN (Tandil, Argentina)
**Autor:** Luka Benjamin Malegni  
[![GitHub Profile](https://img.shields.io/badge/GitHub-benjamalegni-181717?style=flat&logo=github)](https://github.com/benjamalegni)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NVIDIA DLI Certified](https://img.shields.io/badge/NVIDIA%20DLI-Certified-76B900?style=flat&logo=nvidia&logoColor=white)](https://learn.nvidia.com/)

---

## 📌 Descripción General

Este repositorio centraliza y documenta los proyectos, trabajos prácticos, experimentos de investigación y apuntes teóricos desarrollados a lo largo de la **Diplomatura Universitaria en Inteligencia Artificial (DUIA)** dictada por la **Universidad Nacional del Centro de la Provincia de Buenos Aires (UNICEN)** en Tandil, complementados con certificaciones y workshops del **NVIDIA Deep Learning Institute (DLI)** y desarrollo interactivo en Google Colab.

Abarca desde fundamentos estadísticos y algoritmos clásicos de Machine Learning supervisado y no supervisado, hasta arquitecturas avanzadas de Deep Learning convolucionales (CNN con skip connections, ResNet, Transfer Learning), modelos secuenciales para Procesamiento de Lenguaje Natural (RNN, LSTM, Transformers) y optimización heurística (Algoritmos Genéticos).

---

## 🗺️ Estructura del Repositorio

```bash
ML-luka/
├── 01-machine-learning/                    # Aprendizaje estadístico clásico y ensambles
│   ├── 01-regresion-seguros-medicos/       # TP Final ML: Regresión, Regularización y Random Forest
│   └── 02-clasificacion-palmer-penguins/   # Clasificación multiclase, EDA y Scikit-Learn pipelines
├── 02-deep-learning/                       # Redes neuronales convolucionales y arquitecturas profundas
│   ├── 01-tpe-eurosat-satelital/           # TPE: Clasificación satelital EuroSAT (CNN, Skip-ResNet, ResNet-50)
│   └── labs-fundamentos/                   # Gradient descent, redes densas (MLP) y CNN vs Dense (MNIST)
├── 03-nlp-procesamiento-texto/             # Modelos secuenciales y minería de texto
│   ├── 01-redes-recurrentes-rnn/           # Modelado temporal (RNNs, LSTMs bidireccionales, vanishing gradient)
│   ├── 02-formato-y-preprocesamiento/      # Preprocesamiento léxico, regex, lematización, BoW y TF-IDF
│   ├── 03-clasificacion-hate-speech/       # Detección de discurso de odio en medios sociales
│   └── 04-analisis-sintactico-semantico/   # Parsing sintáctico (spaCy/NLTK), WSD y semántica léxica
├── 04-algoritmos-geneticos/                # Optimización combinatoria heurística (Problema de la Mochila)
├── 05-certificaciones-nvidia-dli/          # Certificados oficiales NVIDIA DLI y notebooks de evaluación
│   ├── certificados/                       # Certificados en PDF verificables
│   └── assessments/                        # Evaluaciones de certificación aprobadas
└── apuntes/                                # Base de conocimiento teórico y notas de clase (Obsidian/Markdown)
    ├── 01-intro-ia/                        # Agentes, planning, algoritmos genéticos, reinforcement learning
    ├── 02-machine-learning/                # Regresión, clasificación, clustering y anomalías
    ├── 03-redes-neuronales/                # Backpropagation, CNNs, RNNs, Attention y Transformers
    ├── 04-procesamiento-texto/             # NLP, análisis sintáctico/semántico, BERT, fairness
    ├── 05-vision-computacional/            # PyTorch Deep Learning para visión artificial
    └── media/                              # Diagramas, gráficos y esquemas ilustrativos
```

---

## 🚀 Proyectos Destacados

### 1. 🛰️ Clasificación de Imágenes Satelitales (EuroSAT Sentinel-2)
- **Directorio:** [`02-deep-learning/01-tpe-eurosat-satelital/`](02-deep-learning/01-tpe-eurosat-satelital/)
- **Notebook:** [`TPE_Redes_Neuronales_EuroSAT_Sentinel2.ipynb`](02-deep-learning/01-tpe-eurosat-satelital/TPE_Redes_Neuronales_EuroSAT_Sentinel2.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/02-deep-learning/01-tpe-eurosat-satelital/TPE_Redes_Neuronales_EuroSAT_Sentinel2.ipynb)
- **Problema:** Clasificación multiclase de 10 tipos de cobertura terrestre (LULC) sobre imágenes satelitales multiespectrales Sentinel-2 (European Space Agency).
- **Arquitecturas implementadas y comparadas:**
  1. **CNN Propia (Baseline):** Bloques convolucionales secuenciales con normalización por lotes (`BatchNormalization`) y `Dropout`.
  2. **CNN con Conexiones Residuales (Skip Connections):** Arquitectura custom inspirada en ResNet para acelerar la propagación del gradiente y mejorar la extracción de patrones multiescala.
  3. **Transfer Learning con ResNet-50:** Extracción de características preentrenadas en ImageNet combinada con fine-tuning de cabeza densa personalizada.
- **Resultados e Ingeniería:** Análisis comparativo de métricas (Accuracy, Macro F1, Matrices de Confusión) contra coste computacional (parámetros y latencia de inferencia), concluyendo que la arquitectura con skip-connections propia ofrece el mejor balance de rendimiento/recurso.

---

### 2. 🏥 Predicción de Costos de Seguros Médicos (Regresión y Ensamble)
- **Directorio:** [`01-machine-learning/01-regresion-seguros-medicos/`](01-machine-learning/01-regresion-seguros-medicos/)
- **Notebook:** [`TP_Seguros_Medicos_Regresion.ipynb`](01-machine-learning/01-regresion-seguros-medicos/TP_Seguros_Medicos_Regresion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/01-machine-learning/01-regresion-seguros-medicos/TP_Seguros_Medicos_Regresion.ipynb)
- **Problema:** Predecir con alta precisión los costos facturados a pacientes en base a variables demográficas y de salud (edad, IMC, tabaquismo, cargas familiares, región).
- **Metodología:**
  - Análisis exploratorio de datos (EDA), correlaciones y detección de patrones de no-linealidad.
  - Pipelines reproducibles de Scikit-Learn integrando `ColumnTransformer`, `StandardScaler` y `OneHotEncoder`.
  - Comparativa de modelos lineales regularizados: **Ridge (L2)**, **Lasso (L1)** y **ElasticNet**.
  - Modelo de ensamble no lineal: **Random Forest Regressor** optimizado mediante búsqueda exhaustiva con validación cruzada (`GridSearchCV`, 5-folds).
  - Diagnóstico riguroso de errores: análisis de residuos, $R^2$, RMSE y MAE.

---

### 3. 🐧 Clasificación Multiclase & EDA: Palmer Penguins
- **Directorio:** [`01-machine-learning/02-clasificacion-palmer-penguins/`](01-machine-learning/02-clasificacion-palmer-penguins/)
- **Notebooks:**
  - [`TP_2a_Preprocesamiento_Datos_EDA.ipynb`](01-machine-learning/02-clasificacion-palmer-penguins/TP_2a_Preprocesamiento_Datos_EDA.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/01-machine-learning/02-clasificacion-palmer-penguins/TP_2a_Preprocesamiento_Datos_EDA.ipynb): Análisis exploratorio (EDA) con `ydata-profiling`, pairplots con Seaborn, estrategias de imputación y pipelines iniciales.
  - [`TP_Palmer_Penguins_Clasificacion.ipynb`](01-machine-learning/02-clasificacion-palmer-penguins/TP_Palmer_Penguins_Clasificacion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/01-machine-learning/02-clasificacion-palmer-penguins/TP_Palmer_Penguins_Clasificacion.ipynb): Pipeline modular de Scikit-Learn que realiza imputación diferenciada (`SimpleImputer` con mediana para variables biométricas y moda para sexo), escalado estándar, One-Hot Encoding y validación estratificada (`StratifiedKFold`).

---

### 4. 🔤 Procesamiento de Lenguaje Natural (NLP Completo)
- **Directorio:** [`03-nlp-procesamiento-texto/`](03-nlp-procesamiento-texto/)
- **Notebooks de la Serie:**
  - **Preprocesamiento Léxico & Limpieza:** [`02_NLP_Preprocesamiento_y_Analisis_Lexico.ipynb`](03-nlp-procesamiento-texto/02-formato-y-preprocesamiento/02_NLP_Preprocesamiento_y_Analisis_Lexico.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/03-nlp-procesamiento-texto/02-formato-y-preprocesamiento/02_NLP_Preprocesamiento_y_Analisis_Lexico.ipynb): Limpieza HTML, contracciones, regex, tokenización, stop words, stemming (Porter, Snowball) y lematización (WordNet, spaCy).
  - **Representaciones Tradicionales:** [`03a_NLP_Representaciones_Tradicionales_BoW_TFIDF.ipynb`](03-nlp-procesamiento-texto/02-formato-y-preprocesamiento/03a_NLP_Representaciones_Tradicionales_BoW_TFIDF.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/03-nlp-procesamiento-texto/02-formato-y-preprocesamiento/03a_NLP_Representaciones_Tradicionales_BoW_TFIDF.ipynb): Bag-of-Words binario y por frecuencia, n-gramas, TF-IDF y matrices documento-término.
  - **Análisis Sintáctico:** [`04_NLP_Analisis_Sintactico_POS_spacy_nltk.ipynb`](03-nlp-procesamiento-texto/04-analisis-sintactico-semantico/04_NLP_Analisis_Sintactico_POS_spacy_nltk.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/03-nlp-procesamiento-texto/04-analisis-sintactico-semantico/04_NLP_Analisis_Sintactico_POS_spacy_nltk.ipynb): POS tagging con NLTK/spaCy, shallow parsing (chunking) y árboles de dependencias.
  - **Análisis Semántico y Pragmática:** [`05_NLP_Analisis_Semantico_Discurso_Pragmatica.ipynb`](03-nlp-procesamiento-texto/04-analisis-sintactico-semantico/05_NLP_Analisis_Semantico_Discurso_Pragmatica.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/03-nlp-procesamiento-texto/04-analisis-sintactico-semantico/05_NLP_Analisis_Semantico_Discurso_Pragmatica.ipynb): Semántica léxica (WordNet, synsets, hiperónimos), desambiguación de sentidos léxicos (WSD, algoritmo de Lesk) y similitud semántica.
  - **Modelos Recurrentes (Deep Learning):** [`Lenguaje_Natural_RNN_y_Bidireccional.ipynb`](03-nlp-procesamiento-texto/01-redes-recurrentes-rnn/Lenguaje_Natural_RNN_y_Bidireccional.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/03-nlp-procesamiento-texto/01-redes-recurrentes-rnn/Lenguaje_Natural_RNN_y_Bidireccional.ipynb): Mitigación de *vanishing gradient* con **RNNs Bidireccionales, LSTMs y GRUs**.
  - **Clasificación de Toxicidad:** [`DUIA_NLP_Clasificacion_Hate_Speech.ipynb`](03-nlp-procesamiento-texto/03-clasificacion-hate-speech/DUIA_NLP_Clasificacion_Hate_Speech.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/03-nlp-procesamiento-texto/03-clasificacion-hate-speech/DUIA_NLP_Clasificacion_Hate_Speech.ipynb): Clasificación de hate speech en comentarios de redes.

---

### 5. 🧬 Algoritmos Genéticos (Optimización Heurística)
- **Directorio:** [`04-algoritmos-geneticos/`](04-algoritmos-geneticos/)
- **Notebooks:**
  - [`TP4_Algoritmos_Geneticos_Mochila.ipynb`](04-algoritmos-geneticos/TP4_Algoritmos_Geneticos_Mochila.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/04-algoritmos-geneticos/TP4_Algoritmos_Geneticos_Mochila.ipynb): Resolución del problema de la mochila (*Knapsack Problem*) con biblioteca `geneticalgorithm`.
  - [`Perspectiva_Practica_Algoritmos_Geneticos.ipynb`](04-algoritmos-geneticos/Perspectiva_Practica_Algoritmos_Geneticos.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/04-algoritmos-geneticos/Perspectiva_Practica_Algoritmos_Geneticos.ipynb): Modelado de funciones de fitness y configuración de operadores de mutación/cruzamiento.
- **Dataset:** `geneticos-mochila.csv` con 19 objetos de supervivencia y sus pesos/utilidades.

---

## 🏅 Certificaciones Oficiales NVIDIA DLI

| Certificación | Emisor | Fecha | Verificación Oficial | Evaluación Práctica |
| :--- | :---: | :---: | :---: | :---: |
| **Fundamentals of Deep Learning** | NVIDIA DLI | Ago 2026 | [wbIctS24TAapK3aGhg75gg](https://learn.nvidia.com/certificates?id=wbIctS24TAapK3aGhg75gg) | [`01_Fundamentals_of_Deep_Learning_Assessment.ipynb`](05-certificaciones-nvidia-dli/assessments/01_Fundamentals_of_Deep_Learning_Assessment.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/05-certificaciones-nvidia-dli/assessments/01_Fundamentals_of_Deep_Learning_Assessment.ipynb) |
| **Building Transformer-Based NLP Applications** | NVIDIA DLI | Ago 2026 | [vywAGFnxQJWhHIJ5mK33YQ](https://learn.nvidia.com/certificates?id=vywAGFnxQJWhHIJ5mK33YQ) | [`02_Transformers_Authorship_Attribution_NeMo_BERT.ipynb`](05-certificaciones-nvidia-dli/assessments/02_Transformers_Authorship_Attribution_NeMo_BERT.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjamalegni/ML-luka/blob/main/05-certificaciones-nvidia-dli/assessments/02_Transformers_Authorship_Attribution_NeMo_BERT.ipynb) |

*Los certificados completos en formato PDF se encuentran en [`05-certificaciones-nvidia-dli/certificados/`](05-certificaciones-nvidia-dli/certificados/).*

---

## 🛠️ Tecnologías y Librerías Utilizadas

- **Lenguaje:** Python 3.10+
- **Machine Learning & Estadísticas:** Scikit-Learn, XGBoost, SciPy, NumPy, Pandas
- **Deep Learning & Frameworks:** TensorFlow / Keras, PyTorch, Torchvision, NVIDIA NeMo
- **Visualización & EDA:** Matplotlib, Seaborn
- **Entornos de Desarrollo:** Jupyter Notebooks, Google Colab, VS Code / Cursor

---

## 💻 Instalación y Ejecución Local

Para clonar y reproducir los experimentos en un entorno local:

```bash
# 1. Clonar el repositorio
git clone https://github.com/benjamalegni/ML-luka.git
cd ML-luka

# 2. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Iniciar Jupyter Lab o Notebook
jupyter lab
```

---

## 👤 Autor

**Luka Benjamin Malegni**  
- **GitHub:** [@benjamalegni](https://github.com/benjamalegni)  
- **Email:** lukabenjaminmalegni@gmail.com  
- **Formación:** Estudiante de Ingeniería de Sistemas y Diplomatura Universitaria en Inteligencia Artificial (DUIA) — UNICEN, Tandil, Argentina.
