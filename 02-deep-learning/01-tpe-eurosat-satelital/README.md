# 🛰️ Clasificación de Imágenes Satelitales (EuroSAT Sentinel-2)
### Trabajo Práctico Especial — Redes Neuronales (DUIA, UNICEN)

Este proyecto desarrolla y compara tres arquitecturas de Deep Learning para clasificación multiclase sobre el dataset satelital **EuroSAT (Sentinel-2 RGB)** con 10 clases de uso y cobertura del suelo (*Land Use and Land Cover - LULC*).

---

## 🏗️ Modelos Desarrollados y Comparados

1. **CNN Base (Custom):**
   - Arquitectura convolucional propia con bloques Conv2D + BatchNorm + MaxPooling + Dropout.
   - Enfoque directo para benchmarking y control de sobreajuste.

2. **CNN con Conexiones Residuales (Skip Connections - Estilo ResNet):**
   - Arquitectura personalizada integrando bloques residuales (shortcut / skip connections).
   - Facilita el flujo del gradiente en capas profundas y mitiga el desvanecimiento del gradiente (*vanishing gradient*).

3. **Transfer Learning con ResNet-50:**
   - Backbone preentrenado en **ImageNet** con congelamiento (*freezing*) de capas base y fine-tuning de cabeza clasificadora densa con regularización.

---

## 📊 Métricas y Conclusiones de Ingeniería

| Modelo | Accuracy | F1-Score Macro | Eficiencia de Entrenamiento | Parámetros Entrenables |
| :--- | :---: | :---: | :---: | :---: |
| **CNN Base** | ~85-88% | ~0.86 | Rápido | ~1.2M |
| **ResNet Propia (Skip Connections)** | ~90-92% | ~0.91 | Equilibrado | ~2.5M |
| **ResNet-50 (Transfer Learning)** | **~94-96%** | **~0.95** | Mayor tiempo por época | ~24M (Fine-tuned top) |

> **Conclusión de Ingeniería:** Se analiza en profundidad el trade-off entre coste computacional (FLOPS, memoria, tiempo por época) y ganancia marginal de precisión, demostrando cómo las conexiones residuales propias ofrecen un ratio rendimiento/costo sobresaliente en entornos con recursos acotados.

---

## 📂 Dataset EuroSAT
El dataset contiene 27.000 imágenes Sentinel-2 (64x64) distribuidas en 10 clases:
`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.

- Descarga oficial: [EuroSAT Dataset (GitHub / Zenodo)](https://github.com/phelber/eurosat)
- O vía TensorFlow Datasets: `tfds.load('eurosat/rgb')`
- O vía PyTorch / Torchvision: `torchvision.datasets.EuroSAT`
