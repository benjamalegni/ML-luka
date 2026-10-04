# 🧬 Algoritmos Genéticos - Optimización Heurística

Este módulo aborda la resolución de problemas de optimización combinatoria utilizando **Algoritmos Genéticos (GA)** como parte de la diplomatura DUIA (UNICEN).

## 🎒 Problema de la Mochila (Knapsack Problem)
Optimización de selección de elementos para maximizar la utilidad bajo una restricción estricta de capacidad de peso (15 kg).

### Contenido del módulo:
- `geneticos-mochila.csv`: Dataset con lista de 19 ítems (cuchillo, filtro de agua, pedernal, carpa ligera, raciones, etc.), sus pesos en kilogramos y su valor de utilidad.
- **Cuaderno Interactivo en Google Colab**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CWlLIN9wVooWrRranssWSvkp6Qskrbl0?usp=sharing)

### Conceptos Clave Implementados:
1. **Representación cromosómica (Genotipo)**: Vectores binarios indicando inclusión/exclusión del objeto.
2. **Función de Fitness**: Suma del valor de utilidad con penalización o descarte por exceso de peso.
3. **Operadores Genéticos**:
   - Selección por torneo / ruleta.
   - Cruzamiento en un punto / multipunto.
   - Mutación bit-flip con tasa adaptativa.
4. **Evolución y Convergencia**: Evolución generacional y análisis de curva de fitness máximo y medio a través de las generaciones.
