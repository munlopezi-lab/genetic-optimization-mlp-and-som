# Optimización de Hiperparámetros mediante Algoritmos Genéticos con PyGAD

## Descripción

Este proyecto explora el uso de **algoritmos genéticos** para la optimización automática de hiperparámetros en modelos de aprendizaje automático.

Se desarrollaron dos casos de estudio utilizando la biblioteca **PyGAD**:

1. Optimización de una red neuronal multicapa (**MLPClassifier**) de Scikit-Learn.
2. Optimización de un **Self-Organizing Map (SOM)** implementado con MiniSom.

El objetivo es analizar la capacidad de los algoritmos evolutivos para encontrar configuraciones de hiperparámetros que mejoren el desempeño de modelos supervisados y no supervisados, reduciendo la necesidad de realizar búsquedas manuales exhaustivas.

---

## Objetivos

- Implementar algoritmos genéticos utilizando la biblioteca PyGAD.
- Representar hiperparámetros mediante cromosomas.
- Automatizar la búsqueda de configuraciones óptimas.
- Evaluar el desempeño de diferentes configuraciones generadas durante la evolución.
- Aplicar técnicas de optimización evolutiva tanto a modelos supervisados como no supervisados.

---

## Dataset Utilizado

Se utilizó el **Wine Dataset**, disponible en Scikit-Learn.

Este conjunto de datos contiene información química de tres variedades de vino obtenidas a partir de análisis de laboratorio.

### Características del Dataset

- 178 muestras.
- 13 variables numéricas.
- 3 clases de vino.

Las variables incluyen mediciones como:

- Alcohol
- Ácido málico
- Cenizas
- Alcalinidad de las cenizas
- Magnesio
- Fenoles totales
- Flavonoides
- Intensidad de color
- Prolina

Este conjunto de datos fue utilizado tanto para la optimización del **MLPClassifier** como del **Self-Organizing Map (SOM)**.

---

## Fundamento Teórico

Los algoritmos genéticos son métodos de optimización inspirados en los principios de la evolución biológica y la selección natural.

Cada solución candidata es representada mediante un cromosoma compuesto por genes que codifican parámetros específicos del problema. A lo largo de múltiples generaciones, la población evoluciona mediante operadores genéticos como:

- Selección
- Cruza (Crossover)
- Mutación

El objetivo es maximizar una función de aptitud (*fitness function*) que evalúa la calidad de cada solución.

En este proyecto, cada individuo representa una configuración particular de hiperparámetros para un modelo de aprendizaje automático.

---

# Caso de Estudio 1: Optimización de MLPClassifier

## Descripción

Se implementó un algoritmo genético para optimizar automáticamente los hiperparámetros de una red neuronal multicapa (`MLPClassifier`) de Scikit-Learn.

Cada cromosoma representa una configuración específica del modelo, permitiendo explorar múltiples combinaciones de hiperparámetros de manera eficiente.

## Hiperparámetros Optimizados

Entre los parámetros considerados se encuentran:

- Número de neuronas ocultas.
- Función de activación.
- Optimizador.
- Tasa de aprendizaje.
- Momentum.
- Número máximo de iteraciones.

## Evaluación

La aptitud de cada individuo se calcula a partir del desempeño del clasificador sobre el conjunto de datos utilizado en el experimento.

La función objetivo busca maximizar la capacidad predictiva del modelo.

### Mejor Configuración Encontrada

| Hiperparámetro | Valor Óptimo |
|--------------|-------------|
| Capas ocultas | 26, 30, 40, 59, 9, 17, 38, 54, 11 |
| Función de activación | tanh |
| Optimizador | lbfgs |
| Tasa de aprendizaje | 0.008 |
| Máximo de iteraciones | 280 |
| Momentum (SGD) | 0.81 |

### Resultados

Tras la optimización evolutiva, el modelo alcanzó una precisión del **100%** sobre el conjunto de evaluación utilizado en el experimento.

> **Nota:** Este resultado corresponde a la configuración y partición de datos utilizadas durante el estudio.

---

# Caso de Estudio 2: Optimización de Self-Organizing Maps (SOM)

## Descripción

Se implementó un segundo algoritmo genético para optimizar los hiperparámetros de un Self-Organizing Map utilizando la biblioteca MiniSom.

Los SOM son redes neuronales no supervisadas capaces de proyectar datos de alta dimensión sobre un espacio bidimensional preservando relaciones topológicas entre observaciones.

## Hiperparámetros Optimizados

Entre los parámetros considerados se encuentran:

- Altura del mapa.
- Anchura del mapa.
- Tasa de aprendizaje.
- Radio de vecindad (Sigma).
- Función de vecindad.
- Número de iteraciones.

## Evaluación

La calidad de cada configuración se evalúa mediante métricas de agrupamiento calculadas después del entrenamiento del SOM.

La función objetivo busca maximizar la calidad de los clusters obtenidos.

### Mejor Configuración Encontrada

| Hiperparámetro | Valor Óptimo |
|----------------|--------------|
| Ancho del mapa | 9 |
| Alto del mapa | 24 |
| Radio de vecindad (σ) | 2.07 |
| Tasa de aprendizaje | 0.09 |
| Función de vecindad | gaussian |
| Iteraciones | 1278 |

### Resultados

El mejor modelo obtenido alcanzó un **Adjusted Rand Index (ARI) de 0.967**.

Este valor indica una separación casi perfecta de las tres clases del conjunto de datos.

La topología del mapa (9×24) sugiere que una estructura alargada permitió una mejor discriminación de las características del dataset. Además, la función de vecindad **gaussian** mostró un mejor desempeño que las funciones **triangular**, **mexican hat** y **bubble** para este problema.

---

## Metodología General

El procedimiento seguido en ambos casos de estudio fue:

1. Definición del espacio genético.
2. Codificación de hiperparámetros en cromosomas.
3. Generación de una población inicial.
4. Evaluación de la aptitud de cada individuo.
5. Aplicación de operadores genéticos:
   - Selección
   - Cruza
   - Mutación
6. Evolución de la población durante múltiples generaciones.
7. Obtención de la mejor solución encontrada.

---

## Resultados Generales

### MLPClassifier

| Métrica | Valor |
|----------|----------|
| Accuracy | 100% |

### SOM

| Métrica | Valor |
|----------|----------|
| ARI | 96.70% |

---


## Reporte Técnico

El análisis completo de los experimentos, metodología, resultados y conclusiones puede consultarse en:

[Reporte técnico](reporte_tecnico.pdf)

---

## Tecnologías Utilizadas

- Python
- PyGAD
- Scikit-Learn
- MiniSom
- NumPy
- Matplotlib

---

## Estructura del Proyecto

```text
.
├── GA_MLPClassifier.py
├── GA_SOM.py
├── README.md
├── reporte_tecnico.pdf
```

---

## Conclusiones

Los resultados obtenidos muestran que los algoritmos genéticos pueden utilizarse eficazmente para automatizar la búsqueda de hiperparámetros en modelos supervisados y no supervisados.

En ambos casos de estudio fue posible encontrar configuraciones con alto desempeño sin recurrir a búsquedas exhaustivas manuales. Los resultados sugieren que las técnicas de optimización evolutiva constituyen una alternativa viable para el ajuste automático de modelos de aprendizaje automático.

---

## Aprendizajes

Durante este proyecto se aplicaron conceptos relacionados con:

- Algoritmos Genéticos.
- Optimización Evolutiva.
- Metaheurísticas.
- Búsqueda de Hiperparámetros.
- Redes Neuronales Artificiales.
- Self-Organizing Maps.
- Aprendizaje Supervisado.
- Aprendizaje No Supervisado.
- Evaluación de Modelos de Machine Learning.

---

## Autor

Jairo Isaac Muñoz López

Estudiante de Licenciatura en Matemáticas Aplicadas.

GitHub: https://github.com/munlopezi-lab
