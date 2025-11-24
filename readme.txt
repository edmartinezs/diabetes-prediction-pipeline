# 🏥 Diabetes Prediction Pipeline

Este proyecto implementa un pipeline de Machine Learning "End-to-End" para la predicción de diabetes, diseñado con un enfoque en **arquitectura funcional**, **tipado estricto** y **fundamentos matemáticos**.

El objetivo principal es demostrar la construcción de flujos de datos robustos sin depender excesivamente de abstracciones automáticas, implementando manualmente la lógica de normalización y limpieza.

## ⚙️ Características Técnicas

- **Strict Type Hints:** Uso extensivo de `TypeAlias` y anotaciones de tipo para garantizar la integridad de datos entre funciones (`numpy.ndarray`, `pd.DataFrame`).
- **Preprocesamiento Manual:** Implementación explícita del algoritmo de estandarización Z-Score:
  $$z = \frac{x - \mu}{\sigma}$$
- **Imputación Lógica:** Tratamiento de valores nulos/ceros en variables biológicas (como glucosa o presión arterial) utilizando la media condicional.
- **Validación Defensiva:** Chequeos de dimensionalidad y tipos de datos en cada etapa del pipeline.

## Dataset

Este proyecto utiliza el dataset **Pima Indians Diabetes Database**.

Para ejecutar el modelo con datos reales:
1. Descarga el archivo CSV desde el [repositorio oficial aquí](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).
2. Guarda el archivo en la carpeta principal del proyecto con el nombre: `diabetes.csv`.

> **Nota:** Si el script no encuentra el archivo `diabetes.csv`, generará automáticamente **datos sintéticos** para demostrar la funcionalidad del pipeline sin interrupciones.

##  Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone [https://https://github.com/edmartinezs/diabetes-prediction-pipeline.git](https://github.com/edmartinezs/diabetes-prediction-pipeline.git)
   cd diabetes-prediction-pipeline