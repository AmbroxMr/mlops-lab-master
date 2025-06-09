# Registro de modelos con MLflow y Docker

En esta sección se aborda el entrenamiento y registro de un modelo de clasificación de texto utilizando **MLflow** como sistema de seguimiento y registro de modelos.

Se proporciona una *pipeline* de entrenamiento en el archivo `model_pipeline.py`, cuyo objetivo es registrar manualmente el modelo entrenado, sus métricas y parámetros en un servidor de MLflow. Este registro es esencial para asegurar la reproducibilidad, versionado y trazabilidad del modelo.

## Objetivo

- Implementar el registro manual del modelo, dado que **autolog** no es compatible con el modelo actual.
- Seguir como referencia el ejemplo `diabetes_example.py`, donde sí se usa `mlflow.sklearn.autolog()`, pero adaptar la estrategia al caso actual.
- Completar el script `model_pipeline_incomplete.py`, que ya proporciona la estructura base para realizar el registro de manera manual.
- Registrar todo lo necesario para replicar el experimento:
  - Dataset utilizado
  - Modelo entrenado
  - Métricas de evaluación
  - Parámetros de entrenamiento

## Requisitos

Instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## Despliegue del servidor de MLflow con Docker

Para registrar los modelos localmente, se puede levantar un servidor de MLflow con Docker:

1. Descargar la imagen de MLflow:

```bash
docker pull ghcr.io/mlflow/mlflow:v2.8.1
```

2. Ejecutar el contenedor en segundo plano:

```bash
docker run -d -p 32000:5000 ghcr.io/mlflow/mlflow:v2.8.1 mlflow server -h 0.0.0.0
```

El servidor quedará accesible en `http://localhost:32000`.

## Archivos relevantes

- `model_pipeline.py`: pipeline completa con el entrenamiento del modelo.
- `model_pipeline_incomplete.py`: versión inicial que debes completar añadiendo el registro manual en MLflow.
- `model_pipeline_complete.py`: referencia con el registro implementado correctamente.
- `diabetes_example.py`: ejemplo de uso de `mlflow.sklearn.autolog()` con un modelo compatible.
- `requirements.txt`: librerías necesarias para ejecutar los scripts.

## Documentación útil

- Introducción rápida:  
  https://mlflow.org/docs/latest/getting-started/intro-quickstart/

- API de MLflow (v2.8.1):  
  https://mlflow.org/docs/2.8.1/python_api/mlflow.html

- Registro de modelos en MLflow:  
  https://mlflow.org/docs/latest/model-registry/#api-workflow