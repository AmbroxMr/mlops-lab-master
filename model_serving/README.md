# Despliegue del modelo con Seldon Core

En esta práctica se utiliza **Seldon Core** para desplegar el modelo de clasificación de texto que ha sido previamente entrenado y registrado con **MLflow**. El despliegue se realiza mediante un *wrapper* que integra el modelo directamente desde MLflow y lo expone a través de APIs REST y gRPC.

## Objetivo

- Crear una imagen Docker que contenga el modelo y su lógica de inferencia.
- Usar la clase `TextAIWrapper` para envolver el modelo, implementando al menos los métodos `__init__`, `predict`, y opcionalmente `metrics`.
- Exponer el modelo como un servicio con Seldon Core, permitiendo el acceso vía HTTP y gRPC.
- Exponer métricas del modelo para su monitorización en tiempo real con Prometheus.
- Realizar una prueba de carga usando Locust para simular tráfico y verificar el correcto funcionamiento del servicio.

## Requisitos

Instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## Estructura de archivos

- `TextAIWrapper.py`: clase del modelo ya completada.
- `TextAIWrapper_unsolved.py`: versión inicial sin completar, útil como práctica.
- `Dockerfile`: definición de la imagen Docker con Seldon Core y el modelo.
- `request.py`: ejemplo de cómo realizar peticiones al modelo desplegado.
- `locustfile.py`: script para prueba de carga con Locust.
- `requirements.txt`: dependencias necesarias.

## Construcción y ejecución del contenedor

1. Construir la imagen Docker:

```bash
docker build -t deploy-model .
```

2. Ejecutar el contenedor localmente:

```bash
docker run -p 32005:5000 -p 32006:9000 -p 32007:6000 deploy-model
```

- Puerto 5000: API REST
- Puerto 9000: API gRPC
- Puerto 6000: métricas Prometheus

## Prueba de carga con Locust

1. Instala Locust:

```bash
pip install locust
```

2. Ejecuta la prueba:

```bash
locust -f locustfile.py
```

3. (Opcional) Monitoriza uso de recursos del contenedor:

```bash
docker stats
```

## Documentación útil

- Empaquetado de modelos Python con Seldon Core usando Docker:  
  https://github.com/SeldonIO/seldon-core/blob/master/doc/source/python/python_wrapping_docker.md#packaging-a-python-model-for-seldon-core-using-docker

- Descarga de artefactos desde MLflow:  
  https://mlflow.org/docs/latest/api_reference/python_api/mlflow.artifacts.html