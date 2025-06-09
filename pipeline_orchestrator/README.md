# Orquestación de la pipeline de reentrenamiento con Prefect

En esta etapa se automatiza la pipeline de reentrenamiento del modelo de clasificación de texto usando **Prefect 2.x**. La pipeline descarga los datos desde MinIO, los limpia, vectoriza y vuelve a entrenar el modelo, subiéndolo posteriormente al registro de MLflow.

## Objetivo

- Crear una pipeline modular y orquestada con Prefect.
- Automatizar tareas comunes del ciclo de vida del modelo: ingestión, preprocesado, entrenamiento y subida.
- Ejecutar y desplegar la pipeline con Docker.
- Conectarse a MinIO público para descargar los datos.
- Registrar el modelo final en un servidor MLflow ya levantado.

## Requisitos

- Docker instalado y funcionando.
- El servidor de MLflow debe estar accesible en `http://host.docker.internal:32000`.
- El servidor de Prefect debe estar accesible en `http://host.docker.internal:32001/api`.

## Configuración del entorno

1. Descargar la imagen de Prefect (opcional, ya incluida en el `Dockerfile`):

```bash
docker pull prefecthq/prefect:2.19.2-python3.10
```

2. Lanzar la base de datos PostgreSQL para Prefect Server:

```bash
docker run -d --name prefect-postgres \
  -v prefectdb:/var/lib/postgresql/data \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=prefect \
  postgres:latest
```

3. Levantar el Prefect Server:

```bash
docker run -d -p 32001:5000 \
  -e PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://postgres:postgres@host.docker.internal:5432/prefect \
  prefecthq/prefect:2.19.2-python3.10 \
  prefect server start --port 5000 --host 0.0.0.0
```

## Despliegue de la pipeline

Una vez completado el archivo `retraining_pipeline.py`, puedes construir la imagen:

```bash
docker build -t deploy-retrain-flow .
```

Y ejecutarla:

```bash
docker run deploy-retrain-flow
```

Esto ejecutará el flujo de Prefect cargado en el contenedor y lo conectará con el servidor Prefect y MLflow.

## Sobre el dataset

El método `ingest_data_from_s3` utiliza MinIO público (play.min.io) como fuente de datos. Esta plataforma elimina los archivos tras varios días, por lo que si los datos ya no están disponibles, deberás subirlos de nuevo.

## Notas

- Asegúrate de que los servicios de MLflow y Prefect estén corriendo antes de ejecutar el contenedor.
- Los pasos `data_cleaning`, `vectorize_data` y `train_and_upload_model` deben ser completados para que la pipeline funcione correctamente.