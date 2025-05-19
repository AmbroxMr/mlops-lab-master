# Deploying MLflow server with Docker

Pull the MLflow Docker image:

`docker pull ghcr.io/mlflow/mlflow:v2.8.1`

Run the MLflow Docker container:

`docker run -d -p 32000:5000 ghcr.io/mlflow/mlflow:v2.8.1 mlflow server -h 0.0.0.0`

## Links to docs

Quickstart: https://mlflow.org/docs/latest/getting-started/intro-quickstart/

MLflow 2.8.1 Python API: https://mlflow.org/docs/2.8.1/python_api/mlflow.html

MLflow Model Registry: https://mlflow.org/docs/latest/model-registry/#api-workflow
