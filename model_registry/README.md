# Deploying MLflow server with Docker

Pull the MLflow Docker image:

`docker pull ghcr.io/mlflow/mlflow:v2.8.1`

Run the MLflow Docker container:

`docker run -d -p 32000:5000 ghcr.io/mlflow/mlflow:v2.8.1 mlflow server`