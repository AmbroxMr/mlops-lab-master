# Model deployment using Seldon Core

Build the docker image and run the container using the following commands:

`docker build -t deploy-model .`

`docker run -p 32005:5000 -p 32006:9000 -p 32007:6000 deploy-model`

## Links to docs

- [Packaging a Python model for Seldon Core using Docker](https://github.com/SeldonIO/seldon-core/blob/master/doc/source/python/python_wrapping_docker.md#packaging-a-python-model-for-seldon-core-using-docker)

- [Download artifacts from MLflow](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.artifacts.html)

## Stress test

pip install locust
locust -f locustfile.py
docker stats