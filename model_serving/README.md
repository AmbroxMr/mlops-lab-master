# Model deployment using Seldon Core

Build the docker image and run the container using the following commands:

`docker build -t deploy-model .`

`docker run -p 32005:5000 -p 32006:9000 deploy-model`