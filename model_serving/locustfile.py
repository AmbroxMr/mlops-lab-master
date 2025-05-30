from locust import task, between
from seldon_core.seldon_client import microservice_api_rest_seldon_message, microservice_api_grpc_seldon_message
from locust import User, events
import time

data = b"Initializes the S3 client with the endpoint URL for MinIO."

class SeldonRestUser(User):
    wait_time = between(1, 2)

    @task
    def call_rest(self):
        start_time = time.time()
        try:
            response = microservice_api_rest_seldon_message(
                microservice_endpoint="host.docker.internal:32006",
                str_data=data
            )
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(request_type="REST", name="rest_call", response_time=total_time, response_length=0, context={}, exception=None)
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(request_type="REST", name="rest_call", response_time=total_time, response_length=0, context={}, exception=e)


class SeldonGrpcUser(User):
    wait_time = between(1, 2)

    @task
    def call_grpc(self):
        start_time = time.time()
        try:
            response = microservice_api_grpc_seldon_message(
                microservice_endpoint="host.docker.internal:32005",
                str_data=data
            )
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(request_type="gRPC", name="grpc_call", response_time=total_time, response_length=0, context={}, exception=None)
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(request_type="gRPC", name="grpc_call", response_time=total_time, response_length=0, context={}, exception=e)
