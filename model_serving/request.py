from seldon_core.seldon_client import microservice_api_rest_seldon_message, microservice_api_grpc_seldon_message

data = b"Initializes the S3 client with the endpoint URL for MinIO."

response_rest = microservice_api_rest_seldon_message(microservice_endpoint="127.0.0.1:32006", str_data=data)
response_grpc = microservice_api_grpc_seldon_message(microservice_endpoint="127.0.0.1:32005", str_data=data)

print(response_rest)
print("------------")
print(response_grpc)
