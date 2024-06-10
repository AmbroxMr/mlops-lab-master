# Deploying Prefect server with Docker

Pull the Prefect Docker image:
`docker pull prefecthq/prefect:2.19.2-python3.10`

Run the backend Postgres database, which is required for the Prefect server:
`docker run -d --name prefect-postgres -v prefectdb:/var/lib/postgresql/data -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=prefect postgres:latest`

Run the Prefect server, specifying the connection URL to the Postgres database:
`docker run -d -p 32001:5000 -e PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/prefect prefecthq/prefect:2.19.2-python3.10-conda  prefect server start --port 5000`