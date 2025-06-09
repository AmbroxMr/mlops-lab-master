# Monitorización del modelo con Prometheus

En esta etapa del proyecto se monitoriza el modelo de clasificación de texto desplegado con Seldon Core utilizando **Prometheus**. Esta herramienta permite recopilar métricas expuestas por el modelo en tiempo real, facilitando el seguimiento de su comportamiento en producción.

## Objetivo

- Configurar Prometheus para que recoja métricas del modelo desplegado.
- Visualizar métricas como número de peticiones, tiempos de inferencia, etc.
- Comprobar que el endpoint de métricas definido por Seldon está accesible y emitiendo datos correctamente.

## Configuración de Prometheus

El archivo `prometheus.yml` incluido en esta carpeta define una configuración mínima para hacer *scraping* del modelo expuesto por Seldon Core:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'seldon-service'
    static_configs:
      - targets: ['host.docker.internal:32007']
```

- `scrape_interval: 5s`: Prometheus consultará el servicio cada 5 segundos.
- `targets`: debe coincidir con el puerto expuesto por el contenedor donde se ejecuta el modelo (puerto `32007` en este caso).

## Requisitos

- Tener desplegado previamente el modelo con Seldon Core (ver carpeta `model_serving`).
- Tener Docker instalado y funcionando correctamente.

## Despliegue de Prometheus con Docker

Ejecuta el siguiente comando desde la carpeta que contiene `prometheus.yml`:

```bash
docker run -d --name prometheus -p 9090:9090 -v ${PWD}/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```

Esto lanzará Prometheus en segundo plano y lo dejará accesible en:  
[http://localhost:9090](http://localhost:9090)

## Verificación

1. Accede a la interfaz web de Prometheus.
2. En el menú "Status > Targets", deberías ver el servicio `seldon-service` como `UP`.
3. Puedes consultar métricas como `seldon_api_executor_server_requests_seconds_count`, `seldon_api_executor_server_requests_seconds_sum`, etc.