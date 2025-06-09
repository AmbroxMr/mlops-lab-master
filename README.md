# MLOps Práctico: Clasificación de Texto IA vs Humano

Este repositorio contiene una práctica completa de MLOps basada en un caso de uso sencillo: entrenar y desplegar un modelo de clasificación de texto para detectar si un texto ha sido generado por una inteligencia artificial o por un ser humano.

El proyecto está diseñado para ejecutarse de forma local con Python, sin necesidad de Kubernetes, aunque todo el código es fácilmente adaptable a un entorno orquestado. Se cubre el ciclo de vida completo de un modelo de machine learning, automatizando cada etapa con herramientas modernas del ecosistema MLOps.

## Tecnologías utilizadas

- MLflow – Registro de experimentos y modelos
- Seldon Core – Despliegue del modelo en producción
- Prometheus – Monitorización de métricas del modelo
- Prefect – Orquestación de pipelines
- Docker – Contenerización de servicios
- Streamlit – Frontend para interacción con el modelo

## Referencias recomendadas

- Whitepaper de Google sobre el ciclo de vida del ML:  
  https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf

Durante mi doctorado, escribí dos artículos que están directamente relacionados con la infraestructura utilizada en esta práctica:

- Artículo sobre la infraestructura MLOps que sirve de base para este repositorio:  
  https://ieeexplore.ieee.org/abstract/document/10588954/

- Artículo con un caso de uso completo en imágenes satélite, donde se automatiza el ciclo de vida de un modelo de ML:  
  https://www.sciencedirect.com/science/article/pii/S0167739X24004631

Recomiendo la lectura de ambos si te interesa profundizar en la aplicación práctica de MLOps.

## Estructura del repositorio

Cada carpeta representa una fase del ciclo de vida del modelo. Todas incluyen su propio README.md explicativo. El orden recomendado para explorar el proyecto es:

1. `model_registry/`  
   Entrenamiento y registro del modelo con MLflow.  
   Se utiliza Docker para desplegar el servidor de MLflow y gestionar versiones y reproducibilidad del modelo.

2. `model_serving/`  
   Despliegue del modelo en producción con Seldon Core, utilizando el modelo registrado en MLflow.

3. `model_monitoring/`  
   Monitorización del modelo desplegado mediante Prometheus. Permite evaluar métricas de rendimiento en tiempo real.

4. `model_orchestration/`  
   Orquestación del pipeline de entrenamiento y despliegue usando Prefect. Se definen tareas y dependencias para automatizar el ciclo de vida del modelo.

5. `frontend/`  
   Aplicación web construida con Streamlit para interactuar con el modelo en producción. Permite enviar textos y visualizar si son clasificados como generados por IA o por humanos.

## Dataset

- Dataset completo disponible en Kaggle:  
  https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

Dado que el dataset completo es bastante grande, se ha subido una versión reducida al repositorio para facilitar la ejecución local y acelerar el entrenamiento. Aunque esta versión limitada reduce el rendimiento del modelo, es suficiente para validar el flujo de trabajo y hacer pruebas.

Si lo deseas, puedes sustituir el dataset reducido por el completo descargándolo directamente desde Kaggle.

## Recursos de aprendizaje recomendados

- Curso introductorio de Andrew Ng sobre ML en producción (Coursera, 10 horas):  
  https://www.coursera.org/learn/introduction-to-machine-learning-in-production

- Especialización en MLOps de Duke University (Coursera):  
  https://www.coursera.org/specializations/mlops-machine-learning-duke

- Curso de MLOps en Google Cloud (Coursera):  
  https://www.coursera.org/learn/gcp-production-ml-systems

## Objetivo

Este repositorio tiene un enfoque educativo y práctico, ideal para personas que quieran iniciarse en MLOps aplicando herramientas reales sin necesidad de infraestructura compleja. Cada componente está desacoplado y puede ejecutarse de forma independiente, facilitando la comprensión de cada etapa del proceso.