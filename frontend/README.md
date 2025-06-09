# Interfaz web para clasificación de texto: ¿IA o humano?

En esta parte del proyecto se crea una interfaz web interactiva utilizando **Streamlit**, que permite enviar texto al modelo desplegado con **Seldon Core** y visualizar la predicción de forma sencilla.

La aplicación hace uso del cliente de Seldon Core para conectarse al modelo expuesto por el servicio REST (puerto `32006`) y muestra tanto la entrada enviada como la respuesta generada.

## Objetivo

- Proporcionar una interfaz accesible y visual para probar el modelo en producción.
- Facilitar pruebas rápidas sin necesidad de escribir scripts adicionales.
- Mostrar de forma clara tanto la entrada enviada como la predicción recibida del modelo.

## Requisitos

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Ejecución de la aplicación

Ejecuta el siguiente comando desde esta carpeta:

```bash
streamlit run app.py
```

La interfaz estará disponible en:  
[http://localhost:8501](http://localhost:8501)

## Funcionamiento

1. Escribe un texto en el área de entrada.
2. Pulsa el botón "Predict".
3. La aplicación enviará el texto al modelo desplegado vía REST.
4. Se mostrará:
   - El texto enviado (input)
   - La predicción devuelta por el modelo (output)

## Notas

- Asegúrate de que el modelo esté desplegado con Seldon Core y accesible en `localhost:32006`.
- El endpoint utilizado es el de **microservicio REST** de Seldon Core, usando la función `microservice_api_rest_seldon_message`.