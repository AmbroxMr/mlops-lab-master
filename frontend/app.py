# app.py

import streamlit as st
from seldon_core.seldon_client import microservice_api_rest_seldon_message

st.set_page_config(page_title="Seldon Inference App", layout="centered")

st.write("AI or Human Text Classifier")

# Input de usuario
user_input = st.text_area("Write some text to classify as AI or Human:", 
                          placeholder="Type your text here...", 
                          height=200)

if st.button("Predict"):
    try:

        response_rest = microservice_api_rest_seldon_message(
            microservice_endpoint="localhost:32006",
            str_data=user_input
        )

        # Mostrar campos relevantes
        st.subheader("Prediction response:")
        st.text(response_rest.response.strData)

        st.subheader("Model input:")
        st.text(response_rest.request.strData)

    except Exception as e:
        st.error(f"An error occurred: {e}")
