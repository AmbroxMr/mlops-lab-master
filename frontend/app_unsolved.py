import streamlit as st
from seldon_core.seldon_client import microservice_api_rest_seldon_message

st.set_page_config(page_title="Seldon Inference App", layout="centered")

st.write("AI or Human Text Classifier")

if st.button("Predict"):
    pass