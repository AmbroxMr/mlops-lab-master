FROM prefecthq/prefect:2.19.2-python3.10-conda

COPY requirements.txt .
RUN pip install -r requirements.txt --trusted-host pypi.python.org --no-cache-dir

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

COPY retraining_pipeline.py /opt/prefect/flows/retraining_pipeline.py

ENV MLFLOW_TRACKING_URI="http://127.0.0.1:32000"

CMD ["python", "flows/retraining_pipeline.py"]