import pandas as pd
from minio import Minio
from prefect import flow, task

@flow
def retraining_pipeline(bucket_name: str="hello-world", object_key: str="AI_Human_1k.csv"):
    text, target = ingest_data_from_s3(bucket_name, object_key)
    cleaned_text = data_cleaning(text)
    vectorized_data = vectorize_data(cleaned_text)
    train_and_upload_model(vectorized_data, target)

@task
def ingest_data_from_s3(bucket_name: str, object_key: str) -> tuple[pd.Series, pd.Series]:
    client = Minio("play.min.io",
        access_key="Q3AM3UQ867SPQQA43P2F",
        secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
    )
    http_response = client.get_object(bucket_name=bucket_name, object_name=object_key)
    df = pd.read_csv(http_response)
    return df["text"], df["generated"]

@task
def data_cleaning(text: pd.Series) -> pd.Series:
    raise NotImplementedError

@task
def vectorize_data(clean_text: pd.Series) -> pd.Series:
    raise NotImplementedError

@task
def train_and_upload_model(vectorized_data: pd.Series, target: pd.Series):
    raise NotImplementedError

if __name__ == "__main__":
    retraining_pipeline.serve()
