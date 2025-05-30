from string import punctuation

import joblib
import mlflow
import pandas as pd
from minio import Minio
from prefect import flow, task
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def text_preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(punctuation)
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return " ".join(clean_tokens)

@flow
def retraining_pipeline(bucket_name: str="hello-world", object_key: str="AI_Human_1k.csv"):
    text, target = ingest_data_from_s3(bucket_name, object_key)
    cleaned_text = data_cleaning(text)
    vectorized_data = vectorize_data(cleaned_text)
    train_model(vectorized_data, target)

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
    clean_text = text.apply(text_preprocess)
    return clean_text

@task
def vectorize_data(clean_text: pd.Series) -> pd.Series:

    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
    )
    text_vectorized = vectorizer.fit_transform(clean_text).toarray()
    return text_vectorized

@task
def train_model(vectorized_data: pd.Series, target: pd.Series):

    X_train, X_test, y_train, y_test = train_test_split(vectorized_data, target, test_size=0.2, random_state=42)

    experiment_id = mlflow.get_experiment_by_name("text_AI").experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("vectorizer_max_features", 2000)
        mlflow.log_param("vectorizer_min_df", 5)
        mlflow.log_param("vectorizer_max_df", 0.8)
        mlflow.log_param("vectorizer_ngram_range", "(1, 2)")
        mlflow.log_param("model", "LogisticRegression")

        model = LogisticRegression()
        model.fit(X_train, y_train)

        input_schema = Schema([ColSpec("string","input text")])
        output_schema = Schema([ColSpec("string","output label")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = {"input text": "Transportation is a large necessity in most countries worldwide. With no doubt, cars, buses, and other means of transportation make going from place to place easier and faster. However there's always a negative pollution. Although mobile transportation are a huge part of daily lives, we are endangering the Earth with harmful greenhouse gases, which could be suppressed."}

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="text_AI_model",
            registered_model_name="text_AI_model_logistic_regression",
            signature=signature,
            input_example=input_example,
        )

        y_pred = model.predict(X_test)

        classification_report_str = classification_report(y_test, y_pred)
        print("Classification Report for Logistic Regression:\n", classification_report_str)

        mlflow.log_metric("accuracy", model.score(X_test, y_test))

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        for key, value in report_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_metric(f"{key}_{sub_key}", sub_value)
            else:
                mlflow.log_metric(key, value)

        joblib.dump(model, 'logistic_regression_model.joblib')

        mlflow.log_artifact('logistic_regression_model.joblib', 'text_AI_model')
        mlflow.log_artifact('tfidf_vectorizer.joblib', 'text_AI_model')

if __name__ == "__main__":
    retraining_pipeline.serve()
