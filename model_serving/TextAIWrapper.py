from string import punctuation
import joblib
import mlflow
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import time

def text_preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(punctuation)
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return " ".join(clean_tokens)

class TextAIWrapper:

    def __init__(self, model_version="latest", file_path="files"):

        model_uri = f"models:/text_AI_model_logistic_regression/{model_version}"
        model_file = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/logistic_regression_model.joblib", dst_path=file_path)
        vectorizer_file = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/tfidf_vectorizer.joblib", dst_path=file_path)
        self.model = joblib.load(model_file)
        self.vectorizer = joblib.load(vectorizer_file)

        # Métricas internas
        self.total_requests = 0
        self.total_prediction_time = 0
        self.total_text_length = 0
        self.prediction_counts = {"AI": 0, "Human": 0}

    def predict(self, X, features_names=None):

        start = time.time()
        self.total_requests += 1
        self.total_text_length += len(X)

        X_cleaned = [text_preprocess(X)]
        X_vectorized = self.vectorizer.transform(X_cleaned)
        prediction = self.model.predict(X_vectorized)[0]

        prediction_label = 'AI' if prediction == 1 else 'Human'
        self.prediction_counts[prediction_label] += 1

        self.total_prediction_time += (time.time() - start)
        return prediction_label

    def metrics(self):
        avg_prediction_time = self.total_prediction_time / self.total_requests if self.total_requests else 0
        avg_text_length = self.total_text_length / self.total_requests if self.total_requests else 0

        return [
            {"type": "COUNTER", "key": "requests_total", "value": self.total_requests},
            {"type": "COUNTER", "key": "predicted_ai_total", "value": self.prediction_counts["AI"]},
            {"type": "COUNTER", "key": "predicted_human_total", "value": self.prediction_counts["Human"]},
            {"type": "GAUGE", "key": "average_prediction_time_seconds", "value": avg_prediction_time},
            {"type": "GAUGE", "key": "average_input_text_length", "value": avg_text_length}
        ]
