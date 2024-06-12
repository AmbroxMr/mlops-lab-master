from string import punctuation

import joblib
import mlflow
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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

    def predict(self, X, features_names=None):

        X_cleaned =  [text_preprocess(X)]
        X_vectorized = self.vectorizer.transform(X_cleaned)
        prediction = self.model.predict(X_vectorized)[0]

        return 'AI' if prediction == 1 else 'Human'