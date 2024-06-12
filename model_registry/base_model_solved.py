from string import punctuation

import joblib
import mlflow
import nltk
import pandas as pd
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

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

mlflow.set_tracking_uri("http://host.docker.internal:32000")

df = pd.read_csv('AI_Human_1k.csv')
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, test_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)

dataset_train = mlflow.data.from_pandas(
    train_df, name="AI_Human_1k", targets="generated"
)
dataset_test = mlflow.data.from_pandas(
    test_df, name="AI_Human_1k", targets="generated"
)

experiment_id = mlflow.get_experiment_by_name("text_AI").experiment_id
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_input(dataset_train, context="training")
    mlflow.log_input(dataset_test, context="testing")

    train_df.loc[:, 'clean_text'] = train_df['text'].apply(text_preprocess)
    test_df.loc[:, 'clean_text'] = test_df['text'].apply(text_preprocess)

    X_train = train_df['clean_text']
    X_test = test_df['clean_text']
    y_train = train_df['generated']
    y_test = test_df['generated']

    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

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
        code_paths=["base_model.py"],
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
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

    mlflow.log_artifact('logistic_regression_model.joblib', 'text_AI_model')
    mlflow.log_artifact('tfidf_vectorizer.joblib', 'text_AI_model')

    df_shuffled.to_csv('data.csv', index=False)
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    mlflow.log_artifact('data.csv', 'text_AI_model')
    mlflow.log_artifact('train_data.csv', 'text_AI_model')
    mlflow.log_artifact('test_data.csv', 'text_AI_model')
