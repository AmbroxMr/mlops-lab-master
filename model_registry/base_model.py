from string import punctuation

import joblib
import mlflow
import nltk
import pandas as pd
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

experiment_id = mlflow.get_experiment_by_name("text_AI").experiment_id
with mlflow.start_run(experiment_id=experiment_id):

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

    model = LogisticRegression()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="text_AI_model",
        registered_model_name="text_AI_model_logistic_regression",
    )

    y_pred = model.predict(X_test)

    classification_report_str = classification_report(y_test, y_pred)
    print("Classification Report for Logistic Regression:\n", classification_report_str)

    joblib.dump(model, 'logistic_regression_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

    df_shuffled.to_csv('data.csv', index=False)
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
