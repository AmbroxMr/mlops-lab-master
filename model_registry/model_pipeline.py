# Import punctuation symbols
from string import punctuation

# Import required libraries
import joblib
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

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset containing text and binary labels (0 = human, 1 = AI)
df = pd.read_csv('AI_Human_1k.csv')

# Shuffle the dataset for randomness
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset into training and test sets (80% train, 20% test)
train_df, test_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)

# Apply preprocessing to text in both training and test sets
train_df.loc[:, 'clean_text'] = train_df['text'].apply(text_preprocess)
test_df.loc[:, 'clean_text'] = test_df['text'].apply(text_preprocess)

# Define features (X) and labels (y)
X_train = train_df['clean_text']
X_test = test_df['clean_text']
y_train = train_df['generated']
y_test = test_df['generated']

# Convert text into TF-IDF feature vectors with n-grams (1,2)
vectorizer = TfidfVectorizer(
    max_features=2000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2),
)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Train logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = model.predict(X_test)

# Print classification performance metrics
classification_report_str = classification_report(y_test, y_pred)
print("Classification Report for Logistic Regression:\n", classification_report_str)

# Save model and vectorizer locally for reuse
joblib.dump(model, 'logistic_regression_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Save full, train, and test datasets to CSV
df_shuffled.to_csv('data.csv', index=False)
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)