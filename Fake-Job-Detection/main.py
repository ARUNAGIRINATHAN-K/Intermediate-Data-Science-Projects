# Import libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 1. Load the dataset
file_path = "fake_job_postings.xlsx"  # Update to .xlsx if using Excel
try:
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)  # Use pd.read_excel("fake_job_postings.xlsx") for Excel
        print("Dataset loaded successfully. Shape:", df.shape)
        print("Columns:", df.columns.tolist())
    else:
        raise FileNotFoundError(f"File {file_path} not found in {os.getcwd()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Text Preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Combine relevant text columns
try:
    df['text'] = df['title'].fillna('') + " " + df['description'].fillna('') + " " + \
                 df['requirements'].fillna('') + " " + df['benefits'].fillna('')
except KeyError as e:
    print(f"Column not found: {e}. Available columns: {df.columns.tolist()}")
    exit()

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# 3. Feature Extraction
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_text'])
y = df['fraudulent']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# 6. Predictions and Evaluation
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()