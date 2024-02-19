import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Adding this line to download the required resource

# Load dataset
#df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
df = pd.read_json('mini.jo', lines=True)

# Basic analysis
print(df['category'].value_counts())

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

df['text'] = df['headline'] + ' ' + df['short_description']
df['text'] = df['text'].apply(preprocess_text)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

# Vectorizing text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Building the model
model = MultinomialNB()

# Training the model with a progress bar
with tqdm(total=1, desc="Training Model") as pbar:
    model.fit(X_train_tfidf, y_train)
    pbar.update(1)

# Predicting and evaluating
predictions = model.predict(X_test_tfidf)
print(classification_report(y_test, predictions))
