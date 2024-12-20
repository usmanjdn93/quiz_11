import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

data = pd.read_csv('C:/Users/usman/Downloads/reviews_dataset.csv') 

print(data.shape)
print(data.head)
news = data['news']
type = data['type']

def preprocess_text(text):

    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]

    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

print(news)
news = news.apply(preprocess_text)
print(news)

vectorizer = CountVectorizer(ngram_range=(2, 3))
vector_news = vectorizer.fit_transform(news)

print(vector_news)

x_train, x_test, y_train, y_test = train_test_split(vector_news, type, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = confusion_matrix(y_test, y_pred)
report2 = classification_report(y_test, y_pred)
print(f"accuracy: {accuracy}")
print("confusion matrix:")
print(report)


print('classification score')
print(report2)

