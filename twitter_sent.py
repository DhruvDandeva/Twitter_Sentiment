import zipfile
with zipfile.ZipFile('/content/sentiment140.zip', 'r') as zip_ref:
    zip_ref.extractall('/content')



data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')

data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
data = data[['target', 'text']]

# (0 = Negative, 4 = Positive) and map to 0 and 1
data['target'] = data['target'].replace(4, 1)

data.dropna(inplace=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

data['text'] = data['text'].apply(preprocess_text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1 

max_length = 50  
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=max_length, padding='post')
y = data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

batch_size = 64
epochs = 5
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

model.save('lstm_sentiment_model.keras')
print("Model Saved!")
loaded_model = tf.keras.models.load_model('lstm_sentiment_model.keras')
print("Model Loaded Successfully!")
