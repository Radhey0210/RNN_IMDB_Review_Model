import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model with RELU activation
model = load_model('simple_rnn_model.h5')
model.summary()

# Get the word index from the IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)
    sentiment= 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

## Streamlit App
import streamlit as st

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment")

user_input = st.text_area("Review Text", height=200)

if st.button("Predict"):
    if user_input:
        sentiment, score = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {score:.4f}")
    else:
        st.write("Please enter a review")