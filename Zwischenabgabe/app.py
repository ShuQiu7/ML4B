import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import string
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
nltk.download('stopwords')

# Load your trained model
model = tf.keras.models.load_model('path_to_your_model')

# Define the text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Define the sentiment analysis function
def get_binary_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return 1 if polarity >= 0 else 0

# Define the function to create sequences
def create_sequences(news, price, sentiment, window_size):
    sequences = []
    news_seq = news[-window_size:]
    sentiment_seq = sentiment[-window_size:]
    combined_seq = np.hstack([news_seq, np.array(sentiment_seq).reshape(-1, 1)])        
    sequences.append(combined_seq)
    return np.array(sequences, dtype=np.float32)

# Define the TextVectorization layer (make sure to match parameters from training)
vectorizer = tf.keras.layers.TextVectorization(max_tokens=100000, output_sequence_length=391, output_mode='int')
# You need to adapt the vectorizer to some training data or load the fitted vectorizer from your training script
# For example:
# vectorizer.adapt(your_training_text_data)

# Streamlit interface
st.title('Stock Price Prediction')

# Input panel for news headline
news_headline = st.text_input('News Headline', 'Enter the news headline here')

# Input panel for yesterday's closing price
closing_price_yesterday = st.number_input('Yesterday\'s Closing Price', min_value=0.0, value=150.0, step=0.01)

# Predict button
if st.button('Predict Stock Price'):
    # Preprocess the news headline
    processed_news = preprocess_text(news_headline)
    
    # Vectorize the news headline
    news_sequence = vectorizer(np.array([processed_news]))
    news_sequence = np.array(news_sequence)
    
    # Get sentiment of the news headline
    sentiment = get_binary_sentiment(news_headline)
    sentiment_label = "positive" if sentiment == 1 else "negative"
    
    # Display the sentiment analysis result
    st.write(f'Sentiment Analysis: {sentiment_label}')
    
    # Use the input closing price and the news headline to create the input sequence
    new_price_data = np.array([[closing_price_yesterday]])
    look_back = 5  # Define the look-back period as used during training
    
    # Creating a mock sequence for demonstration purposes
    # You may need to adjust this to fit your actual use case
    recent_prices = np.array([closing_price_yesterday] * (look_back - 1) + [closing_price_yesterday])
    recent_sentiments = np.array([sentiment] * look_back)
    
    # Combine the data into a sequence
    sequence = create_sequences(news_sequence, recent_prices, recent_sentiments, look_back)
    
    # Predict the stock price
    predicted_price = model.predict(sequence)
    
    st.write(f'Predicted Stock Price: ${predicted_price[0][0]:.2f}')

streamlit run app.py
