import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
import requests
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load historical data (replace with your data loading logic)

data = pd.read_csv("C:/Users/Felix/OneDrive/10_FAU/Semester 6/Machine Learning for Business/GOOGLE.csv", encoding="utf-8", delimiter=";")
date_col = "Date"  # Column containing the date
price_col = "Close"  # Column containing the closing price
news_col = "News_Article"  # Column containing the news text (optional)

# Prepare data
data.iloc[:, 4] = data.iloc[:, 2].shift(-1)  # Shift price for prediction
data.dropna(inplace=True)  # Remove rows with missing values

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Text Preprocessing (if using news articles)
def preprocess_text(text):
  # Lowercase text
  text = text.lower()

  # Remove punctuation
  text = text.translate(str.maketrans('', '', string.punctuation))

  # Remove stopwords
  stop_words = stopwords.words('english')
  text = ' '.join([word for word in text.split() if word not in stop_words])

  return text

# Apply preprocessing to each headline
train_news = train_data[news_col].apply(preprocess_text)
test_news = test_data[news_col].apply(preprocess_text)

# Text Vectorization (if using news articles)
max_vocab_size = 10000  # Adjust based on your data
vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_vocab_size)
vectorizer.adapt(train_news.tolist() + test_news.tolist())

train_news_sequences = vectorizer(train_news.tolist())
test_news_sequences = vectorizer(test_news.tolist())

# Combining features (consider appropriate concatenation based on your data)
train_features = {
    "news": train_news_sequences,
    "price": train_data[price_col].values.reshape(-1, 1),  # Reshape for 2D array
    #"keywords": keyword_features, #noch nicht definiert
    #"topics": lda_topics #noch nicht definiert
}
test_features = {
    "news": test_news_sequences,
    "price": test_data[price_col].values.reshape(-1, 1),
    #"keywords": keyword_features, #noch nicht definiert
    #"topics": lda_topics #noch nicht definiert
}

# Define look-back window
look_back = 5  # Number of past days (including news) to consider for prediction

def create_sequences(features, window_size):
  sequences = []
  for i in range(len(features["price"]) - window_size):
    news_sequence = features["news"][i:i+window_size]
    price_sequence = features["price"][i:i+window_size]
    #keyword_sequence = features["keywords"][i:i+window_size]
    #topic_sequence = features["topics"][i:i+window_size].reshape(-1, 1)  # Reshape for concatenation
    sequence = np.concatenate((news_sequence, price_sequence), axis=1)  # Concatenate all sequences, keyword and topic sequence to be added
    sequences.append(sequence)
  return sequences

train_sequences = create_sequences(train_features.copy(), look_back)
test_sequences = create_sequences(test_features.copy(), look_back)

# Convert sequences to numpy arrays
train_sequences = np.array(train_sequences)
test_sequences = np.array(test_sequences)

# Build Transformer model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_vocab_size, output_dim = 150, input_shape=(look_back, None)))  # Embedding for news
#model.add(tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64))  # Adjust hyperparameters as needed; funktioniert noch nicht.
model.add(tf.keras.layers.Dense(units=1))  # Output layer for predicted price

# Compile model
model.compile(loss="mse", optimizer="adam")

# Train the model
model.fit(train_sequences, train_data[price_col][look_back:], epochs=10, batch_size=32)

# Make predictions on test data
#predicted_prices = model.predict(test_sequences)

#####################Neue Nachrichten einbauen

# Define your API key and endpoint
api_key = '5a9f2b08c11f43b0a3fcde731a6b8707'
endpoint = 'https://newsapi.org/v2/everything'

# Parameters for the API request
params = {
    'q': 'finance',
    'apiKey': api_key,
    'language': 'en',
    'sortBy': 'publishedAt'
}

# Function to fetch news and return a DataFrame
def fetch_financial_news(api_endpoint, parameters):
    response = requests.get(api_endpoint, params=parameters)
    news_data = response.json()

    articles = news_data.get('articles', [])
    
    # Extract headlines and publication dates
    data = [(article['title'], article['publishedAt']) for article in articles]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Headline', 'Date'])
    
    # Convert 'Date' from string to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

# Fetch financial news
news_df = fetch_financial_news(endpoint, params)

# Display the DataFrame
print(news_df)

###bzw. bei existierendem DataFrame (wie in unserem Fall)
existing_df = pd.DataFrame(columns=['Headline', 'Date'])  # Or load from an existing file

# Fetch new financial news
new_news_df = fetch_financial_news(endpoint, params)

# Append new rows to the existing DataFrame
updated_df = existing_df.append(new_news_df, ignore_index=True)

# Optionally, save the updated DataFrame to a CSV file
updated_df.to_csv('financial_news.csv', index=False)
