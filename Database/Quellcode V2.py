import pandas as pd
import tensorflow as tf 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, TFBertModel
from sklearn.decomposition import LatentDirichletAllocation

# Load historical data (replace with your data loading logic)
data = pd.read_csv("C:/Users/Felix/OneDrive/10_FAU/Semester 6/Machine Learning for Business/GOOGLE.csv", encoding="utf-8", delimiter=";")

date_col = data.iloc[:, 0]  # Column containing the date
price_col = data.iloc[:, 2]  # Column containing the closing price
news_col = data.iloc[:, 1]  # Column containing the news text (optional)

# Prepare data
data.iloc[:, 4] = data.iloc[:, 2].shift(-1)  # Shift price for prediction
data.dropna(inplace=True)  # Remove rows with missing values

print(data.iloc[1, 4], data.iloc[1, 2])
# Feature engineering (optional)
# You can add additional features based on your data, like news sentiment score

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Text Preprocessing (if using news articles)
def preprocess_text(text):
  # Text cleaning steps like tokenization, lowercasing, etc.
  # ...
  return processed_text

train_news = train_data[news_col].apply(preprocess_text)
test_news = test_data[news_col].apply(preprocess_text)

# Text Vectorization (if using news articles)
max_vocab_size = 10000  # Adjust based on your data
vectorizer = TextVectorization(max_tokens=max_vocab_size)
vectorizer.fit_on_texts(train_news.tolist() + test_news.tolist())

train_news_sequences = vectorizer(train_news.tolist())
test_news_sequences = vectorizer(test_news.tolist())

# Combining features (consider appropriate concatenation based on your data)
train_features = {
    "news": train_news_sequences,
    "price": train_data[price_col].values.reshape(-1, 1)  # Reshape for 2D array
}
test_features = {
    "news": test_news_sequences,
    "price": test_data[price_col].values.reshape(-1, 1)
}

# Define look-back window
look_back = 5  # Number of past days (including news) to consider for prediction

def create_sequences(features, window_size):
  sequences = []
  for i in range(len(features["price"]) - window_size):
    news_sequence = features["news"][i:i+window_size]
    price_sequence = features["price"][i:i+window_size]
    sequence = np.concatenate((news_sequence, price_sequence), axis=1)  # Concatenate news and price
    sequences.append(sequence)
  return sequences

train_sequences = create_sequences(train_features.copy(), look_back)
test_sequences = create_sequences(test_features.copy(), look_back)

# Convert sequences to numpy arrays
train_sequences = np.array(train_sequences)
test_sequences = np.array(test_sequences)

# Build Transformer model
model = Sequential()
model.add(Embedding(max_vocab_size, embedding_dim=128, input_shape=(look_back, None)))  # Embedding for news
model.add(Transformer(num_layers=2, units=64, head_size=8))  # Adjust hyperparameters as needed
model.add(Dense(units=1))  # Output layer for predicted price

# Compile model
model.compile(loss="mse", optimizer="adam")

# Train the model
model.fit(train_sequences, train_data["Future_Price"], epochs=50, batch_size=32)

# Make predictions on test data
predicted_prices = model.predict(test_sequences)

# Evaluate model performance (optional)
# You can use metrics like mean squared error (MSE) to evaluate

# Use the model for future predictions (replace with your new data)
new_news = preprocess_text("Your new news article")  # Preprocess new news article
new_news_sequence = vectorizer(np.array([new_news]))
new_price_data = predicted_prices
