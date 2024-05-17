############################ Transformer
#news Data: Data Cleaning (lowercase, punctuation removal, stop word removal) Preprocessing (ggf. stemming/lemmatzation, pre-trained NER??)
#preprocess the text data to clean and tokenize it (ext is split into individual words)
#define an embedding layer that maps each word to a vector representation (pre-trained Word2Vec).
#encode the entire text sequence (news headline) using the transformer encoder. This step allows the model to capture relationships between words within the headline.
#the encoded representation of the news headline is fed as input to the transformer for further processing alongside other features like TF-IDF

#ggf. TF IDF for news data: Assigns weitghts to words based on their importance in a document and the entire collection (feature engineering/ seperate model for feature extraction)

#Stock prices: missing values, outliers, consistency, ggf normalization
#Standadize or normalize

#=>Encode sequence: each sequence should represent a specific time window (day)
#Incorporate Stock Data
#==> Concatenate (during endocding) or Seperate Inputs/Embeddings 
#!Multi-head Attention with Shared Encoder/ Seperate Embeddings
#Pediction Layer on top of the Encoder: combining them
#OR Multi head Attention with seperate Encoders: Encoded representation of stock prices necessary

#Impact of news articles on MULTIPLE stocks and their corresponding stock prices:
#Multpi-output transformer/ NER and company (entity) embeddings layer (specifically mentioned)
#Multi task Learning: Train seperate models for each company
#
#TOPIC MODELLING (outputs topics represented by clusters of words) o RELEVANT entity recognition
#Latent Dirichlet Allocation (LDA) to identify latent topics (elevant to specific industries)
#feed combines representation (word embeddings + topic proportionns) as input to transformer encoder
#Multi-head attention/ multi-output transformer links them to stock prices
#
#Conditional Encoding with NER only the embeddings for companies mentioned in the headline are fed into the attention layer

#DATE AS FEATURE

#Stock Price Prediction (Regression): final dense layers for regression if you want to predict a continuous value like future stock price

#Predicting Quantile Regression OR Confidence Interval Calculation: 
#potential range of future stock prices based on the news content and other factors considered by the transformer model.

#Relevant Entity recognition????

#ggf ConvTransformer/ ConvBERT: apturing local relationships within the text data (e.g., word order) in addition to long-range dependencies
#Fine Tuning on news before feeding in transformer 

#Questions?
#fine-Tuning pre-trained models: BERT
#(feed this representation (BERT embeddings or combined representation) as input to transformer encoder?)

#Streamlit:
#Depict relevant news (sector/comapny): NER/ Topic Modelling
#Potential Top Mover als interaktives Element: Ergebnisse der Prognose und Anzeigen von Kursen

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# import Dataset
df1 = pd.read_csv("C:/Users/Felix/OneDrive/10_FAU/Semester 6/Machine Learning for Business/Datenblatt1.csv", encoding="ISO-8859-1")

# Alles außer a-z und A-Z entfernen
data1 = df1.iloc[:, 2:27]
data1.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Spaltennamen in Zahlen ändern
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data1.columns = new_index

# Großbuchstaben in Kleinbuchstaben
for i in new_index:
    data1[i] = data1[i].str.lower()

# Daten in jeder Zeile in String konvertieren und zu einem String zusammenfassen
# => Nachrichten pro Tag in einem String zusammengefasstheadlines = []
news_articles = []
for row in range(0, len(data1.index)):
    news_articles.append(' '.join(str(x) for x in data1.iloc[row, 0:25]))

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed

# Fit the vectorizer on all news articles
vectorizer.fit(news_articles)

# Transform news articles into TF-IDF vectors
tfidf_features = vectorizer.transform(news_articles)

######Einschub, um Aktienkurse für den Transformer zu encoden
# Stock price encoding model (feed-forward for this example)
price_model = Sequential()
price_model.add(Dense(32, activation="relu", input_shape=(stock_prices.shape[1],)))
price_model.add(Dense(16, activation="relu"))
price_model.add(Dense(8, activation="relu"))
price_model.compile(loss="mse", optimizer="adam")
price_model.fit(stock_prices, np.zeros((len(stock_prices), 8)), epochs=10)  # Train on dummy target


################### in diesem Abschnitt ist ein "vollständiges" Modellskript unter Einbezug von TF IDF. Die Nummern beziehen sich auf die einzelnen Schritte
# Extract top keywords based on TF-IDF weights
def get_top_keywords(tfidf_vector, num_keywords=10):
  tfidf_array = tfidf_vector.toarray()  # Convert tfidf_vector to a NumPy array
  keyword_indices = tfidf_array.argsort()[:,-num_keywords:]
  keywords = vectorizer.get_feature_names_out()[keyword_indices.ravel()]
  return keywords

###3.Feature Engineering using TF IDF Scores
# Create a feature vector for each news article with top keywords (one-hot encoded or similar)
keyword_features = []
for tfidf_vec in tfidf_features:
  top_keywords = get_top_keywords(tfidf_vec)
  # One-hot encode or create your desired feature representation based on top keywords
  keyword_feature_vector = ...  # Implement your encoding logic
  keyword_features.append(keyword_feature_vector)
    
# Text embedding for raw news text
max_vocab_size = 10000  # Adjust based on your data
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_vocab_size)
text_vectorizer.adapt(news_articles)

# Define look-back window
look_back = 5  # Number of past days (including features) to consider for prediction

def create_sequences(features, window_size):
  sequences = []
  for i in range(len(features[0]) - window_size):
    news_sequence = features[0][i:i+window_size]  # Raw news text sequence
    # Combine news sequence with your chosen feature representation (TF-IDF or topic)
    feature_sequence = features[1][i:i+window_size]  # Feature sequence (keywords or topics)
    sequence = np.concatenate((text_vectorizer(news_sequence), feature_sequence), axis=1)
    sequences.append(sequence)
  return sequences

# Prepare training and testing data
train_size = int(len(news_articles) * 0.8)
train_data, test_data = news_articles[:train_size], news_articles[train_size:]

train_features = news_articles[:train_size], keyword_features[:train_size]  # Adjust feature based on your choice
test_features = news_articles[train_size:], keyword_features[train_size:]

train_sequences = create_sequences(list(train_features), look_back)
test_sequences = create_sequences(list(test_features), look_back)

# Convert sequences to numpy arrays
train_sequences = np.array(train_sequences)
test_sequences = np.array(test_sequences)

# Build Transformer model (adjust hyperparameters as needed)
model = Sequential()
model.add(Embedding(max_vocab_size, embedding_dim=128, input_shape=(look_back, None)))
model.add(Transformer(num_layers=2, units=64, head_size=8))
model.add(Dense(units=1))  # Output layer for predicted price

# Compile and train the model
model.compile(loss="mse", optimizer="adam")
model.fit(train_sequences, train_data["Future_Price"], epochs=50, batch_size=32)

# Make predictions on test data
predicted_prices = model.predict(test_sequences)

# Evaluate model performance (optional)
# You can use metrics like mean squared error (MSE) to evaluate

# Use the model for future predictions (replace with your new data)
new_news_article = "Your new news article"
new_news_sequence = text_vectorizer.predict(np.array([new_news_article]))

# Prepare feature for new data (based on your chosen method)
if using_tf_idf:
  new_keyword_features = get_top_keywords(vectorizer.transform([new_news_article])[0])
  # Encode keywords into your desired feature representation
  new_feature_sequence = ...  # Implement your encoding logic
elif using_topic_modeling:
  new_topic_features = lda_model.transform(vectorizer.transform([new_news_article])[0])

new_sequence = np.concatenate((new_news_sequence, new_feature_sequence), axis=1)
predicted_future_price = model.predict(np.array([new_sequence]))

print(f"Predicted future price: {predicted_future_price[0][0]}")

####################### Alternative für einen Transformer (ohne TF IDF/ Topic Modelling)
import pandas as pd
from tensorflow.keras.layers import TextVectorization, Embedding, Transformer, Dense
from tensorflow.keras.models import Sequential

# Load historical data (replace with your data loading logic)
data = pd.read_csv("your_stock_data.csv")
date_col = "Date"  # Column containing the date
price_col = "Close"  # Column containing the closing price
news_col = "News_Article"  # Column containing the news text (optional)

# Prepare data
data["Future_Price"] = data[price_col].shift(-1)  # Shift price for prediction
data.dropna(inplace=True)  # Remove rows with missing values

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
new_price_data = [data[price_
###########Danach ist es abgebrochen

########### Wie wir ggf. BERT nutzen wollen                       
from transformers import BertTokenizer, TFBertModel

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Function to process news article with BERT
def get_bert_embeddings(article):
  encoded_inputs = tokenizer(article, return_tensors='tf')
  outputs = model(encoded_inputs)
  # Consider using CLS token embedding or pooling strategies
  document_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
  return document_embedding

# Optional: Combine BERT embeddings with topic proportions (as explained earlier)

# Feed BERT embeddings (or combined representation) to Transformer encoder
# ...

#################Combine embedding layers with the stock price data with the news text and feature vectors 
#Option 1: Seperate Embeddings (komplexer, aber flexibler)
# Separate embedding layers
price_embedding_dim = 8  # Adjust embedding dimension for price data

model = Sequential()
model.add(Embedding(max_vocab_size, embedding_dim=128, input_shape=(look_back, None)))  # Embedding for news text
model.add(Embedding(1, price_embedding_dim, input_length=1))  # Embedding for price (one-hot or similar)  
model.add(Transformer(num_layers=2, units=64, head_size=8))
model.add(Dense(units=1))  # Output layer for predicted price# ... (Rest of the transformer architecture)

# Concatenate embeddings before feeding to transformer layers
def create_sequences(features, window_size):
  sequences = []
  for i in range(len(features[0]) - window_size):
    news_sequence = features[0][i:i+window_size]  # Raw news text sequence
    price_sequence = features[2][i:i+window_size].reshape(-1, 1)  # Reshape price for embedding
    news_embedding = text_vectorizer(news_sequence)
    price_embedding = model.layers[1](price_sequence)  # Pass price through price embedding layer
    sequence = np.concatenate((news_embedding, price_embedding), axis=1)
    sequences.append(sequence)
  return sequences

# Update model to handle concatenated embeddings
# ... (Modify initial layers to concatenate embeddings before feeding to transformer)

#######Option 2: Concatenation
# Include stock price data in features
train_features = [news_articles[:train_size], keyword_features[:train_size] if using_tf_idf else topic_features[:train_size], train_data["Close"][:train_size].values.reshape(-1, 1)]  # Reshape price for 2D array
test_features = [news_articles[train_size:], keyword_features[train_size:] if using_tf_idf else topic_features[train_size:], test_data["Close"][train_size:].values.reshape(-1, 1)]

# Concatenate features in the sequence creation
def create_sequences(features, window_size):
  sequences = []
  for i in range(len(features[0]) - window_size):
    news_sequence = features[0][i:i+window_size]  # Raw news text sequence
    feature_sequence = np.concatenate((features[1][i:i+window_size], features[2][i:i+window_size]), axis=1)  # Combine keyword/topic features and price
    sequence = np.concatenate((text_vectorizer(news_sequence), feature_sequence), axis=1)
    sequences.append(sequence)
  return sequences

# Update model input shape to consider all concatenated features
model.add(Embedding(max_vocab_size, embedding_dim=128, input_shape=(look_back, None)))

################# Modification of model to handle concatenated embeddings for stock price prediction task with news articles and TF-IDF features:
# Embedding size (number of dimensions) should be the same for both layers
embedding_dim = 128  # Example embedding size

# Embedding layer for news articles (text data)
news_embedding = tf.keras.layers.Embedding(max_words, embedding_dim)

# Embedding layer for stock prices (numerical data)
price_embedding = tf.keras.layers.Embedding(num_stock_features, embedding_dim)

# Normalize stock prices (example)
def normalize_prices(prices):
  return (prices - prices.mean()) / (prices.std())

normalized_prices = normalize_prices(data[["Open", "High", "Low", "Close"]])  # Assuming these columns contain prices

# Assuming tfidf_features is the TF-IDF matrix from previous steps
tfidf_features = np.array(tfidf_features)  # Convert to NumPy array if necessary

# Concatenate embeddings and features
combined_inputs = tf.keras.layers.concatenate([
    news_embedding(data["Top 1"]),  # Embedded news headlines
    tfidf_features,  # TF-IDF features
    normalized_prices  # Normalized stock prices (if applicable)
])

# Assuming you have a pre-built Transformer encoder
transformer_encoder = tf.keras.layers.TransformerEncoder(...?...)  # Replace with your specific encoder configuration

# Feed the concatenated tensor to the encoder
encoded_outputs = transformer_encoder(combined_inputs)

#Hier geht es darum, den Transformer mit dem Text zu füttern und dann zur prediction zu nutzen
#####1. Create Word Embedding (ich bin mir noch nicht ganz sicher wie wir das verwenden können/sollen)
# Define embedding layer (example using pre-trained Word2Vec)
embedding_dim = 128  # Example embedding size
embeddings_index = dict(open('word2vec_embeddings.txt').read().splitlines())
num_words = min(max(word_count[word] for word in vocabulary), len(embeddings_index))
word_index = {word: i for i, word in enumerate(vocabulary[:num_words])}
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]
embedding_layer = tf.keras.layers.Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)  # Load pre-trained embeddings

# Example using trainable embedding layer
# embedding_layer = tf.keras.layers.Embedding(max_words, embedding_dim)

#####2.Transformer Encoder (Encoding Text sequence)
# Assuming you have a pre-built Transformer encoder
transformer_encoder = tf.keras.layers.TransformerEncoder(...?...)  # Replace with your specific encoder configuration

# Tokenize headlines
tokenized_headlines = [word_index[word] for headline in data["Top 1"] for word in headline]  # Assuming word_index is a dictionary mapping words to integer indices
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokenized_headlines, maxlen=max_len)  # Pad sequences to a fixed length

# Get word embeddings for each token in the sequence
headline_embeddings = embedding_layer(padded_sequences)

# Feed embeddings to Transformer encoder
encoded_outputs = transformer_encoder(headline_embeddings)
                       

########## Multi Head Attention (in diesem Fall mit Topic Modelling zu Branchen, aber wichtiges KonzeptJ
######Within the attention mechanism, each head learns to attend to different parts of the headline encoding based on its relevance to each sector embedding. 
####This allows the model to identify which sectors are most relevant to the news content, even without explicit mentions.

import tensorflow as tf
from tensorflow.keras import layers

# Define number of sectors and head count for attention
num_sectors = 10  # Replace with actual number of sectors
num_heads = 4

# Define embedding functions (replace with your actual implementation)
def create_headline_embedding(headline):
  # ... (Process and embed the headline text)
  return headline_embedding

def create_sector_embedding(sector_id):
  # ... (Load or create embedding for the sector)
  return sector_embedding

# Input layers for headline and sector embeddings
headline_input = layers.Input(shape=(headline_length,))
sector_embeddings = layers.Embedding(num_sectors, embedding_dim)(layers.Input(shape=(1,)))

# Headline embedding
headline_encoding = layers.Embedding(vocab_size, embedding_dim)(headline_input)

# Multi-head attention with learned relevance
def scaled_dot_product_attention(query, key, value):
  # ... (Implement scaled dot product attention)
  return attention_weights

attention_outputs = []
for _ in range(num_heads):
  # Project headline and sector embeddings for this head
  query = layers.Dense(embedding_dim)(headline_encoding)
  key = layers.Dense(embedding_dim)(sector_embeddings)
  value = layers.Dense(embedding_dim)(sector_embeddings)

  # Attention weights based on headline and all sector embeddings
  attention_weights = scaled_dot_product_attention(query, key, value)

  # Weighted sum of sector embeddings based on attention weights
  context_vector = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([attention_weights, value])  
  attention_outputs.append(context_vector)

# Concatenate outputs from all attention heads
attention_output = layers.Concatenate(axis=-1)(attention_outputs)

# Use the attention output for further processing in your model
# ...

############################## Multi Layer Output, um verschiedene Aktienausgaben zu machen
from tensorflow.keras.layers import Dense

# Assuming you have a pre-built Transformer encoder and encoded outputs for the news headlines

# Define a Dense layer for each company's stock price prediction
amazon_output = Dense(1, activation="linear", name="amazon_output")(encoded_outputs)  # Linear activation for regression
google_output = Dense(1, activation="linear", name="google_output")(encoded_outputs)
apple_output = Dense(1, activation="linear", name="apple_output")(encoded_outputs)

# Combine outputs into a list
multi_outputs = [amazon_output, google_output, apple_output]

# Define the model with the Transformer encoder and multi-outputs
model = tf.keras.Model(inputs=[...], outputs=multi_outputs)  # Replace [...] with your transformer encoder input layer(s)

# Compile the model
model.compile(loss="mse", optimizer="adam")  # Mean Squared Error loss for regression

# Train the model on your prepared data
model.fit(X_train, y_train, epochs=...)  # X_train: Encoded news, y_train: Stock prices for all three companies

################################# Konfidenzintervale für den prognostizierten Wert
import tensorflow as tf

# Assuming you have your trained transformer model with two outputs:
# - mean_prediction (predicted mean stock price)
# - std_prediction (predicted standard deviation of stock price)

# Define the confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate the z-score for the chosen confidence level
z_score = tf.constant(1.96)  # Assuming normal distribution

# Calculate the confidence interval bounds
lower_bound = mean_prediction - z_score * std_prediction
upper_bound = mean_prediction + z_score * std_prediction

##################### Alternativ: Quantile Regression

import tensorflow as tf
from tensorflow.keras import layers

# Define the desired quantiles (e.g., 0.025 and 0.975 for 95% confidence interval)
lower_quantile = 0.025
upper_quantile = 0.975

# Define the transformer model (replace with your actual model architecture)
def create_transformer_model():
  # ... (Your transformer model definition here)
  # Include two separate output layers for the lower and upper quantiles
  outputs = layers.Dense(1, activation="linear", name="lower_quantile")(x)
  outputs = layers.Dense(1, activation="linear", name="upper_quantile")(outputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

# Instantiate the model
model = create_transformer_model()

# Define the quantile loss function (using Huber loss for robustness)
def quantile_loss(y_true, y_pred):
  quantiles = tf.constant([lower_quantile, upper_quantile])
  return tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM) * (
      quantiles * tf.maximum(y_true - y_pred, 0) + (1 - quantiles) * tf.maximum(y_pred - y_true, 0))

# Compile the model with the quantile loss function
model.compile(loss=quantile_loss, optimizer="adam")

# Train your model with appropriate training data (news articles and corresponding stock prices)
# ...

# After training, use the model to predict quantiles for new data
predictions = model.predict(new_data)

# Extract the lower and upper quantile predictions
lower_bound = predictions[:, 0]
upper_bound = predictions[:, 1]

# Utilize the lower and upper bounds for further analysis or visualization
# ..


###########NOCH 2 ANSÄTZE ZUM VERKETTEN DER DATEN
################################Feeding news headlines and company information 
######## into a single transformer encoder for stock price prediction
#### Option 1. Concatenation and Single Encoder:
import tensorflow as tf
from tensorflow.keras import layers

# Define maximum headline length and vocabulary size
max_headline_len = 100  # Replace with appropriate value
vocab_size = 10000  # Replace with appropriate value

# Define embedding functions (replace with your actual implementation)
def create_headline_embedding(headline):
  # ... (Process and embed the headline text)
  return headline_embedding

def create_company_embedding(company_id):
  # ... (Encode company ID using one-hot encoding or embedding model)
  return company_embedding

# Input layers for headline and company information
headline_input = layers.Input(shape=(max_headline_len,))
company_id_input = layers.Input(shape=(1,))

# Headline embedding
headline_encoding = layers.Embedding(vocab_size, embedding_dim)(headline_input)

# Company embedding (one-hot encoding for simplicity)
company_embedding = layers.Embedding(num_companies, embedding_dim)(company_id_input)

# Concatenate headline and company embeddings
combined_features = layers.Concatenate(axis=-1)([headline_encoding, company_embedding])

# Single transformer encoder
encoder_output = layers.TransformerEncoder(num_layers=2, d_model=128)(combined_features)

# Downstream layers for regression (stock price prediction)
dense1 = layers.Dense(64, activation="relu")(encoder_output)
output = layers.Dense(1, activation="linear")(dense1)  # Single neuron for regression

# Model definition
model = tf.keras.Model(inputs=[headline_input, company_id_input], outputs=output)

# Compile and train the model (replace with your training data and optimizer)
model.compile(loss="mse", optimizer="adam")
model.fit([headline_data, company_id_data], stock_price_data, epochs=10)

# Use the model for prediction
predicted_price = model.predict([new_headline, new_company_id])


###### Option 2. Multi-Input Transformer with Separate Encoders: 
import tensorflow as tf
from tensorflow.keras import layers

# Define maximum headline length and vocabulary size
max_headline_len = 100  # Replace with appropriate value
vocab_size = 10000  # Replace with appropriate value

# Define embedding functions (replace with your actual implementation)
def create_headline_embedding(headline):
  # ... (Process and embed the headline text)
  return headline_embedding

def create_company_embedding(company_id):
  # ... (Encode company ID using one-hot encoding or embedding model)
  return company_embedding

# Input layers for headline and company information
headline_input = layers.Input(shape=(max_headline_len,))
company_id_input = layers.Input(shape=(1,))

# Headline embedding
headline_encoding = layers.Embedding(vocab_size, embedding_dim)(headline_input)

# Company embedding (one-hot encoding for simplicity)
company_embedding = layers.Embedding(num_companies, embedding_dim)(company_id_input)

# Separate encoders for headline and company information
headline_encoder = layers.TransformerEncoder(num_layers=2, d_model=128)(headline_encoding)
company_encoder = layers.TransformerEncoder(num_layers=1, d_model=64)(company_embedding)

# Attention layer
attention_weights = layers.Attention()([headline_encoder, company_encoder])

# Apply attention weights to headline encoding for company-specific focus
context_vector = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([attention_weights, headline_encoder])

# Combine headline content and company-specific context
combined_representation = layers.Concatenate(axis=-1)([context_vector, company_encoder])

# Downstream layers for regression (stock price prediction)
dense1 = layers.Dense(64, activation="relu")(combined_representation)
output = layers.Dense(1, activation="linear")(dense1)  # Single neuron for regression

# Model definition
model = tf.keras.Model(inputs=[headline_input, company_id_input], outputs=output)

# Compile and train the model (replace with your training data and optimizer)
model.compile(loss="mse", optimizer="adam")
model.fit([headline_data, company_id_data], stock_price_data, epochs=10)

# Use the model for prediction
predicted_price = model.predict([new_headline, new_company_id])




##################################Optional: Topic Modelling mit LDA (to represent each article with its topic proportions)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Preprocess news article text (cleaning, tokenization)
# ...

# Define and fit the LDA model
vectorizer = TfidfVectorizer(max_features=max_words)
lda_model = LatentDirichletAllocation(n_components=k, random_state=42)
lda_model.fit(vectorizer.fit_transform(corpus))

# Get document topic proportions for each news article
def get_topic_proportions(article):
  article_vector = vectorizer.transform([article])
  return lda_model.transform(article_vector)[0]

# Combine word embeddings and topic proportions for each article
def get_combined_representation(article, word_embeddings):
  topic_props = get_topic_proportions(article)
  # Convert article to word indices and lookup word embeddings
  article_embeddings = word_embeddings[np.array([word_index[word] for word in article.split()])]
  # Combine word embeddings and topic proportions (consider weighting)
  combined_representation = np.concatenate([article_embeddings.mean(axis=0), topic_props])
  return combined_representation


#####################Topic Modelling als Teil des Feature Engineering
from sklearn.decomposition import LatentDirichletAllocation

# Define the number of topics
num_topics = 5  # Adjust num_topics as needed

# Create and fit LDA model
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_topics = lda_model.fit_transform(tfidf_features)



