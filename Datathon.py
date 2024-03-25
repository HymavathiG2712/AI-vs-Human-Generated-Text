#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('aihu.csv')


# In[3]:


df.head(4)
#0 - human 
#1 - ai


# In[4]:


# df.rename(columns = {'Data':'text'}, inplace = True) 
df.rename(columns = {'generated':'class'}, inplace = True) 


# In[5]:


# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check for class balance
print("\nClass distribution:")
print(df['class'].value_counts())

# Basic preprocessing: removing missing values
df.dropna(inplace=True)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.2, random_state=42)

# Output the size of the splits
print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])


# In[6]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set maximum number of words to keep, based on word frequency
MAX_WORDS = 10000  
MAX_SEQUENCE_LENGTH = 250  

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert texts to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print("Shape of training data:", X_train_padded.shape)
print("Shape of testing data:", X_test_padded.shape)


# In[7]:


import numpy as np

# Calculate simple linguistic features
X_train_lengths = np.array([len(text.split()) for text in X_train])  # Text length
X_train_avg_word_length = np.array([np.mean([len(word) for word in text.split()]) for text in X_train])  # Average word length

X_test_lengths = np.array([len(text.split()) for text in X_test])
X_test_avg_word_length = np.array([np.mean([len(word) for word in text.split()]) for text in X_test])

# Stack features together for MLP input
X_train_features = np.vstack((X_train_lengths, X_train_avg_word_length)).T
X_test_features = np.vstack((X_test_lengths, X_test_avg_word_length)).T


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Model parameters
VOCAB_SIZE = min(MAX_WORDS, len(tokenizer.word_index) + 1)  # Vocabulary size
EMBEDDING_DIM = 100  # Size of the embedding vectors 
FILTERS = 128  # Number of filters for the Conv layer
KERNEL_SIZE = 5  # Size of the kernel for the Conv layer
DENSE_UNITS = 10  # Number of units in the Dense layer


# In[9]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Dense
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Flatten

# CNN part for text
text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
text_features = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(text_input)
text_features = Conv1D(FILTERS, KERNEL_SIZE, activation='relu')(text_features)
text_features = GlobalMaxPooling1D()(text_features)

# MLP part for linguistic features
linguistic_input = Input(shape=(X_train_features.shape[1],))
linguistic_features = Dense(DENSE_UNITS, activation='relu')(linguistic_input)

# Concatenate the outputs of the two models
combined_features = concatenate([text_features, linguistic_features])

# Final classification layer
classification_output = Dense(1, activation='sigmoid')(combined_features)

model = Model(inputs=[text_input, linguistic_input], outputs=classification_output)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[10]:


from tensorflow.keras.callbacks import EarlyStopping

# Convert labels to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)

# Train the model
history = model.fit([X_train_padded, X_train_features], y_train, 
                    batch_size=32, 
                    epochs=10, 
                    validation_data=([X_test_padded, X_test_features], y_test),
                    callbacks=[early_stopping])

# Note: The batch_size and epochs are set to small numbers for demonstration and might need adjustment.


# In[11]:


import matplotlib.pyplot as plt

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([X_test_padded, X_test_features], y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plotting training history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)


# In[12]:


def predict_text(text):
    # Tokenize and pad the input text
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    text_length = len(text.split())
    avg_word_length = np.mean([len(word) for word in text.split()])
    features = np.array([[text_length, avg_word_length]])
    
    # Make a prediction
    prediction = model.predict([padded_seq, features])
    
    # Interpret the prediction
    if prediction >= 0.5:
        print("The text is predicted to be an AI Generated Essay.")
    else:
        print("The text is predicted to be a Human Written Essay.")

# Example usage
user_input = input("Please enter a sentence to classify: ")
predict_text(user_input)


# In[13]:


user_input = input("Please enter a sentence to classify: ")
predict_text(user_input)

