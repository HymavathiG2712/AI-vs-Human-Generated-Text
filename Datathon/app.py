import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Make sure to import your model and tokenizer here

# Load your trained model and tokenizer
# model = ... (Assuming your model is loaded here)
# tokenizer = ... (And your tokenizer here)

from tensorflow.keras.models import load_model

model = load_model('/home/hygumm/Datathon/model')

# Load your tokenizer
with open('/home/hygumm/Datathon/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_text(text, model, tokenizer, MAX_SEQUENCE_LENGTH):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    text_length = len(text.split())
    avg_word_length = np.mean([len(word) for word in text.split()])
    features = np.array([[text_length, avg_word_length]])
    prediction = model.predict([padded_seq, features])
    return prediction

def main():
    st.title("AI vs Human Written Text Classifier")
    
    user_input = st.text_area("Enter text to classify", "")
    
    if st.button("Predict"):
        # Assuming your constants and model are loaded here
        prediction = predict_text(user_input, model, tokenizer, MAX_SEQUENCE_LENGTH)
        
        if prediction >= 0.5:
            st.write("The text is predicted to be an AI Generated Essay.")
        else:
            st.write("The text is predicted to be a Human Written Essay.")

if __name__ == "__main__":
    main()
