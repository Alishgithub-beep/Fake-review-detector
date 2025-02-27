import streamlit as st
import pickle
from sklearn.pipeline import Pipeline
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings('ignore')

# Download stopwords and wordnet if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Define the same text processing function used when training the model
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word.lower() not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Load the trained model pipeline
model_path = "pipeline.pkl"

try:
    with open(model_path, "rb") as model_file:
        pipeline = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file 'pipeline.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Streamlit UI
st.title("Fake Review Detection App")
st.write("Enter a review below to check if it's real or fake.")

user_input = st.text_area("Enter Review:")

if st.button("Check Review"):
    if user_input.strip():  # Ensure user input is not empty
        try:
            prediction = pipeline.predict([user_input])[0] #pass user_input as a list
            if prediction =='OR':
                st.write("Review is real")
            else:
                st.write("Review is computer-generated")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter a review before checking.")