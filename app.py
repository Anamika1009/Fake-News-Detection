import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import StringIO

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

# -------------------------------
# Download NLTK data (Sirf OOV check ke liye zaroori)
# -------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
download_nltk_data()

# -------------------------------
# Load trained model and tokenizer (Using your filenames)
# -------------------------------
MODEL_PATH = "fake_news_model.h5"  # <-- Your correct file
TOKENIZER_PATH = "tokenizer.pkl" # <-- Your correct file

try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    MAXLEN = 1000  # This was set during your training
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model/tokenizer: {e}")
    st.info("The Live Prediction tab may not function.")
    model_loaded = False

# -------------------------------
# Helper function: clean and preprocess text
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', ' ', text)   # remove URLs
    text = re.sub(r'<[^>]+>', ' ', text)           # remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)      # remove special chars/numbers
    text = re.sub(r'\s+', ' ', text).strip()       # remove extra spaces
    return text

# -------------------------------
# Helper function: Get Model Summary
# -------------------------------
def get_model_summary(model):
    s = StringIO()
    model.summary(print_fn=lambda x: s.write(x + '\n'))
    return s.getvalue()

# -------------------------------
# Streamlit layout
# -------------------------------
st.set_page_config(page_title="📰 Fake News Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4B0082;'>📰 Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("About")
st.sidebar.info("""
This dashboard performs **Live Prediction** to predict if any news article is Real or Fake.
""")

if model_loaded:
    st.sidebar.header("Model Architecture")
    st.sidebar.code(get_model_summary(model))
else:
    st.sidebar.warning("Model not loaded. Architecture cannot be displayed.")

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Main App (Koi Tab Nahin)
# -------------------------------
st.header("🔎 Enter News for Prediction")

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area("Paste news article here:", height=200, value=st.session_state.user_input, key="user_input_area")

if st.button("Predict"):
    st.session_state.user_input = st.session_state.user_input_area # Sync state
    if not model_loaded:
        st.error("Model is not loaded. Cannot perform prediction.")
    elif user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        text_clean = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([text_clean])[0]
        padded = pad_sequences([seq], maxlen=MAXLEN, padding='post', truncating='post')

        # OOV check
        total_words = [t for t in seq if t != 0]
        known_ratio = 1.0
        if len(total_words) > 0:
            known_words = [t for t in total_words if t > 1] # Assuming OOV is 1
            known_ratio = len(known_words) / len(total_words)

        # Predict
        prob_real = float(model.predict(padded)[0][0])
        prob_fake = 1 - prob_real
        label = "🟢 Real News" if prob_real > 0.5 else "🔴 Fake News"

        st.markdown(f"<h2 style='text-align:center;'>{label}</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Prediction Confidence")
            st.bar_chart(pd.DataFrame({'Probability': [prob_fake, prob_real], 'Label': ['Fake', 'Real']}).set_index('Label'))
            if known_ratio < 0.5:
                st.warning(f"Warning: {100*(1-known_ratio):.1f}% of words are OOV (Out-of-Vocabulary). Prediction may be unreliable.")
            else:
                st.success(f"Model recognized {known_ratio*100:.1f}% of the words.")
        with col2:
            st.subheader("WordCloud of Entered Text")
            if text_clean:
                wc_input = WordCloud(width=600, height=300, background_color="white").generate(text_clean)
                fig_input, ax_input = plt.subplots()
                ax_input.imshow(wc_input, interpolation="bilinear")
                ax_input.axis("off")
                st.pyplot(fig_input)
            else:
                st.info("No words to generate WordCloud.")
        
        st.session_state.history.append({'Text': user_input[:50]+"...", 'Prediction': label, 'Prob_Real': round(prob_real, 3), 'Known_Ratio': round(known_ratio, 3)})

st.markdown("---")
st.subheader("📜 Session Prediction History")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history).iloc[::-1]
    st.dataframe(df_history, use_container_width=True)
else:
    st.info("No predictions yet. Enter news above.")

