import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from collections import Counter

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

from io import StringIO

# -------------------------------
# Download NLTK data (Fixes the LookupError)
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
# (This is the correct version)
# -------------------------------
def clean_text(text, for_ngrams=False):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', ' ', text)   # remove URLs
    text = re.sub(r'<[^>]+>', ' ', text)           # remove HTML tags
    if not for_ngrams:
        text = re.sub(r'[^a-z\s]', ' ', text)      # remove special chars/numbers
    else:
        # For n-grams, keep basic punctuation
        text = re.sub(r'[^a-z\s\.\?!]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()       # remove extra spaces
    return text

# -------------------------------
# Load data (for Analysis Tab)
# --- MODIFIED: Loads only 1000 rows to save RAM ---
# -------------------------------
@st.cache_data
def load_data():
    try:
        # Only load 1000 rows to prevent RAM crash
        fake_df = pd.read_csv('fake.csv', nrows=1000) 
        true_df = pd.read_csv('true.csv', nrows=1000)
        
        # --- This logic was missing from your code ---
        fake_df['label'] = 0
        true_df['label'] = 1
        
        df = pd.concat([fake_df, true_df], axis=0)
        
        if 'text' not in df.columns or 'title' not in df.columns:
             st.error("Dataset (sample) is missing 'text' or 'title' column.")
             return pd.DataFrame()

        df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
        
        # Create cleaned columns for different analyses
        df['cleaned_text_wc'] = df['full_text'].apply(lambda x: clean_text(x, for_ngrams=False))
        df['cleaned_text_ngrams'] = df['full_text'].apply(lambda x: clean_text(x, for_ngrams=True))
        
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df # <-- This return is essential
    
    except FileNotFoundError:
        st.sidebar.error("For analysis, 'fake.csv' or 'true.csv' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# This line must be outside the function
df = load_data()

# -------------------------------
# Helper function: Create Word Cloud
# --- ADDED: This function was missing ---
# -------------------------------
def create_wordcloud(text_data, title):
    st.subheader(title)
    if pd.isna(text_data) or not text_data:
        st.info("No text data available for this category.")
        return
    try:
        wc = WordCloud(width=800, height=400, background_color="white", max_words=100, collocations=False).generate(text_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate WordCloud: {e}")

# -------------------------------
# Helper function: Get N-grams
# --- ADDED: This function was missing ---
# -------------------------------
def get_top_ngrams(corpus, n=2, top_k=20):
    tokens = word_tokenize(corpus)
    n_grams = ngrams(tokens, n)
    n_gram_counts = Counter(n_grams)
    top_ngrams = n_gram_counts.most_common(top_k)
    # Format for plotting
    df_ngrams = pd.DataFrame(top_ngrams, columns=['n_gram', 'count'])
    df_ngrams['n_gram'] = df_ngrams['n_gram'].apply(lambda x: ' '.join(x))
    return df_ngrams

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
st.markdown("<h1 style='text-align:center;color:#4B0082;'>📰 Fake News Detection & Analysis</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("About")
st.sidebar.info("""
This dashboard performs two functions:
1.  **Live Prediction:** Predicts if any news article is Real or Fake.
2.  **Dataset Analysis:** Shows an in-depth analysis of the training data.
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
# CREATE TABS
# --- UN-COMMENTED: Both tabs are now active ---
# -------------------------------
tab1, tab2 = st.tabs(["🔎 Live Prediction", "📊 Dataset Analysis"])

# -------------------------------
# TAB 1: LIVE PREDICTION
# -------------------------------
with tab1:
    st.header("🔎 Enter News for Prediction")
    
    # --- ADDED: "Test with Sample" buttons ---
    if not df.empty:
        st.subheader("Test with a Dataset Sample")
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Load Random REAL News Sample"):
            sample_text = df[df['label'] == 1].sample(1)['full_text'].iloc[0]
            st.session_state.user_input = sample_text
        if col_btn2.button("Load Random FAKE News Sample"):
            sample_text = df[df['label'] == 0].sample(1)['full_text'].iloc[0]
            st.session_state.user_input = sample_text
    
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
            text_clean = clean_text(user_input, for_ngrams=False)
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

# -------------------------------
# TAB 2: DATASET ANALYSIS
# --- UN-COMMENTED: This tab is now active ---
# -------------------------------
with tab2:
    st.header("📊 Training Dataset Analysis (Sample of 2000 Articles)")
    
    if df.empty:
        st.error("Could not load dataset. Check 'fake.csv' and 'true.csv' files.")
    else:
        st.info("This analysis is based on a sample of 1000 'Real' and 1000 'Fake' articles from the dataset.")
        
        # --- Section 1: Overview & Distribution ---
        st.subheader("1. Data Sample & Distribution")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df[['full_text', 'label']].head(10))
        with col2:
            dist_data = df['label'].value_counts().rename({0: 'Fake', 1: 'Real'})
            st.bar_chart(dist_data)
            st.write(f"**Total Articles in Sample:** {len(df)}")
            st.write(f"**Real News (Sample):** {dist_data.get('Real', 0)}")
            st.write(f"**Fake News (Sample):** {dist_data.get('Fake', 0)}")
        
        st.markdown("---")

        # --- Section 2: Text Statistics ---
        st.subheader("2. Comparative Text Statistics")
        if 'stats_df' not in st.session_state:
            real_stats = df[df['label'] == 1]['full_text']
            fake_stats = df[df['label'] == 0]['full_text']
            
            stats_data = {
                'Metric': ['Avg. Word Count', 'Avg. Sentence Length', 'Avg. Word Length'],
                'Real News': [
                    np.mean(real_stats.apply(lambda x: len(word_tokenize(x)))),
                    np.mean(real_stats.apply(lambda x: np.mean([len(word_tokenize(sent)) for sent in sent_tokenize(x)]) if sent_tokenize(x) else 0)),
                    np.mean(real_stats.apply(lambda x: np.mean([len(word) for word in word_tokenize(x)]) if word_tokenize(x) else 0))
                ],
                'Fake News': [
                    np.mean(fake_stats.apply(lambda x: len(word_tokenize(x)))),
                    np.mean(fake_stats.apply(lambda x: np.mean([len(word_tokenize(sent)) for sent in sent_tokenize(x)]) if sent_tokenize(x) else 0)),
                    np.mean(fake_stats.apply(lambda x: np.mean([len(word) for word in word_tokenize(x)]) if word_tokenize(x) else 0))
                ]
            }
            stats_df = pd.DataFrame(stats_data).set_index('Metric')
            st.session_state.stats_df = stats_df
        
        st.dataframe(st.session_state.stats_df.style.format("{:.2f}"))

        st.markdown("---")

        # --- Section 3: Word Clouds ---
        st.subheader("3. Word Cloud Analysis")
        col3, col4 = st.columns(2)
        with col3:
            real_text_wc = " ".join(df[df['label'] == 1]['cleaned_text_wc'].dropna())
            create_wordcloud(real_text_wc, "Real News Word Cloud")
        with col4:
            fake_text_wc = " ".join(df[df['label'] == 0]['cleaned_text_wc'].dropna())
            create_wordcloud(fake_text_wc, "Fake News Word Cloud")
            
        st.markdown("---")

        # --- Section 4: N-gram Analysis ---
        st.subheader("4. Common Phrase (N-gram) Analysis")
        st.info("This shows the most common 2-word (bigram) and 3-word (trigram) phrases.")
        
        col5, col6 = st.columns(2)
        
        # Join text for n-gram analysis
        real_text_ngrams = " ".join(df[df['label'] == 1]['cleaned_text_ngrams'].dropna())
        fake_text_ngrams = " ".join(df[df['label'] == 0]['cleaned_text_ngrams'].dropna())

        with col5:
            st.write("**Top 20 Bigrams (2-word) in Real News**")
            df_real_bi = get_top_ngrams(real_text_ngrams, n=2, top_k=20)
            st.dataframe(df_real_bi, use_container_width=True)
            
            st.write("**Top 20 Bigrams (2-word) in Fake News**")
            df_fake_bi = get_top_ngrams(fake_text_ngrams, n=2, top_k=20)
            st.dataframe(df_fake_bi, use_container_width=True)
        
        with col6:
            st.write("**Top 20 Trigrams (3-word) in Real News**")
            df_real_tri = get_top_ngrams(real_text_ngrams, n=3, top_k=20)
            st.dataframe(df_real_tri, use_container_width=True)

            st.write("**Top 20 Trigrams (3-word) in Fake News**")
            df_fake_tri = get_top_ngrams(fake_text_ngrams, n=3, top_k=20)
            st.dataframe(df_fake_tri, use_container_width=True)
