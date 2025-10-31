📰 Fake News Detection Dashboard

This project is a Streamlit web application built for a university mini-project. It uses a Deep Learning (LSTM) model (trained with Keras/TensorFlow) to classify news articles as "Real" or "Fake".

The app provides a simple interface for live prediction and is deployed on Streamlit Community Cloud.

✨ Features

Live Prediction: A text area where any news article can be pasted for instant classification.

Prediction Confidence: A bar chart that displays the model's confidence in its prediction (i.e., the probability of "Fake" vs. "Real").

Input WordCloud: A dynamic WordCloud generated from the text that was entered for prediction.

Session History: A table that keeps a running list of all predictions made during the current session.

Model Architecture: The sidebar displays the structure of the loaded Keras model, showing its layers and parameters.

🚀 How to Run

There are two ways to run this project:

Method 1: View the Live Deployed App (Easiest)

The application is deployed on Streamlit Community Cloud and can be accessed directly from any web browser:

[Enter Your Streamlit App URL Here, e.g., https://www.google.com/search?q=https://fake-news-detection-app.streamlit.app]

Method 2: Run Locally (For Faculty)

To run the project on your local machine, please follow these steps.

1. Prerequisites

Python 3.8 - 3.11

Git

Git LFS (This is required to download the large model file)

2. Clone the Repository

First, clone the repository to your local machine.

# Clone the repository
git clone [Your_GitHub_Repository_URL_Here]

# Navigate into the project folder
cd Fake-News-Detection


3. Install Git LFS & Pull Model

This project uses Git LFS to handle the large model file (.h5). You must have Git LFS installed.

# Install Git LFS (if you haven't already)
git lfs install

# Pull the large files from the repository
git lfs pull


After this step, you should see the fake_news_model.h5 file in your folder (it should be ~90MB, not 1KB).

4. Install Requirements

Install all the necessary Python libraries from the requirements.txt file.

pip install -r requirements.txt


(This will install Streamlit, TensorFlow, Pandas, NLTK, etc.)

5. Run the Streamlit App

You are all set. Run the following command in your terminal to launch the app.

streamlit run app.py


The application will automatically open in your default web browser at http://localhost:8501.

📁 Project Structure

.
├── 📄 app.py                     # The main Streamlit application code
├── 📦 fake_news_model.h5         # The trained Keras (LSTM) model
├── 📦 tokenizer.pkl                # The Keras tokenizer file
├── 📜 model-final-training.ipynb  # Jupyter Notebook with the model training process
├── 📝 requirements.txt             # List of all Python dependencies
├── 📊 fake.csv                    # Original training data (for reference)
├── 📊 true.csv                    # Original training data (for reference)
└── 📖 README.md                    # This file
