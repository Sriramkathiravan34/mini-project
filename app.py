import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Load your trained spam model
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model")

# Load sentiment model (fixed)
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

st.title("Spam Detection + Sentiment Analysis")

text = st.text_input("Enter message")

if st.button("Analyze"):
    if text:
        # Spam prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        spam = "Spam" if prediction == 1 else "Not Spam"

        # Sentiment
        sentiment = sentiment_model(text)[0]['label']

        st.write("Spam:", spam)
        st.write("Sentiment:", sentiment)