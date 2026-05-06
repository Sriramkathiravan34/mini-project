from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Load spam model
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model")

# Sentiment model
sentiment_model = pipeline("sentiment-analysis")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    spam = "Spam" if prediction == 1 else "Not Spam"
    sentiment = sentiment_model(text)[0]['label']

    return spam, sentiment