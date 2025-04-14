from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
import train_model
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize FastAPI
app = FastAPI()

# Load trained model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Define input model
class NewsItem(BaseModel):
    text: str

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.get("/")
def home():
    return {"message": "Fake News Detection API is Running"}

@app.post("/predict/")
def predict(news: NewsItem):
    try:
        processed_text = wordopt(news.text)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)[0]
        return {"prediction": "Fake News" if prediction == 0 else "Not Fake News"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)  # Change port

