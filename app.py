# fake_news_api.py

# Install necessary libraries
# pip install fastapi uvicorn pandas scikit-learn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Initialize FastAPI
app = FastAPI()

# Define a Pydantic model for input validation
class NewsItem(BaseModel):
    text: str

# Load and preprocess data
data_fake = pd.read_csv(r"C:\Users\asus\Desktop\fackNewsDection\datasets\Fake.csv")
data_true = pd.read_csv(r"C:\Users\asus\Desktop\fackNewsDection\datasets\True.csv")

data_fake["class"] = 0
data_true['class'] = 1

data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Randomly shuffle the dataframe
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

# Preprocessing Text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

# Define dependent and independent variables
x = data['text']
y = data['class']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Convert text to vectors
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train models
LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)

RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

# Function to output label
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# FastAPI Endpoint for Prediction
@app.post("/predict")
def predict_news(news_item: NewsItem):
    try:
        # Preprocess the input news text
        news_text = news_item.text
        testing_news = {"text": [news_text]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test['text'] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)

        # Make predictions
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GB = GB.predict(new_xv_test)
        pred_RF = RF.predict(new_xv_test)

        # Prepare the result
        result = {
            "LR Prediction": output_label(pred_LR[0]),
            "DT Prediction": output_label(pred_DT[0]),
            "GBC Prediction": output_label(pred_GB[0]),
            "RFC Prediction": output_label(pred_RF[0])
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)