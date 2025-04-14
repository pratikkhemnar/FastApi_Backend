import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data_fake = pd.read_csv(r"C:\Users\asus\Desktop\fackNewsDection\datasets\Fake.csv")
data_true = pd.read_csv(r"C:\Users\asus\Desktop\fackNewsDection\datasets\True.csv")

# Assign class labels
data_fake["class"] = 0
data_true["class"] = 1

# Merge datasets and drop unnecessary columns
data = pd.concat([data_fake, data_true], axis=0).drop(["title", "subject", "date"], axis=1)
data = data.sample(frac=1).reset_index(drop=True)

# Preprocess function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply preprocessing
data["text"] = data["text"].apply(wordopt)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data["text"], data["class"], test_size=0.1, random_state=42)

# Convert text to vectors
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)

# Train model
model = LogisticRegression()
model.fit(xv_train, y_train)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")
