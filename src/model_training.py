import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib


def load_data():
    X_train = pd.read_csv("./data/X_train.csv")
    y_train = pd.read_csv("./data/y_train.csv")
    return X_train, y_train


def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    classifier = LogisticRegression()
    pipeline = make_pipeline(vectorizer, classifier)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "./models/sentiment_model.pkl")


if __name__ == "__main__":
    X_train, y_train = load_data()
    train_model(X_train["processed_text"], y_train["label"])
