import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_data():
    X_test = pd.read_csv("./data/X_test.csv")
    y_test = pd.read_csv("./data/y_test.csv")
    return X_test, y_test


def evaluate_model(X_test, y_test):
    pipeline = joblib.load("./models/sentiment_model.pkl")
    predictions = pipeline.predict(X_test.squeeze())
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    X_test, y_test = load_data()
    evaluate_model(X_test, y_test)
