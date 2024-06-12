import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("french"))
    tokens = [
        word
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def preprocess_data(df):
    df["processed_text"] = df["text"].apply(preprocess_text)
    return df


def split_data(df):
    return train_test_split(
        df["processed_text"], df["label"], test_size=0.2, random_state=42
    )


if __name__ == "__main__":
    filepath = "../data/reviews.csv"
    df = load_data(filepath)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train.to_csv("../data/X_train.csv", index=False)
    X_test.to_csv("../data/X_test.csv", index=False)
    y_train.to_csv("../data/y_train.csv", index=False)
    y_test.to_csv("../data/y_test.csv", index=False)
