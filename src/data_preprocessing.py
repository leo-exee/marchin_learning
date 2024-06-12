import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.fr.stop_words import STOP_WORDS as stopwords
import spacy
import string

nlp = spacy.load("fr_core_news_sm")


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stopwords and token.text not in string.punctuation
    ]
    return " ".join(tokens)


def preprocess_data(df):
    df["processed_text"] = df["text"].apply(preprocess_text)
    return df


def split_data(df):
    return train_test_split(
        df["processed_text"], df["label"], test_size=0.2, random_state=42
    )


if __name__ == "__main__":
    filepath = "./data/data.csv"
    df = load_data(filepath)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train.to_csv("./data/X_train.csv", index=False)
    X_test.to_csv("./data/X_test.csv", index=False)
    y_train.to_csv("./data/y_train.csv", index=False)
    y_test.to_csv("./data/y_test.csv", index=False)
