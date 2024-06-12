import joblib


def predict_sentiment(text):
    pipeline = joblib.load("./models/sentiment_model.pkl")
    prediction = pipeline.predict([text])
    return prediction[0]


if __name__ == "__main__":
    text = input("Entrez un commentaire: ")
    sentiment = predict_sentiment(text)
    print(f"Le sentiment du commentaire est: {sentiment}")
