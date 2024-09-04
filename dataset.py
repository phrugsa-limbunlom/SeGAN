import pandas as pd


def with_sentiment(df):
    return df[df['daily_sentiment_score'] != 0.0]


def without_sentiment(df):
    return df.drop(['daily_sentiment_score'], axis=1)  # Drop sentiment score column


if __name__ == "__main__":
    data = pd.read_csv("file/preprocessed/preprocessed_date_filled_test.csv")

    data.to_csv("file/dataset/test_all.csv", index=False)

    data_with_sentiment = with_sentiment(data)
    data_with_sentiment.to_csv("file/dataset/test_with_sentiment_score.csv", index=False)

    data_without_sentiment = data.drop(['daily_sentiment_score'], axis=1)
    data_without_sentiment.to_csv("file/dataset/test_without_sentiment_score.csv", index=False)
