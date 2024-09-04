import ast

import numpy as np
import pandas as pd


def transform_score(row):
    if row['sentiment_label'] == 'Positive':
        return row['sentiment_score']
    elif row['sentiment_label'] == 'Negative':
        return -row['sentiment_score']
    elif row['sentiment_label'] == 'Neutral':
        return 0


def incorporate(sentiment_df, stock_df, out_name):
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date

    # Extract the date part from the datetime strings
    sentiment_df['Date'] = sentiment_df['versionCreated'].str[:10]

    # Convert string to date
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date

    # Convert string representations of dictionaries to actual dictionaries
    sentiment_df['sentiment'] = sentiment_df['sentiment'].apply(ast.literal_eval)

    # Extract sentiment score
    sentiment_df['sentiment_score'] = sentiment_df['sentiment'].apply(lambda x: x['score'])

    # Extract sentiment label
    sentiment_df['sentiment_label'] = sentiment_df['sentiment'].apply(lambda x: x['label'])

    sentiment_df['transformed_score'] = sentiment_df.apply(transform_score, axis=1)

    # Aggregate sentiment scores to daily level
    daily_sentiment = sentiment_df.groupby('Date')['transformed_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'daily_sentiment_score']

    # Display the aggregated daily sentiment scores
    print(daily_sentiment.head())

    daily_sentiment.to_csv(out_name, index=False)

    # Merge two dataframe
    merged_df = pd.merge(stock_df, daily_sentiment, on='Date', how='left')

    return merged_df


def clean_up(merged_df, out_name):
    if 'daily_sentiment_score_x' in merged_df.columns and 'daily_sentiment_score_y' in merged_df.columns:
        merged_df['daily_sentiment_score'] = np.where(merged_df['daily_sentiment_score_y'].isna(),
                                                      merged_df['daily_sentiment_score_x'],
                                                      merged_df['daily_sentiment_score_y'])

        merged_df.drop(['daily_sentiment_score_x', 'daily_sentiment_score_y'], axis=1, inplace=True)

    merged_df.to_csv(out_name, index=False)

    return merged_df


if __name__ == "__main__":
    stock_daily = pd.read_csv("../file/stock/stock_daily_test.csv")

    news_sentiment = pd.read_csv("../file/sentiment/sentiment_test.csv")

    file_out_name = "../file/sentiment/sentiment_to_daily_level_test.csv"
    merged_df = incorporate(news_sentiment, stock_daily, file_out_name)

    clean_up(merged_df, "../file/incorporate/sentiment_stock_incorporate_test.csv")
