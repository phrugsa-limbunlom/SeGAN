import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
from statsmodels.tsa.stattools import grangercausalitytests


def stock_price_movement(date, closing_price, label, title):
    # Plot data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date, closing_price, label=label)
    ax.set(xlabel="Date",
           ylabel="Closing Price (USD)",
           title=title)
    date_form = DateFormatter("%Y")
    ax.xaxis.set_major_formatter(date_form)
    plt.savefig("stock_price_movement.png")
    plt.show()


def stock_price_and_sentiment_score_correlation(df):
    # Calculate correlation matrices
    correlation = df[['daily_return', 'daily_sentiment_score']].corr()
    logging.info("Correlation Matrix:")
    logging.info(correlation)

    # Calculate the correlation between today's return and next day's sentiment score
    df['next_day_sentiment_score'] = df['daily_sentiment_score'].shift(-1)
    lagged_correlation = df[['daily_return', 'next_day_sentiment_score']].corr()
    logging.info("Lagged Correlation Matrix (sentiment score):")
    logging.info(lagged_correlation)

    # Calculate the correlation between today's sentiment score and next day's return
    df['next_day_return'] = df['daily_return'].shift(-1)
    correlation_matrix = df[['daily_sentiment_score', 'next_day_return']].corr()
    logging.info("Correlation Matrix of Today's Sentiment Score and Next Day's Return:")
    logging.info(correlation_matrix)

    # Set the style
    sns.set(style="whitegrid")

    # Plot correlation between daily_return and daily_sentiment_score
    plt.figure(figsize=(10, 6))
    sns.regplot(x='daily_sentiment_score', y='daily_return', data=df, scatter_kws={"color": "lightblue"},
                line_kws={"color": "blue"})
    plt.title('Correlation between Daily Return and Sentiment Score')
    plt.xlabel('Daily Sentiment Score')
    plt.ylabel('Daily Return')
    plt.savefig("stock_price_and_sentiment_score_correlation.png")
    plt.show()

    # Plot correlation between today return and next day sentiment score
    plt.figure(figsize=(10, 6))
    sns.regplot(x='next_day_sentiment_score', y='daily_return', data=df, scatter_kws={"color": "lightpink"},
                line_kws={"color": "red"})
    plt.title('Lagged Correlation between Next Daily Sentiment Score and Today Return (1 Day)')
    plt.xlabel('Daily Sentiment Score of the next date')
    plt.ylabel('Daily Return')
    plt.savefig("next_day_sentiment_and_today_return_correlation.png")
    plt.show()

    # Plot correlation between today sentiment score and next day return
    plt.figure(figsize=(10, 6))
    sns.regplot(x='daily_sentiment_score', y='next_day_return', data=df, scatter_kws={"color": "lightgreen"},
                line_kws={"color": "green"})
    plt.title('Lagged Correlation between Next Daily Return and Today Sentiment Score (1 Day)')
    plt.xlabel('Daily Sentiment Score')
    plt.ylabel('Daily Return of the next date')
    plt.savefig("next_day_return_and_today_sentiment_correlation.png")
    plt.show()

    # Perform Granger causality tests
    logging.info("Granger Causality Tests:")

    # Granger causality test to see if sentiment scores predict daily returns
    granger_result_1 = grangercausalitytests(df[['daily_return', 'daily_sentiment_score']].dropna(), maxlag=5)
    logging.info("\nGranger Causality Test Results (Sentiment Score -> Daily Return):")
    for key in granger_result_1:
        logging.info(f"Lag {key}:")
        logging.info(granger_result_1[key][0]['ssr_chi2test'])
        logging.info(granger_result_1[key][0]['params_ftest'])

    # Granger causality test to see if daily returns predict sentiment scores
    granger_result_2 = grangercausalitytests(df[['daily_sentiment_score', 'daily_return']].dropna(), maxlag=5)
    logging.info("\nGranger Causality Test Results (Daily Return -> Sentiment Score):")
    for key in granger_result_2:
        logging.info(f"Lag {key}:")
        logging.info(granger_result_2[key][0]['ssr_chi2test'])
        logging.info(granger_result_2[key][0]['params_ftest'])


if __name__ == "__main__":
    logging.basicConfig(filename='correlation.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    data = pd.read_csv("../file/dataset/train_all.csv")

    logging.info(data.head())

    # Convert String to Date
    data['Date'] = pd.to_datetime(data['Date'])

    logging.info(data.columns)

    # Visualize closing price of Microsoft stock from 2013-2023
    stock_price_movement(data['Date'], data['CLOSE'], 'Microsoft stock', "Microsoft Stock Price (2013-2023)")

    data = pd.read_csv("../file/dataset/train_with_sentiment_score.csv")

    # Convert String to Date
    data['Date'] = pd.to_datetime(data['Date'])

    # Calculate daily returns
    data['daily_return'] = data['CLOSE'].pct_change()

    data.dropna(inplace=True)  # Drop rows with NaN values that result from the percentage change calculation

    # Visualize correlation between daily return and sentiment score
    stock_price_and_sentiment_score_correlation(data)
