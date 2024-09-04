import pandas as pd


def incorporate(sentiment_stock, index_price):
    sentiment_stock['Date'] = pd.to_datetime(sentiment_stock['Date'])

    index_price['Date'] = pd.to_datetime(index_price['Date'])

    # Merge two dataframe
    merged_df = pd.merge(sentiment_stock, index_price, on='Date', how='left')

    # Set Date column to index
    merged_df.set_index('Date', inplace=True)

    # Interpolate Null value with time
    merged_df = merged_df.interpolate(method='time')

    # Fill in NaN
    merged_df = merged_df.fillna(0)

    merged_df = merged_df.round(2)

    # Revert index to Date column
    merged_df.reset_index(inplace=True)

    return merged_df


if __name__ == "__main__":
    sentiment_stock = pd.read_csv("../file/incorporate/sentiment_stock_incorporate_test.csv")

    index_price = pd.read_csv("../file/index/df_index_prices_test.csv")

    merged_df = incorporate(sentiment_stock, index_price)

    merged_df.to_csv("../file/incorporate/incorporate_index_sentiment_stock_test.csv", index=False)