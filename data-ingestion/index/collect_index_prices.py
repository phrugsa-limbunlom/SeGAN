import yfinance as yf
import pandas as pd


def get_index_prices(indexes, start_date, end_date):
    # Fetch historical data for each index
    index_data = {}
    for index_name, ticker in indexes.items():
        data = yf.download(ticker, start=start_date, end=end_date)
        index_data[index_name] = data['Close']

    return index_data


if __name__ == '__main__':
    # Define the tickers for the indexes
    indexes = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ 100': '^NDX',
        'Nikkei 225': '^N225',
        'FTSE 100': '^FTSE',
        'DAX 30': '^GDAXI'
    }

    # Set the start and end dates for the data
    start_date = "2024-01-01"
    end_date = "2024-05-31"

    index_data = get_index_prices(indexes, start_date, end_date)

    # Convert the data to a DataFrame
    df_index_prices = pd.DataFrame(index_data)

    # Print the DataFrame
    print(df_index_prices.head())

    df_index_prices.to_csv('../../file/index/df_index_prices_test.csv')