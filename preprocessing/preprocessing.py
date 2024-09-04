from pickle import dump

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocessing(df):
    # Fill Null values
    return df.fillna(0)


def preprocessing_data_filled(df, start, end, out_name):
    df['Date'] = pd.to_datetime(df['Date'])

    # Create a date range from the minimum to maximum timestamp
    date_range = pd.date_range(start=start, end=end, freq='D')

    # Set Date column to index
    df.set_index('Date', inplace=True)

    # Fill in date column with missing date
    df = df.reindex(date_range)

    # Interpolate Null value with time
    df = df.interpolate(method='time')

    df = df.round(2)

    # Revert index to Date column
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Date'})

    df.to_csv(out_name, index=False)

    return df


def transform_data(data):
    # Drop Date column to scale data
    data = data.drop(columns=["Date"])

    print(data.columns)

    # Extract features and target
    X_value = data[['HIGH', 'LOW', 'OPEN', 'COUNT', 'VOLUME',
                    'daily_sentiment_score', 'S&P 500', 'Dow Jones', 'NASDAQ 100',
                    'Nikkei 225', 'FTSE 100', 'DAX 30']].values

    y_value = data['CLOSE'].values

    # Normalized the data
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaler.fit(X_value)
    y_scaler.fit(y_value.reshape(-1, 1))

    dump(X_scaler, open('X_scaler_test.pkl', 'wb'))
    dump(y_scaler, open('y_scaler_test.pkl', 'wb'))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data)
    dump(scaler, open('scaler_test.pkl', 'wb'))

    return data


if __name__ == "__main__":
    # Train Data
    data = pd.read_csv("../file/incorporate/incorporate_index_sentiment_stock_test.csv")

    print("Before preprocessing")

    data = preprocessing(data)

    # Check Null
    print(data.isnull().sum())

    # Fill missing date
    file_out_name = "../file/preprocessed/preprocessed_date_filled_test.csv"
    start = '2024-01-01'
    end = '2024-05-31'
    data = preprocessing_data_filled(data, start, end, file_out_name)

    print("After preprocessing")

    # Count rows where 'daily_sentiment_score' is not 0.0
    non_zero_count = (data['daily_sentiment_score'] != 0.0).sum()
    print(f"Number of rows where 'daily_sentiment_score' is not 0.0: {non_zero_count}")

    # Transform data
    data = transform_data(data)
