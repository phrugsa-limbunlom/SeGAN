import numpy as np
import pandas as pd


class DataPreprocessor:
    @staticmethod
    def transform(data, X_scaler, y_scaler):
        X = data[['HIGH', 'LOW', 'OPEN', 'COUNT', 'VOLUME',
                  'daily_sentiment_score', 'S&P 500', 'Dow Jones', 'NASDAQ 100',
                  'Nikkei 225', 'FTSE 100', 'DAX 30']]

        X_scaled_data = X_scaler.transform(X.values)

        y = data["CLOSE"]

        y_scaled_data = y_scaler.transform(y.values.reshape(-1, 1))

        X = pd.DataFrame(X_scaled_data, columns=X.columns, index=X.index)

        y = pd.DataFrame(y_scaled_data, columns=["CLOSE"], index=y.index)

        return X, y

    @staticmethod
    def create_sequences(data, time_frame, time_horizon, features, output):
        sequences = []
        future_prices = []
        prices = []
        dates = data['Date'].values
        for i in range(len(data) - time_frame - time_horizon):
            sequences.append(features[i:i + time_frame])
            future_prices.append(output[i + time_frame + (time_horizon - 1)])
            prices.append(output[i:i + time_frame])
        return np.array(sequences), np.array(future_prices), np.array(prices), dates[time_frame:len(data)-1]
