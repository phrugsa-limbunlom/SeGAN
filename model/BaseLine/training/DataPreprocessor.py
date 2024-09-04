import numpy as np
import pandas as pd


class DataPreprocessor:
    def __init__(self, X_scaler, y_scaler, file_path, sequence_length, time_horizon):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.time_horizon = time_horizon
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data

    def preprocess_data(self):
        data = self.load_data()
        dates = data['Date'].values
        features = data.drop(['CLOSE', 'Date'], axis=1).values
        labels = data['CLOSE'].values.reshape(-1, 1)

        scaled_features = self.X_scaler.transform(features)
        scaled_labels = self.y_scaler.transform(labels)

        X = []
        y = []
        for i in range(len(data) - self.sequence_length - self.time_horizon):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(scaled_labels[(i + self.sequence_length):(i + self.sequence_length + self.time_horizon)])
        return np.array(X), np.array(y), dates[self.sequence_length:len(data)-1]

    # def get_train_test_data(self, train_size=0.8):
    #     x, y = self.preprocess_data()
    #     train_size = int(len(x) * train_size)
    #     x_train, y_train = x[:train_size], y[:train_size]
    #     x_test, y_test = x[train_size:], y[train_size:]
    #     return x_train, y_train, x_test, y_test
