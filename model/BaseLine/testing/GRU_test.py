import logging
from pickle import load

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
from sklearn.metrics import root_mean_squared_error

from model.BaseLine.training.DataPreprocessor import DataPreprocessor
from model.BaseLine.training.GRUModel import GRUModel

if __name__ == "__main__":
    logging.basicConfig(filename='GRU_model_test.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    look_back = [7, 15, 30, 60]
    time_horizons = [1]

    folder = ""
    for time_horizon in time_horizons:
        for step in look_back:

            logging.info(f"Train on {step} days period")

            logging.info(f"Prediction {time_horizon} day/days ahead")

            if time_horizon == 1:
                folder = "daily"
            elif time_horizon == 7:
                folder = "weekly"
            elif time_horizon == 30:
                folder = "monthly"

            # Load scalers
            X_scaler = load(open('../../../preprocessing/X_scaler_test.pkl', 'rb'))
            y_scaler = load(open('../../../preprocessing/y_scaler_test.pkl', 'rb'))

            file_path = '../../../file/dataset/test_all.csv'

            df_test = pd.read_csv(file_path)

            # Data Preprocessing
            data_preprocessor = DataPreprocessor(X_scaler, y_scaler, file_path, sequence_length=step,
                                                 time_horizon=time_horizon)

            X_test, y_test, test_dates = data_preprocessor.preprocess_data()

            # Load the GRU model for predictions
            gru_model = GRUModel(input_shape=(X_test.shape[1], X_test.shape[2]), output_shape=1, y_scaler=y_scaler)
            gru_model.compile_model()
            gru_model.model.load_weights(
                f'../training/weight/{folder}/gru_model_{step}days_{time_horizon}days.weights.weights.h5')

            # Predict closing price
            predicted_closing_price = gru_model.model.predict(X_test)

            # Inverse transform the predicted and actual closing prices
            real_closing_price = y_scaler.inverse_transform(y_test.reshape(-1, 1))
            predicted_closing_price = y_scaler.inverse_transform(predicted_closing_price)

            print(y_test.shape)
            print(real_closing_price.shape)

            logging.info(f"(GRU) Predicted Closing Price: {predicted_closing_price}")

            # Evaluate model
            rmse = root_mean_squared_error(real_closing_price, predicted_closing_price)
            logging.info(f'GRU Model RMSE Testing set: {rmse:.3f}')

            test_dates = pd.to_datetime(test_dates)

            # Plot data
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.plot(test_dates, real_closing_price, label='Real Closing Prices')
            ax.plot(test_dates, predicted_closing_price, label='Predicted Closing Prices')
            ax.set(xlabel='Date',
                   ylabel='Closing Price (USD)',
                   title='Real vs Predicted Closing Prices (GRU)')
            date_form = DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            ax.legend(loc='upper right')

            plt.xticks(rotation=45)

            # Show the plot
            plt.savefig(f"../testing/figure/{folder}/gru_closing_prices_{step}days_{time_horizon}days.png")
            plt.show()
